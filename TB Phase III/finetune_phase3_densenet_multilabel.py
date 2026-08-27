"""
Fine-tune torchvision DenseNet121 end-to-end on DICOMs (multi-label BCE).

Uses the same spatial preprocessing as build_phase3_features.py (window→[0,1], optional CLAHE, optional lung crop).
Paths and label order come from an existing Phase III .npz (reference_npz); pixel tensors are rebuilt at --image-size (default 448).

Optimizer: AdamW with separate learning rates for backbone (features) vs classifier head.
Loss: BCEWithLogitsLoss with per-label pos_weight from the train fold (computed once per CV run).

Example (CLAHE, 448, D1–D6, 5-fold):
  python finetune_phase3_densenet_multilabel.py \\
    --reference_npz "<REDACTED_PATH> Phase III\\phase3_features_active_all_260428_d5kw_fix_densenet121_clahe_v27.npz" \\
    --clahe --pretrained --force_close_zero --num_labels 6 \\
    --out_dir "<REDACTED_PATH> Phase III\\artifacts\\phase3_finetune_densenet121_clahe_448_d1_d6"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import build_phase3_features as bpf  # noqa: E402


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def mean_std(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "n": int(a.size),
    }


def make_splits(
    strat_labels: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(strat_labels)
    if n_train + n_val + n_test != n:
        raise ValueError(f"Split count mismatch: {n_train}+{n_val}+{n_test}!={n}")

    all_idx = np.arange(n)
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=int(seed))
    trva_idx, te_idx = next(s1.split(all_idx, strat_labels))

    val_ratio = n_val / (n_train + n_val)
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=int(seed) + 17)
    tr_sub, va_sub = next(s2.split(trva_idx, strat_labels[trva_idx]))
    tr_idx = trva_idx[tr_sub]
    va_idx = trva_idx[va_sub]
    return tr_idx, va_idx, te_idx


def make_phase3_strata(Y6: np.ndarray) -> np.ndarray:
    bits = np.array(["".join(str(int(v)) for v in row.tolist()) for row in Y6], dtype=object)
    _, counts = np.unique(bits, return_counts=True)
    if counts.min() >= 2:
        return bits
    return Y6[:, 5].astype(int)


def load_y6_from_npz(npz_path: Path, force_close_zero: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    d = np.load(npz_path, allow_pickle=True)
    paths = np.asarray(d["paths"], dtype=object)
    Y = np.asarray(d["Y"], dtype=int)
    D5 = np.asarray(d["D5"], dtype=int).ravel()
    if Y.shape[1] != 5:
        raise ValueError(f"Expected Y (N,5), got {Y.shape}")
    Y6 = np.column_stack([Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], D5, Y[:, 4]]).astype(int)

    close_count = 0
    if force_close_zero:
        close_mask = np.array([("close" in Path(str(p)).stem.lower()) for p in paths], dtype=bool)
        close_count = int(close_mask.sum())
        Y6 = Y6.copy()
        Y6[close_mask, :] = 0

    return paths, Y6, close_count


def per_label_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    y = y_train.astype(np.float64)
    n = y.shape[0]
    pos = y.sum(axis=0)
    neg = n - pos
    pw = neg / np.maximum(pos, 1.0)
    return torch.tensor(pw, dtype=torch.float32)


class Phase3DicomDataset(Dataset):
    def __init__(
        self,
        indices: np.ndarray,
        paths: np.ndarray,
        Y: np.ndarray,
        *,
        lung_inferer,
        lung_crop: bool,
        lung_margin: float,
        use_clahe: bool,
        clahe_clip_limit: float,
        clahe_kernel_size: int | None,
        robust_norm: str,
        robust_norm_eps: float,
        robust_norm_clip: float | None,
        robust_norm_order: str,
        transform: torch.nn.Module,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.paths = paths
        self.Y = np.asarray(Y, dtype=np.float32)
        self.lung_inferer = lung_inferer
        self.lung_crop = lung_crop
        self.lung_margin = lung_margin
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_kernel_size = clahe_kernel_size
        self.robust_norm = robust_norm
        self.robust_norm_eps = robust_norm_eps
        self.robust_norm_clip = robust_norm_clip
        self.robust_norm_order = robust_norm_order
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def _preprocess_array(self, p: Path) -> np.ndarray:
        img = bpf.dicom_to_float01(
            p,
            lung_inferer=self.lung_inferer,
            lung_crop=self.lung_crop,
            lung_margin=self.lung_margin,
        )
        rn_on = (self.robust_norm or "").strip().lower() not in ("", "none", "off")
        rn_order = (self.robust_norm_order or "").strip().lower()
        if rn_on and rn_order == "pre_clahe":
            img = bpf.robust_norm_per_image(
                img, method=self.robust_norm, eps=float(self.robust_norm_eps), clip=self.robust_norm_clip
            )
            mn = float(np.min(img))
            mx = float(np.max(img))
            img = (img - mn) / max(mx - mn, 1e-6)
        if self.use_clahe:
            ks = self.clahe_kernel_size
            img = bpf.apply_clahe_float01(img, clip_limit=float(self.clahe_clip_limit), kernel_size=ks)
        if rn_on and rn_order == "post_clahe":
            img = bpf.robust_norm_per_image(
                img, method=self.robust_norm, eps=float(self.robust_norm_eps), clip=self.robust_norm_clip
            )
        return img

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        p = Path(str(self.paths[idx]))
        img = self._preprocess_array(p)
        x = self.transform(img).float()
        y = torch.from_numpy(self.Y[idx]).float()
        return x, y


def build_model(num_labels: int, pretrained: bool) -> nn.Module:
    w = models.DenseNet121_Weights.DEFAULT if pretrained else None
    m = models.densenet121(weights=w)
    in_f = m.classifier.in_features
    m.classifier = nn.Linear(in_f, int(num_labels))
    return m


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        with autocast(enabled=use_amp):
            logits = model(xb)
        prob = torch.sigmoid(logits).float().cpu().numpy()
        probs_all.append(prob)
        y_all.append(yb.numpy())
    return np.concatenate(probs_all, axis=0), np.concatenate(y_all, axis=0)


def train_one_fold(
    *,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    paths: np.ndarray,
    Y: np.ndarray,
    label_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
    lung_inferer,
    tfm,
    seed_run: int,
) -> dict:
    num_labels = len(label_names)
    pw = per_label_pos_weight(Y[tr_idx][:, :num_labels]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    ds_tr = Phase3DicomDataset(
        tr_idx,
        paths,
        Y,
        lung_inferer=lung_inferer,
        lung_crop=args.lung_crop,
        lung_margin=args.lung_margin,
        use_clahe=args.clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_kernel_size=args.clahe_kernel_size if args.clahe_kernel_size > 0 else None,
        robust_norm=args.robust_norm,
        robust_norm_eps=args.robust_norm_eps,
        robust_norm_clip=None if args.robust_norm_clip == 0.0 else float(args.robust_norm_clip),
        robust_norm_order=args.robust_norm_order,
        transform=tfm,
    )
    ds_va = Phase3DicomDataset(
        va_idx,
        paths,
        Y,
        lung_inferer=lung_inferer,
        lung_crop=args.lung_crop,
        lung_margin=args.lung_margin,
        use_clahe=args.clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_kernel_size=args.clahe_kernel_size if args.clahe_kernel_size > 0 else None,
        robust_norm=args.robust_norm,
        robust_norm_eps=args.robust_norm_eps,
        robust_norm_clip=None if args.robust_norm_clip == 0.0 else float(args.robust_norm_clip),
        robust_norm_order=args.robust_norm_order,
        transform=tfm,
    )
    ds_te = Phase3DicomDataset(
        te_idx,
        paths,
        Y,
        lung_inferer=lung_inferer,
        lung_crop=args.lung_crop,
        lung_margin=args.lung_margin,
        use_clahe=args.clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_kernel_size=args.clahe_kernel_size if args.clahe_kernel_size > 0 else None,
        robust_norm=args.robust_norm,
        robust_norm_eps=args.robust_norm_eps,
        robust_norm_clip=None if args.robust_norm_clip == 0.0 else float(args.robust_norm_clip),
        robust_norm_order=args.robust_norm_order,
        transform=tfm,
    )

    nw = int(args.num_workers)
    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=device.type == "cuda")
    loader_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=device.type == "cuda")
    loader_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=device.type == "cuda")

    torch.manual_seed(seed_run)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed_run)

    model = build_model(num_labels, pretrained=args.pretrained).to(device)

    opt = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": float(args.backbone_lr)},
            {"params": model.classifier.parameters(), "lr": float(args.head_lr)},
        ],
        weight_decay=float(args.weight_decay),
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(args.epochs))
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    use_amp = bool(args.amp and device.type == "cuda")

    best_state = None
    best_score = -1.0

    for ep in range(int(args.epochs)):
        model.train()
        for xb, yb in tqdm(loader_tr, desc=f"ep{ep+1}/{args.epochs}", leave=False):
            xb = xb.to(device)
            yb = yb[:, :num_labels].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        sched.step()

        p_va, y_va = predict_probs(model, loader_va, device, use_amp)
        run_aucs = []
        for j in range(num_labels):
            run_aucs.append(safe_auc(y_va[:, j].astype(int), p_va[:, j]))
        macro = float(np.nanmean(np.asarray(run_aucs, dtype=float)))

        if np.isfinite(macro) and macro > best_score:
            best_score = macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    p_te, y_te = predict_probs(model, loader_te, device, use_amp)
    test_aucs: dict[str, float] = {}
    for j, name in enumerate(label_names):
        test_aucs[name] = safe_auc(y_te[:, j].astype(int), p_te[:, j])

    macro_vals = np.array(list(test_aucs.values()), dtype=float)
    macro_vals = macro_vals[np.isfinite(macro_vals)]
    macro_mean = float(np.mean(macro_vals)) if macro_vals.size else float("nan")

    return {
        "seed_run": int(seed_run),
        "best_val_macro_auroc": float(best_score),
        "test_auroc_by_label": test_aucs,
        "test_macro_auroc": macro_mean,
        "pos_weight_train_fold": pw.detach().cpu().numpy().tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune DenseNet121 multi-label on Phase III DICOMs")
    ap.add_argument("--reference_npz", type=Path, required=True, help="NPZ with paths,Y,D5 for cohort order (X ignored)")
    ap.add_argument("--num_labels", type=int, default=6, choices=(5, 6), help="5 = D1..D5 columns of Y6; 6 = full Y6")
    ap.add_argument("--cv_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--force_close_zero", action="store_true")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--backbone_lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--clahe_clip_limit", type=float, default=0.03)
    ap.add_argument("--clahe_kernel_size", type=int, default=0, help="0 = skimage default tile size")
    ap.add_argument("--lung_crop", action="store_true")
    ap.add_argument("--lung_margin", type=float, default=0.15)
    ap.add_argument("--lungmask_force_cpu", action="store_true")
    ap.add_argument("--robust_norm", type=str, default="none")
    ap.add_argument("--robust_norm_eps", type=float, default=1e-6)
    ap.add_argument("--robust_norm_clip", type=float, default=5.0)
    ap.add_argument("--robust_norm_order", type=str, default="post_clahe")
    ap.add_argument("--image_size", type=int, default=448)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="Mixed precision (CUDA only)")
    ap.add_argument("--pipeline_version", type=str, default="", help="Tag stored in JSON output")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    if args.pretrained:
        bpf.configure_https_with_certifi()

    paths, Y6, close_count = load_y6_from_npz(args.reference_npz, force_close_zero=bool(args.force_close_zero))
    n = len(paths)
    if args.num_labels == 5:
        Y = Y6[:, :5].astype(np.float32)
        label_names = ["D1", "D2", "D3", "D4", "D5"]
    else:
        Y = Y6.astype(np.float32)
        label_names = ["D1", "D2", "D3", "D4", "D5", "D6"]

    n_train = int(round(n * float(args.train_ratio)))
    n_val = int(round(n * float(args.val_ratio)))
    n_test = int(n - n_train - n_val)
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Invalid split counts")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device != "auto":
        device = torch.device(args.device)

    use_imagenet_norm = (args.robust_norm or "").strip().lower() in ("", "none", "off")
    tfm = bpf.default_transform(args.image_size, use_imagenet_norm=use_imagenet_norm)

    lung_inferer = None
    if args.lung_crop:
        try:
            from lungmask import LMInferer  # type: ignore[import-untyped]
        except ImportError as e:
            raise SystemExit("lungmask required for --lung-crop") from e
        use_lm_cuda = torch.cuda.is_available() and not args.lungmask_force_cpu
        lung_inferer = LMInferer(force_cpu=not use_lm_cuda, tqdm_disable=True)

    strata = make_phase3_strata(Y6)
    per_run: list[dict] = []
    by_label: dict[str, list[float]] = {k: [] for k in label_names}
    macro_runs: list[float] = []

    for r in range(args.cv_runs):
        rs = int(args.seed + 222 + r * 1009)
        tr, va, te = make_splits(strata, n_train, n_val, n_test, rs)
        out_run = train_one_fold(
            tr_idx=tr,
            va_idx=va,
            te_idx=te,
            paths=paths,
            Y=Y,
            label_names=label_names,
            args=args,
            device=device,
            lung_inferer=lung_inferer,
            tfm=tfm,
            seed_run=rs + 999,
        )
        per_run.append({"run": r + 1, **out_run})
        macro_runs.append(out_run["test_macro_auroc"])
        for k in label_names:
            by_label[k].append(out_run["test_auroc_by_label"][k])

    pv = (args.pipeline_version or "").strip()

    out = {
        "pipeline_train": "densenet121_finetune_multilabel_bce",
        "pipeline_version_tag": pv or None,
        "config": {
            "reference_npz": str(args.reference_npz),
            "num_labels": int(args.num_labels),
            "image_size": int(args.image_size),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "backbone_lr": float(args.backbone_lr),
            "head_lr": float(args.head_lr),
            "weight_decay": float(args.weight_decay),
            "pretrained": bool(args.pretrained),
            "clahe": bool(args.clahe),
            "lung_crop": bool(args.lung_crop),
            "robust_norm": str(args.robust_norm),
            "force_close_zero": bool(args.force_close_zero),
            "cv_runs": int(args.cv_runs),
            "split_counts_train_val_test": [n_train, n_val, n_test],
            "device": str(device),
            "amp": bool(args.amp),
        },
        "cohort": {"n_total": n, "close_count": close_count},
        "phase3": {
            "test_auroc_by_label": {k: mean_std(v) for k, v in by_label.items()},
            "test_macro_auroc": mean_std(macro_runs),
            "per_run": per_run,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jp = args.out_dir / "phase3_finetune_densenet_test_metrics.json"
    tp = args.out_dir / "phase3_finetune_densenet_test_summary.txt"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    p3 = out["phase3"]
    lines = [
        "DenseNet121 fine-tune (multi-label BCE), TEST",
        f"image_size: {args.image_size}, num_labels: {args.num_labels}",
        f"Macro AUROC: {p3['test_macro_auroc']['mean']:.4f} ± {p3['test_macro_auroc']['std']:.4f}",
    ]
    for lab in label_names:
        d = p3["test_auroc_by_label"][lab]
        lines.append(f"{lab} AUROC: {d['mean']:.4f} ± {d['std']:.4f}")
    tp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {jp}")
    print(f"saved: {tp}")


if __name__ == "__main__":
    main()

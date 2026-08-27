"""
Phase III v3.5 — 4-stage pipeline (DenseNet121 @1024 stretch + CLAHE → partial unfreeze → RF D1-D5).

Stage 1 (Warm-up): DenseNet frozen. Only a linear head is trained on TB labels (BCE).
Stage 2 (Partial Unfreeze): Unfreeze last DenseBlock (denseblock4 + norm5) + head.
         Backbone LR << Head LR to avoid destroying pretrained knowledge.
Stage 3 (Feature extraction): Fine-tuned backbone re-extracts features for the full cohort.
Stage 4 (RF): Train RF D1-D5 balanced on new features → same eval as v3.2 for fair comparison.

Comparable baseline:
  v3.2 — DenseNet121 @1024 stretch CLAHE, RF D1-D5, class_weight=balanced
  Macro AUROC D1-D5: 0.8456 ± 0.0103

Pre-caching:
  On first run the script converts all DICOMs (CLAHE + resize) to .npy files in --cache_dir.
  Subsequent runs load from cache (~10-50x faster than on-the-fly DICOM loading per epoch).

Usage (single run, 5-fold CV):
  python train_phase3_v35_finetune_then_rf.py \\
      --embed_root "<REDACTED_PATH>" \\
      --reference_npz "<REDACTED_PATH> Phase III\\phase3_features_active_all_260428_d5kw_fix_densenet121_clahe_1024_v32.npz" \\
      --cache_dir "<REDACTED_PATH> Phase III\\cache_v35_1024_clahe" \\
      --out_dir "<REDACTED_PATH> Phase III\\artifacts\\phase3_active_cv_v35_finetune_rf_1024_d1_d5"
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

import build_phase3_features as bpf  # reuse preprocess helpers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        raise ValueError(f"Split mismatch: {n_train}+{n_val}+{n_test}!={n}")
    all_idx = np.arange(n)
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=int(seed))
    trva_idx, te_idx = next(s1.split(all_idx, strat_labels))
    val_ratio = n_val / (n_train + n_val)
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=int(seed) + 17)
    tr_sub, va_sub = next(s2.split(trva_idx, strat_labels[trva_idx]))
    return trva_idx[tr_sub], trva_idx[va_sub], te_idx


def make_phase3_strata(Y6: np.ndarray) -> np.ndarray:
    bits = np.array(["".join(str(int(v)) for v in row.tolist()) for row in Y6], dtype=object)
    _, counts = np.unique(bits, return_counts=True)
    if counts.min() >= 2:
        return bits
    return Y6[:, 5].astype(int)


def load_y6_paths(npz_path: Path, force_close_zero: bool) -> tuple[np.ndarray, np.ndarray, int]:
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


def per_label_pos_weight(y_tr: np.ndarray) -> torch.Tensor:
    y = y_tr.astype(np.float64)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pw = neg / np.maximum(pos, 1.0)
    return torch.tensor(pw, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Image pre-cache (DICOM → CLAHE+resize → .npy)
# ---------------------------------------------------------------------------

def _cache_key(path: Path, image_size: int, clahe: bool, clip: float) -> str:
    stem = hashlib.md5(str(path).encode()).hexdigest()[:12]
    return f"{stem}_s{image_size}_cl{int(clahe)}_clip{clip:.4f}.npy"


def build_image_cache(
    paths: np.ndarray,
    cache_dir: Path,
    *,
    tfm,
    clahe: bool,
    clahe_clip_limit: float,
    image_size: int,
) -> list[Path]:
    """
    Pre-process all DICOMs → save as (3, H, W) float32 .npy tensors.
    Returns a list of cache file paths parallel to `paths`.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths: list[Path] = []
    need_build: list[int] = []

    for i, p in enumerate(paths):
        cp = cache_dir / _cache_key(Path(str(p)), image_size, clahe, clahe_clip_limit)
        cache_paths.append(cp)
        if not cp.exists():
            need_build.append(i)

    if need_build:
        print(f"[Cache] Building {len(need_build)}/{len(paths)} image cache files in {cache_dir} …")
        for idx in tqdm(need_build, desc="pre-cache", unit="img"):
            p = Path(str(paths[idx]))
            img = bpf.dicom_to_float01(p)
            if clahe:
                img = bpf.apply_clahe_float01(img, clip_limit=float(clahe_clip_limit), kernel_size=None)
            tensor = tfm(img)  # (3, H, W) float
            np.save(str(cache_paths[idx]), tensor.numpy().astype(np.float32))
    else:
        print(f"[Cache] All {len(paths)} cache files found. Skipping preprocessing.")

    return cache_paths


# ---------------------------------------------------------------------------
# Dataset — loads from .npy cache (fast)
# ---------------------------------------------------------------------------

class CachedDataset(Dataset):
    """Loads pre-cached (3, H, W) float32 .npy tensors — no DICOM I/O per epoch."""

    def __init__(
        self,
        indices: np.ndarray,
        cache_paths: list[Path],
        Y: np.ndarray,
    ) -> None:
        self.indices = indices.astype(np.int64)
        self.cache_paths = cache_paths
        self.Y = np.asarray(Y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = torch.from_numpy(np.load(str(self.cache_paths[idx]))).float()
        y = torch.from_numpy(self.Y[idx]).float()
        return x, y


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(num_labels: int, pretrained: bool) -> models.DenseNet:
    w = models.DenseNet121_Weights.DEFAULT if pretrained else None
    m = models.densenet121(weights=w)
    m.classifier = nn.Linear(m.classifier.in_features, int(num_labels))
    return m


def freeze_backbone(model: models.DenseNet) -> None:
    for p in model.features.parameters():
        p.requires_grad_(False)
    for p in model.classifier.parameters():
        p.requires_grad_(True)


def partial_unfreeze_last_denseblock(model: models.DenseNet) -> None:
    """
    Unfreeze:
      - features.denseblock4  (last dense block)
      - features.norm5        (final BN after features)
      - classifier            (linear head)
    Everything before denseblock4 stays frozen.
    """
    for p in model.features.parameters():
        p.requires_grad_(False)
    for p in model.features.denseblock4.parameters():
        p.requires_grad_(True)
    for p in model.features.norm5.parameters():
        p.requires_grad_(True)
    for p in model.classifier.parameters():
        p.requires_grad_(True)


def make_optimizer(
    model: models.DenseNet,
    *,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [
        {"params": head_params, "lr": float(head_lr)},
        {"params": backbone_params, "lr": float(backbone_lr)},
    ]
    return torch.optim.AdamW(groups, weight_decay=float(weight_decay))


# ---------------------------------------------------------------------------
# Training loop (single epoch)
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_macro_auroc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    num_labels: int,
) -> float:
    model.eval()
    probs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        with autocast("cuda", enabled=use_amp):
            logits = model(xb)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(yb.numpy())
    p_all = np.concatenate(probs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    aucs = [safe_auc(y_all[:, j].astype(int), p_all[:, j]) for j in range(num_labels)]
    finite = [a for a in aucs if np.isfinite(a)]
    return float(np.mean(finite)) if finite else float("nan")


# ---------------------------------------------------------------------------
# Feature extraction with fine-tuned backbone
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    model: models.DenseNet,
    cache_paths: list[Path],
    indices: np.ndarray,
    *,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
) -> np.ndarray:
    """Extract 1024-d DenseNet penultimate features for the given indices using cache."""
    feat_dim = model.classifier.in_features
    feats = np.zeros((len(indices), feat_dim), dtype=np.float32)
    ds = CachedDataset(indices, cache_paths, np.zeros((len(cache_paths), 1)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Temporarily replace classifier with identity to get penultimate features
    orig_classifier = model.classifier
    model.classifier = nn.Identity()
    model.eval()

    row = 0
    for xb, _ in loader:
        xb = xb.to(device)
        with autocast("cuda", enabled=use_amp):
            out = model(xb)
        if out.ndim == 4:
            out = torch.nn.functional.adaptive_avg_pool2d(out, 1).flatten(1)
        batch_f = out.float().cpu().numpy()
        feats[row:row + len(batch_f)] = batch_f
        row += len(batch_f)

    model.classifier = orig_classifier
    return feats


# ---------------------------------------------------------------------------
# RF helpers
# ---------------------------------------------------------------------------

def build_rf_balanced(seed: int, n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None, n_jobs=-1,
        random_state=int(seed), class_weight="balanced",
    )


def safe_predict_proba(clf, x: np.ndarray) -> np.ndarray:
    classes = np.asarray(getattr(clf, "classes_", []))
    if classes.size == 0:
        return np.zeros(x.shape[0], dtype=float)
    if classes.size == 1:
        return np.full(x.shape[0], 1.0 if int(classes[0]) == 1 else 0.0, dtype=float)
    pos = np.where(classes == 1)[0]
    if pos.size == 0:
        return np.zeros(x.shape[0], dtype=float)
    return clf.predict_proba(x)[:, int(pos[0])].astype(float)


# ---------------------------------------------------------------------------
# One CV fold: Stage 1 → 2 → 3 → 4
# ---------------------------------------------------------------------------

def run_one_fold(
    *,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    cache_paths: list[Path],
    Y6: np.ndarray,
    label_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    seed_run: int,
) -> dict:
    num_labels = len(label_names)
    Y5 = Y6[:, :num_labels].astype(np.float32)

    # pos_weight from train fold
    pw = per_label_pos_weight(Y5[tr_idx]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    nw = 0
    batch_size = int(args.batch_size)
    ds_tr = CachedDataset(tr_idx, cache_paths, Y5)
    ds_va = CachedDataset(va_idx, cache_paths, Y5)

    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=nw,
                           pin_memory=(device.type == "cuda"))
    loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=nw,
                           pin_memory=(device.type == "cuda"))

    torch.manual_seed(seed_run)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed_run)

    # --- Stage 1: frozen backbone, head only ---
    print(f"  [Stage1] Warm-up: head only ({args.epochs_stage1} epochs, head_lr={args.head_lr:.1e})", flush=True)
    model = build_model(num_labels, pretrained=args.pretrained).to(device)
    freeze_backbone(model)

    opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=float(args.head_lr), weight_decay=float(args.weight_decay))
    scaler = GradScaler("cuda", enabled=use_amp)
    best_s1, best_state = -1.0, None

    for ep in range(int(args.epochs_stage1)):
        loss = train_epoch(model, loader_tr, opt1, criterion, scaler, device, use_amp)
        val_auc = eval_macro_auroc(model, loader_va, device, use_amp, num_labels)
        print(f"    s1 ep{ep+1:02d}  loss={loss:.4f}  val_macro={val_auc:.4f}", flush=True)
        if np.isfinite(val_auc) and val_auc > best_s1:
            best_s1 = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Stage 2: unfreeze last DenseBlock, very low backbone LR ---
    print(f"  [Stage2] Partial unfreeze (denseblock4+norm5, backbone_lr={args.backbone_lr:.1e}, {args.epochs_stage2} epochs)", flush=True)
    partial_unfreeze_last_denseblock(model)
    opt2 = make_optimizer(model, head_lr=float(args.head_lr), backbone_lr=float(args.backbone_lr),
                          weight_decay=float(args.weight_decay))
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=int(args.epochs_stage2))
    scaler = GradScaler("cuda", enabled=use_amp)
    best_s2, best_state2 = -1.0, None

    for ep in range(int(args.epochs_stage2)):
        loss = train_epoch(model, loader_tr, opt2, criterion, scaler, device, use_amp)
        sched2.step()
        val_auc = eval_macro_auroc(model, loader_va, device, use_amp, num_labels)
        print(f"    s2 ep{ep+1:02d}  loss={loss:.4f}  val_macro={val_auc:.4f}", flush=True)
        if np.isfinite(val_auc) and val_auc > best_s2:
            best_s2 = val_auc
            best_state2 = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state2 is not None:
        model.load_state_dict(best_state2)

    print(f"  [Stage2] Best val macro AUROC: {best_s2:.4f}", flush=True)

    # --- Stage 3: extract fine-tuned features ---
    print("  [Stage3] Extracting fine-tuned features for train+test...", flush=True)
    X_tr = extract_features(model, cache_paths, tr_idx, device=device, use_amp=use_amp, batch_size=batch_size * 2)
    X_te = extract_features(model, cache_paths, te_idx, device=device, use_amp=use_amp, batch_size=batch_size * 2)

    # --- Stage 4: RF on fine-tuned features ---
    print("  [Stage4] Training RF D1-D5 on fine-tuned features...", flush=True)
    run_aucs: dict[str, float] = {}
    for j, name in enumerate(label_names):
        y_tr = Y6[tr_idx, j].astype(int)
        y_te = Y6[te_idx, j].astype(int)
        uniq = np.unique(y_tr)
        if uniq.size < 2:
            run_aucs[name] = float("nan")
            continue
        rf = build_rf_balanced(seed=seed_run + (j + 1) * 13, n_estimators=int(args.rf_estimators))
        rf.fit(X_tr, y_tr)
        p = safe_predict_proba(rf, X_te)
        run_aucs[name] = safe_auc(y_te, p)
        print(f"    {name} AUROC={run_aucs[name]:.4f}", flush=True)

    macro_vals = np.array(list(run_aucs.values()), dtype=float)
    macro_vals = macro_vals[np.isfinite(macro_vals)]
    macro = float(np.mean(macro_vals)) if macro_vals.size else float("nan")
    print(f"  [Stage4] Test Macro AUROC (D1-D5): {macro:.4f}", flush=True)

    return {
        "seed_run": int(seed_run),
        "best_val_macro_auroc_stage1": float(best_s1),
        "best_val_macro_auroc_stage2": float(best_s2),
        "test_auroc_by_label": run_aucs,
        "test_macro_auroc_d1_d5": macro,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase III — Partial-unfreeze DenseNet121@1024+CLAHE → RF D1-D5 (v3.5+; use --pipeline_version_tag / --artifact_slug for runs)"
    )
    ap.add_argument("--embed_root", type=Path, required=True,
                    help="Root dir with DICOM files (D5KW_FIX embed)")
    ap.add_argument("--reference_npz", type=Path, required=True,
                    help="NPZ supplying paths/Y/D5 label ordering (X ignored; use v3.2 NPZ for same cohort)")
    ap.add_argument("--cache_dir", type=Path,
                    default=Path(r"D:\TB Phase III\cache_v35_1024_clahe"),
                    help="Directory for pre-cached .npy image tensors (built once, reused across epochs)")
    ap.add_argument("--out_dir", type=Path,
                    default=Path(r"D:\TB Phase III\artifacts\phase3_active_cv_v35_finetune_rf_1024_d1_d5"))
    ap.add_argument("--cv_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--force_close_zero", action="store_true")
    ap.add_argument("--image_size", type=int, default=1024,
                    help="Input resolution (must match reference version; default 1024 = v3.2 baseline)")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--clahe_clip_limit", type=float, default=0.03)
    ap.add_argument("--pretrained", action="store_true")
    # Stage 1
    ap.add_argument("--epochs_stage1", type=int, default=15,
                    help="Warm-up epochs with frozen backbone (head only)")
    # Stage 2
    ap.add_argument("--epochs_stage2", type=int, default=25,
                    help="Partial-unfreeze epochs (denseblock4 + head)")
    ap.add_argument("--head_lr", type=float, default=3e-3)
    ap.add_argument("--backbone_lr", type=float, default=1e-5,
                    help="Very small LR for unfrozen backbone layers (default 1e-5 ≈ 300x lower than head)")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Batch size (keep small at 1024 res for VRAM)")
    ap.add_argument("--rf_estimators", type=int, default=600)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--amp", action="store_true", help="Mixed precision (recommended on CUDA)")
    ap.add_argument(
        "--pipeline_version_tag",
        type=str,
        default="3.5",
        help='Recorded in JSON metadata (e.g. "3.5", "3.6")',
    )
    ap.add_argument(
        "--artifact_slug",
        type=str,
        default="v35",
        help='Output filenames: phase3_{slug}_finetune_rf_test_metrics.json (e.g. v35, v36)',
    )
    args = ap.parse_args()

    if args.pretrained:
        bpf.configure_https_with_certifi()

    paths, Y6, close_count = load_y6_paths(args.reference_npz, force_close_zero=bool(args.force_close_zero))
    n = len(paths)
    n_train = int(round(n * args.train_ratio))
    n_val = int(round(n * args.val_ratio))
    n_test = int(n - n_train - n_val)
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Invalid split counts")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device not in ("auto", "cuda", "cpu"):
        device = torch.device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"

    print(f"Device: {device}  AMP: {use_amp}", flush=True)

    tfm = bpf.default_transform_with_resize_mode(
        args.image_size, resize_mode="stretch", use_imagenet_norm=True
    )

    # --- Pre-cache all images once ---
    cache_paths = build_image_cache(
        paths, args.cache_dir,
        tfm=tfm, clahe=args.clahe, clahe_clip_limit=args.clahe_clip_limit,
        image_size=args.image_size,
    )

    label_names = ["D1", "D2", "D3", "D4", "D5"]
    strata = make_phase3_strata(Y6)

    per_run: list[dict] = []
    by_label: dict[str, list[float]] = {k: [] for k in label_names}
    macro_runs: list[float] = []

    for r in range(args.cv_runs):
        rs = int(args.seed + 222 + r * 1009)
        print(f"\n=== CV Run {r+1}/{args.cv_runs} (seed {rs}) ===", flush=True)
        tr, va, te = make_splits(strata, n_train, n_val, n_test, rs)
        result = run_one_fold(
            tr_idx=tr, va_idx=va, te_idx=te,
            cache_paths=cache_paths, Y6=Y6, label_names=label_names,
            args=args, device=device, use_amp=use_amp, seed_run=rs + 999,
        )
        per_run.append({"run": r + 1, **result})
        macro_runs.append(result["test_macro_auroc_d1_d5"])
        for k in label_names:
            by_label[k].append(result["test_auroc_by_label"][k])

    tag = str(args.pipeline_version_tag).strip()
    slug = str(args.artifact_slug).strip().replace("/", "_").replace("\\", "_")
    out = {
        "pipeline_train": f"{tag}_densenet121_1024_clahe_partial_unfreeze_then_rf",
        "pipeline_version_tag": tag,
        "stages": {
            "stage1_desc": f"DenseNet frozen, head only warm-up ({args.epochs_stage1} epochs, LR={args.head_lr:.1e})",
            "stage2_desc": f"Unfreeze denseblock4+norm5, backbone LR={args.backbone_lr:.1e}, head LR={args.head_lr:.1e}, {args.epochs_stage2} epochs CosineAnnealing",
            "stage3_desc": "Feature extraction from fine-tuned backbone (1024-d, same as DenseNet penultimate)",
            "stage4_desc": f"RF D1-D5 balanced, {args.rf_estimators} trees",
        },
        "config": {
            "reference_npz": str(args.reference_npz),
            "image_size": int(args.image_size),
            "resize_mode": "stretch",
            "clahe": bool(args.clahe),
            "pretrained": bool(args.pretrained),
            "epochs_stage1": int(args.epochs_stage1),
            "epochs_stage2": int(args.epochs_stage2),
            "head_lr": float(args.head_lr),
            "backbone_lr": float(args.backbone_lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "rf_estimators": int(args.rf_estimators),
            "force_close_zero": bool(args.force_close_zero),
            "device": str(device),
            "amp": bool(use_amp),
            "cv_runs": int(args.cv_runs),
            "split_counts_train_val_test": [n_train, n_val, n_test],
        },
        "cohort": {
            "n_total": n,
            "close_count": close_count,
            "label_positive_counts_d1_d5": {k: int(Y6[:, j].sum()) for j, k in enumerate(label_names)},
        },
        "phase3": {
            "test_auroc_by_label": {k: mean_std(v) for k, v in by_label.items()},
            "test_macro_auroc_d1_d5": mean_std(macro_runs),
            "per_run": per_run,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jp = args.out_dir / f"phase3_{slug}_finetune_rf_test_metrics.json"
    tp = args.out_dir / f"phase3_{slug}_finetune_rf_test_summary.txt"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    p3 = out["phase3"]
    mm = p3["test_macro_auroc_d1_d5"]
    lines = [
        f"Phase III v{tag} — Partial-unfreeze DenseNet121@1024 → RF D1-D5 — TEST",
        f"Stage1: {out['stages']['stage1_desc']}",
        f"Stage2: {out['stages']['stage2_desc']}",
        f"Baseline v3.2 (frozen @1024 stretch RF): Macro 0.8456 ± 0.0103",
        f"Macro AUROC (D1..D5): {mm['mean']:.4f} ± {mm['std']:.4f}",
    ]
    for lab in label_names:
        d = p3["test_auroc_by_label"][lab]
        lines.append(f"{lab} AUROC: {d['mean']:.4f} ± {d['std']:.4f}")
    tp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved: {jp}")
    print(f"saved: {tp}")


if __name__ == "__main__":
    main()

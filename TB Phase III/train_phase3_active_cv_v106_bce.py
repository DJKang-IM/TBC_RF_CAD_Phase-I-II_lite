"""
Phase III v1.06 — frozen features + multi-label head trained with BCEWithLogitsLoss.

Policy (v2.6 training):
  - Feature NPZ: build with CLAHE (and --pipeline-version 2.6 in meta).
  - Labels: D1–D6 (six sigmoid heads).
  - Loss: nn.BCEWithLogitsLoss with per-label pos_weight = n_neg / n_pos on TRAIN fold.
    Rarest positives (often D6, then D5) get the largest weights.

Splits / cohort handling mirror train_phase3_active_cv_v103.py (same strata, train/val/test sizes).
Reports macro AUROC over D1–D6 and D6 EER at 0.5 threshold (same notion as v103).

Example feature build (CLAHE only, DenseNet121, matches typical v2.1-style preprocessing):
  python build_phase3_features.py --in_dir <EMBED_ROOT> --out <out.npz> \\
    --model densenet121 --pretrained --clahe --pipeline-version 2.6

Example train:
  python train_phase3_active_cv_v106_bce.py --phase3_all_npz <out.npz> \\
    --force_close_zero --out_dir <artifacts/phase3_active_cv_v26_clahe_bce_densenet121_d1_d6>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def eer(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(int)
    if y.min() == y.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def cm(y: np.ndarray, s: np.ndarray, thr: float = 0.5) -> np.ndarray:
    yhat = (s >= float(thr)).astype(int)
    return confusion_matrix(y.astype(int), yhat, labels=[0, 1]).astype(float)


def stack_cm_stats(cms: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.stack(cms, axis=0)
    m = np.mean(arr, axis=0)
    s = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(m)
    return m, s


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


def load_phase3(npz_path: Path, force_close_zero: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    d = np.load(npz_path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["Y"], dtype=int)
    D5 = np.asarray(d["D5"], dtype=int).ravel()
    paths = np.asarray(d["paths"], dtype=object)
    if Y.shape[1] != 5:
        raise ValueError(f"Expected phase3 Y shape (N,5), got {Y.shape}")
    if D5.shape[0] != Y.shape[0]:
        raise ValueError("D5 length mismatch")
    if paths.shape[0] != Y.shape[0]:
        raise ValueError("paths length mismatch")
    Y6 = np.column_stack([Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], D5, Y[:, 4]]).astype(int)

    close_count = 0
    if force_close_zero:
        close_mask = np.array([("close" in Path(str(p)).stem.lower()) for p in paths], dtype=bool)
        close_count = int(close_mask.sum())
        Y6 = Y6.copy()
        Y6[close_mask, :] = 0

    return X, Y6, paths, close_count


def per_label_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """
    y_train: (N, L) with values in {0,1}. Returns shape (L,) for BCEWithLogitsLoss pos_weight.
    """
    y = y_train.astype(np.float64)
    n = y.shape[0]
    pos = y.sum(axis=0)
    neg = n - pos
    pw = neg / np.maximum(pos, 1.0)
    return torch.tensor(pw, dtype=torch.float32)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, n_labels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        hid = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hid, int(n_labels)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def predict_proba_logits(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        batch = torch.from_numpy(X[i : i + batch_size]).to(device)
        logits = model(batch)
        prob = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        out.append(prob)
    return np.concatenate(out, axis=0)


def train_one_run(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
) -> MLPHead:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    in_dim = X_tr.shape[1]
    model = MLPHead(in_dim, n_labels=y_tr.shape[1], hidden=hidden, dropout=dropout).to(device)
    pos_w = per_label_pos_weight(y_tr).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    ds_tr = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr),
    )
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

    best_loss = float("inf")
    stale = 0
    best_state: dict | None = None

    for epoch in range(int(max_epochs)):
        model.train()
        for xb, yb in loader_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        va_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in loader_va:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                va_losses.append(float(criterion(logits, yb).item()))
        va_mean = float(np.mean(va_losses)) if va_losses else float("inf")

        if va_mean + float(min_delta) < best_loss:
            best_loss = va_mean
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase III v1.06 — D1–D6 BCE head on frozen features (v2.6-style)")
    ap.add_argument("--phase3_all_npz", type=Path, required=True)
    ap.add_argument("--cv_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--force_close_zero", action="store_true")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(r"D:\TB Phase III\artifacts\phase3_active_cv_v26_clahe_bce_densenet121_d1_d6"),
    )
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min_delta", type=float, default=1e-5)
    ap.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    args = ap.parse_args()

    X, Y6, _, close_count = load_phase3(args.phase3_all_npz, force_close_zero=bool(args.force_close_zero))
    Y6f = Y6.astype(np.float32)
    n = len(X)
    n_train = int(round(n * float(args.train_ratio)))
    n_val = int(round(n * float(args.val_ratio)))
    n_test = int(n - n_train - n_val)
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Invalid split counts")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    label_names = ["D1", "D2", "D3", "D4", "D5", "D6"]
    strata = make_phase3_strata(Y6)

    by_label_auc: dict[str, list[float]] = {k: [] for k in label_names}
    macro_aurocs: list[float] = []
    d6_eers: list[float] = []
    d6_cms: list[np.ndarray] = []
    per_run: list[dict] = []

    for r in range(args.cv_runs):
        rs = int(args.seed + 222 + r * 1009)
        tr, va, te = make_splits(strata, n_train, n_val, n_test, rs)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr]).astype(np.float32)
        X_va = scaler.transform(X[va]).astype(np.float32)
        X_te = scaler.transform(X[te]).astype(np.float32)

        y_tr = Y6f[tr]
        y_va = Y6f[va]
        y_te = Y6f[te]

        y_tr_np = np.asarray(y_tr, dtype=np.float32)
        pos_w_vec = per_label_pos_weight(y_tr_np)
        pos_counts = y_tr_np.sum(axis=0).tolist()

        model = train_one_run(
            X_tr,
            np.asarray(y_tr, dtype=np.float32),
            X_va,
            np.asarray(y_va, dtype=np.float32),
            seed=rs + 333,
            device=device,
            hidden=args.hidden,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
        )

        p_te = predict_proba_logits(model, X_te, device, batch_size=max(64, args.batch_size))

        run_aucs: dict[str, float] = {}
        for j, name in enumerate(label_names):
            y_col = np.asarray(y_te[:, j], dtype=int)
            pj = p_te[:, j].astype(float)
            a = safe_auc(y_col, pj)
            run_aucs[name] = a
            by_label_auc[name].append(a)

        y6_te = np.asarray(y_te[:, 5], dtype=int)
        p6 = p_te[:, 5].astype(float)
        d6_e = eer(y6_te, p6)
        d6_cm = cm(y6_te, p6)
        d6_eers.append(d6_e)
        d6_cms.append(d6_cm)

        macro_vals = np.array([run_aucs[k] for k in label_names], dtype=float)
        macro_vals = macro_vals[np.isfinite(macro_vals)]
        macro_aurocs.append(float(np.mean(macro_vals)) if macro_vals.size else float("nan"))

        per_run.append(
            {
                "run": r + 1,
                "seed": rs,
                "pos_weight_train_fold": pos_w_vec.detach().cpu().numpy().tolist(),
                "train_positive_counts_d1_d6": pos_counts,
                "test_auroc_by_label": run_aucs,
                "test_d6_eer": d6_e,
                "test_d6_cm": d6_cm.tolist(),
            }
        )

    d6_cm_mean, d6_cm_std = stack_cm_stats(d6_cms)

    out = {
        "version": "1.06",
        "pipeline_train": "2.6_bce_d1_d6",
        "loss": "BCEWithLogitsLoss",
        "pos_weight_rule": "per_label n_neg/n_pos on train fold (rarest positive class → largest weight, often D6)",
        "config": {
            "phase3_all_npz": str(args.phase3_all_npz),
            "cv_runs": int(args.cv_runs),
            "seed": int(args.seed),
            "split_counts_train_val_test": [int(n_train), int(n_val), int(n_test)],
            "force_close_zero": bool(args.force_close_zero),
            "hidden": int(args.hidden),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "device": str(device),
            "feature_standardize": "StandardScaler fit on train fold only",
        },
        "cohort": {
            "n_total": int(n),
            "close_count": int(close_count),
            "label_positive_counts_d1_d6": {k: int(Y6[:, i].sum()) for i, k in enumerate(label_names)},
        },
        "phase3": {
            "test_auroc_by_label": {k: mean_std(v) for k, v in by_label_auc.items()},
            "test_macro_auroc_d1_d6": mean_std(macro_aurocs),
            "test_d6_eer": mean_std(d6_eers),
            "test_d6_cm_mean": d6_cm_mean.tolist(),
            "test_d6_cm_std": d6_cm_std.tolist(),
            "per_run": per_run,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "phase3_active_cv_v106_bce_test_metrics.json"
    txt_path = args.out_dir / "phase3_active_cv_v106_bce_test_summary.txt"
    csv_path = args.out_dir / "phase3_active_cv_v106_bce_per_run.csv"

    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    p3 = out["phase3"]
    lines: list[str] = []
    lines.append("v1.06 Phase III active CV — D1..D6 BCE head on frozen features — TEST")
    lines.append(f"pipeline_train: {out['pipeline_train']}")
    lines.append(f"N total: {n}")
    lines.append(f"force_close_zero: {bool(args.force_close_zero)}")
    if args.force_close_zero:
        lines.append(f"close_count: {close_count}")
    lines.append(f"Split counts (train/val/test): {n_train}/{n_val}/{n_test}")
    mmacro = p3["test_macro_auroc_d1_d6"]
    lines.append(f"Macro AUROC (D1..D6): {mmacro['mean']:.4f} ± {mmacro['std']:.4f}")
    for lab in label_names:
        d = p3["test_auroc_by_label"][lab]
        lines.append(f"{lab} AUROC: {d['mean']:.4f} ± {d['std']:.4f}")
    d6e = p3["test_d6_eer"]
    lines.append(f"D6 EER (BCE prob @ 0.5): {d6e['mean']:.4f} ± {d6e['std']:.4f}")
    lines.append("D6 CM mean [[TN,FP],[FN,TP]]:")
    lines.append(np.array2string(np.asarray(p3["test_d6_cm_mean"]), precision=2))
    lines.append("D6 CM std [[TN,FP],[FN,TP]]:")
    lines.append(np.array2string(np.asarray(p3["test_d6_cm_std"]), precision=2))

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run",
                "seed",
                "D1",
                "D2",
                "D3",
                "D4",
                "D5",
                "D6",
                "macro_d1_d6",
                "d6_eer",
                "d6_tn",
                "d6_fp",
                "d6_fn",
                "d6_tp",
            ]
        )
        for r in p3["per_run"]:
            aucs = r["test_auroc_by_label"]
            mv = np.array([aucs[k] for k in label_names], dtype=float)
            mv = mv[np.isfinite(mv)]
            macro_r = float(np.mean(mv)) if mv.size else float("nan")
            d6cm = np.asarray(r["test_d6_cm"], dtype=float)
            w.writerow(
                [
                    r["run"],
                    r["seed"],
                    aucs["D1"],
                    aucs["D2"],
                    aucs["D3"],
                    aucs["D4"],
                    aucs["D5"],
                    aucs["D6"],
                    macro_r,
                    r["test_d6_eer"],
                    d6cm[0, 0],
                    d6cm[0, 1],
                    d6cm[1, 0],
                    d6cm[1, 1],
                ]
            )

    print(f"saved: {json_path}")
    print(f"saved: {txt_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Phase I CXR triage: Normal vs Abnormal (frozen DenseNet121 features + RandomForest).

Labels:
  - Normal:   all *.dcm under --normal_root (recursive)
  - Abnormal: all *.dcm under each --abnormal_roots path (recursive)

Preprocessing (aligned with build_phase3_features.py):
  - DICOM window → [0,1], optional CLAHE (skimage), stretch or letterbox resize,
    ImageNet normalization, DenseNet121 pretrained backbone → 1024-D vector.

Split / evaluation:
  - Optional stratified hold-out test (--holdout_test_fraction, default 0.15).
  - Stratified K-fold CV (--cv_folds, default 5) on the development subset (or all data if holdout=0).
  - Final RF trained on full dev set and evaluated on hold-out test when holdout > 0.

Example:
  python train_phase1_triage_rf_cv.py \\
    --normal_root "<REDACTED_PATH> CXR_NORMAL_ext.251212_251101~251208" \\
    --abnormal_roots \\
      "<REDACTED_PATH> CXR_INACTIVE_ext.260121_Exam.220101~251231" \\
      "<REDACTED_PATH> CXR_Active Image" \\
    --image_size 1024 --clahe --pretrained \\
    --out_dir "<REDACTED_PATH> Phase III\\artifacts\\phase1_triage_rf_densenet121_clahe_1024"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import build_phase3_features as bpf  # noqa: E402


def collect_dicoms(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        if not r.is_dir():
            continue
        for p in sorted(r.rglob("*.dcm")):
            key = str(p.resolve()).lower()
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_acc(y: np.ndarray, pred: np.ndarray) -> float:
    return float(accuracy_score(y.astype(int), pred.astype(int)))


def cm_2x2(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Binary confusion matrix with labels [0=Normal, 1=Abnormal].
    Rows = true class, cols = predicted class.
      [[TN, FP],
       [FN, TP]]
    """
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1])
    return {
        "labels_true_rows": [0, 1],
        "labels_pred_cols": [0, 1],
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
        "matrix": [[int(cm[0, 0]), int(cm[0, 1])], [int(cm[1, 0]), int(cm[1, 1])]],
    }


def abnormal_error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    AER: among true Abnormal (y=1), fraction predicted Normal (pred=0).
    FN / (TP + FN). NaN if no abnormal samples.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    m = y_true == 1
    denom = int(m.sum())
    if denom == 0:
        return float("nan")
    fn = int(np.sum(y_pred[m] == 0))
    return float(fn / denom)


def overall_error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1 - accuracy."""
    return float(1.0 - accuracy_score(y_true.astype(int), y_pred.astype(int)))


def extract_features(
    pairs: list[tuple[Path, int]],
    backbone: torch.nn.Module,
    tfm,
    device: str,
    *,
    use_clahe: bool,
    clahe_clip_limit: float,
    clahe_kernel_size: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feats: list[np.ndarray] = []
    labels: list[int] = []
    ok_paths: list[str] = []
    for p, lab in tqdm(pairs, desc="extract DenseNet121", unit="file"):
        try:
            v = bpf.extract_one(
                backbone,
                tfm,
                p,
                device,
                lung_inferer=None,
                lung_crop=False,
                lung_margin=0.15,
                use_clahe=bool(use_clahe),
                clahe_clip_limit=float(clahe_clip_limit),
                clahe_kernel_size=clahe_kernel_size,
                robust_norm="none",
            )
            feats.append(v)
            labels.append(int(lab))
            ok_paths.append(str(p))
        except Exception as e:
            print(f"[skip] {p}: {e}", file=sys.stderr)
    if not feats:
        raise RuntimeError("No features extracted (all DICOM reads failed?).")
    X = np.stack(feats, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, ok_paths


def build_rf(seed: int, n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=None,
        n_jobs=-1,
        random_state=int(seed),
        class_weight="balanced",
    )


def proba_positive(clf: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(X)
    classes = np.asarray(clf.classes_)
    pos = np.where(classes == 1)[0]
    if pos.size == 0:
        return np.zeros((X.shape[0],), dtype=float)
    return np.asarray(proba[:, int(pos[0])], dtype=float)


def mean_std(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "n": int(a.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase I triage: RF on frozen DenseNet121 + CLAHE features.")
    ap.add_argument("--normal_root", type=Path, required=True, help="Root folder whose DICOMs are label Normal (0).")
    ap.add_argument(
        "--abnormal_roots",
        type=Path,
        nargs="+",
        required=True,
        help="One or more roots whose DICOMs are label Abnormal (1).",
    )
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--features_npz", type=Path, default=None, help="Cache/load X,y,paths to skip re-extraction.")
    ap.add_argument("--rebuild_features", action="store_true", help="Ignore cache and re-extract.")

    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--pretrained", action="store_true", help="Torchvision ImageNet weights for DenseNet121.")
    ap.add_argument("--weights", type=str, default=None, help="Optional state_dict path (overrides pretrained).")
    ap.add_argument("--resize_mode", type=str, default="stretch", choices=("stretch", "letterbox"))
    ap.add_argument("--letterbox_pad_value", type=float, default=0.0)
    ap.add_argument("--clahe", action="store_true", help="Apply CLAHE before resize (recommended).")
    ap.add_argument("--clahe_clip_limit", type=float, default=0.03)
    ap.add_argument("--clahe_kernel_size", type=int, default=0, help="0 = skimage default tile size.")

    ap.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--holdout_test_fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rf_estimators", type=int, default=600)
    ap.add_argument("--max_files_normal", type=int, default=0, help="0 = all (debug cap per class).")
    ap.add_argument("--max_files_abnormal", type=int, default=0, help="0 = all.")
    ap.add_argument("--save_model", action="store_true", help="Save final RF (dev-trained) as joblib.")
    args = ap.parse_args()

    bpf.configure_https_with_certifi()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.features_npz
    if cache_path is None:
        cache_path = args.out_dir / "phase1_triage_features.npz"

    use_cache = cache_path.is_file() and not args.rebuild_features

    if use_cache:
        d = np.load(cache_path, allow_pickle=True)
        X = np.asarray(d["X"], dtype=np.float32)
        y = np.asarray(d["y"], dtype=np.int64).ravel()
        paths = [str(x) for x in np.asarray(d["paths"], dtype=object).tolist()]
        print(f"Loaded cached features: {cache_path} X={X.shape}", file=sys.stderr)
    else:
        normal_paths = collect_dicoms([args.normal_root])
        ab_paths = collect_dicoms(list(args.abnormal_roots))
        if args.max_files_normal and args.max_files_normal > 0:
            normal_paths = normal_paths[: int(args.max_files_normal)]
        if args.max_files_abnormal and args.max_files_abnormal > 0:
            ab_paths = ab_paths[: int(args.max_files_abnormal)]

        pairs: list[tuple[Path, int]] = [(p, 0) for p in normal_paths] + [(p, 1) for p in ab_paths]

        dev = args.device.strip().lower()
        if dev == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = dev

        backbone, feat_dim = bpf.build_backbone("densenet121", bool(args.pretrained), args.weights)
        backbone = backbone.to(device)

        ks = int(args.clahe_kernel_size) if int(args.clahe_kernel_size) > 0 else None
        tfm = bpf.default_transform_with_resize_mode(
            int(args.image_size),
            resize_mode=str(args.resize_mode),
            letterbox_pad_value=float(args.letterbox_pad_value),
            use_imagenet_norm=True,
        )

        X, y, paths = extract_features(
            pairs,
            backbone,
            tfm,
            device,
            use_clahe=bool(args.clahe),
            clahe_clip_limit=float(args.clahe_clip_limit),
            clahe_kernel_size=ks,
        )
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            paths=np.array(paths, dtype=object),
            meta=json.dumps(
                {
                    "feat_dim": int(feat_dim),
                    "image_size": int(args.image_size),
                    "clahe": bool(args.clahe),
                    "resize_mode": str(args.resize_mode),
                    "backbone": "densenet121",
                }
            ),
        )
        print(f"Saved features: {cache_path}", file=sys.stderr)

    n = len(y)
    n_pos = int(y.sum())
    n_neg = int(n - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("Need both Normal (0) and Abnormal (1) samples.")

    hold_frac = float(args.holdout_test_fraction)
    if hold_frac < 0 or hold_frac >= 1:
        raise SystemExit("holdout_test_fraction must be in [0, 1).")
    rng = int(args.seed)

    idx_all = np.arange(n)
    idx_dev = idx_all
    idx_test = np.array([], dtype=int)
    if hold_frac > 0 and hold_frac < 1:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=hold_frac, random_state=rng)
        idx_dev, idx_test = next(sss.split(idx_all, y))

    X_dev = X[idx_dev]
    y_dev = y[idx_dev]
    X_test = X[idx_test] if idx_test.size else None
    y_test = y[idx_test] if idx_test.size else None

    cv_folds = int(args.cv_folds)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rng)

    fold_rows: list[dict] = []
    aucs: list[float] = []
    accs: list[float] = []
    aers: list[float] = []
    oers: list[float] = []

    for fold_id, (tr, va) in enumerate(skf.split(X_dev, y_dev)):
        rs = rng + fold_id * 997
        clf = build_rf(rs, args.rf_estimators)
        clf.fit(X_dev[tr], y_dev[tr])
        s_va = proba_positive(clf, X_dev[va])
        pred_va = clf.predict(X_dev[va])
        y_va = y_dev[va]
        auc_v = safe_auc(y_va, s_va)
        acc_v = safe_acc(y_va, pred_va)
        aer_v = abnormal_error_rate(y_va, pred_va)
        oer_v = overall_error_rate(y_va, pred_va)
        aucs.append(auc_v)
        accs.append(acc_v)
        aers.append(aer_v)
        oers.append(oer_v)
        fold_rows.append(
            {
                "fold": fold_id + 1,
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "val_auroc": auc_v,
                "val_accuracy": acc_v,
                "val_overall_error_rate": oer_v,
                "val_aer_abnormal_miss_rate": aer_v,
                "val_confusion_matrix": cm_2x2(y_va, pred_va),
            }
        )

    final_metrics: dict = {
        "definitions": {
            "AUROC": "Area under ROC curve; positive class = Abnormal (1).",
            "AER": "Abnormal error rate = FN/(TP+FN): among true Abnormal, fraction predicted Normal (miss rate).",
            "overall_error_rate": "1 - accuracy.",
            "confusion_matrix": "Rows=true Normal/Abnormal (0/1), cols=pred Normal/Abnormal; TN,FP / FN,TP.",
        },
        "cv_val_auroc": mean_std(aucs),
        "cv_val_accuracy": mean_std(accs),
        "cv_val_overall_error_rate": mean_std(oers),
        "cv_val_aer_abnormal_miss_rate": mean_std(aers),
        "folds": fold_rows,
    }

    # Fit on full dev, optional test
    clf_final = build_rf(rng + 424242, args.rf_estimators)
    clf_final.fit(X_dev, y_dev)

    if args.save_model:
        try:
            import joblib

            joblib.dump(clf_final, args.out_dir / "phase1_triage_rf_final.joblib")
            print(f"Saved model: {args.out_dir / 'phase1_triage_rf_final.joblib'}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] joblib save failed: {e}", file=sys.stderr)

    if X_test is not None and len(idx_test) > 0:
        s_te = proba_positive(clf_final, X_test)
        pred_te = clf_final.predict(X_test)
        final_metrics["holdout_test"] = {
            "n": int(len(idx_test)),
            "auroc": safe_auc(y_test, s_te),
            "accuracy": safe_acc(y_test, pred_te),
            "overall_error_rate": overall_error_rate(y_test, pred_te),
            "aer_abnormal_miss_rate": abnormal_error_rate(y_test, pred_te),
            "confusion_matrix": cm_2x2(y_test, pred_te),
        }

    summary = {
        "phase": "Phase_I_triage",
        "task": "Normal(0)_vs_Abnormal(1)",
        "normal_root": str(args.normal_root.resolve()),
        "abnormal_roots": [str(p.resolve()) for p in args.abnormal_roots],
        "n_total": n,
        "n_normal": n_neg,
        "n_abnormal": n_pos,
        "feature_cache": str(cache_path.resolve()),
        "config": {
            "image_size": int(args.image_size),
            "backbone": "densenet121",
            "clahe": bool(args.clahe),
            "clahe_clip_limit": float(args.clahe_clip_limit),
            "resize_mode": str(args.resize_mode),
            "cv_folds": cv_folds,
            "holdout_test_fraction": hold_frac,
            "seed": rng,
            "rf_estimators": int(args.rf_estimators),
        },
        "metrics": final_metrics,
    }

    (args.out_dir / "phase1_triage_rf_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    import csv

    csv_path = args.out_dir / "phase1_triage_rf_cv_folds.csv"
    if fold_rows:
        flat_rows: list[dict] = []
        for r in fold_rows:
            cm = r["val_confusion_matrix"]
            rr = {k: v for k, v in r.items() if k != "val_confusion_matrix"}
            rr["val_TN"] = cm["TN"]
            rr["val_FP"] = cm["FP"]
            rr["val_FN"] = cm["FN"]
            rr["val_TP"] = cm["TP"]
            flat_rows.append(rr)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            w.writeheader()
            w.writerows(flat_rows)

    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(f"\nWrote: {args.out_dir / 'phase1_triage_rf_metrics.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

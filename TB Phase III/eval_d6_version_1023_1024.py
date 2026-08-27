"""
Phase III D6-only, INITIAL cohort (Version 1.01 — NOT Vector Balanced):
  Version 1.023 — RF (ref) + LR + XGBoost

All outputs (PNG + JSON) default to:
  <REDACTED_PATH> 1.023(D6_XGboost)/  (flat; filenames prefixed with v1023_ where applicable).

Features: phase3_features_*.npz · RF: rf_phase3_active_vs_inactive.joblib.
Reuses run_track / write_meta from eval_d6_version_1021_1022.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import eval_d6_version_1021_1022 as v12
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_xgb_scores(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from xgboost import XGBClassifier

    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    spw = float(neg / max(pos, 1))
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
    )
    clf.fit(X_tr, y_tr)
    return (
        clf.predict_proba(X_tr)[:, 1].astype(float),
        clf.predict_proba(X_va)[:, 1].astype(float),
        clf.predict_proba(X_te)[:, 1].astype(float),
    )


DEFAULT_OUT = Path(r"D:\artifact") / "ver. 1.023(D6_XGboost)"


def ensure_output_dir(out_dir: Path) -> Path:
    p = out_dir.expanduser()
    p.mkdir(parents=True, exist_ok=True)
    if not p.is_dir():
        raise RuntimeError(f"Cannot create or access output directory: {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=r"Default: D:\artifact\ver. 1.023(D6_XGboost)",
    )
    args = ap.parse_args()
    root: Path = args.root
    out = ensure_output_dir(args.out_dir or DEFAULT_OUT)
    print("output_dir (created if needed):", out.resolve())

    train_p = root / "phase3_features_train.npz"
    val_p = root / "phase3_features_val.npz"
    test_p = root / "phase3_features_test.npz"
    rf_path = root / "rf_phase3_active_vs_inactive.joblib"

    d_tr = np.load(train_p, allow_pickle=True)
    d_va = np.load(val_p, allow_pickle=True)
    d_te = np.load(test_p, allow_pickle=True)
    X_tr, Y_tr = d_tr["X"].astype(np.float32), d_tr["Y"].astype(int)
    X_va, Y_va = d_va["X"].astype(np.float32), d_va["Y"].astype(int)
    X_te, Y_te = d_te["X"].astype(np.float32), d_te["Y"].astype(int)
    j = 4
    y_tr, y_va, y_te = Y_tr[:, j], Y_va[:, j], Y_te[:, j]

    rf = joblib.load(rf_path)
    score_rf_tr = rf.predict_proba(X_tr)[j][:, 1].astype(float)
    score_rf_va = rf.predict_proba(X_va)[j][:, 1].astype(float)
    score_rf_te = rf.predict_proba(X_te)[j][:, 1].astype(float)

    lr = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42, solver="lbfgs")
    lr.fit(X_tr, y_tr)
    score_lr_tr = lr.predict_proba(X_tr)[:, 1].astype(float)
    score_lr_va = lr.predict_proba(X_va)[:, 1].astype(float)
    score_lr_te = lr.predict_proba(X_te)[:, 1].astype(float)

    score_xgb_tr, score_xgb_va, score_xgb_te = fit_xgb_scores(X_tr, y_tr, X_va, X_te)

    v12.write_meta(
        out / "version_1_023_meta.json",
        {
            "version": "1.023",
            "d6_classifier": "XGBoost (binary; scale_pos_weight = neg/pos on train)",
            "shared": "Initial Phase III features (v1.01); RF = active_vs_inactive D6 head + LR.",
            "output_dir": str(out),
        },
    )
    sts = [
        ("RF (ref, D6 head)", score_rf_tr, score_rf_va, score_rf_te),
        ("Logistic regression", score_lr_tr, score_lr_va, score_lr_te),
        ("XGBoost", score_xgb_tr, score_xgb_va, score_xgb_te),
    ]
    v12.run_track(
        root=root,
        out_dir=out,
        version_label="1.023",
        file_tag="v1023",
        display_title="Version 1.023",
        sts=sts,
        y_va=y_va,
        y_te=y_te,
        metrics_json_name="v1023_metrics_d6.json",
    )


if __name__ == "__main__":
    main()

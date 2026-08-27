"""
Train Phase III v1.03: 6-output RF (D1 through D5, plus RF slot for D6) + XGBoost replacing D6 at inference.

Expects feature .npz from build_phase3_features.py with:
  X, Y shape (N,5) for [D1,D2,D3,D4,D6], D5 array (N,) cavitary 0/1.

Outputs (default <REDACTED_PATH> Phase III\\artifacts\\phase3_v1_03_d5_d6_xgb\\):
  phase3_rf_v1_03_d5_d6_xgb_hybrid.joblib
  d6_ntm_xgb.json
  version_1_03_d5_d6_meta.json

Requires: pip install -e <REDACTED_PATH>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

try:
    from tbc_cad_ver_103.hybrid_phase3 import D6_INDEX, Phase3HybridD6XGB
except ImportError as e:
    raise SystemExit(
        "Import tbc_cad_ver_103 failed. Run: pip install -e <REDACTED_PATH>"
    ) from e


def build_xgb_d6(X_tr: np.ndarray, y6_tr: np.ndarray, seed: int) -> XGBClassifier:
    neg = int((y6_tr == 0).sum())
    pos = int((y6_tr == 1).sum())
    spw = float(neg / max(pos, 1))
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=spw,
        random_state=int(seed),
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
    )


def merge_y6(npz: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    Y = np.asarray(npz["Y"], dtype=np.int64)
    if Y.ndim != 2 or Y.shape[1] != 5:
        raise ValueError(f"Expected Y shape (N,5), got {Y.shape}")
    if "D5" not in npz:
        print("[warn] 'D5' not in npz — using zeros. Re-run build_phase3_features on DICOMs with D5 tags.")
        d5 = np.zeros((Y.shape[0],), dtype=np.int64)
    else:
        d5 = np.asarray(npz["D5"]).ravel().astype(np.int64)
        if d5.shape[0] != Y.shape[0]:
            raise ValueError("D5 length must match Y rows")
    # Y columns: D1,D2,D3,D4,D6 last
    y6 = np.column_stack([Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], d5, Y[:, 4]])
    return Y, y6


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Phase III v1.03 hybrid (D1-D5 RF + D6 XGB).")
    ap.add_argument("--train_npz", type=Path, required=True, help="Training .npz from build_phase3_features")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "phase3_v1_03_d5_d6_xgb",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--max_depth", type=int, default=0, help="0 = None")
    args = ap.parse_args()

    data = np.load(args.train_npz, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    _, Y6 = merge_y6(data)
    if X.shape[0] != Y6.shape[0]:
        raise ValueError("X and Y row count mismatch")

    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth == 0 else args.max_depth,
        n_jobs=-1,
        random_state=args.seed,
        class_weight="balanced",
    )
    rf.fit(X, Y6)
    print("RF fit done. Y6 shape:", Y6.shape, "pos counts per column:", Y6.sum(axis=0).tolist())

    y_d6 = Y6[:, D6_INDEX]
    xgb = build_xgb_d6(X, y_d6, seed=args.seed)
    xgb.fit(X, y_d6)

    hybrid = Phase3HybridD6XGB(rf, xgb, d6_threshold=0.5)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hybrid_path = out_dir / "phase3_rf_v1_03_d5_d6_xgb_hybrid.joblib"
    joblib.dump(hybrid, hybrid_path)

    xgb_json = out_dir / "d6_ntm_xgb.json"
    xgb.save_model(str(xgb_json))

    rf_only = out_dir / "rf_phase3_v103_d1_d6_6out.joblib"
    joblib.dump(rf, rf_only)

    # quick D6 AUROC on training set
    p6 = hybrid.predict_proba(X)[D6_INDEX][:, 1].astype(float)
    auc = float("nan")
    if y_d6.min() < y_d6.max():
        auc = float(roc_auc_score(y_d6, p6))

    meta = {
        "version": "1.03",
        "train_npz": str(args.train_npz.resolve()),
        "n_samples": int(X.shape[0]),
        "hybrid_joblib": str(hybrid_path.resolve()),
        "rf_6out_joblib": str(rf_only.resolve()),
        "d6_xgb_json": str(xgb_json.resolve()),
        "d6_train_auroc": auc,
        "y6_columns": ["D1", "D2", "D3", "D4", "D5", "D6"],
    }
    (out_dir / "version_1_03_d5_d6_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved:", hybrid_path)
    print("saved:", rf_only)
    print("saved:", xgb_json)
    print("train D6 AUROC (sanity):", auc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

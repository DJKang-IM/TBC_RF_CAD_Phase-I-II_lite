"""Evaluate Phase III v1.02 (D6_XGBoost hybrid) on one or more feature .npz files."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score


def load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return d["X"], d["Y"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="phase3_rf_v1_02_d6_xgb_hybrid.joblib")
    ap.add_argument("--features", required=True, nargs="+")
    args = ap.parse_args()

    model = joblib.load(args.model)
    names = ["D1", "D2", "D3", "D4", "D6"]

    for feat_path in args.features:
        X, Y = load_npz(feat_path)
        pred = model.predict(X)
        probas = model.predict_proba(X)
        print("")
        print(f"== {Path(feat_path).name} ==")
        for j, name in enumerate(names):
            print("")
            print(f"[{name}]")
            print(classification_report(Y[:, j], pred[:, j], digits=4, zero_division=0))
            try:
                p1 = probas[j][:, 1]
                auc = float(roc_auc_score(Y[:, j], p1))
                print(f"AUROC={auc:.4f}")
            except Exception as e:  # noqa: BLE001
                print(f"AUROC=NA ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score


def load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf_model", required=True, help="Path to RF .joblib")
    ap.add_argument("--features", required=True, nargs="+", help="One or more feature .npz files to evaluate")
    args = ap.parse_args()

    rf = joblib.load(args.rf_model)

    for feat_path in args.features:
        X, y = load_npz(feat_path)
        pred = rf.predict(X)
        name = Path(feat_path).name
        print("")
        print(f"== {name} ==")
        print(classification_report(y, pred, digits=4, zero_division=0))
        # AUC if predict_proba available (binary)
        try:
            proba = rf.predict_proba(X)[:, 1]
            auc = float(roc_auc_score(y, proba))
            print(f"AUC={auc:.4f}")
        except Exception:
            pass

    out = {
        "rf_model": str(Path(args.rf_model)),
        "features": [str(Path(p)) for p in args.features],
    }
    print("")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


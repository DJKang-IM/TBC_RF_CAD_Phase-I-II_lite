import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_stage1.npz", help="Input features .npz")
    ap.add_argument("--out", default="rf_stage1.joblib", help="Output model path")
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--max_depth", type=int, default=0, help="0 = None")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth == 0 else args.max_depth,
        n_jobs=-1,
        random_state=args.seed,
        class_weight="balanced",
    )
    rf.fit(X, y)

    pred = rf.predict(X)
    print("Train-set report (no internal validation):")
    print(classification_report(y, pred, digits=4))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()


import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="phase3_features.npz", help="Input .npz from build_phase3_features.py")
    ap.add_argument("--out", default="rf_phase3_multioutput.joblib", help="Output model path")
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--max_depth", type=int, default=0, help="0 = None")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]  # shape (N,5) for [D1,D2,D3,D4,D6]

    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth == 0 else args.max_depth,
        n_jobs=-1,
        random_state=args.seed,
        class_weight="balanced",
    )
    rf.fit(X, Y)

    pred = rf.predict(X)
    print("Train-set report (no internal validation):")
    for j, name in enumerate(["D1", "D2", "D3", "D4", "D6"]):
        print("")
        print(f"[{name}]")
        print(classification_report(Y[:, j], pred[:, j], digits=4, zero_division=0))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()


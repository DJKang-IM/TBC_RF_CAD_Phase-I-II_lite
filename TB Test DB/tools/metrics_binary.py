import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"]


def eer_from_scores(y_true: np.ndarray, scores: np.ndarray) -> dict:
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    return {
        "eer": eer,
        "threshold": float(thr[i]),
        "fpr": float(fpr[i]),
        "fnr": float(fnr[i]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf_model", required=True)
    ap.add_argument("--features", required=True, nargs="+")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    rf = joblib.load(args.rf_model)
    out = {
        "rf_model": str(Path(args.rf_model)),
        "threshold": float(args.threshold),
        "sets": [],
    }

    for feat in args.features:
        p = Path(feat)
        X, y = load_npz(p)
        proba = rf.predict_proba(X)[:, 1].astype(float)
        pred = (proba >= args.threshold).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        out["sets"].append(
            {
                "name": p.name,
                "n": int(y.shape[0]),
                "confusion_matrix": {
                    "labels": [0, 1],
                    "matrix": cm.astype(int).tolist(),  # [[TN,FP],[FN,TP]]
                },
                "eer": eer_from_scores(y.astype(int), proba),
                "support": {"n0": int((y == 0).sum()), "n1": int((y == 1).sum())},
            }
        )

    payload = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out_json:
        Path(args.out_json).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()


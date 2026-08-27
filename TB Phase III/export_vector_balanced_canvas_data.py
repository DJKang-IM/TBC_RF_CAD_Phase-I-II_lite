"""Emit JSON for embedding Vector Balanced v1 ROC/AUC/EER/CM in a Cursor canvas."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve


def subsample_xy(fpr: np.ndarray, tpr: np.ndarray, max_pts: int = 14) -> list[list[float]]:
    n = len(fpr)
    if n <= max_pts:
        return [[float(fpr[i]), float(tpr[i])] for i in range(n)]
    idx = np.unique(np.linspace(0, n - 1, max_pts, dtype=int))
    return [[float(fpr[i]), float(tpr[i])] for i in idx]


def eer_point(fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray) -> dict:
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return {
        "eer": float((fpr[i] + fnr[i]) / 2.0),
        "threshold": float(thr[i]),
        "fpr": float(fpr[i]),
        "fnr": float(fnr[i]),
        "tpr": float(tpr[i]),
    }


def cm_thr(y: np.ndarray, s: np.ndarray, th: float) -> list[list[int]]:
    yhat = (s >= th).astype(int)
    m = confusion_matrix(y.astype(int), yhat, labels=[0, 1])
    return m.astype(int).tolist()


def main() -> None:
    root = Path(__file__).resolve().parent
    names = ["D1", "D2", "D3", "D4", "D6"]
    rf = joblib.load(root / "rf_phase3_vector_balanced_version_1.joblib")
    out: dict = {
        "model": "rf_phase3_vector_balanced_version_1.joblib",
        "dataset_title": "Vector Balanced Version 1",
        "cm_threshold": 0.5,
        "splits": [],
    }
    for split, fname in [
        ("val", "phase3_vector_balanced_version_1_features_val.npz"),
        ("test", "phase3_vector_balanced_version_1_features_test.npz"),
    ]:
        path = root / fname
        d = np.load(path, allow_pickle=True)
        X, Y = d["X"], d["Y"]
        probas = rf.predict_proba(X)
        block: dict = {"split": split, "path": fname, "n": int(Y.shape[0]), "labels": []}
        for j, nk in enumerate(names):
            y = Y[:, j].astype(int)
            s = probas[j][:, 1].astype(float)
            fpr, tpr, thr = roc_curve(y, s)
            ep = eer_point(fpr, tpr, thr)
            block["labels"].append(
                {
                    "key": nk,
                    "auc": float(auc(fpr, tpr)),
                    "roc": subsample_xy(fpr, tpr, 12),
                    "eer": ep,
                    "cm_at_05": cm_thr(y, s, 0.5),
                    "cm_at_eer": cm_thr(y, s, ep["threshold"]),
                    "support": {"n0": int((y == 0).sum()), "n1": int((y == 1).sum())},
                }
            )
        out["splits"].append(block)

    dst = root / "artifacts" / "vector_balanced_version_1" / "canvas_embed_data.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(dst)


if __name__ == "__main__":
    main()

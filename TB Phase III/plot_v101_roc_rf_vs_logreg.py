"""
Phase III Version 1.01: ROC curves on Val/Test — Random Forest (trained model) vs
Logistic Regression refit on the same training features (per label).

AUROC is the area under each ROC curve; this script plots the curves and prints AUC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve


LABEL_NAMES = ["D1 AFB", "D2 TB PCR", "D3 Solid", "D4 Liquid", "D6 NTM"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Default: <root>/artifacts/version_1_01/plots",
    )
    args = ap.parse_args()
    root: Path = args.root
    out_dir = args.out_dir or (root / "artifacts" / "version_1_01" / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    train = np.load(root / "phase3_features_train.npz", allow_pickle=True)
    X_tr, Y_tr = train["X"].astype(np.float32), train["Y"].astype(int)
    rf = joblib.load(root / "rf_phase3_active_vs_inactive.joblib")
    rf_name = "Random Forest (v1.01)"

    for split in ("val", "test"):
        path = root / f"phase3_features_{split}.npz"
        d = np.load(path, allow_pickle=True)
        X, Y = d["X"].astype(np.float32), d["Y"].astype(int)
        rf_proba = rf.predict_proba(X)

        fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
        fig.suptitle(
            f"Phase III Version 1.01 — ROC: RF vs Logistic Regression ({split})\n"
            f"(LR refit per label on train; curves on {path.name})",
            fontsize=11,
        )
        axes_flat = axes.ravel()
        for j in range(Y.shape[1]):
            ax = axes_flat[j]
            y = Y[:, j]
            if y.max() == y.min():
                ax.text(0.5, 0.5, "single class", ha="center", va="center")
                ax.set_title(LABEL_NAMES[j])
                continue

            # LR trained only on train split for this label
            lr = LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                random_state=42,
                solver="lbfgs",
            )
            lr.fit(X_tr, Y_tr[:, j])
            lr_score = lr.predict_proba(X)[:, 1]

            rf_score = rf_proba[j][:, 1].astype(float)

            for scores, name, style in (
                (rf_score, rf_name, "-"),
                (lr_score, "Logistic regression (train→eval)", "--"),
            ):
                fpr, tpr, _ = roc_curve(y, scores)
                a = float(auc(fpr, tpr))
                ax.plot(fpr, tpr, linestyle=style, lw=1.8, label=f"{name} (AUC={a:.4f})")

            ax.plot([0, 1], [0, 1], color="gray", lw=0.8, linestyle=":", label="chance")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title(LABEL_NAMES[j])
            ax.legend(loc="lower right", fontsize=7)
            ax.grid(True, alpha=0.25)
        axes_flat[-1].axis("off")
        out_png = out_dir / f"v101_roc_rf_vs_logreg_{split}.png"
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print("saved:", out_png)


if __name__ == "__main__":
    main()

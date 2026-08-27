"""
Phase III multi-output RF: ROC (threshold sweep), AUROC, EER point, confusion matrices.

Note: AUROC is a single scalar; the curve whose area is AUROC is the ROC curve built by
varying the classification threshold on the positive-class score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

from metrics_multioutput import NAMES, eer_from_scores, load_npz


def cm_at_threshold(y: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    y = y.astype(int)
    yhat = (scores >= thr).astype(int)
    return confusion_matrix(y, yhat, labels=[0, 1])


def plot_split(
    rf,
    npz_path: Path,
    out_png: Path,
    title: str,
    cm_threshold: float,
) -> None:
    X, Y = load_npz(npz_path)
    probas = rf.predict_proba(X)
    n_labels = Y.shape[1]
    fig, axes = plt.subplots(n_labels, 3, figsize=(14, 3.0 * n_labels), constrained_layout=True)
    # subplots(1, 3) returns shape (3,); ensure (n_labels, 3)
    axes = np.atleast_2d(axes)
    if axes.shape[1] != 3:
        axes = axes.T

    fig.suptitle(title, fontsize=12, fontweight="bold")

    for j in range(n_labels):
        name = NAMES[j]
        y = Y[:, j].astype(int)
        scores = probas[j][:, 1].astype(float)
        fpr, tpr, thr = roc_curve(y, scores)
        roc_auc = float(auc(fpr, tpr))
        eer_info = eer_from_scores(y, scores)
        eer_thr = eer_info["threshold"]
        eer = eer_info["eer"]

        ax_roc = axes[j, 0]
        ax_roc.plot(fpr, tpr, color="C0", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="chance")
        ax_roc.scatter(
            [eer_info["fpr"]],
            [1.0 - eer_info["fnr"]],
            color="C1",
            s=80,
            zorder=5,
            label=f"EER point (EER={eer:.3f})",
        )
        ax_roc.set_xlim(0.0, 1.0)
        ax_roc.set_ylim(0.0, 1.05)
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title(f"{name} — ROC (threshold sweep)")
        ax_roc.legend(loc="lower right", fontsize=8)
        ax_roc.grid(True, alpha=0.3)
        ax_roc.text(
            0.02,
            0.08,
            f"EER thr={eer_thr:.4f}\nCM@0.5 thr={cm_threshold:.2f}",
            transform=ax_roc.transAxes,
            fontsize=8,
            verticalalignment="bottom",
        )

        def draw_cm(ax, cm: np.ndarray, thr: float, subtitle: str) -> None:
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0)
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticklabels(["True 0", "True 1"])
            thresh = cm.max() / 2.0 if cm.size else 0
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(
                        c,
                        r,
                        format(cm[r, c], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[r, c] > thresh else "black",
                    )
            ax.set_ylabel("True label")
            ax.set_xlabel("Predicted label")
            ax.set_title(f"{name} — {subtitle}\n(thr={thr:.4f})")

        cm05 = cm_at_threshold(y, scores, cm_threshold)
        draw_cm(axes[j, 1], cm05, cm_threshold, "Confusion matrix")

        cm_eer = cm_at_threshold(y, scores, eer_thr)
        draw_cm(axes[j, 2], cm_eer, eer_thr, "Confusion matrix @ EER threshold")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"saved: {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf_model", required=True)
    ap.add_argument("--features", required=True, nargs="+")
    ap.add_argument("--out_dir", required=True, help="Directory for PNGs")
    ap.add_argument("--dataset_title", default="", help="Prefix for figure suptitle")
    ap.add_argument(
        "--cm_threshold",
        type=float,
        default=0.5,
        help="Threshold for middle-column confusion matrix (previous JSON used 0.5)",
    )
    args = ap.parse_args()

    rf = joblib.load(args.rf_model)
    out_dir = Path(args.out_dir)
    prefix = (args.dataset_title + " — ") if args.dataset_title else ""

    for feat in args.features:
        p = Path(feat)
        tag = p.stem.replace("phase3_", "").replace("_features", "")
        n_samples = int(np.load(p, allow_pickle=True)["Y"].shape[0])
        title = f"{prefix}{p.name} (n={n_samples})"
        out_png = out_dir / f"roc_eer_cm__{tag}.png"
        plot_split(rf, p, out_png, title=title, cm_threshold=args.cm_threshold)


if __name__ == "__main__":
    main()

"""
Phase I & II: confusion matrix heatmaps (Val | Test) + ROC curves (RF vs LR, Val | Test).

Reads metrics JSON for CM (threshold 0.5). ROC from rf joblib + feature npz; LR refit on train.

Outputs (default):
  artifacts/plots_phase1/phase1_cm_heatmap_val_test.png
  artifacts/plots_phase1/phase1_roc_rf_vs_logreg_val_test.png
  artifacts/plots_phase2/phase2_cm_heatmap_val_test.png
  artifacts/plots_phase2/phase2_roc_rf_vs_logreg_val_test.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    if "y" in d.files:
        y = np.asarray(d["y"])
    elif "Y" in d.files:
        y = np.asarray(d["Y"])
    else:
        raise KeyError(f"No y/Y in {path}")
    y = y.astype(int).ravel()
    return X, y


def pick_sets(sets: list) -> tuple[dict, dict]:
    val = next(s for s in sets if "_val." in s["name"] or s["name"].endswith("_val.npz"))
    test = next(s for s in sets if "_test." in s["name"] or s["name"].endswith("_test.npz"))
    return val, test


def _annotate_text_color_for_heatmap(
    *,
    rgba: tuple[float, float, float, float],
    dark_text: str = "black",
    light_text: str = "white",
) -> str:
    """
    Choose readable annotation text color against a heatmap cell color.

    Uses relative luminance (WCAG-style) of the RGBA facecolor produced by the colormap.
    """
    r, g, b, _a = rgba
    # sRGB luminance
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return light_text if lum < 0.55 else dark_text


def draw_cm(ax, cm: np.ndarray, title: str) -> None:
    vmax = float(cm.max()) if cm.size else 1.0
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))
    im = ax.imshow(cm, cmap="Blues", norm=norm, interpolation="nearest")
    cmap = im.cmap
    for i in range(2):
        for j in range(2):
            v = int(cm[i, j])
            face = cmap(norm(float(v)))
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color=_annotate_text_color_for_heatmap(rgba=face),
                fontsize=16,
                fontweight="bold",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"], fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_cm_pair(
    out_png: Path,
    title: str,
    metrics_path: Path,
    split_titles: tuple[str, str] = ("Validation", "Test"),
) -> None:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    thr = float(data.get("threshold", 0.5))
    val_s, test_s = pick_sets(data["sets"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    fig.suptitle(f"{title}\nConfusion matrix heatmaps · threshold = {thr} · {metrics_path.name}", fontsize=12)
    for ax, s, st in zip(axes, (val_s, test_s), split_titles):
        cm = np.array(s["confusion_matrix"]["matrix"], dtype=float)
        draw_cm(ax, cm, f"{st} (n={s['n']})")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("saved:", out_png)


def plot_roc_pair(
    out_png: Path,
    title: str,
    rf_path: Path,
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    rf_label: str,
) -> None:
    rf = joblib.load(rf_path)
    X_tr, y_tr = load_xy(train_npz)
    X_va, y_va = load_xy(val_npz)
    X_te, y_te = load_xy(test_npz)

    lr = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    lr.fit(X_tr, y_tr)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    fig.suptitle(f"{title}\nROC: {rf_label} vs logistic regression (LR fit on train)", fontsize=12)

    for ax, X, y, st in (
        (axes[0], X_va, y_va, "Validation"),
        (axes[1], X_te, y_te, "Test"),
    ):
        if y.max() == y.min():
            ax.text(0.5, 0.5, "single class", ha="center", va="center")
            ax.set_title(st)
            continue
        rf_p = rf.predict_proba(X)[:, 1].astype(float)
        lr_p = lr.predict_proba(X)[:, 1].astype(float)
        for scores, name, ls in (
            (rf_p, rf_label, "-"),
            (lr_p, "Logistic regression", "--"),
        ):
            fpr, tpr, _ = roc_curve(y, scores)
            a = float(auc(fpr, tpr))
            ax.plot(fpr, tpr, ls, lw=2, label=f"{name} (AUC={a:.4f})")
        ax.plot([0, 1], [0, 1], ":", color="gray", lw=0.9, label="chance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"{st} (n={len(y)})")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("saved:", out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = ap.parse_args()
    root: Path = args.root
    art = root / "artifacts"

    phase1 = {
        "metrics": art / "metrics_phase1.json",
        "rf": art / "rf_phase1_huge.joblib",
        "train": art / "phase1_features_train.npz",
        "val": art / "phase1_features_val.npz",
        "test": art / "phase1_features_test.npz",
        "plot_dir": art / "plots_phase1",
        "name": "Phase I",
        "rf_short": "Random Forest (Phase I)",
    }
    phase2 = {
        "metrics": art / "metrics_phase2.json",
        "rf": art / "rf_phase2_huge.joblib",
        "train": art / "phase2_features_train.npz",
        "val": art / "phase2_features_val.npz",
        "test": art / "phase2_features_test.npz",
        "plot_dir": art / "plots_phase2",
        "name": "Phase II",
        "rf_short": "Random Forest (Phase II)",
    }

    for cfg, tag in (
        (phase1, "phase1"),
        (phase2, "phase2"),
    ):
        cfg["plot_dir"].mkdir(parents=True, exist_ok=True)
        plot_cm_pair(
            cfg["plot_dir"] / f"{tag}_cm_heatmap_val_test.png",
            cfg["name"],
            cfg["metrics"],
        )
        plot_roc_pair(
            cfg["plot_dir"] / f"{tag}_roc_rf_vs_logreg_val_test.png",
            cfg["name"],
            cfg["rf"],
            cfg["train"],
            cfg["val"],
            cfg["test"],
            cfg["rf_short"],
        )


if __name__ == "__main__":
    main()

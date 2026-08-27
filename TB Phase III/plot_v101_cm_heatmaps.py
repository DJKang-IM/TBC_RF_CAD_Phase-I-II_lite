"""
Phase III Version 1.01 — confusion matrices @ threshold 0.5 as heatmaps (Val / Test).

Reads artifacts/metrics_phase3_version_1_01.json (same CM as metrics_multioutput).
Output next to ROC plots: artifacts/version_1_01/plots/v101_cm_heatmap_val.png, ..._test.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pick_set(sets: list, split: str) -> dict:
    for s in sets:
        n = s["name"]
        if split == "val" and ("_val." in n or n.endswith("_val.npz")):
            return s
        if split == "test" and ("_test." in n or n.endswith("_test.npz")):
            return s
    raise KeyError(f"no {split} set in metrics JSON")


def draw_cm(ax, cm: np.ndarray, title: str) -> None:
    im = ax.imshow(cm, cmap="Blues", vmin=0, interpolation="nearest")
    vmax = float(cm.max()) if cm.size else 1.0
    thr = vmax / 2.0
    for i in range(2):
        for j in range(2):
            v = int(cm[i, j])
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color="white" if v > thr else "black",
                fontsize=14,
                fontweight="bold",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"], fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Default: <root>/artifacts/metrics_phase3_version_1_01.json",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Default: <root>/artifacts/version_1_01/plots",
    )
    args = ap.parse_args()
    root: Path = args.root
    metrics_path = args.metrics or (root / "artifacts" / "metrics_phase3_version_1_01.json")
    out_dir = args.out_dir or (root / "artifacts" / "version_1_01" / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    thr = float(data.get("threshold", 0.5))
    sets = data["sets"]

    for split in ("val", "test"):
        s = pick_set(sets, split)
        n = int(s["n"])
        per = s["per_label"]
        fig, axes = plt.subplots(2, 3, figsize=(11, 7.2), constrained_layout=True)
        fig.suptitle(
            f"Phase III Version 1.01 — confusion matrix heatmaps ({split}, n={n})\n"
            f"Threshold = {thr} · [[TN, FP], [FN, TP]] · {metrics_path.name}",
            fontsize=11,
        )
        for idx, p in enumerate(per):
            ax = axes.ravel()[idx]
            cm = np.array(p["confusion_matrix"]["matrix"], dtype=float)
            draw_cm(ax, cm, f"{p['key']} — {p['name']}")
        axes.ravel()[-1].axis("off")
        out_png = out_dir / f"v101_cm_heatmap_{split}.png"
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print("saved:", out_png)


if __name__ == "__main__":
    main()

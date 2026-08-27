"""
Draw the end-to-end TB pipeline architecture (Phases I–III + hybrid D6) as one PNG.

Default output:
  <REDACTED_PATH> 1.023(D6_XGboost)/architecture_pipeline_v101_d6_xgb1023.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DEFAULT_OUT = Path(r"D:\artifact") / "ver. 1.023(D6_XGboost)"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def box(ax, xy, w, h, text, fc, ec="0.35", fontsize=9):
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
        mutation_aspect=0.4,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="medium",
        wrap=True,
    )
    return (x, y, w, h)


def arrow(ax, a, b, color="0.25", style="->"):
    arr = FancyArrowPatch(
        a,
        b,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=1.3,
        color=color,
        shrinkA=4,
        shrinkB=4,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_png",
        type=Path,
        default=None,
        help="Default: <DEFAULT_OUT>/architecture_pipeline_v101_d6_xgb1023.png",
    )
    args = ap.parse_args()
    out_dir = ensure_dir(DEFAULT_OUT)
    out_png = args.out_png or (out_dir / "architecture_pipeline_v101_d6_xgb1023.png")

    fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    fig.suptitle(
        "TB pipeline architecture — Phase I & II (v1.01 RF) + Phase III hybrid\n"
        "(D1–D4: multi-output RF v1.01 · D6 NTM: XGBoost v1.023, same feature matrix X)",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    # --- Phase I row (y ~ 8.2) ---
    y1 = 8.0
    b1 = box(ax, (0.35, y1), 2.5, 0.95, "Phase I\nfeatures\n(.npz X)", "#E3F2FD")
    b2 = box(ax, (3.4, y1), 2.8, 0.95, "Random Forest\n(v1.01, Phase I)", "#BBDEFB")
    b3 = box(ax, (6.8, y1), 2.6, 0.95, "Binary score /\nclass\n(gate I)", "#90CAF9")
    arrow(ax, (b1[0] + b1[2], b1[1] + b1[3] / 2), (b2[0], b2[1] + b2[3] / 2))
    arrow(ax, (b2[0] + b2[2], b2[1] + b2[3] / 2), (b3[0], b3[1] + b3[3] / 2))

    ax.text(0.35, y1 + 1.15, "Phase I", fontsize=11, fontweight="bold", color="#1565C0")

    # --- Phase II row (y ~ 6.0) ---
    y2 = 5.85
    p1 = box(ax, (0.35, y2), 2.5, 0.95, "Phase II\nfeatures\n(.npz X)", "#E8F5E9")
    p2 = box(ax, (3.4, y2), 2.8, 0.95, "Random Forest\n(v1.01, Phase II)", "#C8E6C9")
    p3 = box(ax, (6.8, y2), 2.6, 0.95, "Binary score /\nclass\n(gate II)", "#A5D6A7")
    arrow(ax, (p1[0] + p1[2], p1[1] + p1[3] / 2), (p2[0], p2[1] + p2[3] / 2))
    arrow(ax, (p2[0] + p2[2], p2[1] + p2[3] / 2), (p3[0], p3[1] + p3[3] / 2))

    ax.text(0.35, y2 + 1.15, "Phase II", fontsize=11, fontweight="bold", color="#2E7D32")

    # --- Phase III block (y ~ 0.5–4.5) ---
    ax.text(0.35, 4.55, "Phase III (active vs inactive cohort, v1.01 features)", fontsize=11, fontweight="bold", color="#6A1B9A")

    fx = box(ax, (0.35, 3.35), 2.6, 0.9, "Phase III\nfeatures X\n(train / val / test)", "#F3E5F5")
    rf = box(ax, (3.45, 2.85), 3.15, 1.85, "Multi-output Random Forest\nrf_phase3_active_vs_inactive\n(v1.01, 5 heads)", "#E1BEE7")

    # D1–D4 outputs (right of RF)
    outs_y = [3.95, 3.35, 2.75, 2.15]
    labels = ["D1 AFB", "D2 TB PCR", "D3 Solid", "D4 Liquid"]
    colors = ["#CE93D8", "#CE93D8", "#CE93D8", "#CE93D8"]
    out_boxes = []
    for i, (ly, lab, c) in enumerate(zip(outs_y, labels, colors)):
        ob = box(ax, (7.35, ly - 0.28), 2.15, 0.56, f"{lab}\nP(y=1|X) from RF", c)
        out_boxes.append(ob)

    # XGB D6 branch (below RF, parallel path)
    xgb = box(ax, (3.45, 0.55), 3.15, 1.05, "XGBoost classifier\n(v1.023 D6: same X)\nscale_pos_weight,\ntree_method=hist", "#B2DFDB")
    d6 = box(ax, (7.35, 0.72), 2.15, 0.72, "D6 NTM\nP(y=1|X) from\nXGBoost (replaces\nRF D6 head)", "#80CBC4")

    arrow(ax, (fx[0] + fx[2], fx[1] + fx[3] / 2), (rf[0], rf[1] + rf[3] * 0.65))
    arrow(ax, (fx[0] + fx[2], fx[1] + fx[3] * 0.35), (xgb[0], xgb[1] + xgb[3]))

    # RF -> D1..D4
    rf_cx, rf_cy = rf[0] + rf[2], rf[1] + rf[3] / 2
    for ob in out_boxes:
        mid_y = ob[1] + ob[3] / 2
        arrow(ax, (rf_cx, rf_cy), (ob[0], mid_y), color="#4A148C")

    arrow(ax, (xgb[0] + xgb[2], xgb[1] + xgb[3] / 2), (d6[0], d6[1] + d6[3] / 2), color="#00695C")

    # Annotation: hybrid predict_proba
    note = (
        "Inference (hybrid): predict_proba returns five heads —\n"
        "indices 0–3 from RF unchanged; index 4 (D6) from XGBoost only."
    )
    ax.text(
        10.05,
        3.5,
        note,
        fontsize=8.8,
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFDE7", edgecolor="#F9A825", linewidth=1),
    )

    # Dashed: unused RF D6 head (conceptual)
    ax.plot(
        [rf[0] + rf[2] * 0.55, 7.0],
        [rf[1] + rf[3] * 0.12, 1.35],
        linestyle="--",
        color="#9E9E9E",
        linewidth=1.0,
    )
    ax.text(7.05, 1.38, "RF D6 head\n(not used)", fontsize=7.5, color="#616161", style="italic")

    # Footer
    ax.text(
        0.35,
        0.12,
        "Data: TB Test DB (Phase I/II) · TB Phase III (Phase III npz + RF joblib). "
        "Plots & metrics default folder: <REDACTED_PATH> 1.023(D6_XGboost)\\",
        fontsize=7.5,
        color="#424242",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", out_png.resolve())


if __name__ == "__main__":
    main()

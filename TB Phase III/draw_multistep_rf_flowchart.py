"""
Publication-quality flowchart: multistep Random Forest binary classification framework.

Outputs a sharp PNG (default 300 DPI) on a clean white background.

Usage:
  python draw_multistep_rf_flowchart.py
  python draw_multistep_rf_flowchart.py --out_png <REDACTED_PATH> --dpi 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


def rounded_box(ax, xy, w, h, text, *, fc, ec, lw=1.0, fontsize=8.5, text_color="0.1"):
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        mutation_aspect=0.35,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=text_color)
    return (x, y, w, h)


def diamond(ax, cx, cy, rw, rh, text, *, fc="white", ec="0.25", lw=1.1, fontsize=8):
    verts = [(cx, cy + rh), (cx + rw, cy), (cx, cy - rh), (cx - rw, cy)]
    poly = Polygon(verts, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, fontweight="medium", color="0.1")
    return verts


def arrow(ax, a, b, *, color="0.2", lw=1.15, rad=0.0):
    arr = FancyArrowPatch(
        a,
        b,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        shrinkA=3,
        shrinkB=3,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def draw_simple_cxr_icon(ax, cx, cy, w, h):
    """Minimal PA chest schematic: lung fields + mediastinum (not a face)."""
    x0, y0 = cx - w / 2, cy - h / 2
    frame = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="0.35",
        facecolor="#FAFAFA",
    )
    ax.add_patch(frame)
    # clavicles
    ax.plot([cx - 0.35, cx - 0.08], [y0 + h * 0.78, y0 + h * 0.72], color="0.4", lw=1.0)
    ax.plot([cx + 0.08, cx + 0.35], [y0 + h * 0.72, y0 + h * 0.78], color="0.4", lw=1.0)
    # mediastinal stripe
    ax.add_patch(
        FancyBboxPatch(
            (cx - 0.06, y0 + h * 0.28),
            0.12,
            h * 0.38,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor="#E0E0E0",
            edgecolor="0.45",
            linewidth=0.7,
        )
    )
    # lung fields
    for sign in (-1, 1):
        ell = FancyBboxPatch(
            (cx + sign * 0.34 - 0.2, y0 + h * 0.22),
            0.34,
            h * 0.48,
            boxstyle="round,pad=0.02,rounding_size=0.22",
            facecolor="#EEEEEE",
            edgecolor="0.42",
            linewidth=0.85,
        )
        ax.add_patch(ell)


def draw_gears(ax, x, y):
    ax.text(x - 0.11, y, "\u2699", fontsize=13, ha="center", va="center", color="0.2")
    ax.text(x + 0.11, y, "\u2699", fontsize=13, ha="center", va="center", color="0.2")


def draw_check_badge(ax, cx, cy, r=0.11):
    circ = Circle((cx, cy), r, facecolor="0.12", edgecolor="0.05", linewidth=0.8)
    ax.add_patch(circ)
    ax.plot(
        [cx - 0.04, cx - 0.01, cx + 0.055],
        [cy - 0.01, cy - 0.04, cy + 0.04],
        color="white",
        lw=1.4,
        solid_capstyle="round",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_png",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "multistep_rf_binary_framework.png",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--out_pdf",
        type=Path,
        default=None,
        help="Optional vector PDF (journal print). Default: sibling of PNG with .pdf extension.",
    )
    args = ap.parse_args()
    if args.out_pdf is None:
        args.out_pdf = args.out_png.with_suffix(".pdf")
    args.out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.linewidth": 0.8,
        }
    )

    fig_w, fig_h = 15.0, 4.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi, facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    y_m = 1.85
    y_u = 3.55
    box_h = 0.62
    box_w_n = 1.35

    fig.text(
        0.5,
        0.94,
        "Suggested model: multistep Random Forest (RF) binary classification framework",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="0.05",
    )

    draw_simple_cxr_icon(ax, 0.85, y_m + 0.05, 1.0, 1.05)
    ax.text(0.85, y_m - 0.75, "X-ray image", ha="center", va="top", fontsize=9, color="0.15")

    x_d1, x_d2, x_d3 = 2.55, 5.45, 8.35
    rw, rh = 0.62, 0.48

    diamond(ax, x_d1, y_m, rw, rh, "Step 1:\nNormal / Abnormal\n(RF model)", fontsize=7.2)

    n1 = rounded_box(
        ax,
        (x_d1 - box_w_n / 2, y_u),
        box_w_n,
        box_h,
        "Normal\n10,000 cases",
        fc="#F5F7FA",
        ec="0.35",
        fontsize=7.5,
    )
    arrow(ax, (x_d1, y_m + rh + 0.02), (n1[0] + n1[2] / 2, n1[1] - 0.02), rad=0.12)
    e1 = Circle((n1[0] + n1[2] / 2, y_u + box_h + 0.35), 0.18, facecolor="white", edgecolor="0.35", linewidth=1.0)
    ax.add_patch(e1)
    ax.text(n1[0] + n1[2] / 2, y_u + box_h + 0.35, "End", ha="center", va="center", fontsize=7.5)
    arrow(ax, (n1[0] + n1[2] / 2, n1[1] + n1[3] + 0.02), (n1[0] + n1[2] / 2, y_u + box_h + 0.14))

    ab1 = rounded_box(ax, (3.55, y_m - box_h / 2), 1.45, box_h, "Abnormal\n10,000 cases", fc="#EEF2F7", ec="0.35")
    arrow(ax, (x_d1 + rw + 0.05, y_m), (ab1[0] - 0.02, y_m))

    diamond(ax, x_d2, y_m, rw, rh, "Step 2:\nActivity\n(RF model)", fontsize=7.2)
    arrow(ax, (ab1[0] + ab1[2] + 0.02, y_m), (x_d2 - rw - 0.05, y_m))

    n2 = rounded_box(
        ax,
        (x_d2 - box_w_n / 2, y_u),
        box_w_n,
        box_h,
        "Inactive\n5,000 cases",
        fc="#F5F7FA",
        ec="0.35",
        fontsize=7.5,
    )
    arrow(ax, (x_d2, y_m + rh + 0.02), (n2[0] + n2[2] / 2, n2[1] - 0.02), rad=0.12)
    e2 = Circle((n2[0] + n2[2] / 2, y_u + box_h + 0.35), 0.18, facecolor="white", edgecolor="0.35", linewidth=1.0)
    ax.add_patch(e2)
    ax.text(n2[0] + n2[2] / 2, y_u + box_h + 0.35, "End", ha="center", va="center", fontsize=7.5)
    arrow(ax, (n2[0] + n2[2] / 2, n2[1] + n2[3] + 0.02), (n2[0] + n2[2] / 2, y_u + box_h + 0.14))

    ab2 = rounded_box(ax, (6.45, y_m - box_h / 2), 1.35, box_h, "Active\n5,000 cases", fc="#EEF2F7", ec="0.35")
    arrow(ax, (x_d2 + rw + 0.05, y_m), (ab2[0] - 0.02, y_m))

    diamond(ax, x_d3, y_m, rw, rh, "Step 3:\nLesion type\n(RF model)", fontsize=7.2)
    arrow(ax, (ab2[0] + ab2[2] + 0.02, y_m), (x_d3 - rw - 0.05, y_m))

    n3w = 1.55
    n3 = rounded_box(
        ax,
        (x_d3 - n3w / 2, y_u),
        n3w,
        box_h,
        "NTM\n(nontuberculous mycobacteria)",
        fc="#F5F7FA",
        ec="0.35",
        fontsize=7.0,
    )
    arrow(ax, (x_d3, y_m + rh + 0.02), (n3[0] + n3[2] / 2, n3[1] - 0.02), rad=0.12)
    e3 = Circle((n3[0] + n3[2] / 2, y_u + box_h + 0.35), 0.18, facecolor="white", edgecolor="0.35", linewidth=1.0)
    ax.add_patch(e3)
    ax.text(n3[0] + n3[2] / 2, y_u + box_h + 0.35, "End", ha="center", va="center", fontsize=7.5)
    arrow(ax, (n3[0] + n3[2] / 2, n3[1] + n3[3] + 0.02), (n3[0] + n3[2] / 2, y_u + box_h + 0.14))

    ab3 = rounded_box(ax, (9.35, y_m - box_h / 2), 1.25, box_h, "Active TB", fc="#EEF2F7", ec="0.35", fontsize=8)
    arrow(ax, (x_d3 + rw + 0.05, y_m), (ab3[0] - 0.02, y_m))

    ix, iy = 11.15, y_m - box_h / 2
    iw, ih = 2.35, 0.95
    draw_gears(ax, ix + iw / 2, iy + ih + 0.12)
    inf = FancyBboxPatch(
        (ix, iy),
        iw,
        ih,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=2.0,
        edgecolor="#B71C1C",
        facecolor="#FFEBEE",
    )
    ax.add_patch(inf)
    ax.text(
        ix + iw / 2,
        iy + ih / 2,
        "Infectivity prediction model\n(inference infectivity vector)",
        ha="center",
        va="center",
        fontsize=7.8,
        color="0.12",
    )
    arrow(ax, (ab3[0] + ab3[2] + 0.02, y_m), (ix - 0.02, y_m))

    fw, fh = 1.45, box_h
    fx = ix + iw + 0.55
    draw_check_badge(ax, fx + fw / 2, y_m + fh + 0.38)
    final = rounded_box(ax, (fx, y_m - box_h / 2), fw, fh, "Final result", fc="#E8F5E9", ec="0.35", fontsize=8.5)
    arrow(ax, (ix + iw + 0.02, y_m), (final[0] - 0.02, y_m))

    arrow(ax, (1.35, y_m), (x_d1 - rw - 0.05, y_m))

    fig.savefig(args.out_png, dpi=args.dpi, facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(args.out_pdf, format="pdf", facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print("saved:", args.out_png.resolve())
    print("saved:", args.out_pdf.resolve())


if __name__ == "__main__":
    main()

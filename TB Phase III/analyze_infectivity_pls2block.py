"""
Infectivity composite — two-block PLS-PM + higher-order construct (HOC).

Why a PLS branch (separate from polychoric / tetrachoric)
---------------------------------------------------------
The polychoric and tetrachoric branches are reflective latent-trait models:
they posit an underlying continuous Infectivity that *causes* each indicator,
and they correct the Pearson attenuation that pulled Cavity's CFA-SEM loading
down. They still rest on the assumption that one (or two correlated) latent
traits exist.

PLS-PM (Partial Least Squares Path Modelling, Wold 1982) takes a different
stance: the construct is a **composite** of its indicators (formative-friendly)
rather than a hidden cause. There is no distributional assumption on the items
and no global ML fit; instead, block scores are extracted iteratively (NIPALS)
so that they maximise the inner-model correlation. For this study that gives
the user a principled answer to "how do I weight the blocks?": the weights are
not hand-picked but emerge from the data via outer-loadings within each block
and a higher-order composite step between blocks.

Model
-----
  Block 1 — Imaging       : { Cavity }
  Block 2 — Microbiology  : { AFB_Smear, TB_PCR, Solid_Culture, Liquid_Culture }

  Stage 1: within-block scores via Mode A (reflective) NIPALS.
           F_imaging = z(Cavity) trivially; w_imaging = [1].
           F_micro   = PC1-aligned composite of the four sputum tests;
                       w_micro = the converged outer weights (unit norm).

  Stage 2: higher-order Infectivity composite via PC1 on (F_imaging, F_micro):
           F_inf = v[0] * F_imaging + v[1] * F_micro
           v     = leading unit eigenvector of corr(F_imaging, F_micro).

Effective indicator weight on Infectivity (the "비중" the user wanted explicit):
  w_Cavity_eff           = 1 * v[0]
  w_AFB_eff              = w_micro[0] * v[1]
  w_TB_PCR_eff           = w_micro[1] * v[1]
  w_Solid_Culture_eff    = w_micro[2] * v[1]
  w_Liquid_Culture_eff   = w_micro[3] * v[1]

Reliability and validity:
  Cronbach alpha and composite reliability rho_c per block; AVE per block;
  Fornell-Larcker discriminant validity (sqrt(AVE) vs |corr|).

Inference:
  Nonparametric bootstrap on the entire pipeline (default 500 resamples)
  -> percentile 95 percent CI for every outer / inner / effective weight and
  for the AVE / rho_c reliability statistics.

Outputs (--out)
---------------
  outer_weights_microbio.csv     w_micro (mean, SE, CI) per indicator
  hoc_outer_v.csv                v[0], v[1] (mean, SE, CI)
  effective_weights.csv          per-indicator weight on Infectivity (incl. share = w_eff^2 / sum)
  per_patient_scores.csv         F_imaging, F_micro, F_inf per row (kept in artifacts/)
  reliability.csv                Cronbach alpha, rho_c, AVE per block + Fornell-Larcker check
  path_diagram_pls.png           PLS-PM diagram with annotated weights
  bar_effective_weights.png      effective indicator weights with 95% CI error bars
  report.txt, summary.json

Usage
-----
  python analyze_infectivity_pls2block.py \
      --npz <.npz> --meta-main <meta.csv> [--meta-d5 <meta_d5.csv>] \
      [--bootstrap 500] [--seed 42] \
      --out artifacts/infectivity_pls

Dependencies: numpy, scipy, pandas, matplotlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_infectivity_latent import (  # noqa: E402
    COLS,
    load_from_csv,
    load_from_npz,
    meta_complete_case_mask,
)


# ---------------------------------------------------------------------------
# Block PLS-PM (Mode A reflective)
# ---------------------------------------------------------------------------


def _zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def mode_a_outer_weights(Z_block: np.ndarray, max_iter: int = 200, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Single-block Mode A reflective NIPALS.

    Returns (w, F) where w are unit-norm outer weights and F is the unit-norm
    block composite (after scaling Z_block @ w to unit variance).
    For a single-indicator block this returns w = [1] and F = z_indicator.
    For a multi-indicator block this converges to the leading singular direction
    of Z_block (i.e., PC1) with sign chosen by sum(w) > 0.
    """
    n, p = Z_block.shape
    if p == 1:
        F = Z_block[:, 0].copy()
        # Already standardized; scale to unit norm in n
        F = F / max(np.std(F, ddof=1), 1e-12)
        return np.array([1.0]), F
    w = np.ones(p) / np.sqrt(p)
    for _ in range(max_iter):
        F = Z_block @ w
        F = F / max(np.linalg.norm(F), 1e-12)
        w_new = Z_block.T @ F  # = corr(Z_j, F) up to scaling when Z is standardized
        w_new = w_new / max(np.linalg.norm(w_new), 1e-12)
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
    if float(np.sum(w)) < 0:
        w = -w
    F = Z_block @ w
    # Rescale F to unit variance for downstream interpretability
    F = F / max(np.std(F, ddof=1), 1e-12)
    return w, F


def hoc_composite(F_block_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Higher-order composite from block-score matrix (n, B) via leading
    eigenvector of the inter-block correlation matrix. Returns (v, F_hoc) where
    v is the unit-norm outer vector and F_hoc has unit variance."""
    Z = _zscore(F_block_scores)
    R = np.corrcoef(Z.T)
    if R.ndim == 0:
        R = np.array([[1.0]])
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = int(np.argmax(eigvals))
    v = eigvecs[:, idx]
    if float(np.sum(v)) < 0:
        v = -v
    F_hoc = Z @ v
    F_hoc = F_hoc / max(np.std(F_hoc, ddof=1), 1e-12)
    return v, F_hoc


# ---------------------------------------------------------------------------
# Reliability / validity
# ---------------------------------------------------------------------------


def cronbach_alpha(Z_block: np.ndarray) -> float:
    n, p = Z_block.shape
    if p < 2:
        return float("nan")
    var_total = float(np.var(Z_block.sum(axis=1), ddof=1))
    var_items = float(np.sum(np.var(Z_block, axis=0, ddof=1)))
    if var_total <= 0:
        return float("nan")
    return float(p / (p - 1) * (1 - var_items / var_total))


def composite_reliability(loadings: np.ndarray) -> float:
    """rho_c = (Σλ)^2 / [(Σλ)^2 + Σ(1 - λ^2)]; reflective."""
    l = np.asarray(loadings, dtype=float)
    if l.size < 2:
        return float("nan")
    num = float(np.sum(l)) ** 2
    den = num + float(np.sum(1.0 - l**2))
    return float(num / den) if den > 0 else float("nan")


def ave(loadings: np.ndarray) -> float:
    l = np.asarray(loadings, dtype=float)
    if l.size == 0:
        return float("nan")
    return float(np.mean(l**2))


# ---------------------------------------------------------------------------
# Pipeline + bootstrap
# ---------------------------------------------------------------------------


IMAGING_NAMES = ["Cavity"]
MICRO_NAMES = ["AFB_Smear", "TB_PCR", "Solid_Culture", "Liquid_Culture"]


def run_pipeline(X: np.ndarray) -> dict:
    """One pass through the full PLS-PM + HOC pipeline. X is (n, 5) raw 0/1."""
    df_idx = {c: i for i, c in enumerate(COLS)}
    Z = _zscore(X.astype(float))

    Z_imaging = Z[:, [df_idx[c] for c in IMAGING_NAMES]]
    Z_micro = Z[:, [df_idx[c] for c in MICRO_NAMES]]

    w_im, F_im = mode_a_outer_weights(Z_imaging)
    w_mi, F_mi = mode_a_outer_weights(Z_micro)

    # Outer LOADINGS (Pearson r between each indicator and its block score)
    L_im = np.array([np.corrcoef(Z_imaging[:, j], F_im)[0, 1] for j in range(Z_imaging.shape[1])])
    L_mi = np.array([np.corrcoef(Z_micro[:, j], F_mi)[0, 1] for j in range(Z_micro.shape[1])])

    # HOC step
    F_blocks = np.column_stack([F_im, F_mi])
    v, F_inf = hoc_composite(F_blocks)
    # Path coefficients block -> Infectivity = corr(F_block, F_inf)
    beta_im = float(np.corrcoef(F_im, F_inf)[0, 1])
    beta_mi = float(np.corrcoef(F_mi, F_inf)[0, 1])

    # Effective per-indicator weights on Infectivity
    eff = {}
    for j, name in enumerate(IMAGING_NAMES):
        eff[name] = float(w_im[j] * v[0])
    for j, name in enumerate(MICRO_NAMES):
        eff[name] = float(w_mi[j] * v[1])

    # Reliability per block
    alpha_im = cronbach_alpha(Z_imaging)
    alpha_mi = cronbach_alpha(Z_micro)
    rho_im = composite_reliability(L_im)
    rho_mi = composite_reliability(L_mi)
    ave_im = ave(L_im)
    ave_mi = ave(L_mi)

    inter_block_corr = float(np.corrcoef(F_im, F_mi)[0, 1])

    return {
        "z_block": {"imaging": Z_imaging, "micro": Z_micro},
        "outer_weights_imaging": w_im.tolist(),
        "outer_weights_micro": w_mi.tolist(),
        "outer_loadings_imaging": L_im.tolist(),
        "outer_loadings_micro": L_mi.tolist(),
        "v_hoc": v.tolist(),
        "beta_imaging_to_inf": beta_im,
        "beta_micro_to_inf": beta_mi,
        "effective_weights": eff,
        "F_imaging": F_im,
        "F_micro": F_mi,
        "F_infectivity": F_inf,
        "inter_block_correlation": inter_block_corr,
        "reliability": {
            "imaging": {"alpha": alpha_im, "rho_c": rho_im, "ave": ave_im},
            "micro": {"alpha": alpha_mi, "rho_c": rho_mi, "ave": ave_mi},
        },
    }


def bootstrap_pipeline(X: np.ndarray, B: int = 500, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # Storage
    w_mi_bs = np.zeros((B, len(MICRO_NAMES)))
    v_bs = np.zeros((B, 2))
    eff_bs = {c: np.zeros(B) for c in COLS}
    beta_im_bs = np.zeros(B)
    beta_mi_bs = np.zeros(B)
    rho_mi_bs = np.zeros(B)
    ave_mi_bs = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        # Guard against constant columns under resampling
        if any(np.unique(Xb[:, j]).size < 2 for j in range(Xb.shape[1])):
            # Skip this iteration by copying a NaN placeholder
            w_mi_bs[b] = np.nan
            v_bs[b] = np.nan
            for c in COLS:
                eff_bs[c][b] = np.nan
            beta_im_bs[b] = beta_mi_bs[b] = np.nan
            rho_mi_bs[b] = ave_mi_bs[b] = np.nan
            continue
        res = run_pipeline(Xb)
        w_mi_bs[b] = res["outer_weights_micro"]
        v_bs[b] = res["v_hoc"]
        for c in COLS:
            eff_bs[c][b] = res["effective_weights"][c]
        beta_im_bs[b] = res["beta_imaging_to_inf"]
        beta_mi_bs[b] = res["beta_micro_to_inf"]
        rho_mi_bs[b] = res["reliability"]["micro"]["rho_c"]
        ave_mi_bs[b] = res["reliability"]["micro"]["ave"]
    return {
        "B": int(B),
        "outer_weights_micro": w_mi_bs,
        "v_hoc": v_bs,
        "effective_weights": eff_bs,
        "beta_imaging_to_inf": beta_im_bs,
        "beta_micro_to_inf": beta_mi_bs,
        "rho_c_micro": rho_mi_bs,
        "ave_micro": ave_mi_bs,
    }


def _summarise(boot_array: np.ndarray) -> dict:
    a = np.asarray(boot_array, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "mean": float(np.mean(a)),
        "se": float(np.std(a, ddof=1)) if a.size > 1 else float("nan"),
        "ci_low": float(np.quantile(a, 0.025)),
        "ci_high": float(np.quantile(a, 0.975)),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_path_diagram(res: dict, boot_summary: dict | None, out_png: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    w_im = res["outer_weights_imaging"]
    w_mi = res["outer_weights_micro"]
    v = res["v_hoc"]
    beta_im = res["beta_imaging_to_inf"]
    beta_mi = res["beta_micro_to_inf"]

    fig, ax = plt.subplots(figsize=(11.5, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    hoc = (10.0, 4.0)
    ax.add_patch(plt.Circle(hoc, 0.7, fc="lightyellow", ec="black", lw=2, zorder=3))
    ax.text(*hoc, "Infectivity\n(HOC)", ha="center", va="center", fontsize=10, weight="bold")

    f_im = (6.5, 6.4)
    f_mi = (6.5, 1.6)
    ax.add_patch(plt.Circle(f_im, 0.55, fc="#dde7ff", ec="black", lw=2, zorder=3))
    ax.text(*f_im, "Imaging\ncomposite", ha="center", va="center", fontsize=9, weight="bold")
    ax.add_patch(plt.Circle(f_mi, 0.55, fc="#ffe6dd", ec="black", lw=2, zorder=3))
    ax.text(*f_mi, "Microbio\ncomposite", ha="center", va="center", fontsize=9, weight="bold")

    # Imaging indicators
    indicator_y_im = [6.4]
    for j, name in enumerate(IMAGING_NAMES):
        bx, by = 2.0, indicator_y_im[j]
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (bx - 0.85, by - 0.28), 1.7, 0.56, boxstyle="round,pad=0.02", ec="black", fc="white"
            )
        )
        ax.text(bx, by, name.replace("_", " "), ha="center", va="center", fontsize=8)
        ax.annotate(
            "",
            xy=f_im,
            xytext=(bx + 0.85, by),
            arrowprops=dict(arrowstyle="->", color="#1f3a93", lw=1.6),
        )
        mid_x, mid_y = (bx + 0.85 + f_im[0]) / 2, (by + f_im[1]) / 2
        ax.text(mid_x, mid_y + 0.15, f"w = {w_im[j]:.3f}", fontsize=8, color="#1f3a93")

    # Microbio indicators
    indicator_y_mi = np.linspace(0.5, 2.7, len(MICRO_NAMES))
    for j, name in enumerate(MICRO_NAMES):
        bx, by = 2.0, float(indicator_y_mi[j])
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (bx - 0.85, by - 0.24), 1.7, 0.48, boxstyle="round,pad=0.02", ec="black", fc="white"
            )
        )
        ax.text(bx, by, name.replace("_", " "), ha="center", va="center", fontsize=8)
        ax.annotate(
            "",
            xy=f_mi,
            xytext=(bx + 0.85, by),
            arrowprops=dict(arrowstyle="->", color="#a93226", lw=1.6),
        )
        mid_x, mid_y = (bx + 0.85 + f_mi[0]) / 2, (by + f_mi[1]) / 2
        ax.text(mid_x, mid_y + 0.15, f"w = {w_mi[j]:.3f}", fontsize=8, color="#a93226")

    # HOC arrows
    ax.annotate(
        "", xy=hoc, xytext=(f_im[0] + 0.55, f_im[1] - 0.2),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.8),
    )
    ax.text(
        (f_im[0] + hoc[0]) / 2, (f_im[1] + hoc[1]) / 2 + 0.2,
        f"v = {v[0]:.3f}   beta = {beta_im:.3f}", fontsize=8, color="black",
    )
    ax.annotate(
        "", xy=hoc, xytext=(f_mi[0] + 0.55, f_mi[1] + 0.2),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.8),
    )
    ax.text(
        (f_mi[0] + hoc[0]) / 2, (f_mi[1] + hoc[1]) / 2 - 0.4,
        f"v = {v[1]:.3f}   beta = {beta_mi:.3f}", fontsize=8, color="black",
    )

    # Inter-block dotted line
    ax.plot([f_im[0], f_mi[0]], [f_im[1], f_mi[1]], color="purple", linestyle=":", lw=1.0, alpha=0.6)
    ax.text(
        f_im[0] - 1.3, (f_im[1] + f_mi[1]) / 2,
        f"corr(F_im, F_mi) = {res['inter_block_correlation']:.3f}", fontsize=8, color="purple",
    )

    title = (
        "Two-block PLS-PM + HOC Infectivity\n"
        f"Cronbach alpha(micro) = {res['reliability']['micro']['alpha']:.3f}    "
        f"rho_c(micro) = {res['reliability']['micro']['rho_c']:.3f}    "
        f"AVE(micro) = {res['reliability']['micro']['ave']:.3f}"
    )
    ax.set_title(title, fontsize=10)
    ax.legend(
        handles=[
            mpatches.Patch(color="#1f3a93", label="Imaging block outer (w)"),
            mpatches.Patch(color="#a93226", label="Microbiology block outer (w)"),
            mpatches.Patch(color="black", label="HOC outer (v) and path (beta)"),
        ],
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_effective_weights_bar(
    eff_point: dict[str, float], eff_summary: dict[str, dict[str, float]] | None, out_png: Path
) -> None:
    import matplotlib.pyplot as plt

    names = COLS
    point = np.array([eff_point[c] for c in names], dtype=float)
    if eff_summary is not None:
        err_lo = np.array([eff_summary[c]["mean"] - eff_summary[c]["ci_low"] for c in names])
        err_hi = np.array([eff_summary[c]["ci_high"] - eff_summary[c]["mean"] for c in names])
        yerr = np.vstack([np.abs(err_lo), np.abs(err_hi)])
    else:
        yerr = None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, point, color=["#1f3a93", "#a93226", "#a93226", "#a93226", "#a93226"])
    if yerr is not None:
        ax.errorbar(np.arange(len(names)), point, yerr=yerr, fmt="none", color="black", capsize=4)
    ax.axhline(0, color="gray", lw=0.7)
    for bar, val in zip(bars, point):
        offset = 0.005 if val >= 0 else -0.02
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val:.3f}",
                ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    ax.set_ylabel("Effective weight on Infectivity composite")
    ax.set_title("PLS-PM + HOC effective indicator weights"
                 + (" (mean +- 95% bootstrap CI)" if eff_summary is not None else ""))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Two-block PLS-PM + HOC composite for infectivity")
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--meta-main", type=Path, default=None)
    ap.add_argument("--meta-d5", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=500, help="Bootstrap reps (0 to disable).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if (args.npz is None) == (args.csv is None):
        print("Provide exactly one of --npz or --csv", file=sys.stderr)
        return 2

    meta_info = None
    if args.npz is not None:
        df, paths = load_from_npz(args.npz)
        if args.meta_main is not None:
            if paths is None:
                print("NPZ missing 'paths'; cannot apply meta filter.", file=sys.stderr)
                return 2
            mask, meta_info = meta_complete_case_mask(paths, args.meta_main, args.meta_d5)
            df = df.loc[mask].reset_index(drop=True)
    else:
        df, _ = load_from_csv(args.csv)

    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    X = df[COLS].values.astype(int)
    n = len(df)
    if n < 30:
        print(f"[warn] only n = {n} rows.", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)

    res = run_pipeline(X)
    boot_summary = None
    if args.bootstrap > 0:
        boot = bootstrap_pipeline(X, B=args.bootstrap, seed=args.seed)
        boot_summary = {
            "outer_weights_micro": [_summarise(boot["outer_weights_micro"][:, j]) for j in range(len(MICRO_NAMES))],
            "v_hoc": [_summarise(boot["v_hoc"][:, j]) for j in range(2)],
            "effective_weights": {c: _summarise(boot["effective_weights"][c]) for c in COLS},
            "beta_imaging_to_inf": _summarise(boot["beta_imaging_to_inf"]),
            "beta_micro_to_inf": _summarise(boot["beta_micro_to_inf"]),
            "rho_c_micro": _summarise(boot["rho_c_micro"]),
            "ave_micro": _summarise(boot["ave_micro"]),
        }

    # --- Save tables --------------------------------------------------------
    df_micro_w = pd.DataFrame(
        {
            "indicator": MICRO_NAMES,
            "outer_weight": res["outer_weights_micro"],
            "outer_loading": res["outer_loadings_micro"],
        }
    )
    if boot_summary is not None:
        df_micro_w["boot_mean"] = [s["mean"] for s in boot_summary["outer_weights_micro"]]
        df_micro_w["boot_se"] = [s["se"] for s in boot_summary["outer_weights_micro"]]
        df_micro_w["ci_low"] = [s["ci_low"] for s in boot_summary["outer_weights_micro"]]
        df_micro_w["ci_high"] = [s["ci_high"] for s in boot_summary["outer_weights_micro"]]
    df_micro_w.to_csv(args.out / "outer_weights_microbio.csv", index=False, encoding="utf-8")

    hoc_rows = [
        {"block": "Imaging", "v": res["v_hoc"][0], "beta_path": res["beta_imaging_to_inf"]},
        {"block": "Microbiology", "v": res["v_hoc"][1], "beta_path": res["beta_micro_to_inf"]},
    ]
    if boot_summary is not None:
        for i, block in enumerate(("Imaging", "Microbiology")):
            row = next(r for r in hoc_rows if r["block"] == block)
            s = boot_summary["v_hoc"][i]
            row["v_ci_low"] = s["ci_low"]
            row["v_ci_high"] = s["ci_high"]
            row["v_se"] = s["se"]
        bsim = boot_summary["beta_imaging_to_inf"]
        bsmi = boot_summary["beta_micro_to_inf"]
        hoc_rows[0].update({"beta_ci_low": bsim["ci_low"], "beta_ci_high": bsim["ci_high"], "beta_se": bsim["se"]})
        hoc_rows[1].update({"beta_ci_low": bsmi["ci_low"], "beta_ci_high": bsmi["ci_high"], "beta_se": bsmi["se"]})
    pd.DataFrame(hoc_rows).to_csv(args.out / "hoc_outer_v.csv", index=False, encoding="utf-8")

    eff_rows = []
    eff_vals = np.array([res["effective_weights"][c] for c in COLS], dtype=float)
    share = (eff_vals**2) / max(float(np.sum(eff_vals**2)), 1e-12)
    for c, w, s in zip(COLS, eff_vals, share):
        row = {"indicator": c, "effective_weight": float(w), "share_of_variance": float(s)}
        if boot_summary is not None:
            row.update(
                {
                    "boot_mean": boot_summary["effective_weights"][c]["mean"],
                    "boot_se": boot_summary["effective_weights"][c]["se"],
                    "ci_low": boot_summary["effective_weights"][c]["ci_low"],
                    "ci_high": boot_summary["effective_weights"][c]["ci_high"],
                }
            )
        eff_rows.append(row)
    pd.DataFrame(eff_rows).to_csv(args.out / "effective_weights.csv", index=False, encoding="utf-8")

    # Per-patient composite scores (kept in artifacts/, ignored by .gitignore)
    scores_df = pd.DataFrame(
        {
            "F_imaging": res["F_imaging"],
            "F_micro": res["F_micro"],
            "F_infectivity": res["F_infectivity"],
        }
    )
    scores_df.to_csv(args.out / "per_patient_scores.csv", index=False, encoding="utf-8")

    rel = res["reliability"]
    reliab_df = pd.DataFrame(
        [
            {"block": "Imaging", "cronbach_alpha": rel["imaging"]["alpha"],
             "rho_c": rel["imaging"]["rho_c"], "AVE": rel["imaging"]["ave"]},
            {"block": "Microbiology", "cronbach_alpha": rel["micro"]["alpha"],
             "rho_c": rel["micro"]["rho_c"], "AVE": rel["micro"]["ave"]},
        ]
    )
    # Fornell-Larcker: sqrt(AVE) per block vs |inter-block correlation|
    sqrt_ave_micro = float(np.sqrt(max(rel["micro"]["ave"], 0.0)))
    sqrt_ave_imaging = float(np.sqrt(max(rel["imaging"]["ave"], 0.0))) if not np.isnan(rel["imaging"]["ave"]) else float("nan")
    reliab_df["sqrt_AVE_vs_interblock_corr_ok"] = [
        sqrt_ave_imaging > abs(res["inter_block_correlation"]) if not np.isnan(sqrt_ave_imaging) else "n/a (single item)",
        sqrt_ave_micro > abs(res["inter_block_correlation"]),
    ]
    reliab_df.to_csv(args.out / "reliability.csv", index=False, encoding="utf-8")

    plot_path_diagram(res, boot_summary, args.out / "path_diagram_pls.png")
    plot_effective_weights_bar(
        res["effective_weights"],
        boot_summary["effective_weights"] if boot_summary is not None else None,
        args.out / "bar_effective_weights.png",
    )

    # --- Report -------------------------------------------------------------
    lines = [
        "=== Infectivity composite — PLS-PM branch (two blocks + HOC) ===",
        f"n = {n}",
        "",
        "Block 1 (Imaging) indicators: " + ", ".join(IMAGING_NAMES),
        "Block 2 (Microbiology) indicators: " + ", ".join(MICRO_NAMES),
        "",
        "Outer weights (Mode A, NIPALS converged):",
        f"   Imaging   : Cavity w = {res['outer_weights_imaging'][0]:.4f} (trivial — single indicator)",
        "   Microbiology:",
    ]
    for n_, w_, l_ in zip(MICRO_NAMES, res["outer_weights_micro"], res["outer_loadings_micro"]):
        lines.append(f"     {n_:>16s}   w = {w_:+.4f}   loading = {l_:+.4f}")
    lines += [
        "",
        f"Inter-block correlation corr(F_imaging, F_micro) = {res['inter_block_correlation']:+.4f}",
        "",
        "Higher-order Infectivity composite (PC1 of block scores):",
        f"   v_imaging = {res['v_hoc'][0]:+.4f}    beta(Imaging -> Inf)   = {res['beta_imaging_to_inf']:+.4f}",
        f"   v_micro   = {res['v_hoc'][1]:+.4f}    beta(Microbio -> Inf)  = {res['beta_micro_to_inf']:+.4f}",
        "",
        "Effective per-indicator weights on Infectivity composite:",
    ]
    for r in sorted(eff_rows, key=lambda x: -abs(x["effective_weight"])):
        ci = f"   95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]" if "ci_low" in r else ""
        lines.append(
            f"   {r['indicator']:>16s}   w_eff = {r['effective_weight']:+.4f}   share = {r['share_of_variance']:.4f}{ci}"
        )
    lines += [
        "",
        "Reliability:",
        f"   Imaging block    : alpha = {rel['imaging']['alpha']}    rho_c = {rel['imaging']['rho_c']}    AVE = {rel['imaging']['ave']}",
        f"   Microbiology     : alpha = {rel['micro']['alpha']:.4f}    rho_c = {rel['micro']['rho_c']:.4f}    AVE = {rel['micro']['ave']:.4f}",
        "",
        "Fornell-Larcker discriminant validity:",
        f"   sqrt(AVE_micro)  = {sqrt_ave_micro:.4f}   vs  |corr(F_im, F_mi)| = {abs(res['inter_block_correlation']):.4f}"
        + ("   PASS" if sqrt_ave_micro > abs(res["inter_block_correlation"]) else "   FAIL"),
    ]
    if meta_info is not None:
        lines.extend(["", "Meta exclusion summary:", json.dumps(meta_info, indent=2, ensure_ascii=False)])
    (args.out / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "branch": "stat/pls",
        "method": "two-block PLS-PM (Mode A) + higher-order composite (PC1 of block scores)",
        "n": int(n),
        "imaging_block": IMAGING_NAMES,
        "microbio_block": MICRO_NAMES,
        "outer_weights_imaging": res["outer_weights_imaging"],
        "outer_weights_micro": res["outer_weights_micro"],
        "outer_loadings_micro": res["outer_loadings_micro"],
        "v_hoc": res["v_hoc"],
        "beta_imaging_to_inf": res["beta_imaging_to_inf"],
        "beta_micro_to_inf": res["beta_micro_to_inf"],
        "inter_block_correlation": res["inter_block_correlation"],
        "effective_weights": res["effective_weights"],
        "share_of_variance": dict(zip(COLS, share.tolist())),
        "reliability": rel,
        "bootstrap": {"B": int(args.bootstrap), "seed": int(args.seed), "summary": boot_summary},
        "meta_exclusion": meta_info,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved: {args.out / 'report.txt'}")
    print(f"Saved: {args.out / 'summary.json'}")
    print(f"Saved: {args.out / 'outer_weights_microbio.csv'}, hoc_outer_v.csv, effective_weights.csv")
    print(f"Saved: {args.out / 'reliability.csv'}, per_patient_scores.csv")
    print(f"Saved: {args.out / 'path_diagram_pls.png'}, bar_effective_weights.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

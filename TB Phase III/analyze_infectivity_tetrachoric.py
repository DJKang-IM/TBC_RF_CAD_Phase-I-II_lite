"""
Infectivity latent — TETRACHORIC correlation + DWLS factor extraction (1F / 2F).

Why a dedicated tetrachoric branch (separate from polychoric)
-----------------------------------------------------------
All five Phase III infectivity indicators (Cavity, AFB_Smear, TB_PCR,
Solid_Culture, Liquid_Culture) are coded as 0/1. For purely binary data the
polychoric MLE collapses onto the special case of the TETRACHORIC correlation
(Pearson, 1900 / Brown, 1977). This branch implements that special case directly
with three deliberate differences from `stat/polychoric`:

  1. Closed-form 2x2 likelihood (one cell probability per pair) so the per-pair
     MLE is faster and numerically cleaner; explicit Yates 0.5 continuity
     correction for cells with zero counts (rather than a marginal-only
     correction).
  2. Asymptotic standard error and Wald 95 percent CI per tetrachoric rho via
     the Fisher information of the binomial-like cell likelihood, plus an
     optional non-parametric bootstrap.
  3. Factor extraction by DWLS (diagonally weighted least squares) using those
     per-pair asymptotic variances as weights on the off-diagonal residuals.
     ULS (unit-weight) results are also reported for comparison.

Same two-factor confirmatory structure as the polychoric branch:
    F1 (Imaging)      = Cavity
    F2 (Microbiology) = AFB_Smear, TB_PCR, Solid_Culture, Liquid_Culture
    phi(F1, F2) free.
Higher-order Infectivity via Schmid-Leiman: g_i = lambda_i * sqrt(phi).

Outputs (--out)
---------------
  tet_corr.csv              tetrachoric correlation matrix
  tet_se.csv                asymptotic SE matrix (off-diagonal)
  tet_ci.csv                95 percent Wald CI (off-diagonal, lower/upper)
  pearson_vs_tetrachoric.png
  loadings_1f_uls.csv / loadings_1f_dwls.csv
  loadings_2f_uls.csv / loadings_2f_dwls.csv
  path_diagram_2f_dwls.png
  prevalence_diag.png       why Pearson r underestimates as p moves from 0.5
  report.txt, summary.json

Usage
-----
  python analyze_infectivity_tetrachoric.py \
      --npz <Phase III NPZ> --meta-main <meta.csv> [--meta-d5 <meta_d5.csv>] \
      [--bootstrap 0]  --out artifacts/infectivity_tetrachoric

Dependencies: numpy, scipy, pandas, matplotlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import multivariate_normal, norm

sys.path.insert(0, str(Path(__file__).parent))
from analyze_infectivity_latent import (  # noqa: E402
    COLS,
    load_from_csv,
    load_from_npz,
    meta_complete_case_mask,
)

# ---------------------------------------------------------------------------
# Tetrachoric correlation: 2x2 specific
# ---------------------------------------------------------------------------


def _bvn_cdf(rho: float, h: float, k: float) -> float:
    """P(X <= h, Y <= k) for std bivariate normal with correlation rho."""
    rho = float(np.clip(rho, -0.999999, 0.999999))
    rv = multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
    return float(rv.cdf([float(h), float(k)]))


def _bvn_pdf(rho: float, h: float, k: float) -> float:
    """phi_2(h, k; rho) — standard bivariate normal density at (h, k)."""
    rho = float(np.clip(rho, -0.999999, 0.999999))
    z = (h * h - 2 * rho * h * k + k * k) / (2 * (1 - rho * rho))
    return float(np.exp(-z) / (2 * np.pi * np.sqrt(1 - rho * rho)))


def _2x2_counts(x: np.ndarray, y: np.ndarray, yates: bool = True) -> tuple[np.ndarray, int]:
    n11 = int(np.sum((x == 1) & (y == 1)))
    n10 = int(np.sum((x == 1) & (y == 0)))
    n01 = int(np.sum((x == 0) & (y == 1)))
    n00 = int(np.sum((x == 0) & (y == 0)))
    nij = np.array([[n00, n01], [n10, n11]], dtype=float)
    # Yates 0.5 continuity correction when any cell == 0 (Brown, 1977 §3.6).
    if yates and (nij == 0).any():
        nij = nij + 0.5
    return nij, int(n00 + n01 + n10 + n11)


def tetrachoric_pair(
    x: np.ndarray, y: np.ndarray, yates: bool = True
) -> dict:
    """Tetrachoric correlation rho for a binary pair (Brown, 1977 MLE).

    Returns rho, SE (Fisher information based), 95 percent Wald CI, threshold
    estimates, and the 2x2 contingency table used.
    """
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    nij, n = _2x2_counts(x, y, yates=yates)
    if n == 0 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return {
            "rho": 0.0,
            "se": float("nan"),
            "ci": (float("nan"), float("nan")),
            "tau_x": float("nan"),
            "tau_y": float("nan"),
            "n": int(n),
            "table": nij.tolist(),
            "reason": "constant variable",
        }

    # Marginal thresholds
    px = float(nij[1, :].sum()) / nij.sum()  # P(x = 1) -> tau_x = Phi^-1(1 - px)
    py = float(nij[:, 1].sum()) / nij.sum()
    # Clip to avoid +-inf
    px = min(max(px, 0.5 / n), 1 - 0.5 / n)
    py = min(max(py, 0.5 / n), 1 - 0.5 / n)
    tau_x = float(norm.ppf(1 - px))
    tau_y = float(norm.ppf(1 - py))

    def cell_probs(rho: float) -> np.ndarray:
        # P(X = 0, Y = 0) = Phi_2(tau_x, tau_y; rho)
        p00 = _bvn_cdf(rho, tau_x, tau_y)
        p10 = float(norm.cdf(tau_y)) - p00  # P(X = 1, Y = 0)
        p01 = float(norm.cdf(tau_x)) - p00  # P(X = 0, Y = 1)
        p11 = 1.0 - p00 - p10 - p01
        P = np.array([[p00, p01], [p10, p11]], dtype=float)
        return np.clip(P, 1e-15, 1.0)

    def neg_loglik(rho: float) -> float:
        P = cell_probs(rho)
        return -float(np.sum(nij * np.log(P)))

    res = optimize.minimize_scalar(
        neg_loglik, bounds=(-0.99, 0.99), method="bounded", options={"xatol": 1e-6}
    )
    rho_hat = float(res.x)

    # Asymptotic SE: -d^2 logL / drho^2 at rho_hat (information).
    # d/drho P(00) = phi_2(tau_x, tau_y; rho); P(10) and P(01) depend on it
    # negatively, P(11) positively.
    phi2 = _bvn_pdf(rho_hat, tau_x, tau_y)
    P = cell_probs(rho_hat)
    # First derivatives wrt rho:
    dP = np.array(
        [
            [phi2, -phi2],
            [-phi2, phi2],
        ],
        dtype=float,
    )
    # Score = sum nij * (1/P_ij) * dP_ij/drho. Fisher info ~ sum n * (dP/drho)^2 / P.
    info = float(np.sum(n * (dP**2) / P))
    se = float(np.sqrt(1.0 / max(info, 1e-12)))
    ci_lo = float(np.clip(rho_hat - 1.96 * se, -1.0, 1.0))
    ci_hi = float(np.clip(rho_hat + 1.96 * se, -1.0, 1.0))

    return {
        "rho": rho_hat,
        "se": se,
        "ci": (ci_lo, ci_hi),
        "tau_x": tau_x,
        "tau_y": tau_y,
        "n": int(n),
        "table": nij.tolist(),
        "loglik": float(-res.fun),
        "marginal_p_x1": float(px),
        "marginal_p_y1": float(py),
    }


def tetrachoric_matrix(
    X: np.ndarray, names: list[str], bootstrap: int = 0, rng: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Return (R, SE, CI) matrices and per-pair metadata."""
    p = X.shape[1]
    R = np.eye(p, dtype=float)
    SE = np.zeros((p, p), dtype=float)
    CI_lo = np.zeros((p, p), dtype=float)
    CI_hi = np.zeros((p, p), dtype=float)
    meta: list[dict] = []
    rng_obj = np.random.default_rng(rng)
    n = X.shape[0]
    for i in range(p):
        for j in range(i + 1, p):
            res = tetrachoric_pair(X[:, i], X[:, j])
            R[i, j] = R[j, i] = res["rho"]
            SE[i, j] = SE[j, i] = res["se"]
            CI_lo[i, j] = CI_lo[j, i] = res["ci"][0]
            CI_hi[i, j] = CI_hi[j, i] = res["ci"][1]
            row = {
                "i": names[i],
                "j": names[j],
                "rho": res["rho"],
                "se_asymptotic": res["se"],
                "ci_low_wald": res["ci"][0],
                "ci_high_wald": res["ci"][1],
                "n": res["n"],
            }
            if bootstrap > 0:
                bs = []
                for _ in range(bootstrap):
                    idx = rng_obj.integers(0, n, size=n)
                    bs.append(tetrachoric_pair(X[idx, i], X[idx, j])["rho"])
                bs = np.asarray(bs, dtype=float)
                row["ci_low_boot"] = float(np.quantile(bs, 0.025))
                row["ci_high_boot"] = float(np.quantile(bs, 0.975))
                row["se_boot"] = float(np.std(bs, ddof=1))
            meta.append(row)
    return R, SE, (CI_lo, CI_hi), meta


# ---------------------------------------------------------------------------
# Factor extraction: ULS and DWLS
# ---------------------------------------------------------------------------


def _uls_one_factor(R: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Single-factor PAF iteration; if weights (shape pxp upper-tri) is given,
    compute weighted residual SS using those off-diag weights (DWLS-style).

    weights[i,j] = 1 / Var(rho_ij_asymp)  for DWLS; equal weights (1) for ULS.
    """
    p = R.shape[0]
    iu = np.triu_indices(p, k=1)
    if weights is None:
        w = np.ones_like(R, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    # Init communality
    try:
        Rinv = np.linalg.pinv(R + 1e-8 * np.eye(p))
        h2 = 1.0 - 1.0 / np.diag(Rinv)
        h2 = np.clip(h2, 0.0, 0.99)
    except np.linalg.LinAlgError:
        h2 = np.full(p, 0.5)
    # Iterate PAF (DWLS via reweighted PAF is approximate but stable; the
    # standard exact DWLS is below in _dwls_one_factor.)
    for _ in range(200):
        Rred = R.copy()
        np.fill_diagonal(Rred, h2)
        eigvals, eigvecs = np.linalg.eigh(Rred)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        lam1 = float(max(eigvals[0], 0.0))
        L_new = eigvecs[:, 0] * np.sqrt(lam1)
        h2_new = np.clip(L_new**2, 0.0, 0.99)
        if np.max(np.abs(h2_new - h2)) < 1e-9:
            h2 = h2_new
            break
        h2 = h2_new
    L = eigvecs[:, 0] * np.sqrt(lam1)
    if float(np.sum(L)) < 0:
        L = -L
    Rhat = np.outer(L, L)
    np.fill_diagonal(Rhat, 1.0)
    resid = R - Rhat
    srmr = float(np.sqrt(np.mean(resid[iu] ** 2)))
    wsrmr = float(np.sqrt(np.sum(w[iu] * resid[iu] ** 2) / max(w[iu].sum(), 1e-12)))
    return {
        "loadings": L.tolist(),
        "communalities": h2.tolist(),
        "srmr": srmr,
        "weighted_srmr": wsrmr,
        "implied_R": Rhat,
    }


def _dwls_one_factor(R: np.ndarray, W_diag_offdiag: np.ndarray) -> dict:
    """One-factor model fit by weighted least squares on the off-diagonal
    residual vector with weights W_diag_offdiag (positive). Parameters:
    loadings (p,) only; communalities follow as lambda^2.
    """
    p = R.shape[0]
    iu = np.triu_indices(p, k=1)
    r_obs = R[iu]
    w = W_diag_offdiag

    def loss(theta: np.ndarray) -> float:
        L = theta
        R_hat = np.outer(L, L)
        return float(np.sum(w * (r_obs - R_hat[iu]) ** 2))

    # Init from PAF
    init = np.asarray(_uls_one_factor(R)["loadings"], dtype=float)
    res = optimize.minimize(
        loss, init, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-10}
    )
    L = res.x
    if float(np.sum(L)) < 0:
        L = -L
    Rhat = np.outer(L, L)
    np.fill_diagonal(Rhat, 1.0)
    resid = R - Rhat
    srmr = float(np.sqrt(np.mean(resid[iu] ** 2)))
    wsrmr = float(np.sqrt(np.sum(w * resid[iu] ** 2) / max(w.sum(), 1e-12)))
    return {
        "loadings": L.tolist(),
        "communalities": (L**2).tolist(),
        "srmr": srmr,
        "weighted_srmr": wsrmr,
        "loss_wls": float(res.fun),
        "implied_R": Rhat,
    }


def fit_two_factor_constrained(
    R: np.ndarray,
    names: list[str],
    imaging_idx: list[int],
    micro_idx: list[int],
    weights_offdiag: np.ndarray | None = None,
) -> dict:
    """Constrained 2-factor CFA fit by (D)ULS depending on weights_offdiag."""
    p = R.shape[0]
    pat = np.zeros((p, 2), dtype=int)
    pat[imaging_idx, 0] = 1
    pat[micro_idx, 1] = 1
    free_mask = pat.astype(bool)
    n_free = int(free_mask.sum())
    iu = np.triu_indices(p, k=1)
    r_obs = R[iu]
    w = np.ones_like(r_obs) if weights_offdiag is None else np.asarray(weights_offdiag, dtype=float)

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, float]:
        L = np.zeros((p, 2), dtype=float)
        L[free_mask] = theta[:n_free]
        phi = float(np.tanh(theta[n_free]))
        return L, phi

    def loss(theta: np.ndarray) -> float:
        L, phi = unpack(theta)
        Phi = np.array([[1.0, phi], [phi, 1.0]], dtype=float)
        Rhat = L @ Phi @ L.T
        return float(np.sum(w * (r_obs - Rhat[iu]) ** 2))

    # Init from 1F PAF abs loadings
    L0 = np.asarray(_uls_one_factor(R)["loadings"], dtype=float)
    theta0 = np.concatenate([np.abs(L0) + 1e-3, [np.arctanh(0.6)]])
    res = optimize.minimize(
        loss, theta0, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-10}
    )
    L_hat, phi = unpack(res.x)
    for k in range(2):
        if float(np.sum(L_hat[:, k])) < 0:
            L_hat[:, k] = -L_hat[:, k]

    Phi = np.array([[1.0, phi], [phi, 1.0]], dtype=float)
    Rhat = L_hat @ Phi @ L_hat.T
    np.fill_diagonal(Rhat, 1.0)
    resid = R - Rhat
    srmr = float(np.sqrt(np.mean(resid[iu] ** 2)))
    wsrmr = float(np.sqrt(np.sum(w * resid[iu] ** 2) / max(w.sum(), 1e-12)))

    g = np.zeros(p)
    if phi > 0:
        for k in range(2):
            g += L_hat[:, k] * np.sqrt(phi)

    return {
        "loadings_F1": L_hat[:, 0].tolist(),
        "loadings_F2": L_hat[:, 1].tolist(),
        "factor_correlation_phi": float(phi),
        "srmr": srmr,
        "weighted_srmr": wsrmr,
        "loss_wls": float(res.fun),
        "schmid_leiman_general_loadings": g.tolist(),
        "names": names,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_corr_compare(R_pearson: np.ndarray, R_tet: np.ndarray, names: list[str], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, M, title in zip(axes, [R_pearson, R_tet], ["Pearson r (phi-coef)", "Tetrachoric r"]):
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.7, label="corr")
    fig.suptitle("Pearson (phi) vs Tetrachoric correlation — 5 infectivity items")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prevalence_diagnostic(X: np.ndarray, names: list[str], out_png: Path) -> None:
    """Bar chart of marginal P(item = 1) — shows which items deviate from p = 0.5,
    explaining where Pearson (phi-coefficient) underestimates the latent rho."""
    import matplotlib.pyplot as plt

    p = X.shape[1]
    prev = X.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, prev, color="steelblue")
    ax.axhline(0.5, color="red", lw=1, linestyle="--", label="p = 0.5 (Pearson unbiased)")
    for bar, p1 in zip(bars, prev):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{p1:.2f}",
                ha="center", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(item = 1)")
    ax.set_title("Marginal prevalence — distance from 0.5 indexes Pearson r attenuation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_path_diagram_2f(res_2f: dict, names: list[str], out_png: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    L1 = np.asarray(res_2f["loadings_F1"], dtype=float)
    L2 = np.asarray(res_2f["loadings_F2"], dtype=float)
    g = np.asarray(res_2f["schmid_leiman_general_loadings"], dtype=float)
    phi = float(res_2f["factor_correlation_phi"])

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    hoc = (10.0, 4.0)
    ax.add_patch(plt.Circle(hoc, 0.7, fc="lightyellow", ec="black", lw=2, zorder=3))
    ax.text(*hoc, "Infectivity\n(HOC)", ha="center", va="center", fontsize=10, weight="bold")
    f1 = (6.5, 6.2)
    f2 = (6.5, 1.8)
    ax.add_patch(plt.Circle(f1, 0.55, fc="#dde7ff", ec="black", lw=2, zorder=3))
    ax.text(*f1, "F1\nImaging", ha="center", va="center", fontsize=9, weight="bold")
    ax.add_patch(plt.Circle(f2, 0.55, fc="#ffe6dd", ec="black", lw=2, zorder=3))
    ax.text(*f2, "F2\nMicrobio", ha="center", va="center", fontsize=9, weight="bold")

    ax.annotate(
        "",
        xy=(f1[0] - 0.6, f1[1] - 0.4),
        xytext=(f2[0] - 0.6, f2[1] + 0.4),
        arrowprops=dict(arrowstyle="<->", color="purple", lw=1.6),
    )
    ax.text(f1[0] - 1.3, (f1[1] + f2[1]) / 2, f"phi = {phi:.3f}", color="purple", fontsize=9)

    indicator_y = np.linspace(0.6, 7.4, len(names))
    for i, name in enumerate(names):
        bx, by = 2.0, float(indicator_y[i])
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (bx - 0.75, by - 0.28), 1.5, 0.56, boxstyle="round,pad=0.02", ec="black", fc="white"
            )
        )
        ax.text(bx, by, name.replace("_", " "), ha="center", va="center", fontsize=8)
        if abs(L1[i]) > abs(L2[i]):
            target = f1
            est = float(L1[i])
            color = "#1f3a93"
        else:
            target = f2
            est = float(L2[i])
            color = "#a93226"
        ax.annotate(
            "",
            xy=target,
            xytext=(bx + 0.75, by),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )
        mid_x = (bx + 0.75 + target[0]) / 2
        mid_y = (by + target[1]) / 2
        ax.text(mid_x, mid_y + 0.1, f"{est:.3f}", fontsize=8, color=color)
        ax.annotate(
            "",
            xy=hoc,
            xytext=(bx + 0.75, by),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8, linestyle="dashed", alpha=0.7),
        )

    ax.annotate(
        "", xy=hoc, xytext=(f1[0] + 0.55, f1[1] - 0.2),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.4),
    )
    ax.annotate(
        "", xy=hoc, xytext=(f2[0] + 0.55, f2[1] + 0.2),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.4),
    )
    ax.set_title(
        f"Two-factor CFA on TETRACHORIC R (DWLS) + Schmid-Leiman HOC\n"
        f"SRMR(off-diag) = {res_2f['srmr']:.4f}    wSRMR = {res_2f['weighted_srmr']:.4f}    "
        f"phi(F1, F2) = {phi:.3f}",
        fontsize=10,
    )
    legend = [
        mpatches.Patch(color="#1f3a93", label="loading on F1 (Imaging)"),
        mpatches.Patch(color="#a93226", label="loading on F2 (Microbiology)"),
        mpatches.Patch(color="gray", label="Schmid-Leiman general (dashed)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.9)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def influence_share(values: list[float]) -> list[float]:
    a = np.abs(np.asarray(values, dtype=float))
    s = float(a.sum())
    if s <= 0:
        return [1.0 / len(a)] * len(a)
    return (a / s).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description="Tetrachoric R + 1F/2F CFA (ULS and DWLS)")
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--meta-main", type=Path, default=None)
    ap.add_argument("--meta-d5", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=0, help="Nonparametric bootstrap reps per pair (0 to disable).")
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
                print("NPZ missing 'paths' key; cannot apply meta filter.", file=sys.stderr)
                return 2
            mask, meta_info = meta_complete_case_mask(paths, args.meta_main, args.meta_d5)
            df = df.loc[mask].reset_index(drop=True)
    else:
        df, _ = load_from_csv(args.csv)

    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    X = df[COLS].values.astype(int)
    n = len(df)
    if n < 30:
        print(f"[warn] only n = {n} rows; tetrachoric MLE is noisy at this size.", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)

    # --- Correlations + asymptotics ----------------------------------------
    R_pearson = np.corrcoef(X.T.astype(float))
    R_tet, SE_tet, (CI_lo, CI_hi), meta_pairs = tetrachoric_matrix(X, COLS, bootstrap=args.bootstrap)
    pd.DataFrame(R_tet, index=COLS, columns=COLS).to_csv(args.out / "tet_corr.csv", encoding="utf-8")
    pd.DataFrame(SE_tet, index=COLS, columns=COLS).to_csv(args.out / "tet_se.csv", encoding="utf-8")
    pd.DataFrame(CI_lo, index=COLS, columns=COLS).to_csv(args.out / "tet_ci_low.csv", encoding="utf-8")
    pd.DataFrame(CI_hi, index=COLS, columns=COLS).to_csv(args.out / "tet_ci_high.csv", encoding="utf-8")
    pd.DataFrame(meta_pairs).to_csv(args.out / "tet_pairs_detail.csv", index=False, encoding="utf-8")

    plot_corr_compare(R_pearson, R_tet, COLS, args.out / "pearson_vs_tetrachoric.png")
    plot_prevalence_diagnostic(X, COLS, args.out / "prevalence_diag.png")

    # --- 1-factor: ULS and DWLS --------------------------------------------
    p = R_tet.shape[0]
    iu = np.triu_indices(p, k=1)
    # DWLS off-diagonal weights = 1 / Var(r_ij)
    var_offdiag = SE_tet[iu] ** 2
    var_offdiag = np.where(np.isfinite(var_offdiag) & (var_offdiag > 0), var_offdiag, np.nan)
    var_offdiag = np.where(np.isnan(var_offdiag), np.nanmedian(var_offdiag), var_offdiag)
    w_offdiag = 1.0 / var_offdiag

    res_1f_uls = _uls_one_factor(R_tet)
    res_1f_dwls = _dwls_one_factor(R_tet, w_offdiag)

    for tag, res in [("uls", res_1f_uls), ("dwls", res_1f_dwls)]:
        L = np.asarray(res["loadings"], dtype=float)
        infl = influence_share(L.tolist())
        df_1f = pd.DataFrame(
            {
                "indicator": COLS,
                "loading_F": L,
                "communality": res["communalities"],
                "abs_loading": np.abs(L),
                "relative_influence": infl,
            }
        ).sort_values("relative_influence", ascending=False)
        df_1f.to_csv(args.out / f"loadings_1f_{tag}.csv", index=False, encoding="utf-8")

    # --- 2-factor constrained: ULS and DWLS --------------------------------
    imaging_idx = [COLS.index("Cavity")]
    micro_idx = [COLS.index(c) for c in ("AFB_Smear", "TB_PCR", "Solid_Culture", "Liquid_Culture")]
    res_2f_uls = fit_two_factor_constrained(R_tet, COLS, imaging_idx, micro_idx, None)
    res_2f_dwls = fit_two_factor_constrained(R_tet, COLS, imaging_idx, micro_idx, w_offdiag)

    for tag, res in [("uls", res_2f_uls), ("dwls", res_2f_dwls)]:
        g = np.asarray(res["schmid_leiman_general_loadings"], dtype=float)
        infl_g = influence_share(g.tolist())
        pd.DataFrame(
            {
                "indicator": COLS,
                "loading_F1_Imaging": res["loadings_F1"],
                "loading_F2_Microbio": res["loadings_F2"],
                "schmid_leiman_general": g.tolist(),
                "abs_general": np.abs(g),
                "relative_influence_HOC": infl_g,
            }
        ).to_csv(args.out / f"loadings_2f_{tag}.csv", index=False, encoding="utf-8")

    plot_path_diagram_2f(res_2f_dwls, COLS, args.out / "path_diagram_2f_dwls.png")

    # --- Report -------------------------------------------------------------
    lines = [
        "=== Infectivity latent — TETRACHORIC branch ===",
        f"n = {n}",
        "",
        "Marginal prevalence P(item = 1):",
    ]
    for c, p1 in zip(COLS, X.mean(axis=0)):
        lines.append(f"   {c:>14s}  {p1:.4f}")
    lines += [
        "",
        "Mean off-diagonal correlations:",
        f"   Pearson (phi-coef)  = {float(np.mean(np.abs(R_pearson[iu]))):.4f}",
        f"   Tetrachoric         = {float(np.mean(np.abs(R_tet[iu]))):.4f}",
        "",
        "1-factor results:",
        f"   ULS  loadings = {[round(v, 4) for v in res_1f_uls['loadings']]}    SRMR = {res_1f_uls['srmr']:.4f}",
        f"   DWLS loadings = {[round(v, 4) for v in res_1f_dwls['loadings']]}    SRMR = {res_1f_dwls['srmr']:.4f}   wSRMR = {res_1f_dwls['weighted_srmr']:.4f}",
        "",
        "2-factor constrained (F1 = Cavity ; F2 = AFB/PCR/Solid/Liquid):",
        f"   ULS   phi = {res_2f_uls['factor_correlation_phi']:.4f}    SRMR = {res_2f_uls['srmr']:.4f}",
        f"   DWLS  phi = {res_2f_dwls['factor_correlation_phi']:.4f}    SRMR = {res_2f_dwls['srmr']:.4f}    wSRMR = {res_2f_dwls['weighted_srmr']:.4f}",
        "",
        "Schmid-Leiman HOC general loadings (DWLS):",
    ]
    g = np.asarray(res_2f_dwls["schmid_leiman_general_loadings"], dtype=float)
    infl_g = influence_share(g.tolist())
    for name, gi, share in sorted(zip(COLS, g, infl_g), key=lambda x: -x[2]):
        lines.append(f"   {name:>14s}  g = {gi:+.4f}   share = {share:.4f}")
    if meta_info is not None:
        lines.extend(["", "Meta exclusion summary:", json.dumps(meta_info, indent=2, ensure_ascii=False)])
    (args.out / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "branch": "stat/tetrachoric",
        "method": "tetrachoric correlation + ULS and DWLS 1F/2F constrained CFA with Schmid-Leiman HOC",
        "n": int(n),
        "columns": COLS,
        "marginal_prevalence": dict(zip(COLS, X.mean(axis=0).tolist())),
        "pearson_corr": R_pearson.tolist(),
        "tetrachoric_corr": R_tet.tolist(),
        "tetrachoric_se": SE_tet.tolist(),
        "one_factor_uls": res_1f_uls,
        "one_factor_dwls": res_1f_dwls,
        "two_factor_uls": res_2f_uls,
        "two_factor_dwls": res_2f_dwls,
        "relative_influence_HOC_dwls": dict(zip(COLS, infl_g)),
        "meta_exclusion": meta_info,
    }
    # Strip non-serialisable numpy items from res dicts (implied_R arrays).
    for key in ("one_factor_uls", "one_factor_dwls"):
        summary[key].pop("implied_R", None)

    def _np_default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_np_default), encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"\nSaved: {args.out / 'report.txt'}")
    print(f"Saved: {args.out / 'summary.json'}")
    print(f"Saved: {args.out / 'tet_corr.csv'}, tet_se.csv, tet_ci_low.csv, tet_ci_high.csv")
    print(f"Saved: {args.out / 'pearson_vs_tetrachoric.png'}, prevalence_diag.png, path_diagram_2f_dwls.png")
    print(f"Saved: {args.out / 'loadings_1f_uls.csv'}, loadings_1f_dwls.csv, loadings_2f_uls.csv, loadings_2f_dwls.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Infectivity latent — polychoric correlation + ULS factor extraction (1F / 2F CFA).

Why polychoric:
  The default CFA-SEM pipeline (analyze_infectivity_latent.py) treats binary
  indicators as continuous and uses Pearson covariance via MLW. For 0/1 data
  with skewed marginals (e.g., Cavity prevalence ≠ 0.5), Pearson r underestimates
  the true latent association. This makes the cavity standardized loading look
  smaller than it really is, and biases the relative-influence weights.

  Polychoric correlation assumes each observed ordinal/binary item has an
  underlying continuous latent following a bivariate normal, with thresholds set
  by the marginal cumulative proportions. The MLE on the contingency table gives
  the latent correlation that is invariant to marginal skew. For binary data
  polychoric reduces to tetrachoric (handled by the same routine here).

Model
-----
  Indicators:
    Cavity         ↔ D5  (imaging)
    AFB_Smear      ↔ D1  (microbiology / sputum)
    TB_PCR         ↔ D2
    Solid_Culture  ↔ D3
    Liquid_Culture ↔ D4

  Two-factor structure (the user's intended 2FA):
    F1 (Imaging)       : Cavity
    F2 (Microbiology)  : AFB_Smear, TB_PCR, Solid_Culture, Liquid_Culture
    Factor correlation : phi
    Higher-order (HOC) Infectivity from Schmid-Leiman transformation:
        General loading on indicator i = lambda_i * sqrt(phi)   (positive phi)

Outputs (--out)
---------------
  poly_corr.csv           polychoric correlation matrix
  pearson_vs_poly.png     side-by-side heatmaps (sanity check)
  loadings_1f.csv         1-factor ULS loadings + relative influence
  loadings_2f.csv         2-factor constrained CFA (ULS) loadings, phi, HOC weights
  path_diagram_2f.png     CFA-style diagram with Schmid-Leiman general loadings
  report.txt              textual summary
  summary.json            machine-readable summary

Usage
-----
  python analyze_infectivity_polychoric.py \
      --npz "<.npz with Y(N,5)=D1..D4,D6 and D5>" \
      --meta-main "<meta.csv>" --meta-d5 "<meta_d5.csv>" \
      --out "artifacts/infectivity_polychoric"

Requires: numpy, scipy, pandas, matplotlib (no semopy needed).
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

# Reuse loaders + meta filter from the original latent script (kept on this branch).
sys.path.insert(0, str(Path(__file__).parent))
from analyze_infectivity_latent import (  # noqa: E402
    COLS,
    load_from_csv,
    load_from_npz,
    meta_complete_case_mask,
)

# ---------------------------------------------------------------------------
# Polychoric / tetrachoric correlation per pair
# ---------------------------------------------------------------------------


def _marginal_thresholds(x: np.ndarray) -> np.ndarray:
    """Thresholds tau_1 < ... < tau_{k-1} from cumulative marginals of an integer-coded
    ordinal vector x in {0, 1, ..., k-1}. With Yates-style 0.5 continuity correction
    when any cell hits 0/N (prevents tau = +/- inf)."""
    vals = np.unique(x[~np.isnan(x.astype(float))]) if x.dtype != int else np.unique(x)
    vals = np.sort(vals.astype(int))
    n = len(x)
    cum = 0.0
    taus: list[float] = []
    for v in vals[:-1]:
        cum += float(np.sum(x == v))
        p = cum / n
        # Continuity: avoid 0 or 1 exact
        p = min(max(p, 0.5 / n), 1 - 0.5 / n)
        taus.append(float(norm.ppf(p)))
    return np.asarray(taus, dtype=float)


_INF = 8.0  # numerical +inf used for normal CDF tails (Phi(8) ~ 1 - 6e-16)


def _bvn_cdf_grid(rho: float, x_pts: np.ndarray, y_pts: np.ndarray) -> np.ndarray:
    """F(x_i, y_j) for the standard bivariate normal with correlation rho.

    Boundary handling: x = -inf or y = -inf -> 0; x = +inf -> Phi(y); y = +inf -> Phi(x);
    both +inf -> 1. Internally we cap +/-inf at +/-_INF and call multivariate_normal.cdf.
    """
    rho = float(np.clip(rho, -0.999999, 0.999999))
    rv = multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
    F = np.zeros((len(x_pts), len(y_pts)), dtype=float)
    for i, x in enumerate(x_pts):
        xv = float(x)
        if not np.isfinite(xv):
            xv = float(np.sign(xv) * _INF)
        for j, y in enumerate(y_pts):
            yv = float(y)
            if not np.isfinite(yv):
                yv = float(np.sign(yv) * _INF)
            if xv <= -_INF or yv <= -_INF:
                F[i, j] = 0.0
            elif xv >= _INF and yv >= _INF:
                F[i, j] = 1.0
            elif xv >= _INF:
                F[i, j] = float(norm.cdf(yv))
            elif yv >= _INF:
                F[i, j] = float(norm.cdf(xv))
            else:
                F[i, j] = float(rv.cdf([xv, yv]))
    return F


def polychoric_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    """MLE polychoric correlation for an ordinal-by-ordinal contingency table.

    Two-step Olsson (1979):
      Step 1: marginal thresholds from observed margins.
      Step 2: maximize log-likelihood over rho with thresholds fixed.

    For binary x and y this is the tetrachoric correlation (Brown 1977 limit).
    Implementation note: for each candidate rho we evaluate the bivariate normal
    CDF on the (k_x+1) x (k_y+1) threshold grid once, then form per-cell
    probabilities by 2D differencing (inclusion-exclusion) — far fewer CDF calls
    than a per-cell rectangle.
    """
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    xs = np.sort(np.unique(x))
    ys = np.sort(np.unique(y))
    nij = np.zeros((len(xs), len(ys)), dtype=int)
    for ii, vx in enumerate(xs):
        for jj, vy in enumerate(ys):
            nij[ii, jj] = int(np.sum((x == vx) & (y == vy)))
    if len(xs) < 2 or len(ys) < 2:
        return 0.0, {"reason": "constant variable", "n": int(nij.sum())}

    tau_x = _marginal_thresholds(x)
    tau_y = _marginal_thresholds(y)
    x_pts = np.concatenate([[-np.inf], tau_x, [np.inf]])
    y_pts = np.concatenate([[-np.inf], tau_y, [np.inf]])

    def neg_loglik(rho: float) -> float:
        F = _bvn_cdf_grid(rho, x_pts, y_pts)
        P = F[1:, 1:] - F[:-1, 1:] - F[1:, :-1] + F[:-1, :-1]
        P = np.clip(P, 1e-15, 1.0)
        return -float(np.sum(nij * np.log(P)))

    res = optimize.minimize_scalar(
        neg_loglik, bounds=(-0.99, 0.99), method="bounded", options={"xatol": 1e-5}
    )
    rho_hat = float(res.x)
    info = {
        "n": int(nij.sum()),
        "rows": [int(r) for r in nij.sum(axis=1).tolist()],
        "cols": [int(c) for c in nij.sum(axis=0).tolist()],
        "thresholds_x": tau_x.tolist(),
        "thresholds_y": tau_y.tolist(),
        "loglik": float(-res.fun),
    }
    return rho_hat, info


def polychoric_corr_matrix(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Symmetric polychoric correlation matrix on integer-coded columns of X."""
    p = X.shape[1]
    R = np.eye(p, dtype=float)
    diag = []
    for i in range(p):
        for j in range(i + 1, p):
            r, _info = polychoric_pair(X[:, i], X[:, j])
            R[i, j] = R[j, i] = r
        diag.append({"col": names[i], "n_unique": int(len(np.unique(X[:, i])))})
    info_df = pd.DataFrame(diag)
    return R, info_df


# ---------------------------------------------------------------------------
# Factor extraction
# ---------------------------------------------------------------------------


def uls_one_factor(R: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> dict:
    """One-factor ULS (minimum residual) via iterated principal axis with squared
    multiple correlation as initial communalities."""
    p = R.shape[0]
    # Initial communality: SMC = 1 - 1/diag(R^-1)
    try:
        Rinv = np.linalg.pinv(R + 1e-8 * np.eye(p))
        h2 = 1.0 - 1.0 / np.diag(Rinv)
        h2 = np.clip(h2, 0.0, 0.99)
    except np.linalg.LinAlgError:
        h2 = np.full(p, 0.5)

    L = np.zeros(p, dtype=float)
    for _ in range(max_iter):
        Rred = R.copy()
        np.fill_diagonal(Rred, h2)
        eigvals, eigvecs = np.linalg.eigh(Rred)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        lam1 = float(max(eigvals[0], 0.0))
        L_new = eigvecs[:, 0] * np.sqrt(lam1)
        h2_new = L_new**2
        h2_new = np.clip(h2_new, 0.0, 0.99)
        if np.max(np.abs(h2_new - h2)) < tol:
            L = L_new
            h2 = h2_new
            break
        L = L_new
        h2 = h2_new

    # Sign: orient so the sum of loadings is positive.
    if float(np.sum(L)) < 0:
        L = -L

    R_hat = np.outer(L, L)
    np.fill_diagonal(R_hat, 1.0)
    resid = R - R_hat
    p_tri = p * (p - 1) // 2
    iu = np.triu_indices(p, k=1)
    srmr = float(np.sqrt(np.mean(resid[iu] ** 2)))

    return {
        "loadings": L.tolist(),
        "communalities": h2.tolist(),
        "srmr": srmr,
        "explained_total": float(np.sum(L**2) / p),
        "implied_R": R_hat,
    }


def fit_two_factor_constrained(
    R: np.ndarray, names: list[str], imaging_idx: list[int], micro_idx: list[int]
) -> dict:
    """Two-factor confirmatory ULS with the constrained pattern:
        F1 loads on `imaging_idx` (e.g., Cavity)
        F2 loads on `micro_idx`   (AFB, PCR, Solid, Liquid)
        No cross-loadings.
        F1, F2 standardized (var = 1), correlated by phi in (-0.99, 0.99).
    Parameters: loadings (len = |imaging_idx| + |micro_idx|) + phi.
    Minimize ULS = sum_{i<j}(R_ij - R_hat_ij)^2.
    """
    p = R.shape[0]
    assert sorted(imaging_idx + micro_idx) == list(range(p)), "indices must cover all items"

    pat = np.zeros((p, 2), dtype=int)
    pat[imaging_idx, 0] = 1
    pat[micro_idx, 1] = 1
    free_mask = pat.astype(bool)
    n_free = int(free_mask.sum())  # = p in this 1-per-row design

    iu = np.triu_indices(p, k=1)
    R_off = R[iu]

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, float]:
        L = np.zeros((p, 2), dtype=float)
        L[free_mask] = theta[:n_free]
        phi = float(np.tanh(theta[n_free]))  # smooth bound (-1, 1)
        return L, phi

    def implied(theta: np.ndarray) -> np.ndarray:
        L, phi = unpack(theta)
        Phi = np.array([[1.0, phi], [phi, 1.0]], dtype=float)
        return L @ Phi @ L.T  # off-diag of this is the implied correlation

    def loss(theta: np.ndarray) -> float:
        Rhat = implied(theta)
        return float(np.sum((R_off - Rhat[iu]) ** 2))

    # Init: 1-factor loadings as start, phi = 0.6
    res_1f = uls_one_factor(R)
    L0 = np.asarray(res_1f["loadings"], dtype=float)
    theta0 = np.concatenate([np.abs(L0[free_mask.any(axis=1)]) + 1e-3, [np.arctanh(0.6)]])
    # Above indexing returns p loadings since each row has exactly one free slot
    res = optimize.minimize(
        loss,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10},
    )
    L_hat, phi = unpack(res.x)

    # Flip signs per factor if sum negative (identification)
    for k in range(2):
        if float(np.sum(L_hat[:, k])) < 0:
            L_hat[:, k] = -L_hat[:, k]

    Rhat = implied(res.x)
    np.fill_diagonal(Rhat, 1.0)
    resid = R - Rhat
    srmr = float(np.sqrt(np.mean(resid[iu] ** 2)))

    # Schmid-Leiman general loadings (HOC Infectivity ~ F1 + F2)
    # For 2 first-order factors with corr phi (>0), the unit-norm direction of the
    # general factor in the F-space is (1,1)/sqrt(2) up to scaling. The
    # Schmid-Leiman general loading on indicator i is the sum of paths through
    # both group factors weighted by the general-to-group loadings. With phi as
    # the only HOC information, the standard formula yields g_i = lambda_i * sqrt(phi)
    # for indicators loading on either F1 or F2 (positive phi).
    if phi > 0:
        g = np.zeros(p, dtype=float)
        for k in range(2):
            g += L_hat[:, k] * np.sqrt(phi)
    else:
        g = np.zeros(p, dtype=float)

    return {
        "loadings_F1": L_hat[:, 0].tolist(),
        "loadings_F2": L_hat[:, 1].tolist(),
        "factor_correlation_phi": float(phi),
        "srmr": srmr,
        "loss_uls": float(res.fun),
        "iterations": int(res.nit) if hasattr(res, "nit") else -1,
        "schmid_leiman_general_loadings": g.tolist(),
        "implied_R": Rhat,
        "names": names,
    }


def influence_share(values: list[float]) -> list[float]:
    a = np.abs(np.asarray(values, dtype=float))
    s = float(a.sum())
    if s <= 0:
        return [1.0 / len(a)] * len(a)
    return (a / s).tolist()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_corr_compare(R_pearson: np.ndarray, R_poly: np.ndarray, names: list[str], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, M, title in zip(axes, [R_pearson, R_poly], ["Pearson", "Polychoric"]):
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.7, label="corr")
    fig.suptitle("Pearson vs Polychoric correlation (5 infectivity indicators)")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as _plt

    _plt.close(fig)


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

    # Higher-order node
    hoc = (10.0, 4.0)
    ax.add_patch(plt.Circle(hoc, 0.7, fc="lightyellow", ec="black", lw=2, zorder=3))
    ax.text(*hoc, "Infectivity\n(HOC)", ha="center", va="center", fontsize=10, weight="bold")

    f1 = (6.5, 6.2)
    f2 = (6.5, 1.8)
    ax.add_patch(plt.Circle(f1, 0.55, fc="#dde7ff", ec="black", lw=2, zorder=3))
    ax.text(*f1, "F1\nImaging", ha="center", va="center", fontsize=9, weight="bold")
    ax.add_patch(plt.Circle(f2, 0.55, fc="#ffe6dd", ec="black", lw=2, zorder=3))
    ax.text(*f2, "F2\nMicrobio", ha="center", va="center", fontsize=9, weight="bold")

    # Phi double-arrow
    ax.annotate(
        "",
        xy=(f1[0] - 0.6, f1[1] - 0.4),
        xytext=(f2[0] - 0.6, f2[1] + 0.4),
        arrowprops=dict(arrowstyle="<->", color="purple", lw=1.6),
    )
    ax.text(f1[0] - 1.3, (f1[1] + f2[1]) / 2, f"phi = {phi:.3f}", color="purple", fontsize=9)

    # Schmid-Leiman general arrows (from each indicator to HOC) shown as dashed
    indicator_y = np.linspace(0.6, 7.4, len(names))
    for i, name in enumerate(names):
        bx, by = 2.0, float(indicator_y[i])
        # Box
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (bx - 0.75, by - 0.28),
                1.5,
                0.56,
                boxstyle="round,pad=0.02",
                ec="black",
                fc="white",
            )
        )
        ax.text(bx, by, name.replace("_", " "), ha="center", va="center", fontsize=8)

        # F1 or F2 arrow (group factor loading)
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
        # Label loading
        mid_x = (bx + 0.75 + target[0]) / 2
        mid_y = (by + target[1]) / 2
        ax.text(mid_x, mid_y + 0.1, f"{est:.3f}", fontsize=8, color=color)

        # Schmid-Leiman general loading (dashed thin) directly to HOC
        ax.annotate(
            "",
            xy=hoc,
            xytext=(bx + 0.75, by),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8, linestyle="dashed", alpha=0.7),
        )

    # F1, F2 to HOC paths (loadings = sqrt(phi) if symmetric; we show phi label)
    ax.annotate(
        "",
        xy=hoc,
        xytext=(f1[0] + 0.55, f1[1] - 0.2),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.4),
    )
    ax.annotate(
        "",
        xy=hoc,
        xytext=(f2[0] + 0.55, f2[1] + 0.2),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.4),
    )

    ax.set_title(
        f"Two-factor CFA on POLYCHORIC R + Schmid-Leiman HOC Infectivity\n"
        f"SRMR(off-diag) = {res_2f['srmr']:.4f}    phi(F1,F2) = {phi:.3f}",
        fontsize=10,
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color="#1f3a93", label="loading on F1 (Imaging)"),
        mpatches.Patch(color="#a93226", label="loading on F2 (Microbiology)"),
        mpatches.Patch(color="gray", label="Schmid-Leiman general (dashed)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as _plt

    _plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Polychoric R + 1F/2F CFA for infectivity indicators")
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--meta-main", type=Path, default=None)
    ap.add_argument("--meta-d5", type=Path, default=None)
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
                print("NPZ missing 'paths' key; cannot apply meta 미검 filter.", file=sys.stderr)
                return 2
            mask, meta_info = meta_complete_case_mask(paths, args.meta_main, args.meta_d5)
            df = df.loc[mask].reset_index(drop=True)
    else:
        df, _ = load_from_csv(args.csv)

    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    X = df[COLS].values.astype(int)
    n = len(df)
    if n < 30:
        print(f"[warn] only n={n} rows after NA drop — polychoric MLE is noisy at this size.", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)

    # --- Correlations -------------------------------------------------------
    R_pearson = np.corrcoef(X.T.astype(float))
    R_poly, _info = polychoric_corr_matrix(X, COLS)

    pd.DataFrame(R_poly, index=COLS, columns=COLS).to_csv(args.out / "poly_corr.csv", encoding="utf-8")
    pd.DataFrame(R_pearson, index=COLS, columns=COLS).to_csv(
        args.out / "pearson_corr.csv", encoding="utf-8"
    )
    plot_corr_compare(R_pearson, R_poly, COLS, args.out / "pearson_vs_poly.png")

    # --- 1-factor ULS on polychoric R ---------------------------------------
    res_1f = uls_one_factor(R_poly)
    L1f = np.asarray(res_1f["loadings"], dtype=float)
    inf_1f = influence_share(L1f.tolist())
    df_1f = pd.DataFrame(
        {
            "indicator": COLS,
            "loading_F": L1f,
            "communality": res_1f["communalities"],
            "abs_loading": np.abs(L1f),
            "relative_influence": inf_1f,
        }
    ).sort_values("relative_influence", ascending=False)
    df_1f.to_csv(args.out / "loadings_1f.csv", index=False, encoding="utf-8")

    # --- 2-factor constrained ULS (Imaging vs Microbiology) -----------------
    imaging_idx = [COLS.index("Cavity")]
    micro_idx = [COLS.index(c) for c in ("AFB_Smear", "TB_PCR", "Solid_Culture", "Liquid_Culture")]
    res_2f = fit_two_factor_constrained(R_poly, COLS, imaging_idx, micro_idx)

    g = np.asarray(res_2f["schmid_leiman_general_loadings"], dtype=float)
    inf_g = influence_share(g.tolist())
    df_2f = pd.DataFrame(
        {
            "indicator": COLS,
            "loading_F1_Imaging": res_2f["loadings_F1"],
            "loading_F2_Microbio": res_2f["loadings_F2"],
            "schmid_leiman_general": g.tolist(),
            "abs_general": np.abs(g),
            "relative_influence_HOC": inf_g,
        }
    )
    df_2f.to_csv(args.out / "loadings_2f.csv", index=False, encoding="utf-8")

    plot_path_diagram_2f(res_2f, COLS, args.out / "path_diagram_2f.png")

    # --- Report -------------------------------------------------------------
    lines = [
        "=== Infectivity latent — POLYCHORIC branch ===",
        f"n = {n}",
        "",
        "1) Correlation comparison (off-diagonal magnitudes):",
        f"   mean |Pearson r|     = {float(np.mean(np.abs(R_pearson[np.triu_indices(5, 1)]))):.4f}",
        f"   mean |Polychoric r|  = {float(np.mean(np.abs(R_poly[np.triu_indices(5, 1)]))):.4f}",
        "   (Polychoric removes the binary-truncation bias of Pearson; expect"
        "  larger magnitudes for unbalanced indicators such as Cavity.)",
        "",
        "2) One-factor ULS on POLYCHORIC R",
        df_1f.to_string(index=False),
        f"   SRMR(off-diag) = {res_1f['srmr']:.4f}    mean h^2 = {float(np.mean(res_1f['communalities'])):.4f}",
        "",
        "3) Two-factor constrained CFA (F1=Cavity ; F2=AFB/PCR/Solid/Liquid) on POLYCHORIC R",
        df_2f.to_string(index=False),
        f"   phi(F1,F2)       = {res_2f['factor_correlation_phi']:.4f}",
        f"   SRMR(off-diag)   = {res_2f['srmr']:.4f}",
        f"   ULS objective    = {res_2f['loss_uls']:.6f}",
        "",
        "Schmid-Leiman general loadings rescale each indicator onto a single",
        "higher-order Infectivity dimension (g_i = lambda_i * sqrt(phi) for phi>0).",
        "Relative influence on HOC Infectivity (sums to 1):",
    ]
    for name, share in sorted(zip(COLS, inf_g), key=lambda x: -x[1]):
        lines.append(f"   {name:>14s}  {share:.4f}")

    if meta_info is not None:
        lines.extend(["", "Meta exclusion summary:", json.dumps(meta_info, indent=2, ensure_ascii=False)])

    (args.out / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "branch": "stat/polychoric",
        "method": "polychoric correlation + ULS factor extraction (1F and constrained 2F + Schmid-Leiman HOC)",
        "n": int(n),
        "columns": COLS,
        "pearson_corr": R_pearson.tolist(),
        "polychoric_corr": R_poly.tolist(),
        "one_factor": {
            "loadings": L1f.tolist(),
            "communalities": res_1f["communalities"],
            "srmr_offdiag": res_1f["srmr"],
            "relative_influence": dict(zip(COLS, inf_1f)),
        },
        "two_factor_constrained": {
            "imaging_indicators": [COLS[i] for i in imaging_idx],
            "microbio_indicators": [COLS[i] for i in micro_idx],
            "loadings_F1_imaging": res_2f["loadings_F1"],
            "loadings_F2_microbio": res_2f["loadings_F2"],
            "phi": res_2f["factor_correlation_phi"],
            "srmr_offdiag": res_2f["srmr"],
            "schmid_leiman_general_loadings": res_2f["schmid_leiman_general_loadings"],
            "relative_influence_HOC": dict(zip(COLS, inf_g)),
        },
        "meta_exclusion": meta_info,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"\nSaved: {args.out / 'report.txt'}")
    print(f"Saved: {args.out / 'summary.json'}")
    print(f"Saved: {args.out / 'poly_corr.csv'}")
    print(f"Saved: {args.out / 'pearson_vs_poly.png'}")
    print(f"Saved: {args.out / 'loadings_1f.csv'}")
    print(f"Saved: {args.out / 'loadings_2f.csv'}")
    print(f"Saved: {args.out / 'path_diagram_2f.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

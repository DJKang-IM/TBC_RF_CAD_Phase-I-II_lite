"""
Block-level Imaging vs Microbiology contribution decomposition.

Two questions the indicator-level "relative influence" (14% Cavity, 86% Micro)
cannot answer:

  1. PLS-PM:   How is R^2(F_infectivity) split between Imaging and Micro blocks?
               Standard decomposition R^2 = sum_j beta_j * r_jY where
                 beta_j = standardized regression coefficient of F_inf on F_block_j
                 r_jY  = simple correlation corr(F_block_j, F_inf)
               Reported "beta" in the PLS branch is r_jY (path = correlation in a
               2-block reflective HOC); the multivariate beta has to be recovered
               by solving the 2x2 inner-block normal equations.

  2. Polychoric (and tetrachoric) HOC:
               What is the standardized gamma_k = path(HOC -> F_k) for each
               first-order factor k? With phi(F1, F2) only, the just-identified
               model gives gamma_1 = gamma_2 = sqrt(phi). To account for the
               (very) different reliability between F1 (single indicator
               Cavity) and F2 (four sputum tests), we also report
               reliability-corrected effective contributions:
                   eff_k = gamma_k * sqrt(rho_F_k)
               where rho_F_k is the composite reliability of the block.

Both decompositions are reported relative to the total (sum-to-1 share).

Inputs (--art-root):
    artifacts/infectivity_pls/summary.json
    artifacts/infectivity_polychoric/summary.json
    artifacts/infectivity_tetrachoric/summary.json   (optional)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def pls_beta_r_decomposition(pls: dict) -> dict:
    """Recover multivariate beta from the reported path coefs (which equal
    simple correlations in the 2-block HOC composite) and the inter-block
    correlation, then return both the raw beta*r and the share."""
    r_im_y = float(pls["beta_imaging_to_inf"])      # corr(F_im, F_inf)
    r_mi_y = float(pls["beta_micro_to_inf"])        # corr(F_mi, F_inf)
    rho = float(pls["inter_block_correlation"])     # corr(F_im, F_mi)

    # 2x2 multivariate normal equations:
    # [1   rho] [beta_im]   [r_im_y]
    # [rho 1  ] [beta_mi] = [r_mi_y]
    det = 1.0 - rho * rho
    beta_im = (r_im_y - rho * r_mi_y) / det
    beta_mi = (r_mi_y - rho * r_im_y) / det

    cont_im = beta_im * r_im_y
    cont_mi = beta_mi * r_mi_y
    r2 = cont_im + cont_mi

    share_im = cont_im / r2
    share_mi = cont_mi / r2

    return {
        "r_im_y": r_im_y,
        "r_mi_y": r_mi_y,
        "inter_block_rho": rho,
        "beta_im_multivariate": beta_im,
        "beta_mi_multivariate": beta_mi,
        "contribution_im_betaXr": cont_im,
        "contribution_mi_betaXr": cont_mi,
        "R2_F_infectivity": r2,
        "share_im": share_im,
        "share_mi": share_mi,
    }


def polychoric_gamma_decomposition(poly: dict, pls: dict) -> dict:
    """Schmid-Leiman gamma_k = sqrt(phi) for both factors when the HOC is
    identified with only phi. Reliability-weighted version uses each first-order
    factor's composite reliability rho_c (block 1 = Cavity -> ave=1.0; block 2 =
    micro -> rho_c from PLS branch since this depends only on indicator
    loadings within block, which both branches agree on)."""
    phi = float(poly["two_factor_constrained"]["phi"])
    gamma_naive = math.sqrt(phi)
    gamma_im = gamma_naive
    gamma_mi = gamma_naive

    # Block reliabilities (composite reliability or AVE) -- from PLS branch
    # (single-item Imaging block: AVE = 1.0 by definition, alpha undefined).
    rho_c_im = 1.0  # single indicator -> taken as 1.0; conservative
    rho_c_mi = float(pls["reliability"]["micro"]["rho_c"])
    ave_im = 1.0
    ave_mi = float(pls["reliability"]["micro"]["ave"])

    eff_im_rho = gamma_im * math.sqrt(rho_c_im)
    eff_mi_rho = gamma_mi * math.sqrt(rho_c_mi)
    eff_im_ave = gamma_im * math.sqrt(ave_im)
    eff_mi_ave = gamma_mi * math.sqrt(ave_mi)

    return {
        "phi_F1_F2": phi,
        "gamma_imaging_naive": gamma_im,
        "gamma_micro_naive": gamma_mi,
        "share_im_naive": gamma_im / (gamma_im + gamma_mi),
        "share_mi_naive": gamma_mi / (gamma_im + gamma_mi),
        "rho_c_imaging": rho_c_im,
        "rho_c_micro": rho_c_mi,
        "ave_imaging": ave_im,
        "ave_micro": ave_mi,
        "eff_im_rho_c": eff_im_rho,
        "eff_mi_rho_c": eff_mi_rho,
        "share_im_rho_c_weighted": eff_im_rho / (eff_im_rho + eff_mi_rho),
        "share_mi_rho_c_weighted": eff_mi_rho / (eff_im_rho + eff_mi_rho),
        "eff_im_ave": eff_im_ave,
        "eff_mi_ave": eff_mi_ave,
        "share_im_ave_weighted": eff_im_ave / (eff_im_ave + eff_mi_ave),
        "share_mi_ave_weighted": eff_mi_ave / (eff_im_ave + eff_mi_ave),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-root", type=Path, default=Path("artifacts"),
                    help="Root containing infectivity_pls / infectivity_polychoric / infectivity_tetrachoric")
    ap.add_argument("--out", type=Path, default=Path("artifacts/block_decomposition"))
    args = ap.parse_args()

    pls = _load(args.art_root / "infectivity_pls" / "summary.json")
    poly = _load(args.art_root / "infectivity_polychoric" / "summary.json")
    tet = _load(args.art_root / "infectivity_tetrachoric" / "summary.json")
    if not pls or not poly:
        raise SystemExit("PLS or polychoric summary.json not found under --art-root")

    pls_dec = pls_beta_r_decomposition(pls)
    poly_dec = polychoric_gamma_decomposition(poly, pls)

    tet_dec = None
    if tet:
        # Tetrachoric and polychoric phi should match on binary data
        tet_phi = float(tet.get("two_factor_dwls", {}).get("factor_correlation_phi", float("nan")))
        if not math.isnan(tet_phi):
            tet_dec = {
                "phi_F1_F2_tetrachoric": tet_phi,
                "gamma_naive_tet": math.sqrt(max(tet_phi, 0.0)),
                "share_each_naive": 0.5,
            }

    args.out.mkdir(parents=True, exist_ok=True)
    out_json = {
        "n": pls.get("n"),
        "PLS_block_R2_decomposition_betaXr": pls_dec,
        "Polychoric_HOC_gamma_decomposition": poly_dec,
        "Tetrachoric_phi_match": tet_dec,
    }
    (args.out / "block_decomposition.json").write_text(
        json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Human-readable report
    lines = [
        "=== Block-level Imaging vs Microbiology contribution ===",
        f"n = {pls.get('n')}",
        "",
        "--- PLS-PM: R^2(F_infectivity) = sum beta_j * r_jY ---",
        f"  r(F_im, F_inf)            = {pls_dec['r_im_y']:.4f}",
        f"  r(F_mi, F_inf)            = {pls_dec['r_mi_y']:.4f}",
        f"  rho(F_im, F_mi)           = {pls_dec['inter_block_rho']:.4f}",
        f"  multivariate beta_im      = {pls_dec['beta_im_multivariate']:.4f}",
        f"  multivariate beta_mi      = {pls_dec['beta_mi_multivariate']:.4f}",
        f"  beta_im * r_im_y          = {pls_dec['contribution_im_betaXr']:.4f}",
        f"  beta_mi * r_mi_y          = {pls_dec['contribution_mi_betaXr']:.4f}",
        f"  R^2(F_infectivity)        = {pls_dec['R2_F_infectivity']:.4f}",
        f"  share(Imaging)            = {pls_dec['share_im']*100:.2f}%",
        f"  share(Microbiology)       = {pls_dec['share_mi']*100:.2f}%",
        "",
        "--- Polychoric HOC: gamma_k = path(HOC -> F_k) ---",
        f"  phi(F1, F2)                       = {poly_dec['phi_F1_F2']:.4f}",
        f"  gamma_imaging (naive sqrt(phi))   = {poly_dec['gamma_imaging_naive']:.4f}",
        f"  gamma_micro   (naive sqrt(phi))   = {poly_dec['gamma_micro_naive']:.4f}",
        f"  share(Imaging) naive              = {poly_dec['share_im_naive']*100:.2f}%",
        f"  share(Microbiology) naive         = {poly_dec['share_mi_naive']*100:.2f}%",
        "",
        "  Reliability-corrected eff_k = gamma_k * sqrt(rho_c_k):",
        f"    rho_c(Imaging block)            = {poly_dec['rho_c_imaging']:.4f}  (single indicator: assumed 1.0)",
        f"    rho_c(Micro block)              = {poly_dec['rho_c_micro']:.4f}",
        f"    eff_imaging                     = {poly_dec['eff_im_rho_c']:.4f}",
        f"    eff_micro                       = {poly_dec['eff_mi_rho_c']:.4f}",
        f"    share(Imaging) rho_c-weighted   = {poly_dec['share_im_rho_c_weighted']*100:.2f}%",
        f"    share(Micro)   rho_c-weighted   = {poly_dec['share_mi_rho_c_weighted']*100:.2f}%",
        "",
        "  AVE-weighted eff_k = gamma_k * sqrt(AVE_k):",
        f"    AVE(Imaging)                    = {poly_dec['ave_imaging']:.4f}",
        f"    AVE(Micro)                      = {poly_dec['ave_micro']:.4f}",
        f"    share(Imaging) AVE-weighted     = {poly_dec['share_im_ave_weighted']*100:.2f}%",
        f"    share(Micro)   AVE-weighted     = {poly_dec['share_mi_ave_weighted']*100:.2f}%",
    ]
    if tet_dec is not None:
        lines += [
            "",
            "--- Tetrachoric sanity ---",
            f"  phi(F1, F2) tetrachoric         = {tet_dec['phi_F1_F2_tetrachoric']:.4f}",
            f"  gamma naive (tet)               = {tet_dec['gamma_naive_tet']:.4f}",
        ]
    txt = "\n".join(lines)
    (args.out / "block_decomposition.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\nSaved: {args.out / 'block_decomposition.json'}")
    print(f"Saved: {args.out / 'block_decomposition.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

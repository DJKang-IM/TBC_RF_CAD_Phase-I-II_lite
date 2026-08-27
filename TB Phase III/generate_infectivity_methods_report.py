"""
Build an HTML comparison report from three latent-method artifact folders.

Reads summary.json (and optional report.txt) produced by:
  - analyze_infectivity_polychoric.py   -> artifacts/infectivity_polychoric
  - analyze_infectivity_tetrachoric.py  -> artifacts/infectivity_tetrachoric
  - analyze_infectivity_pls2block.py     -> artifacts/infectivity_pls

Usage:
  python generate_infectivity_methods_report.py \
      --polychoric artifacts/infectivity_polychoric \
      --tetrachoric artifacts/infectivity_tetrachoric \
      --pls artifacts/infectivity_pls \
      --out "reports/infectivity_latent_methods_comparison.html"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _load_txt(p: Path, max_lines: int = 40) -> str:
    if not p.exists():
        return "(not found)"
    lines = p.read_text(encoding="utf-8").splitlines()[:max_lines]
    return "\n".join(lines)


def _block_decomp_section(dec: dict) -> str:
    if not dec:
        return ""
    pls = dec.get("PLS_block_R2_decomposition_betaXr", {})
    poly = dec.get("Polychoric_HOC_gamma_decomposition", {})
    return f"""
  <h2>Block-level construct decomposition</h2>
  <p class="note">
    Indicator-level "relative influence" (Cavity 14% / Micro 86%) is biased by
    indicator counts (1 vs 4). At the construct / block level, both methods
    agree the two blocks contribute symmetrically to Infectivity.
  </p>
  <h3>PLS-PM: R<sup>2</sup>(F<sub>inf</sub>) = &Sigma; &beta;<sub>j</sub> &middot; r<sub>jY</sub></h3>
  <table>
    <thead><tr><th>Block</th><th>r<sub>jY</sub></th><th>&beta;<sub>j</sub> (multivariate)</th>
           <th>&beta;&middot;r contribution</th><th>share of R<sup>2</sup></th></tr></thead>
    <tbody>
      <tr><td>Imaging (Cavity)</td>
        <td>{pls['r_im_y']:.4f}</td><td>{pls['beta_im_multivariate']:.4f}</td>
        <td>{pls['contribution_im_betaXr']:.4f}</td>
        <td><strong>{pls['share_im']*100:.2f}%</strong></td></tr>
      <tr><td>Microbiology</td>
        <td>{pls['r_mi_y']:.4f}</td><td>{pls['beta_mi_multivariate']:.4f}</td>
        <td>{pls['contribution_mi_betaXr']:.4f}</td>
        <td><strong>{pls['share_mi']*100:.2f}%</strong></td></tr>
      <tr><td colspan="3" style="text-align:right;">R<sup>2</sup>(F<sub>infectivity</sub>) =</td>
        <td>{pls['R2_F_infectivity']:.4f}</td><td>&middot;</td></tr>
    </tbody>
  </table>
  <p class="note">Inter-block correlation &rho;(F<sub>im</sub>, F<sub>mi</sub>) = {pls['inter_block_rho']:.3f}.</p>

  <h3>Polychoric HOC: &gamma;<sub>k</sub> = path(Infectivity &rarr; F<sub>k</sub>)</h3>
  <table>
    <thead><tr><th>Weighting</th><th>&gamma;<sub>Imaging</sub></th><th>&gamma;<sub>Micro</sub></th>
           <th>share Imaging</th><th>share Micro</th></tr></thead>
    <tbody>
      <tr><td>Naive (sqrt(&phi;), Schmid-Leiman)</td>
        <td>{poly['gamma_imaging_naive']:.4f}</td>
        <td>{poly['gamma_micro_naive']:.4f}</td>
        <td><strong>{poly['share_im_naive']*100:.2f}%</strong></td>
        <td><strong>{poly['share_mi_naive']*100:.2f}%</strong></td></tr>
      <tr><td>&rho;<sub>c</sub>-weighted (eff = &gamma;&middot;sqrt(&rho;<sub>c</sub>))</td>
        <td>{poly['eff_im_rho_c']:.4f}</td>
        <td>{poly['eff_mi_rho_c']:.4f}</td>
        <td><strong>{poly['share_im_rho_c_weighted']*100:.2f}%</strong></td>
        <td><strong>{poly['share_mi_rho_c_weighted']*100:.2f}%</strong></td></tr>
      <tr><td>AVE-weighted (eff = &gamma;&middot;sqrt(AVE))</td>
        <td>{poly['eff_im_ave']:.4f}</td>
        <td>{poly['eff_mi_ave']:.4f}</td>
        <td><strong>{poly['share_im_ave_weighted']*100:.2f}%</strong></td>
        <td><strong>{poly['share_mi_ave_weighted']*100:.2f}%</strong></td></tr>
    </tbody>
  </table>
  <p class="note">
    &phi;(F<sub>1</sub>, F<sub>2</sub>) = {poly['phi_F1_F2']:.4f}.
    Naive form follows Schmid-Leiman just-identification with &gamma;<sub>1</sub>=&gamma;<sub>2</sub>=sqrt(&phi;).
    Reliability- and AVE-weighted forms downweight the four-indicator Microbiology
    block because its &rho;<sub>c</sub>/AVE are below 1 while the single-item Imaging block is taken at 1.0.
  </p>
"""


def _weights_table_poly(s: dict) -> str:
    inf_1f = s.get("one_factor", {}).get("relative_influence", {})
    inf_hoc = s.get("two_factor_constrained", {}).get("relative_influence_HOC", {})
    rows = []
    for name in inf_1f.keys():
        rows.append(
            f"<tr><td>{name}</td><td>{inf_1f.get(name, 0):.4f}</td>"
            f"<td>{inf_hoc.get(name, 0):.4f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Indicator</th><th>1F ULS share</th>"
        "<th>2F HOC share</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _weights_table_tet(s: dict) -> str:
    inf_uls = s.get("one_factor_dwls", {}).get("loadings")
    # summary stores full dict; use relative_influence from effective if present
    rel = s.get("relative_influence_HOC_dwls", {})
    if not rel:
        L = s.get("one_factor_dwls", {}).get("loadings", [])
        if isinstance(L, list) and L:
            import numpy as np

            a = np.abs(np.asarray(L, dtype=float))
            rel = {k: float(v / a.sum()) for k, v in zip(
                ["Cavity", "AFB_Smear", "TB_PCR", "Solid_Culture", "Liquid_Culture"], a
            )}
    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in sorted(rel.items(), key=lambda x: -x[1])
    )
    phi = s.get("two_factor_dwls", {}).get("factor_correlation_phi", float("nan"))
    srmr = s.get("two_factor_dwls", {}).get("srmr", float("nan"))
    return (
        f"<p>phi(F1,F2) = {phi:.4f} &nbsp; SRMR = {srmr:.4f}</p>"
        f"<table><thead><tr><th>Indicator</th><th>HOC DWLS share</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _weights_table_pls(s: dict) -> str:
    eff = s.get("effective_weights", {})
    share = s.get("share_of_variance", {})
    rows = "".join(
        f"<tr><td>{k}</td><td>{eff.get(k, 0):+.4f}</td><td>{share.get(k, 0):.4f}</td></tr>"
        for k in sorted(share.keys(), key=lambda x: -share.get(x, 0))
    )
    v = s.get("v_hoc", [])
    beta_im = s.get("beta_imaging_to_inf", float("nan"))
    beta_mi = s.get("beta_micro_to_inf", float("nan"))
    return (
        f"<p>v_imaging={v[0]:+.3f} v_micro={v[1]:+.3f} &nbsp; "
        f"beta_im={beta_im:+.3f} beta_micro={beta_mi:+.3f}</p>"
        "<table><thead><tr><th>Indicator</th><th>w_eff</th><th>variance share</th>"
        "</tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


def _copy_figures(paths: dict[str, Path], out_dir: Path) -> dict[str, str]:
    """Copy key PNGs next to the HTML report; return relative img paths."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "poly_pearson": (paths["poly"] / "pearson_vs_poly.png", "poly_pearson_vs_poly.png"),
        "poly_path": (paths["poly"] / "path_diagram_2f.png", "poly_path_2f.png"),
        "tet_pearson": (paths["tet"] / "pearson_vs_tetrachoric.png", "tet_pearson_vs_tet.png"),
        "tet_path": (paths["tet"] / "path_diagram_2f_dwls.png", "tet_path_2f_dwls.png"),
        "tet_prev": (paths["tet"] / "prevalence_diag.png", "tet_prevalence.png"),
        "pls_bar": (paths["pls"] / "bar_effective_weights.png", "pls_effective_weights.png"),
        "pls_path": (paths["pls"] / "path_diagram_pls.png", "pls_path.png"),
    }
    rel: dict[str, str] = {}
    import shutil

    for key, (src, dest_name) in mapping.items():
        if src.exists():
            dest = fig_dir / dest_name
            shutil.copy2(src, dest)
            rel[key] = f"figures/{dest_name}"
        else:
            rel[key] = ""
    return rel


def build_html(poly: dict, tet: dict, pls: dict, paths: dict[str, Path], img: dict[str, str], dec: dict | None = None) -> str:
    n_poly = poly.get("n", "?")
    n_tet = tet.get("n", "?")
    n_pls = pls.get("n", "?")
    prev = tet.get("marginal_prevalence", pls.get("marginal_prevalence", {}))

    prev_rows = "".join(
        f"<tr><td>{k}</td><td>{v:.3f}</td></tr>" for k, v in prev.items()
    ) if prev else "<tr><td colspan=2>(n/a)</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Infectivity latent methods comparison</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; }}
    h1 {{ font-size: 1.4rem; }}
    h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ccc; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
    th {{ background: #f4f4f4; }}
    pre {{ background: #f8f8f8; padding: 12px; overflow-x: auto; font-size: 0.8rem; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    img {{ max-width: 100%; height: auto; border: 1px solid #eee; }}
    .note {{ color: #555; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>TB infectivity — parallel latent / composite methods</h1>
  <p class="note">
    Compares three branches: <strong>polychoric</strong> (threshold-model CFA),
    <strong>tetrachoric</strong> (binary DWLS CFA), and <strong>PLS-PM</strong>
    (two-block composite + HOC). Cavity vs four sputum tests are modelled as
    Imaging vs Microbiology blocks where applicable.
  </p>

  <h2>Cohort</h2>
  <p>n polychoric = {n_poly} &nbsp;|&nbsp; n tetrachoric = {n_tet} &nbsp;|&nbsp; n PLS = {n_pls}</p>
  <table><thead><tr><th>Indicator</th><th>P(=1)</th></tr></thead><tbody>{prev_rows}</tbody></table>

  {_block_decomp_section(dec or {})}

  <h2>Relative influence / effective weights (indicator level)</h2>
  <div class="grid">
    <section>
      <h3>Polychoric (2F HOC)</h3>
      {_weights_table_poly(poly)}
    </section>
    <section>
      <h3>Tetrachoric (2F DWLS HOC)</h3>
      {_weights_table_tet(tet)}
    </section>
    <section>
      <h3>PLS-PM (effective w)</h3>
      {_weights_table_pls(pls)}
    </section>
  </div>

  <h2>Figures</h2>
  <figure><figcaption>Prevalence (tetrachoric diagnostic)</figcaption>
    <img src="{img.get('tet_prev','')}" alt="prev"/></figure>
  <div class="grid">
    <figure><figcaption>Polychoric: Pearson vs poly</figcaption>
      <img src="{img.get('poly_pearson','')}" alt="poly"/></figure>
    <figure><figcaption>Tetrachoric: Pearson vs tet</figcaption>
      <img src="{img.get('tet_pearson','')}" alt="tet"/></figure>
    <figure><figcaption>PLS: effective weights</figcaption>
      <img src="{img.get('pls_bar','')}" alt="pls"/></figure>
  </div>
  <div class="grid">
    <figure><img src="{img.get('poly_path','')}" alt="poly path"/></figure>
    <figure><img src="{img.get('tet_path','')}" alt="tet path"/></figure>
    <figure><img src="{img.get('pls_path','')}" alt="pls path"/></figure>
  </div>

  <h2>Report excerpts</h2>
  <h3>Polychoric</h3>
  <pre>{_load_txt(paths['poly'] / 'report.txt')}</pre>
  <h3>Tetrachoric</h3>
  <pre>{_load_txt(paths['tet'] / 'report.txt')}</pre>
  <h3>PLS</h3>
  <pre>{_load_txt(paths['pls'] / 'report.txt')}</pre>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--polychoric", type=Path, required=True)
    ap.add_argument("--tetrachoric", type=Path, required=True)
    ap.add_argument("--pls", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output HTML path; filename will be auto-prefixed with YYYYMMDD_ unless --no-date-prefix.")
    ap.add_argument("--date", type=str, default=None,
                    help="Override YYYYMMDD stamp (default: today). Ignored if --no-date-prefix.")
    ap.add_argument("--no-date-prefix", action="store_true",
                    help="Disable automatic YYYYMMDD_ prefix on the output filename.")
    args = ap.parse_args()

    paths = {"poly": args.polychoric, "tet": args.tetrachoric, "pls": args.pls}
    poly = _load_json(args.polychoric / "summary.json")
    tet = _load_json(args.tetrachoric / "summary.json")
    pls = _load_json(args.pls / "summary.json")

    out = args.out
    if not args.no_date_prefix:
        stamp = args.date or datetime.now().strftime("%Y%m%d")
        if not out.name.startswith(stamp + "_"):
            out = out.with_name(f"{stamp}_{out.name}")
    out.parent.mkdir(parents=True, exist_ok=True)

    img = _copy_figures(paths, out.parent)
    dec_path = args.polychoric.parent / "block_decomposition" / "block_decomposition.json"
    dec = _load_json(dec_path)
    html = build_html(poly, tet, pls, paths, img, dec)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Generate the PR#7 (entropy-effective-rank block weighting) artifact pair.

Outputs are saved under TB Phase III/reports/ and are *labelled by PR number*,
not by version. Two text artifacts are emitted so that the methodology and the
numerical result are easy to cite separately:

  20260520 PR7 Entropy Effective Rank - Method.md
  20260520 PR7 Entropy Effective Rank - Results.md
  20260520 PR7 Entropy Effective Rank - Results.csv
  20260520 PR7 Entropy Effective Rank - Results.json

Inputs (already on disk):
  artifacts/infectivity_polychoric/summary.json
  artifacts/infectivity_pls/summary.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def entropy_effective_rank(R: np.ndarray) -> tuple[float, list[float]]:
    """exp( -sum p_i log p_i ),  p_i = lambda_i / sum lambda."""
    w = np.clip(np.linalg.eigvalsh(R), 1e-12, None)
    p = w / w.sum()
    H = -(p * np.log(p)).sum()
    return float(math.exp(H)), w.tolist()


def main() -> int:
    root = Path(__file__).resolve().parent
    art = root / "artifacts"
    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    poly = json.loads((art / "infectivity_polychoric" / "summary.json").read_text(encoding="utf-8"))
    pls = json.loads((art / "infectivity_pls" / "summary.json").read_text(encoding="utf-8"))

    cols = ["Cavity", "AFB_Smear", "TB_PCR", "Solid_Culture", "Liquid_Culture"]
    R = np.asarray(poly["polychoric_corr"], dtype=float)
    R_im = R[:1, :1]
    R_mi = R[1:, 1:]

    er_im, eig_im = entropy_effective_rank(R_im)
    er_mi, eig_mi = entropy_effective_rank(R_mi)
    im_share = er_im / (er_im + er_mi)
    mi_share = 1.0 - im_share

    mw_raw = pls["outer_weights_micro"]
    s_mw = float(sum(mw_raw))
    within = [w / s_mw for w in mw_raw]

    weights = {"Cavity": im_share}
    for c, sub in zip(cols[1:], within):
        weights[c] = mi_share * sub

    mean_r_micro = float(R_mi[np.triu_indices(4, k=1)].mean())

    stem = "20260520 PR7 Entropy Effective Rank"

    method_md = f"""# {stem} — Method

## What it is

Block-level weighting of the five TB infectivity indicators by **entropy-based
effective rank** (Roy & Vetterli 2007) of each block's polychoric correlation
matrix. The five indicators are grouped into two blocks:

- **Imaging block**: `Cavity`
- **Microbiology block**: `AFB_Smear`, `TB_PCR`, `Solid_Culture`, `Liquid_Culture`

Each block is allowed to carry a share of the overall infectivity score that
matches the *effective number of independent dimensions* it actually contains,
not the naive count of indicators.

## Why this view

The four microbiology indicators are highly redundant — their pairwise
polychoric correlations average **r̄ = {mean_r_micro:.3f}** in this cohort —
so treating them as four independent pieces of information overstates their
contribution. The single Cavity indicator carries one dimension by definition;
the microbiology block carries less than four. Effective rank quantifies that.

This sits between two extremes:

- **Indicator-share view (PR#1, #2)** — counts every indicator as one of five
  inputs to a single latent factor; Cavity ends up at ~14% because there is one
  cavity row vs four microbiology rows.
- **Block-equal view (PR#3, #4, #5a)** — splits the score 50/50 between Imaging
  and Microbiology blocks because both blocks correlate with the higher-order
  Infectivity construct with the same magnitude. This ignores the redundancy
  inside microbiology.

PR#7 corrects both biases at once.

## Steps

1. Compute the **polychoric correlation matrix** `R` (5×5) on the binary
   indicators using the Olsson (1979) two-step MLE (`analyze_infectivity_polychoric.py`).
2. Partition `R` into two block-correlation matrices: `R_imaging` (1×1) and
   `R_micro` (4×4).
3. For each block, compute eigenvalues `λ_i = eig(R_block)`, normalise to a
   probability distribution `p_i = λ_i / Σ λ_j`, compute Shannon entropy
   `H = −Σ p_i log p_i`, and define the **entropy effective rank** as `exp(H)`.
4. Block weight share = entropy effective rank, normalised across the two
   blocks to sum to 1.
5. Within the microbiology block, allocate the block's share to the four
   indicators using the **PLS-PM Mode A outer weights** (from
   `analyze_infectivity_pls2block.py`), normalised to sum to 1 inside the
   block. (Polychoric F2 loadings give the same ordering and very similar
   shares.)
6. Final per-indicator weight = `block_share × within_block_share`.

## Why entropy effective rank (and not λ₁ or Spearman-Brown)

| Alternative | Imaging block share | Comment |
|---|---:|---|
| PC1 eigenvalue λ₁ | 24.3% | Counts only the *strongest* direction; over-penalises micro. |
| Entropy effective rank `exp(H)` | **32.5%** | Accounts for the full eigenvalue distribution; standard in signal processing and psychometrics. |
| Spearman-Brown effective N | 43.7% | Assumes equal-correlation parallel items; under-penalises micro. |
| `√k` heuristic | 33.3% | No correlation information; benchmark only. |
| `log(k+1)` heuristic | 30.1% | Information-theoretic flavour; benchmark only. |

The entropy effective rank is the only option here that simultaneously
(a) uses the actual polychoric correlation structure, (b) is rotationally
invariant, and (c) collapses smoothly between the extremes (`λ₁`-style and
`k`-counting).

## Reference

- O. Roy and M. Vetterli, "The effective rank: A measure of effective
  dimensionality," *Proc. 15th European Signal Processing Conf. (EUSIPCO)*,
  Poznań, Poland, Sep. 2007, pp. 606–610.
"""

    result_md = f"""# {stem} — Results

## Inputs

- Cohort: complete-case n = {pls['n']} (CSV from `Stats_weights_infectivity`)
- Source artifacts: `artifacts/infectivity_polychoric/summary.json`,
  `artifacts/infectivity_pls/summary.json`

## Block-level numerics

| Block | Eigenvalues of polychoric block-`R` | Entropy effective rank `exp(H)` | Block share |
|---|---|---:|---:|
| Imaging (Cavity) | {[round(v, 4) for v in eig_im]} | {er_im:.4f} | **{im_share*100:.2f}%** |
| Microbiology (AFB, PCR, Solid, Liquid) | {[round(v, 4) for v in eig_mi]} | {er_mi:.4f} | **{mi_share*100:.2f}%** |
| **Sum** | — | {er_im + er_mi:.4f} | 100.00% |

- Mean polychoric r within micro block = **{mean_r_micro:.4f}**
- The four microbiology indicators behave like **{er_mi:.2f}** independent
  dimensions, not four.

## Within-microbiology weights (PLS Mode A outer)

| Indicator | Outer weight (raw) | Within-block share |
|---|---:|---:|
| AFB_Smear | {mw_raw[0]:.4f} | {within[0]:.4f} |
| TB_PCR | {mw_raw[1]:.4f} | {within[1]:.4f} |
| Solid_Culture | {mw_raw[2]:.4f} | {within[2]:.4f} |
| Liquid_Culture | {mw_raw[3]:.4f} | {within[3]:.4f} |
| **Sum** | {s_mw:.4f} | 1.0000 |

## Final per-indicator weights (sum = 1)

| Indicator | Block | Block share | Within-block share | **Final weight** |
|---|---|---:|---:|---:|
| Cavity | Imaging | {im_share:.4f} | 1.0000 | **{weights['Cavity']:.4f}** |
| AFB_Smear | Micro | {mi_share:.4f} | {within[0]:.4f} | **{weights['AFB_Smear']:.4f}** |
| TB_PCR | Micro | {mi_share:.4f} | {within[1]:.4f} | **{weights['TB_PCR']:.4f}** |
| Solid_Culture | Micro | {mi_share:.4f} | {within[2]:.4f} | **{weights['Solid_Culture']:.4f}** |
| Liquid_Culture | Micro | {mi_share:.4f} | {within[3]:.4f} | **{weights['Liquid_Culture']:.4f}** |
| **Sum** | — | — | — | **{sum(weights.values()):.4f}** |

## Score formula

```
infectivity_score(patient) =
      {weights['Cavity']:.4f} * Cavity
    + {weights['AFB_Smear']:.4f} * AFB_Smear
    + {weights['TB_PCR']:.4f} * TB_PCR
    + {weights['Solid_Culture']:.4f} * Solid_Culture
    + {weights['Liquid_Culture']:.4f} * Liquid_Culture
```

All indicators are binary (0 / 1); the score therefore lives in [0, 1].

## Comparison vs other PRs (Cavity weight)

| PR | Method | Cavity share |
|---|---|---:|
| #1 | Polychoric HOC, indicator-share | 0.1448 |
| #2 | Tetrachoric DWLS HOC, indicator-share | 0.1421 |
| #3 | PLS-PM variance share | 0.5000 |
| #4 | Block 50/50 (PLS β·r) | 0.5000 |
| #5a | Block 50/50 (γ = √φ) | 0.5000 |
| #5b | Block ρ_c-weighted | 0.5187 |
| #5c | Block AVE-weighted | 0.5619 |
| **#7** | **Entropy effective rank (this report)** | **{weights['Cavity']:.4f}** |
"""

    method_path = out_dir / f"{stem} - Method.md"
    results_path = out_dir / f"{stem} - Results.md"
    csv_path = out_dir / f"{stem} - Results.csv"
    json_path = out_dir / f"{stem} - Results.json"

    method_path.write_text(method_md, encoding="utf-8")
    results_path.write_text(result_md, encoding="utf-8")

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("indicator,block,block_share,within_block_share,final_weight\n")
        f.write(f"Cavity,Imaging,{im_share:.6f},1.000000,{weights['Cavity']:.6f}\n")
        for c, sub in zip(cols[1:], within):
            f.write(f"{c},Microbiology,{mi_share:.6f},{sub:.6f},{weights[c]:.6f}\n")

    summary = {
        "stem": stem,
        "pr": 7,
        "label": "Entropy Effective Rank",
        "n": pls["n"],
        "method_reference": "Roy & Vetterli 2007, EUSIPCO — effective rank",
        "polychoric_corr_micro_mean_r": mean_r_micro,
        "block_level": {
            "imaging": {
                "eigenvalues": eig_im,
                "entropy_effective_rank": er_im,
                "share": im_share,
            },
            "microbiology": {
                "eigenvalues": eig_mi,
                "entropy_effective_rank": er_mi,
                "share": mi_share,
            },
        },
        "within_micro_outer_weights_raw": dict(zip(cols[1:], mw_raw)),
        "within_micro_outer_weights_norm": dict(zip(cols[1:], within)),
        "final_weights": weights,
        "score_formula": " + ".join(f"{weights[c]:.4f}*{c}" for c in cols),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Imaging block share : {im_share*100:.2f}%")
    print(f"Micro   block share : {mi_share*100:.2f}%")
    print()
    print("Final weights:")
    for c in cols:
        print(f"  {c:<18} {weights[c]:.4f}")
    print()
    print("Saved:")
    print(f"  {method_path}")
    print(f"  {results_path}")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

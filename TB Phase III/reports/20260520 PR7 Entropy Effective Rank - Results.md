# 20260520 PR7 Entropy Effective Rank — Results

## Inputs

- Cohort: complete-case n = 459 (CSV from `Stats_weights_infectivity`)
- Source artifacts: `artifacts/infectivity_polychoric/summary.json`,
  `artifacts/infectivity_pls/summary.json`

## Block-level numerics

| Block | Eigenvalues of polychoric block-`R` | Entropy effective rank `exp(H)` | Block share |
|---|---|---:|---:|
| Imaging (Cavity) | [1.0] | 1.0000 | **32.49%** |
| Microbiology (AFB, PCR, Solid, Liquid) | [0.1108, 0.2271, 0.5521, 3.1099] | 2.0777 | **67.51%** |
| **Sum** | — | 3.0777 | 100.00% |

- Mean polychoric r within micro block = **0.7010**
- The four microbiology indicators behave like **2.08** independent
  dimensions, not four.

## Within-microbiology weights (PLS Mode A outer)

| Indicator | Outer weight (raw) | Within-block share |
|---|---:|---:|
| AFB_Smear | 0.5152 | 0.2581 |
| TB_PCR | 0.4567 | 0.2287 |
| Solid_Culture | 0.5354 | 0.2681 |
| Liquid_Culture | 0.4892 | 0.2450 |
| **Sum** | 1.9965 | 1.0000 |

## Final per-indicator weights (sum = 1)

| Indicator | Block | Block share | Within-block share | **Final weight** |
|---|---|---:|---:|---:|
| Cavity | Imaging | 0.3249 | 1.0000 | **0.3249** |
| AFB_Smear | Micro | 0.6751 | 0.2581 | **0.1742** |
| TB_PCR | Micro | 0.6751 | 0.2287 | **0.1544** |
| Solid_Culture | Micro | 0.6751 | 0.2681 | **0.1810** |
| Liquid_Culture | Micro | 0.6751 | 0.2450 | **0.1654** |
| **Sum** | — | — | — | **1.0000** |

## Score formula

```
infectivity_score(patient) =
      0.3249 * Cavity
    + 0.1742 * AFB_Smear
    + 0.1544 * TB_PCR
    + 0.1810 * Solid_Culture
    + 0.1654 * Liquid_Culture
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
| **#7** | **Entropy effective rank (this report)** | **0.3249** |

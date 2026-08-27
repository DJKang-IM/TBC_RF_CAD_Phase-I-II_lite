# 20260520 PR7 Entropy Effective Rank — Method

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
polychoric correlations average **r̄ = 0.701** in this cohort —
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

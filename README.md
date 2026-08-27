# TBC_RF_CAD_Phase-I-II_lite

Full corpus available under NDA for research collaboration / employment. Contact: drkangim@naver.com

Original data containing identifiable information cannot be released publicly without IRB approval.


**Public lite release** of TB CAD / infectivity research code.

| | |
|--|--|
| Code | Full pipeline source (sanitized paths; no PHI) |
| JSON samples | **3** synthetic (`data/samples/`) |
| Preview images | **10** random 256px coarse (`data/coarse_256/`) |
| Full data / DICOM | **Not included** — gated: [DJKang-IM/TBC_RF_CAD_Phase-I-II](https://github.com/DJKang-IM/TBC_RF_CAD_Phase-I-II) |

See [`SERVING_RULES.md`](SERVING_RULES.md). Copied code files: ~167.

---

**CC BY-NC-SA 4.0, Non-commercial, Citation required. Full version gated — request access via GitHub Issues.**

**This is a coarse preview. For research collaboration contact the lite repo Issues page.**

---

# TBC_CAD_INFECTIVITY

Code-only repository for the tuberculosis (TB) CAD infectivity grading project.  
Covers CAD package development (v1.02 → v1.03) and multi-phase model research (Phase I → Phase III).

> **Data policy:** All DICOM images, feature matrices, trained model files, and output artifacts are excluded from this repository. Only source code, scripts, and configuration files are tracked. See `.gitignore` for the full exclusion list.

---

## Repo layout & branch map (2026-05-22)

This repository is the **infectivity statistics / weighting** side of the project. Image-classifier training (Phase III v7.x, including v7.011 production) lives in a **separate** repo: [TBC_Phase_III](http<REDACTED_PATH>

### Production weighting: Entropy Effective Rank (EER)

The infectivity score adopted for Phase III v7.011 is the **Entropy Effective Rank** (EER) method (block-level eigenvalue entropy → effective rank → block share; within-block PCA normalised loadings). "PR#7" is the historical alias from the merge request that introduced this method on this repository — the same weights are reported under both names.

Final EER weights (sum = 1):

| Indicator | Head | Weight |
|---|---|---:|
| Cavity | D5 | **0.3249** |
| Solid culture | D3 | 0.1810 |
| AFB smear | D1 | 0.1742 |
| Liquid culture | D4 | 0.1654 |
| TB-PCR | D2 | 0.1544 |

Source files (now on `main`):

- `TB Phase III/reports/20260520 PR7 Entropy Effective Rank - Results.{json,md,csv}`
- `TB Phase III/reports/20260520 PR7 Entropy Effective Rank - Method.md`
- `TB Phase III/reports/20260520_infectivity_latent_methods_comparison.html`
- `TB Phase III/reports/figures/*.png`
- Pipeline: `TB Phase III/build_pr7_outputs.py`, `decompose_block_contributions.py`, `generate_infectivity_methods_report.py`, plus the three latent-analysis scripts `analyze_infectivity_{pls2block,polychoric,tetrachoric}.py`.

### Branch map

| Branch | Role | Status |
|---|---|---|
| **`main`** | CAD packages v1.02/v1.03, Phase I triage, Phase III research scripts + **EER pipeline & results (PLS estimator, merged 2026-05-22)** | Active |
| `stat/pls` | PLS-PM two-block + HOC composite estimator used in production EER weights | Now merged into `main`; kept as historical reference |
| `stat/polychoric` | Threshold-model CFA on polychoric correlations (sensitivity / alternative binary-correlation estimator for EER) | Sensitivity branch |
| `stat/tetrachoric` | 2×2 MLE tetrachoric + ULS/DWLS factor extraction (sensitivity / alternative estimator for EER) | Sensitivity branch |

The three `stat/*` branches differ only in the **binary-correlation estimator** for D1–D5; the downstream EER pipeline is shared. `stat/pls` was promoted to `main` because PLS is the headline estimator referenced in the production weights. The other two stay open as estimator-sensitivity branches.

### Reproducing EER on `main`

```powershell
cd "TB Phase III"
.\run_all_infectivity_latent.sh     # or run_pls.sh
python build_pr7_outputs.py         # produces final weights JSON
python generate_infectivity_methods_report.py
```

### Companion repo

| Repo | Role |
|---|---|
| [TBC_CAD_INFECTIVITY](http<REDACTED_PATH> (this) | Infectivity statistics + **EER** weighting (Phase I/II/III research scripts) |
| [TBC_Phase_III](http<REDACTED_PATH> | Phase III v7.x image classifiers — **v7.011 production** (tag `v7.011-production`); applies EER weights to v7.011 predictions in `eval_v7011_eer_infectivity_score.py` |

---

## Repository Structure

```
TBC_CAD_INFECTIVITY/
├── tbc-cad-ver-1.02/        # CAD package v1.02
├── tbc-cad-ver-1.03/        # CAD package v1.03
├── TB Phase III/            # Phase III model research
│   ├── build_phase3_features.py
│   ├── train_phase3_active_cv_v103.py
│   ├── train_phase3_active_cv_v106_bce.py
│   ├── train_phase3_active_rf_d1d5_cv.py
│   ├── train_phase3_v35_finetune_then_rf.py
│   ├── finetune_phase3_densenet_multilabel.py
│   ├── scripts/             # PowerShell automation scripts (run_phase3_vXX_*.ps1)
│   └── reports/             # Version cheat sheet & notes
└── TB Test DB/tools/        # Phase I / II utilities
```

---

## CAD Package Versions

### `tbc-cad-ver-1.02`
Initial production CAD package. Implements the core TB infectivity grading pipeline from DICOM input to per-label scoring.

### `tbc-cad-ver-1.03`
Incremental update over v1.02. Refactored feature extraction, improved DICOM handling, and minor scoring fixes.

---

## Phase I / II — `TB Test DB/tools/`

Utility scripts used in the early research phases:

| Script | Purpose |
|--------|---------|
| Feature extraction tools | Extract radiomic / deep features from DICOM images |
| RF stage-1 tools | Train and evaluate Random Forest classifiers |
| Plotting utilities | Curve plotting, confusion matrices, score distributions |

---

## Phase III — `TB Phase III/`

End-to-end experimental pipeline.  
**Task:** multi-label binary classification of TB infectivity indicators **D1–D5** (D6 tracked but excluded from primary reporting).  
**Cohort:** Phase III active TB cases, 2294 subjects (`[EMBED_ACTIVE_260428_D5KW_FIX]`).  
**Primary metric:** Macro AUROC averaged over D1–D5 (5-fold cross-validation).

---

### Core Scripts

| Script | Role |
|--------|------|
| `build_phase3_features.py` | DICOM → CLAHE → resize → DenseNet121 feature extraction → `.npz`. Supports `stretch` and `letterbox` resize modes. |
| `train_phase3_active_cv_v103.py` | RF (D1–D5) + XGBoost (D6) heads on frozen features. Baseline multi-label CV. |
| `train_phase3_active_cv_v106_bce.py` | MLP heads with `BCEWithLogitsLoss` + per-label `pos_weight` on frozen features. D1–D6. |
| `train_phase3_active_rf_d1d5_cv.py` | RF heads D1–D5 only. Supports `balanced` or `bce_pos_weight` class weighting. |
| `train_phase3_v35_finetune_then_rf.py` | **4-stage fine-tune pipeline** (see v3.5 below). |
| `finetune_phase3_densenet_multilabel.py` | End-to-end DenseNet121 fine-tuning with BCE head (experimental). |

---

### Pipeline versions — detailed cheat sheet

Authoritative paths, NPZ names, and artifact folders: **`TB Phase III/reports/phase3_pipeline_versions.txt`** (update this file when you add a run).

**Evaluation defaults:** cohort **N = 2294**, split **1606 / 344 / 344** (train / val / test), **`force_close_zero`** where noted, **Macro AUROC** = mean of per-label AUROC over **D1–D5** (5-fold CV, TEST fold reported below).

---

#### Leaderboard — frozen DenseNet → RF (D1–D5, `balanced`)

Comparable high-resolution line (stretch unless noted):

| Order | Tag | Configuration (short) | Macro AUROC D1–D5 |
|------:|-----|----------------------|-------------------|
| 1 | **v3.7** | 1024 + CLAHE + **lung crop** (margin 15%) | **0.8472 ± 0.0084** |
| 2 | v3.8 | v3.7 + **IQR** after CLAHE (ImageNet norm off on tensors) | 0.8460 ± 0.0091 |
| 3 | v3.2 | 1024 + CLAHE, full field | 0.8456 ± 0.0103 |
| — | v3.4 | 1024 + CLAHE + **letterbox** | 0.8377 ± 0.0090 *(lowest σ — often most stable)* |

Other resolutions (Macro AUROC not duplicated here): **v2.9** @448, **v3.1** @512, **v3.3** @2048 letterbox — see cheat sheet / local `artifacts/`.

---

#### Leaderboard — partial-unfreeze DenseNet → RF (D1–D5)

| Tag | Stage1 / Stage2 (epochs) | Macro AUROC D1–D5 |
|-----|--------------------------|-------------------|
| v3.6 | 25 / 45 | 0.8200 ± 0.0100 |
| v3.5 | 15 / 25 | 0.8161 ± 0.0075 |

Trainer: `train_phase3_v35_finetune_then_rf.py` (CLI `--pipeline_version_tag`, `--artifact_slug`). Image cache: `cache_v35_1024_clahe/`.

---

#### Full version index (Phase III)

Backbone is **DenseNet121** (ImageNet pretrained) unless noted. Training scripts live under **`TB Phase III/scripts/`**.

| Tag | Input | Resize | Preprocessing / notes | Classifier | Labels | Macro D1–D5 (TEST) | Automation |
|-----|-------|--------|----------------------|------------|--------|-------------------|------------|
| v1.06 | 224 | stretch | No CLAHE | RF + XGB | D1–D6 | — | manual `build_phase3_features.py` + `train_phase3_active_cv_v103.py` |
| v2.1 | 224 | stretch | CLAHE | RF + XGB | D1–D6 | — | ↑ |
| v2.2 | 224 | stretch | lung crop + CLAHE | RF + XGB | D1–D6 | — | ↑ |
| v2.3 | 224 | stretch | lung crop + CLAHE + IQR post | RF + XGB | D1–D6 | — | ↑ |
| v2.4 | 224 | stretch | CLAHE + IQR post | RF + XGB | D1–D6 | — | ↑ |
| v2.5 | 224 | stretch | IQR pre → CLAHE | RF + XGB | D1–D6 | — | ↑ |
| v2.6 | 224 | stretch | Same **X** as v2.1 CLAHE | **MLP**, BCE + pos_weight | D1–D6 | — | `scripts/run_phase3_v26_clahe_bce.ps1` |
| v2.7 | 224 | stretch | Same **X** as v2.1 | RF `balanced` | **D1–D5** | — | `scripts/run_phase3_v27_v28_rf.ps1` |
| v2.8 | 224 | stretch | Same **X** as v2.7 | RF `bce_pos_weight` | **D1–D5** | — | ↑ |
| v2.9 | 448 | stretch | CLAHE | RF `balanced` | D1–D5 | *(see artifacts)* | `scripts/run_phase3_v29_full_448.ps1` |
| v3.1 | 512 | stretch | CLAHE | RF `balanced` | D1–D5 | *(see artifacts)* | `scripts/run_phase3_v31_full_512.ps1` |
| v3.2 | 1024 | stretch | CLAHE | RF `balanced` | D1–D5 | **0.8456 ± 0.0103** | `scripts/run_phase3_v32_full_1024.ps1` |
| v3.3 | 2048 | letterbox | CLAHE | RF `balanced` | D1–D5 | *(see artifacts)* | `scripts/run_phase3_v33_full_2048_letterbox.ps1` |
| v3.4 | 1024 | letterbox | CLAHE | RF `balanced` | D1–D5 | **0.8377 ± 0.0090** | `scripts/run_phase3_v34_full_1024_letterbox.ps1` |
| v3.5 | 1024 | stretch | CLAHE; **finetune** 15+25 ep → RF | RF `balanced` | D1–D5 | 0.8161 ± 0.0075 | `scripts/run_phase3_v35_finetune_rf.ps1` |
| v3.6 | 1024 | stretch | Same as v3.5, **25+45** ep | RF `balanced` | D1–D5 | 0.8200 ± 0.0100 | `scripts/run_phase3_v36_finetune_rf.ps1` |
| v3.7 | 1024 | stretch | CLAHE + lung crop (margin **0.15**) | RF `balanced` | D1–D5 | **0.8472 ± 0.0084** | `scripts/run_phase3_v37_full_1024_lungcrop.ps1` |
| v3.8 | 1024 | stretch | v3.7 + **IQR post-CLAHE** (no ImageNet norm) | RF `balanced` | D1–D5 | 0.8460 ± 0.0091 | `scripts/run_phase3_v38_full_1024_lungcrop_robustnorm.ps1` |

**Glossary**

- **Stretch**: `Resize((S,S))`. **Letterbox**: aspect-preserving resize + pad to `S×S`.
- **Lung crop**: `lungmask` bbox on HU before windowing; margin expands each edge by `lung_margin ×` bbox side (runs use **0.15**). CXR masks can fail → full-field fallback in code.
- **IQR robust norm**: `(x − median) / (p75 − p25)`, clipped (default ±5). When enabled, **`Normalize(mean/std)` is disabled** so tensors are not double-scaled.

---

#### Fine-tune stages (v3.5 / v3.6)

| Stage | What happens |
|-------|----------------|
| 1 | Backbone **frozen**; train linear head only (BCE + pos_weight, head LR 3e-3). |
| 2 | Unfreeze **`denseblock4` + `norm5`**; backbone LR 1e-5; cosine schedule on stage 2. |
| 3 | Extract **1024-d** penultimate features per fold. |
| 4 | RF D1–D5 **`balanced`**, 600 trees. |

---

## Excluded from this repository

| Type | Pattern |
|------|---------|
| DICOM images | `*.dcm` |
| Feature matrices | `*.npz` |
| Trained models | `*.joblib`, `*.pt`, `*.pth`, `*.ckpt` |
| Output artifacts | `artifacts/`, `artifact/` |
| Image cache | `cache_*/` |
| Exports | `*.csv`, `*.xlsx`, `*.json` |

See `.gitignore` for the complete list.

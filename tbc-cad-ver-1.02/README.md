# tbc-cad-ver-1.02 (tbc-cad-ver.1.02)

Integrated **TB CAD** inference stack:

| Stage | Model | Output |
|------|--------|--------|
| **Phase I** (v1.01) | `RandomForestClassifier` on ResNet18 pooled features | Binary: TB vs non-TB (training definition) |
| **Phase II** (v1.01) | `RandomForestClassifier` (same feature space) | Binary: inactive vs active cohort (training definition) |
| **Phase III** (v1.01 RF + v1.023-style D6) | Multi-output RF for **D1–D4** + **D6 (NTM)** | Five binary heads: AFB (D1), TB-PCR (D2), solid culture (D3), liquid culture (D4), NTM (D6) |

D6 uses **XGBoost** probabilities; D1–D4 use the **Phase III v1.01** RF heads (`rf_phase3_active_vs_inactive.joblib`). This matches the hybrid design documented in `TB Phase III/plot_pipeline_v101_d6_xgb1023_bootstrap5.py`.

## Requirements

- Python ≥ 3.10  
- PyTorch / torchvision (CPU or CUDA)  
- scikit-learn, XGBoost, pydicom, joblib, numpy, certifi  

Install (editable):

```bash
cd d:\tbc-cad-ver-1.02
pip install -e .
```

## Model files (you provide paths)

Place or point to:

1. **TB Test DB** `artifacts/`: `rf_phase1_huge.joblib`, `rf_phase2_huge.joblib`  
2. **TB Phase III** root: `rf_phase3_active_vs_inactive.joblib`  
3. **D6 XGBoost** trained in the v1.023 style (same feature matrix as Phase III). Save once from your training code, e.g.:

   ```python
   xgb.fit(X_train, y_d6)
   xgb.save_model("d6_ntm_xgb.json")  # XGBoost JSON
   ```

   Or pass a **hybrid** `joblib` produced by `TB Phase III/phase3_v102_d6_xgb_cv_train.py` (`phase3_rf_v1_02_d6_xgb_hybrid.joblib`) — this package registers a pickle compatibility alias so that object can load.

4. Optional: **ResNet** `state_dict` `.pt` if you did not use stock torchvision weights (same as feature extraction used for the `.npz` training runs).

## CLI

```bash
tbc-cad-infer --dicom "D:\path\image.dcm" ^
  --tb-artifacts "D:\TB Test DB\artifacts" ^
  --phase3-root "D:\TB Phase III" ^
  --xgb-d6-json "D:\models\d6_ntm_xgb.json"
```

JSON is printed to stdout (UTF-8).

Use `--hybrid-joblib` instead of `--xgb-d6-json` if you already have `phase3_rf_v1_02_d6_xgb_hybrid.joblib`.

## Python API

```python
from pathlib import Path
from tbc_cad_ver_102 import IntegratedCad, IntegratedCadConfig

cad = IntegratedCad(IntegratedCadConfig(
    tb_artifacts_dir=Path(r"D:\TB Test DB\artifacts"),
    phase3_root=Path(r"D:\TB Phase III"),
    xgb_d6_model_path=Path(r"D:\models\d6_ntm_xgb.json"),
))
out = cad.predict(Path(r"D:\path\image.dcm"))
print(out.as_dict())
```

## Notes

- **One** 512-D feature vector (default ResNet18) is computed per study image and shared by all heads — same assumption as the original offline pipelines.  
- Phase II was trained on a **specific positive/negative definition** (see your Phase II manifest); scores are reported for all inputs.  
- If pickle loading fails, prefer **RF + `xgb.save_model` JSON** (no `joblib` hybrid).

## Offline 5× repeated-split CV (Phase I / II / III D1–D4)

`scripts/cv5_phase123_unified.py` refits **clone(RF)** from your shipped joblibs on each run (same pool sizes as `phase*_features_{train,val,test}.npz`), evaluates on the held-out **test** slice, and writes **`artifacts/cv5_phase123_unified/cv5_metrics.json`** plus ROC and confusion-matrix PNGs (operating point default **0.5**). This matches the repeated-split scheme used for Phase III D6 in `TB Phase III/phase3_v102_d6_xgb_cv_train.py` (not patient-level k-fold). D6 is **not** re-run here; use the existing `cv_fold_metrics.json` for the XGBoost head.

```bash
python scripts/cv5_phase123_unified.py ^
  --tb-artifacts "D:\TB Test DB\artifacts" ^
  --phase3-root "D:\TB Phase III" ^
  --out-dir "D:\tbc-cad-ver-1.02\artifacts\cv5_phase123_unified"
```

## Version

Package `tbc-cad-ver-1-02` / model line **tbc-cad-ver.1.02** — integration layer only; underlying RF/XGB weights remain your v1.01 / v1.023 training artifacts.

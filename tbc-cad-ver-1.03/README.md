# tbc-cad-ver-1.03 (tbc-cad-ver.1.03)

Same layout as **1.02**: one ResNet embedding, shared by all heads.

| Stage | Model | Output |
|-------|--------|--------|
| Phase I (v1.01) | RF on pooled features | Binary |
| Phase II (v1.01) | RF | Binary |
| Phase III v1.03 | Multi-output RF **D1–D5** + **XGBoost D6** | Six binary heads: AFB (D1), TB-PCR (D2), solid (D3), liquid (D4), **Cavitary lesion (D5)**, NTM (D6) |

**D1–D5** are predicted by a **six-output** RF (indices 0–4: D1–D5, index 5: RF D6 channel replaced at inference by XGBoost, same idea as 1.02’s five-output RF with D6 replaced by XGB).

## Install

```bat
cd D:\tbc-cad-ver-1.03
pip install -e .
```

## Artifacts

1. **Phase I/II** (unchanged): `TB Test DB/artifacts/`: `rf_phase1_huge.joblib`, `rf_phase2_huge.joblib`
2. **Phase III v1.03** — train from `D:\TB Phase III\phase3_v103_d5_d6_hybrid_train.py` using `.npz` with `X`, `Y` (D1–D4, D6), and `D5` (cavitary), or use the saved **hybrid**:
   - `TB Phase III/artifacts/phase3_v1_03_d5_d6_xgb/phase3_rf_v1_03_d5_d6_xgb_hybrid.joblib`
3. If not using the hybrid, provide **6-output** `rf_phase3_v103_d1_d6_6out.joblib` and **D6** `d6_ntm_xgb.json` (v1.023-style) under `phase3_root`.

## CLI

```bat
tbc-cad-infer-103 --dicom "D:\path\image.dcm" ^
  --tb-artifacts "D:\TB Test DB\artifacts" ^
  --phase3-root "D:\TB Phase III" ^
  --hybrid-joblib "D:\TB Phase III\artifacts\phase3_v1_03_d5_d6_xgb\phase3_rf_v1_03_d5_d6_xgb_hybrid.joblib"
```

JSON: `model_line` is `tbc-cad-ver.1.03`; `phase3` has keys `D1`..`D6`.

## API

```python
from pathlib import Path
from tbc_cad_ver_103 import IntegratedCad, IntegratedCadConfig

cad = IntegratedCad(IntegratedCadConfig(
    tb_artifacts_dir=Path(r"D:\TB Test DB\artifacts"),
    phase3_root=Path(r"D:\TB Phase III"),
    hybrid_joblib_path=Path(r"D:\TB Phase III\artifacts\phase3_v1_03_d5_d6_xgb\phase3_rf_v1_03_d5_d6_xgb_hybrid.joblib"),
))
print(cad.predict(Path(r"D:\x.dcm")).as_dict())
```

## Notes

- v1.02 **hybrid** (5 outputs) is **not** compatible; you must retrain v1.03 or use the new hybrid file.
- D6 XGB can remain the same **v1.023** `d6_ntm_xgb.json` if re-fitted on the same label column; the **RF** must be retrained with **6** columns including D5.

# v2.7: RF D1-D5, class_weight balanced (same RF policy as v103 heads).
# v2.8: RF D1-D5, class_weight {0:1, 1:neg/pos} per label (BCE pos_weight analogue).
# D6 not trained — no XGB.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path $Root)) { $Root = "D:\TB Phase III" }

$Train = Join-Path $Root "train_phase3_active_rf_d1d5_cv.py"
$Npz27 = Join-Path $Root "phase3_features_active_all_260428_d5kw_fix_densenet121_clahe_v27.npz"
$Npz28 = Join-Path $Root "phase3_features_active_all_260428_d5kw_fix_densenet121_clahe_v28.npz"
$Out27 = Join-Path $Root "artifacts\phase3_active_cv_v27_rf_clahe_densenet121_d1_d5"
$Out28 = Join-Path $Root "artifacts\phase3_active_cv_v28_rf_clahe_densenet121_d1_d5_bce_weight"

python $Train --phase3_all_npz $Npz27 --rf_class_weight balanced --pipeline_version 2.7 --force_close_zero --out_dir $Out27
python $Train --phase3_all_npz $Npz28 --rf_class_weight bce_pos_weight --pipeline_version 2.8 --force_close_zero --out_dir $Out28
Write-Host "v2.7: $Out27\phase3_active_rf_d1d5_test_summary.txt"
Write-Host "v2.8: $Out28\phase3_active_rf_d1d5_test_summary.txt"

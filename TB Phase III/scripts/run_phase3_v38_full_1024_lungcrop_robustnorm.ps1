# Phase III v3.8 — v3.7 + per-image robust normalization (IQR, post-CLAHE).
#
# Pipeline: lung crop (margin 15%) → window → [0,1] → CLAHE → robust IQR norm → stretch @1024 → DenseNet121 → RF D1-D5 balanced.
#
# Note (build_phase3_features.py): when --robust-norm iqr is on, ImageNet mean/std normalization is OFF;
# backbone sees IQR-scaled maps (same pattern as legacy v2.3/v2.4 experiments).
#
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path $Root)) { $Root = "D:\TB Phase III" }

$Embed = "D:\[EMBED_ACTIVE_260428_D5KW_FIX]"
$Build = Join-Path $Root "build_phase3_features.py"
$Train = Join-Path $Root "train_phase3_active_rf_d1d5_cv.py"
$Npz = Join-Path $Root "phase3_features_active_all_260428_d5kw_fix_densenet121_clahe_lungcrop_robust_iqr_1024_v38.npz"
$Out = Join-Path $Root "artifacts\phase3_active_cv_v38_rf_clahe_densenet1024_lungcrop_robust_iqr_d1_d5"

python $Build --in_dir $Embed --out $Npz --model densenet121 --pretrained --clahe `
    --lung-crop --lung-margin 0.15 `
    --robust-norm iqr --robust-norm-order post_clahe --robust-norm-clip 5.0 `
    --pipeline-version 3.8 --image_size 1024 --resize_mode stretch --max_files 0

if ($LASTEXITCODE -ne 0) {
    Write-Error "Feature build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

python $Train --phase3_all_npz $Npz --rf_class_weight balanced --pipeline_version 3.8 `
    --force_close_zero --out_dir $Out

if ($LASTEXITCODE -ne 0) {
    Write-Error "RF CV failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "Done: $Out\phase3_active_rf_d1d5_test_summary.txt"

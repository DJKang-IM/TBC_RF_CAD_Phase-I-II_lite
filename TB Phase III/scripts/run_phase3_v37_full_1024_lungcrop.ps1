# Phase III v3.7 — DenseNet121 + CLAHE @1024 + lung crop → RF D1-D5 (balanced). D6/XGB 없음.
# Same cohort/embed as v3.2; adds lungmask bbox crop on HU before window→[0,1]→CLAHE→stretch resize.
#
# Lung bbox margin: do NOT crop tight — expand each edge by margin_ratio × (bbox height or width).
# Clinical-style padding is often 10–15%; this run uses 0.15 (15%) per edge (upper end of that range).
#
# Note: lungmask is CT-trained; on CXR masks can be noisy — pipeline falls back to full field if mask tiny.
#
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path $Root)) { $Root = "D:\TB Phase III" }

$Embed = "D:\[EMBED_ACTIVE_260428_D5KW_FIX]"
$Build = Join-Path $Root "build_phase3_features.py"
$Train = Join-Path $Root "train_phase3_active_rf_d1d5_cv.py"
$Npz = Join-Path $Root "phase3_features_active_all_260428_d5kw_fix_densenet121_clahe_lungcrop_1024_v37.npz"
$Out = Join-Path $Root "artifacts\phase3_active_cv_v37_rf_clahe_densenet1024_lungcrop_d1_d5"

python $Build --in_dir $Embed --out $Npz --model densenet121 --pretrained --clahe `
    --lung-crop --lung-margin 0.15 `
    --pipeline-version 3.7 --image_size 1024 --resize_mode stretch --max_files 0

if ($LASTEXITCODE -ne 0) {
    Write-Error "Feature build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

python $Train --phase3_all_npz $Npz --rf_class_weight balanced --pipeline_version 3.7 `
    --force_close_zero --out_dir $Out

if ($LASTEXITCODE -ne 0) {
    Write-Error "RF CV failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "Done: $Out\phase3_active_rf_d1d5_test_summary.txt"

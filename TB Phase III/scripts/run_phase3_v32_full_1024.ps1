# Phase III v3.2 — 전 코호트 DenseNet121 + CLAHE @1024 → RF D1-D5 (balanced). D6/XGB 없음.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path $Root)) { $Root = "D:\TB Phase III" }

# D5가 반영된 embed 폴더를 사용
$Embed = "D:\[EMBED_ACTIVE_260428_D5KW_FIX]"
$Build = Join-Path $Root "build_phase3_features.py"
$Train = Join-Path $Root "train_phase3_active_rf_d1d5_cv.py"
$Npz = Join-Path $Root "phase3_features_active_all_260428_d5kw_fix_densenet121_clahe_1024_v32.npz"
$Out = Join-Path $Root "artifacts\phase3_active_cv_v32_rf_clahe_densenet1024_d1_d5"

python $Build --in_dir $Embed --out $Npz --model densenet121 --pretrained --clahe `
    --pipeline-version 3.2 --image_size 1024 --max_files 0

python $Train --phase3_all_npz $Npz --rf_class_weight balanced --pipeline_version 3.2 `
    --force_close_zero --out_dir $Out

Write-Host "Done: $Out\phase3_active_rf_d1d5_test_summary.txt"


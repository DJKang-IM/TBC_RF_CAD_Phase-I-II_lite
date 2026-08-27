# Phase III v2.9 — 전 코호트 DenseNet121 + CLAHE @448 → RF D1-D5 (balanced). D6/XGB 없음.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path $Root)) { $Root = "D:\TB Phase III" }

# D5는 별도 태그/규칙 반영 폴더 필요 — 일반 EMBED_ACTIVE_260428 는 D5 항상 0일 수 있음
$Embed = "D:\[EMBED_ACTIVE_260428_D5KW_FIX]"
$Build = Join-Path $Root "build_phase3_features.py"
$Train = Join-Path $Root "train_phase3_active_rf_d1d5_cv.py"
$Npz = Join-Path $Root "phase3_features_active_all_260428_d5kw_fix_densenet121_clahe_448_v29.npz"
$Out = Join-Path $Root "artifacts\phase3_active_cv_v29_rf_clahe_densenet448_d1_d5"

python $Build --in_dir $Embed --out $Npz --model densenet121 --pretrained --clahe `
    --pipeline-version 2.9 --image_size 448 --max_files 0

python $Train --phase3_all_npz $Npz --rf_class_weight balanced --pipeline_version 2.9 `
    --force_close_zero --out_dir $Out

Write-Host "Done: $Out\phase3_active_rf_d1d5_test_summary.txt"

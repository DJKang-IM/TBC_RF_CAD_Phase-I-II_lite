# Phase III v3.3 — DenseNet121 + CLAHE @2048 with letterbox resize → RF D1-D5 (balanced). D6/XGB 없음.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path $Root)) { $Root = "D:\TB Phase III" }

$Embed = "D:\[EMBED_ACTIVE_260428_D5KW_FIX]"
$Build = Join-Path $Root "build_phase3_features.py"
$Train = Join-Path $Root "train_phase3_active_rf_d1d5_cv.py"

$Npz = Join-Path $Root "phase3_features_active_all_260428_d5kw_fix_densenet121_clahe_2048_letterbox_v33.npz"
$Out = Join-Path $Root "artifacts\phase3_active_cv_v33_rf_clahe_densenet2048_letterbox_d1_d5"

python $Build --in_dir $Embed --out $Npz --model densenet121 --pretrained --clahe `
    --pipeline-version 3.3 --image_size 2048 --resize_mode letterbox --letterbox_pad_value 0.0 --max_files 0

python $Train --phase3_all_npz $Npz --rf_class_weight balanced --pipeline_version 3.3 `
    --force_close_zero --out_dir $Out

Write-Host "Done: $Out\phase3_active_rf_d1d5_test_summary.txt"


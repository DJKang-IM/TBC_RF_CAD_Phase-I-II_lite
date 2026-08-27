# Phase III v2.6 — DenseNet121 + CLAHE features, BCE head D1–D6 (six sigmoid outputs).
# NPZ meta pipeline_version=2.6 (see reports\phase3_pipeline_versions.txt).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path $Root)) { $Root = "D:\TB Phase III" }

$Npz = Join-Path $Root "phase3_features_active_all_260428_d5kw_fix_densenet121_clahe_v26.npz"
$Out = Join-Path $Root "artifacts\phase3_active_cv_v26_clahe_bce_densenet121_d1_d6"
$Build = Join-Path $Root "build_phase3_features.py"
$Train = Join-Path $Root "train_phase3_active_cv_v106_bce.py"

$EmbedRoot = "D:\[EMBED_ACTIVE_260428_D5KW_FIX]"

if (-not (Test-Path $Npz)) {
    Write-Host "NPZ not found. Building from DICOM (slow)..." -ForegroundColor Yellow
    python $Build --in_dir $EmbedRoot --out $Npz --model densenet121 --pretrained --clahe --pipeline-version 2.6
}

python $Train --phase3_all_npz $Npz --force_close_zero --out_dir $Out
Write-Host "Done. Summary: $Out\phase3_active_cv_v106_bce_test_summary.txt" -ForegroundColor Green

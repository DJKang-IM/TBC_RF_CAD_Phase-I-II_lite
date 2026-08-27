<#
  Phase III v3.6 — Same pipeline as v3.5 with longer training (more epochs).

  v3.5 : Stage1 15 ep + Stage2 25 ep  → Macro D1-D5 test: 0.8161 ± 0.0075
  v3.6 : Stage1 25 ep + Stage2 45 ep (same LR / partial-unfreeze / RF)

  Uses train_phase3_v35_finetune_then_rf.py with --pipeline_version_tag 3.6 --artifact_slug v36

  Baseline (frozen features):
    v3.2   Macro AUROC D1-D5: 0.8456 ± 0.0103

  Cache  : same as v3.5 (D:\TB Phase III\cache_v35_1024_clahe)
  Expected runtime: ~1.7–2.1× v3.5 (~4–5 h on RTX 4070 class GPU, 5-fold)
#>
Set-StrictMode -Version Latest

$PY = "python"

$EmbedRoot  = "D:\[EMBED_ACTIVE_260428_D5KW_FIX]"
$RefNPZ     = "D:\TB Phase III\phase3_features_active_all_260428_d5kw_fix_densenet121_clahe_1024_v32.npz"
$CacheDir   = "D:\TB Phase III\cache_v35_1024_clahe"
$OutDir     = "D:\TB Phase III\artifacts\phase3_active_cv_v36_finetune_rf_1024_d1_d5"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

& $PY "D:\TB Phase III\train_phase3_v35_finetune_then_rf.py" `
    --embed_root   "$EmbedRoot"   `
    --reference_npz "$RefNPZ"     `
    --cache_dir    "$CacheDir"    `
    --out_dir      "$OutDir"      `
    --pipeline_version_tag "3.6" `
    --artifact_slug        "v36" `
    --cv_runs      5              `
    --seed         42             `
    --image_size   1024           `
    --clahe                        `
    --clahe_clip_limit 0.03       `
    --pretrained                   `
    --epochs_stage1 25            `
    --epochs_stage2 45            `
    --head_lr      3e-3           `
    --backbone_lr  1e-5           `
    --weight_decay 1e-4           `
    --batch_size   4              `
    --rf_estimators 600           `
    --amp

if ($LASTEXITCODE -ne 0) {
    Write-Error "v3.6 training exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "=== v3.6 done. Results at: $OutDir ==="
Get-Content "$OutDir\phase3_v36_finetune_rf_test_summary.txt"

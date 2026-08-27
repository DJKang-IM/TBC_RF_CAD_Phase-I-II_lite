<#
  Phase III v3.5 — Partial-unfreeze DenseNet121@1024 + CLAHE → RF D1-D5 (stretch)

  Stage 1 : frozen backbone  + linear head warm-up  (15 epochs, head LR = 3e-3)
  Stage 2 : unfreeze denseblock4 + norm5             (25 epochs, backbone LR = 1e-5)
  Stage 3 : feature extraction from fine-tuned backbone
  Stage 4 : RF D1-D5  class_weight=balanced, 600 trees

  Baseline reference (same cohort / same 1024 stretch / same CV protocol):
    v3.2   Macro AUROC D1-D5: 0.8456 ± 0.0103
    v3.4   Macro AUROC D1-D5: 0.8377 ± 0.0090   (letterbox variant)

  Requires   : 3.2 NPZ (paths + labels; features unused)
               CUDA strongly recommended (1024-res DICOM loading on CPU is very slow)
  Expected   : ~3-8 h depending on GPU / number of CV runs
#>
Set-StrictMode -Version Latest

$PY = "python"

# ---------- paths ----------
$EmbedRoot  = "D:\[EMBED_ACTIVE_260428_D5KW_FIX]"
$RefNPZ     = "D:\TB Phase III\phase3_features_active_all_260428_d5kw_fix_densenet121_clahe_1024_v32.npz"
$CacheDir   = "D:\TB Phase III\cache_v35_1024_clahe"   # pre-processed .npy (built once)
$OutDir     = "D:\TB Phase III\artifacts\phase3_active_cv_v35_finetune_rf_1024_d1_d5"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# ---------- run ----------
& $PY "D:\TB Phase III\train_phase3_v35_finetune_then_rf.py" `
    --embed_root   "$EmbedRoot"   `
    --reference_npz "$RefNPZ"     `
    --cache_dir    "$CacheDir"    `
    --out_dir      "$OutDir"      `
    --pipeline_version_tag "3.5" `
    --artifact_slug        "v35" `
    --cv_runs      5              `
    --seed         42             `
    --image_size   1024           `
    --clahe                        `
    --clahe_clip_limit 0.03       `
    --pretrained                   `
    --epochs_stage1 15            `
    --epochs_stage2 25            `
    --head_lr      3e-3           `
    --backbone_lr  1e-5           `
    --weight_decay 1e-4           `
    --batch_size   4              `
    --rf_estimators 600           `
    --amp

if ($LASTEXITCODE -ne 0) {
    Write-Error "v3.5 training exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "=== v3.5 done. Results at: $OutDir ==="
Get-Content "$OutDir\phase3_v35_finetune_rf_test_summary.txt"

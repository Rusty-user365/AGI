# Fine-tune both 5k checkpoints with the mixed masking objective.
#
# Starting point: the completed token-masked checkpoints. One factor changes:
#   train.mask_mode = "block", block_mask_prob = 0.5
# The mixture keeps the random-token signal (grammar) while teaching whole-block
# infill (the repair operation). Everything else is held fixed: 5,000 docs,
# LoRA rank 8, learning rate, seed. The two runs are sequential because the
# local card cannot hold both at once.
#
# Restart-safe: a run whose train_summary.json exists is skipped, and an
# interrupted run resumes from its own last checkpoint instead of starting over.
# train.py reads train_progress.json from the output directory and skips epochs
# already completed, so a resumed run finishes the 4A+1B schedule rather than
# repeating it. The checkpoint records `resumed_from` and `prior_epochs` so the
# total number of epochs actually seen is visible in the summary.

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$cgrOut = "checkpoints/diffugpt_5k_e10_cgr_blockft"
$baseOut = "checkpoints/diffugpt_5k_e10_backbone_blockft"

if (-not (Test-Path "$cgrOut/train_summary.json")) {
  $cgrResume = if (Test-Path "$cgrOut/stage_b.pt") {
    "$cgrOut/stage_b.pt"
  } else {
    "checkpoints/diffugpt_5k_e10_cgr/stage_b.pt"
  }
  Write-Host "[blockft] CGR fine-tune starting (resume from $cgrResume)"
  python -u -m cgr.train --preset diffugpt --stage-a-epochs 4 --stage-b-epochs 1 `
    --mask-mode block --block-mask-prob 0.5 --device cuda `
    --resume $cgrResume `
    --out $cgrOut --log-every 100 *> results/train_cgr_blockft.log
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[blockft] CGR failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
  }
} else {
  Write-Host "[blockft] CGR already complete; skipping"
}

if (-not (Test-Path "$baseOut/train_summary.json")) {
  $baseResume = if (Test-Path "$baseOut/stage_b.pt") {
    "$baseOut/stage_b.pt"
  } else {
    "checkpoints/diffugpt_5k_e10_backbone/stage_b.pt"
  }
  Write-Host "[blockft] LoRA-only fine-tune starting (resume from $baseResume)"
  python -u -m cgr.train --preset diffugpt --stage-a-epochs 4 --stage-b-epochs 1 `
    --mask-mode block --block-mask-prob 0.5 --objective backbone --device cuda `
    --resume $baseResume `
    --out $baseOut --log-every 100 *> results/train_backbone_blockft.log
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[blockft] baseline failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
  }
} else {
  Write-Host "[blockft] LoRA-only already complete; skipping"
}

Write-Host "[blockft] both runs complete"

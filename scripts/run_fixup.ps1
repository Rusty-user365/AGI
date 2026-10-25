# Fix-up runs: resume the 0.380 checkpoint with the side channel live.
#
# Two arms, identical in every respect except where `cond` comes from:
#
#   reasoner   the per-document steering vector (the CGR claim)
#   constant   one learned vector for the whole batch, same parameter count
#              and same losses, carrying no information about the input
#
# Without the second arm a gain cannot be attributed to steering: both arms
# train for the same extra epochs, so the difference is the information in
# `cond`, not the extra compute.
#
# Safety:
#   * each arm writes to its own directory; the source checkpoint is only read
#   * a run whose train_summary.json exists is skipped
#   * an interrupted arm resumes from its own stage_b.pt and skips completed
#     epochs, because train.py reads train_progress.json from the output dir
#   * checkpoints and train_diagnostics.json are written atomically every 500
#     steps, so a power loss costs at most 500 steps

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$src = "checkpoints/diffugpt_5k_e10_cgr_blockft/stage_b.pt"

$arms = @(
  @{ name = "reasoner"; cond = "reasoner";
     out = "checkpoints/diffugpt_5k_e10_cgr_fixup_reasoner";
     log = "results/train_fixup_reasoner.log" },
  @{ name = "constant"; cond = "learned";
     out = "checkpoints/diffugpt_5k_e10_cgr_fixup_constant";
     log = "results/train_fixup_constant.log" }
)

if (-not (Test-Path $src)) {
  Write-Host "[fixup] missing source checkpoint $src"
  exit 1
}

foreach ($a in $arms) {
  if (Test-Path "$($a.out)/train_summary.json") {
    Write-Host "[fixup] $($a.name) already complete; skipping"
    continue
  }
  $resume = if (Test-Path "$($a.out)/stage_b.pt") {
    "$($a.out)/stage_b.pt"
  } else {
    $src
  }
  Write-Host "[fixup] $($a.name) starting (cond=$($a.cond), resume from $resume)"
  python -u -m cgr.train --preset diffugpt --stage-a-epochs 2 --stage-b-epochs 0 `
    --mask-mode block --block-mask-prob 0.5 --cond-mode $a.cond --device cuda `
    --resume $resume --out $a.out --log-every 100 *> $a.log
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[fixup] $($a.name) failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
  }
}

Write-Host "[fixup] both arms complete"

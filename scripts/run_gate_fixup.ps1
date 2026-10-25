# Gate runs for the two fix-up arms.
#
# Six runs, serialised because the card has 4 GB and training already proved
# that a concurrent eval risks an OOM kill.
#
#   reasoner arm, reasoner-free   test, unseen   <- comparable to the 0.380 run
#   reasoner arm, --use-reasoner  test, unseen   <- the deployed CGR path
#   constant arm, reasoner-free   test, unseen   <- the capacity-matched control
#
# `--use-reasoner` is deliberately NOT run on the constant arm: that arm was
# trained with a single learned vector and never saw a per-document cond, so
# evaluating it through the reasoner path would measure an out-of-distribution
# input, not a fair control.
#
# A run whose JSON already exists is skipped, so an interrupted sweep resumes.

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$r = "checkpoints/diffugpt_5k_e10_cgr_fixup_reasoner/stage_b.pt"
$c = "checkpoints/diffugpt_5k_e10_cgr_fixup_constant/stage_b.pt"

$runs = @(
  @{ tag = "fixup_reasoner_test";          ck = $r; split = "test";   reas = $false },
  @{ tag = "fixup_constant_test";          ck = $c; split = "test";   reas = $false },
  @{ tag = "fixup_reasoner_test_reasoner"; ck = $r; split = "test";   reas = $true  },
  @{ tag = "fixup_reasoner_unseen";        ck = $r; split = "unseen"; reas = $false },
  @{ tag = "fixup_constant_unseen";        ck = $c; split = "unseen"; reas = $false },
  @{ tag = "fixup_reasoner_unseen_reasoner"; ck = $r; split = "unseen"; reas = $true }
)

foreach ($x in $runs) {
  $json = "results/gate_$($x.tag).json"
  if (Test-Path $json) {
    Write-Host "[gate] $($x.tag) already done; skipping"
    continue
  }
  $extra = if ($x.reas) { "--use-reasoner" } else { "" }
  Write-Host "[gate] $($x.tag) starting $(Get-Date -Format 'HH:mm:ss')"
  python -u scripts/check_backbone.py --preset diffugpt --checkpoint $x.ck `
    --split $x.split --n 200 --steps 8 --device cuda $extra `
    --json-out $json *> "results/check_backbone_$($x.tag).txt"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[gate] $($x.tag) failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
  }
  Write-Host "[gate] $($x.tag) done $(Get-Date -Format 'HH:mm:ss')"
}

Write-Host "[gate] all runs complete"

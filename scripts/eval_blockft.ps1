# Re-run the editor gate on the mixed-mask fine-tunes, with the original 5k
# checkpoints as the before-condition.
#
# The gate is `on-thesis >= 0.20` on one-block infill. `test` and `unseen` are
# evaluated separately and never pooled: `unseen` is the held-out-thesis split
# and a number from it cannot be substituted for the ordinary test result.
#
# Every run writes both a console log and a JSON payload. The JSON is what the
# paper table cites; the log is for reading.

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$runs = @(
  @{ name = "cgr";      tag = "blockft"; ckpt = "checkpoints/diffugpt_5k_e10_cgr_blockft/stage_b.pt" },
  @{ name = "backbone"; tag = "blockft"; ckpt = "checkpoints/diffugpt_5k_e10_backbone_blockft/stage_b.pt" },
  @{ name = "cgr";      tag = "5k";      ckpt = "checkpoints/diffugpt_5k_e10_cgr/stage_b.pt" },
  @{ name = "backbone"; tag = "5k";      ckpt = "checkpoints/diffugpt_5k_e10_backbone/stage_b.pt" }
)

foreach ($r in $runs) {
  if (-not (Test-Path $r.ckpt)) {
    Write-Host "[gate] missing $($r.ckpt); skipping $($r.name)/$($r.tag)"
    continue
  }
  foreach ($split in @("test", "unseen")) {
    $log = "results/check_backbone_$($r.name)_$($r.tag)_$split.txt"
    $js  = "results/gate_$($r.name)_$($r.tag)_$split.json"
    Write-Host "[gate] $($r.name) / $($r.tag) / $split"
    python -u scripts/check_backbone.py --preset diffugpt `
      --checkpoint $r.ckpt --split $split --n 200 --steps 8 --device cuda `
      --json-out $js *> $log
    if ($LASTEXITCODE -ne 0) {
      Write-Host "[gate] $($r.name)/$($r.tag)/$split failed with exit code $LASTEXITCODE"
      exit $LASTEXITCODE
    }
  }
}

# The deployed CGR editor conditions the infill on the reasoner. The gate above
# uses the reasoner-free path so it stays comparable with the original 5k gate;
# this extra run reports the deployed path for the fine-tuned CGR checkpoint.
foreach ($split in @("test", "unseen")) {
  $log = "results/check_backbone_cgr_blockft_${split}_reasoner.txt"
  $js  = "results/gate_cgr_blockft_${split}_reasoner.json"
  Write-Host "[gate] cgr+reasoner / blockft / $split"
  python -u scripts/check_backbone.py --preset diffugpt `
    --checkpoint checkpoints/diffugpt_5k_e10_cgr_blockft/stage_b.pt `
    --split $split --n 200 --steps 8 --device cuda --use-reasoner `
    --json-out $js *> $log
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[gate] cgr+reasoner/$split failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
  }
}

Write-Host "[gate] done"

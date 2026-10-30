# CGR Head Ablation Study

## Start here

This folder tracks the controlled study of whether CGR's auxiliary head
supervision contributes to the reported repair gain beyond parameter count and
generic auxiliary training. Read this file before launching a run. The reusable
implementation is intentionally kept in the package modules under `cgr/`; the
run protocol, logs, and study record live here.

## Research question

The study asks whether correctly aligned auxiliary targets help the shared
representation, beyond adding trainable capacity or generic head computation.
The heads predict auxiliary properties from the shared pooled features. They
do not feed their predictions directly into the Gate or repair decoder; their
training influence is through gradients from `h_loss` into shared features.

The seven outputs in the current code are `value`, `outcome`, `tau`,
`epistemic`, `scope`, `source`, and `disagreement`. There is no separately
trained `surprise` head in the current implementation.

## Implemented controls

Use `python -m cgr.train --ablation-mode MODE`. The default is `none`, which
preserves the usual CGR head and loss path.

| Mode | Behavior |
|---|---|
| `none` | Standard CGR training with learned heads and correctly paired labels. |
| `random_heads` | Same head architecture, initialized from the run seed and frozen. Its head loss remains active, so gradients can pass through the fixed readouts into shared features. |
| `shuffled_labels` | Independently permutes each head's labels across the current batch. The permutation is deterministic for a run seed, stage, epoch, batch, and head. It does not swap targets between heads. |
| `no_loss` | Keeps the head forward pass but freezes head parameters and omits `h_loss` from the total objective. |
| `matched_lora` | Selects the existing backbone-only denoising objective and disables conditioning. Set `--lora-rank` to the desired rank. |

The rank override is applied to the config before `CGRSystem` construction. It
preserves the preset's `lora_alpha / lora_rank` scale and records the resulting
rank and alpha in the run metadata.

## Parameter-count note

For the 12-layer, 768-wide DiffuGPT model, the two LoRA adapters per layer add
`55,296 × rank` adapter parameters. Rank 8 is 442,368; rank 146 is 8,073,216;
rank 147 is 8,128,512. These are adapter-only counts. Before describing a run
as parameter-matched, compare the CGR run's recorded total trainable count with
the matched arm's count in `train_summary.json`; rank 146 does not by itself
prove that every trainable parameter was matched.

## Run protocol

Run each arm from the same DiffuGPT pretrained source, with identical corpus,
seed, stage schedule, masking policy, batch size, and optimizer settings. Do
not resume one ablation from another arm's trained checkpoint. In particular,
rank 146 adapters cannot load a rank 8 LoRA checkpoint, and loading trained
head weights defeats the random-head and no-loss controls.

Example PowerShell commands below use the DiffuGPT preset and block masking;
replace stage counts only when the preregistered run schedule requires it.
Run these on the GPU machine after the active training job is finished.

```powershell
python -u -m cgr.train --preset diffugpt --ablation-mode none --lora-rank 8 `
  --stage-a-epochs 4 --stage-b-epochs 1 --mask-mode block --block-mask-prob 0.5 `
  --device cuda --out checkpoints/head_ablation/none *> experiments/head_ablation/logs/none.log

python -u -m cgr.train --preset diffugpt --ablation-mode random_heads --lora-rank 8 `
  --stage-a-epochs 4 --stage-b-epochs 1 --mask-mode block --block-mask-prob 0.5 `
  --device cuda --out checkpoints/head_ablation/random_heads *> experiments/head_ablation/logs/random_heads.log

python -u -m cgr.train --preset diffugpt --ablation-mode shuffled_labels --lora-rank 8 `
  --stage-a-epochs 4 --stage-b-epochs 1 --mask-mode block --block-mask-prob 0.5 `
  --device cuda --out checkpoints/head_ablation/shuffled_labels *> experiments/head_ablation/logs/shuffled_labels.log

python -u -m cgr.train --preset diffugpt --ablation-mode no_loss --lora-rank 8 `
  --stage-a-epochs 4 --stage-b-epochs 1 --mask-mode block --block-mask-prob 0.5 `
  --device cuda --out checkpoints/head_ablation/no_loss *> experiments/head_ablation/logs/no_loss.log

python -u -m cgr.train --preset diffugpt --ablation-mode matched_lora --lora-rank 146 `
  --stage-a-epochs 4 --stage-b-epochs 1 --mask-mode block --block-mask-prob 0.5 `
  --device cuda --out checkpoints/head_ablation/matched_lora *> experiments/head_ablation/logs/matched_lora.log
```

`matched_lora` automatically uses the backbone-only objective. Do not also
pass `--objective cgr` expecting a CGR run; the mode selects the control
objective. The output directory is under `checkpoints/`, which is ignored by
Git in this repository. Text logs are kept under this study folder.

## Evaluation and interpretation

Only one-step DirectML compatibility pilots have run so far; no full ablation
training or evaluation has been completed. Do not fill the results table with
estimates. After each full run, record the exact command, commit, checkpoint
path, trainable parameter count, seed, runtime, device, evaluation split,
metric, and confidence interval in `logs/run_log.md`.

The existing Exp8 evaluator compares a CGR checkpoint and a backbone checkpoint
with different reasoner-conditioning behavior; it is not a multi-arm head
ablation evaluator. Use the same repair benchmark and evaluation conditions
for every arm, and report paired per-example results. Add a dedicated paired
evaluation protocol before claiming a head-ablation effect.

An outcome table can be added here after results are generated. Keep raw
per-example outputs and aggregate summaries together; record all seeds and
confidence intervals. A single score difference is evidence about this run,
not a mathematical proof that the head design is the only cause.

## Implementation map

- `cgr/ablations.py`: frozen random heads and deterministic per-head label
  shuffling.
- `cgr/model/system.py`: selects the requested head implementation while
  preserving the normal default.
- `cgr/train.py`: CLI options, mode-specific loss handling, matched-LoRA
  objective selection, run banner, and checkpoint/summary metadata.
- `cgr/config.py`: rank setter applies rank and preserves adapter scale before
  model construction.
- `logs/run_log.md`: development and run history. Append entries; do not
  rewrite past run records.

## Intel Iris Xe local pilot

The i5 laptop's Intel Iris Xe is accessible through DirectML in the isolated
`.venv/iris` environment. Install it with
`pip install -r requirements-iris.txt`, then select it using
`--device directml`. This is a compatibility path for local development; it is
not the GPU configuration used for the paper runs. DirectML reports the device
as `privateuseone:0` in PyTorch.

On this machine a batch of 8 at sequence length 128 ran out of DirectML shared
GPU memory. Batch size 1 completed. The logs in `logs/iris_pilot/` are therefore
one-step execution pilots (`--batch-size 1 --max-batches 1 --docs 1`, one Stage
A epoch, no Stage B), not training results. Their losses are smoke evidence
only: they do not support comparisons between modes or benchmark claims. The
larger documented runs should use the designated GPU machine and the agreed
full schedule. A pilot checkpoint has `partial_run: true` and the trainer
intentionally refuses to resume it as a full run.

Use this command shape if you need to reproduce the local compatibility pilot:

```powershell
python -u -m cgr.train --preset diffugpt --device directml `
  --ablation-mode random_heads --lora-rank 8 --batch-size 1 `
  --max-batches 1 --docs 1 --stage-a-epochs 1 --stage-b-epochs 0 `
  --out checkpoints/head_ablation/iris_pilot/random_heads `
  *> experiments/head_ablation/logs/iris_pilot/random_heads.log
```

Read `logs/run_log.md` for the actual commands, pilot losses, parameter counts,
backend limitation, and the distinction between these pilots and the full
ablation study.

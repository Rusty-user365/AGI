# Head Ablation Study Log

Append dated entries. A run entry must distinguish a launched run from a
completed run and link its console log, checkpoint, and evaluation output.

## 2026-10-25 — Ablation wiring prepared

- Added `--ablation-mode` choices: `none`, `random_heads`,
  `shuffled_labels`, `no_loss`, and `matched_lora`.
- Added `--lora-rank`; the rank is set before model creation and the preset's
  alpha/rank multiplier is preserved.
- Random heads are frozen while keeping the differentiable path to shared
  features. Shuffled labels use reproducible, independent permutations for
  each head. No-loss mode still computes head predictions but excludes their
  loss. Matched-LoRA mode selects the backbone-only objective.
- Checkpoint metadata and training summaries now record the ablation mode and
  LoRA settings; the trainer prints the selected mode at startup.
- **Training/evaluation status:** not run. There are no ablation scores or
  conclusions yet. The currently discussed 0.380 vs 0.245 result is prior
  project context, not a result produced by this study folder.
- **Validation status:** Python syntax compilation and `git diff --check` passed.
  The test suite was not run, and no GPU job was launched as part of preparing
  this folder.

## 2026-10-25 — Launch requested; local preflight blocked

- Checked the project virtual environment: PyTorch is `2.14.1+cpu`,
  `torch.cuda.is_available()` is false, and CUDA device count is zero.
- `nvidia-smi` is unavailable, and there are no run logs or run directories in
  this study folder yet.
- **No ablation training was started.** The configured DiffuGPT study is a
  multi-arm 127M-model training run; running it on this CPU-only laptop would
  not produce the requested GPU experiment in a reasonable time. Launch the
  documented commands on the GTX 1050 machine when that device is available.

## 2026-10-25 — Intel Iris Xe DirectML execution pilots

- The user clarified that this laptop has Intel Iris integrated graphics. Set
  up an isolated `.venv/iris` Python environment with Python 3.11.16,
  `torch 2.4.1+cpu`, `torch-directml 0.2.5.dev240914`, and
  `transformers 4.57.6`. The DirectML runtime enumerated
  `Intel(R) Iris(R) Xe Graphics` as `privateuseone:0`. The original `.venv`
  was left unchanged.
- A real forward/backward/optimizer step for the DiffuGPT CGR model was run for
  each of the five modes. All commands used the same preset seed and source
  checkpoint, batch size 1, one batch, one Stage A epoch, zero Stage B epochs,
  and one generated training document. Each run is marked `partial_run: true`.
- Exact invocation template (replace MODE/RANK for each arm):

  ```powershell
  python -u -m cgr.train --preset diffugpt --device directml `
    --ablation-mode MODE --lora-rank RANK --batch-size 1 `
    --max-batches 1 --docs 1 --stage-a-epochs 1 --stage-b-epochs 0 `
    --out checkpoints/head_ablation/iris_pilot/MODE `
    *> experiments/head_ablation/logs/iris_pilot/MODE.log
  ```

- Pilot observations (single-step objective values; **not evaluation scores**):

  | Mode | Rank | Trainable parameters | Step loss | Step time | Status |
  |---|---:|---:|---:|---:|---|
  | `none` | 8 | 22,324,637 | 17.6558 | 7.21 s | completed, partial |
  | `random_heads` | 8 | 19,854,474 | 17.6558 | 6.91 s | completed, partial |
  | `shuffled_labels` | 8 | 22,324,637 | 17.6558 | 6.94 s | completed, partial |
  | `no_loss` | 8 | 19,854,474 | 11.4156 | 6.12 s | completed, partial |
  | `matched_lora` | 146 | 8,073,216 | 4.0027 | 3.46 s | completed, partial |

- The matched arm has 8.07M trainable LoRA parameters. The current full CGR
  default reports 22.32M trainable parameters, so the planned `r=146` control
  matches the adapter count claimed in the study plan, **not the full CGR
  trainable count**. Resolve this parameter-count mismatch before treating it
  as a fully parameter-matched comparison.
- A batch of 8 at 128 tokens OOMed on this shared-memory DirectML device; batch
  size 1 succeeded. PyTorch DirectML warned that AdamW's `aten::lerp.Scalar_out`
  falls back to CPU, which may slow training. This local setup is useful for
  compatibility checks, but not a practical substitute for the planned full
  multi-epoch, multi-document study.
- The first `random_heads` pilot exposed a DirectML autograd failure through
  frozen readout parameters. `cgr/train.py` now temporarily enables those
  parameters for graph construction on DirectML, discards their gradients
  after backward, and keeps them out of the optimizer. The retry completed;
  this preserves frozen readout weights while allowing gradients into the
  shared features.
- Run logs: [`none`](iris_pilot/none.log),
  [`random_heads`](iris_pilot/random_heads.log),
  [`shuffled_labels`](iris_pilot/shuffled_labels.log),
  [`no_loss`](iris_pilot/no_loss.log),
  [`matched_lora`](iris_pilot/matched_lora.log). Checkpoints and summaries are
  under the ignored `checkpoints/head_ablation/iris_pilot/` directory.
- **Scientific status:** no benchmark evaluation, full epoch, or ablation
  conclusion was produced. These five one-step pilots only confirm that each
  training branch can execute on this Iris/DirectML setup.

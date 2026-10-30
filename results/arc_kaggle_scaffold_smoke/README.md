# Local ARC scaffold verification

`notebook_run.log` is the captured live notebook transcript;
`verification_results.json` contains the generated answers, the diagnostic
record and the explicit checks; `arc_demo_predictions.json` is synthetic output.
These files come from a local CPU run. They are not a Kaggle run or an ARC
benchmark result.

The notebook, this directory and `ARC_PRIZE_2026_KAGGLE_PLAN.md` are released
under **MIT-0** (`LICENSE-MIT-0.txt`). The rest of the repository remains MIT.

## What is checked

| Check | Guarantee |
|---|---|
| `multiple_test_inputs` | one entry per test input, `attempt_1`/`attempt_2` present, second attempt distinct |
| `invalid_grid_rejections` | ragged, out-of-range, boolean and empty grids raise `ValueError` |
| `fallback` | no fitting transform falls back to identity; a 1x1 grid is recorded as admitting no distinct second attempt |
| `partial_fit_second_attempt` | the runner-up by training-example fit supplies `attempt_2`, not a duplicate |
| `task_error_recovery` | a malformed task degrades to identity in competition mode and re-raises in strict mode |
| `placeholder_submission` | the pre-solve placeholder is a complete, valid mapping |
| `synthetic_distinct_attempts` | the toy task emits two distinct grids |
| `cache` | cropped-cache continuation matches a fresh recomputation (argmax and logit delta) |
| `factorial` | base and coder answers plus the three-case verifier |

## Reproduce on this machine

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'D:\Research Paper\.adapter_compare_deps'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:CGR_LIVE_QWEN = '1'
$env:CGR_LIVE_ADAPTER = '1'
$env:CGR_ADAPTER_LICENCE_ACK = '1'
$env:CGR_QWEN_MODEL_DIR = 'C:\Users\abhis\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca'
$env:CGR_CODER_ADAPTER_DIR = 'D:\Research Paper\notebooks\coder_qwen3_0.6b_q_mlp_r16_v1___Coder Adapter\checkpoint-1000'
& '.venv\Scripts\python.exe' 'results\arc_kaggle_scaffold_smoke\verify_scaffold.py'
```

Running it replaces the log and results. Paths are local, not portable; a
reviewer needs the dependencies and the corresponding model and adapter files.

## Limits

- The revision check reads the snapshot directory name. A differently named
  copy of the weights reports `revision_verified: false`; it is not a
  cryptographic attestation.
- `PYTHONPATH=.adapter_compare_deps` supplies both `transformers 4.57.6` and
  `peft 0.21.2`. The base `.venv` carries `transformers 5.20.2`, which the cache
  API guard would reject, and no `peft` at all. The recorded run therefore mixes
  `.venv` `torch 2.14.1+cpu` with the dependency directory; that is stated in
  `verification_results.json` as `cache.torch`.
- The fp16 tolerance (`0.125`) has never been exercised — every run so far is
  CPU float32 — so it is recorded with `tolerance_validated: false`.
- The factorial verifier runs in a restricted subprocess, not a security
  sandbox.
- `peft` is not installed in `.venv`; it is loaded from `.adapter_compare_deps`
  through `PYTHONPATH`, together with `transformers 4.57.6`.

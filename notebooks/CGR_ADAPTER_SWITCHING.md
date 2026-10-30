# Local CGR adapter switching integration test

Open `http://127.0.0.1:8765/switching` after starting `Start Adapter Lab.cmd`.
The same server and resident model also serve the existing A/B page at `/`.
Here, **base** means the original chat-capable `Qwen/Qwen3-0.6B` with the
specialist disabled, not the separate `Qwen3-0.6B-Base` checkpoint.

The runtime is `cgr/control/adapter_runtime.py`. It uses the existing
`CommitmentDAG` and `Controller`, with task metadata and deterministic verifiers
from `cgr/control/adapter_tasks.py`. It implements the first, metadata-driven
router in `ADAPTER_ROUTING_PLAN.md`; it does not load the legacy diffusion
backbone, its learned gate, or an untrained Layer-20 classifier onto Qwen.

## What to run

- **Switching sequence:** base READY, coder clamp, base ACK, coder sum-even,
  base DONE. This exercises repeated base-to-specialist and specialist-to-base
  transitions on one model and one committed conversation.
- **Planned coder nodes:** the declared skill activates coder; a separate base
  diagnostic uses the same prefix. Each attempt is separately verified.
- **Base first / repair:** base is used first. If a coder node fails its
  verifier, the failed draft is rejected and the coder gets one bounded attempt
  from that node's saved prefix. A base pass avoids a specialist call.
- **Injected failure / rollback control:** the harness explicitly inserts
  `return a - b` for an addition task. The verifier must reject it and the
  controller must restore its ancestor. The real coder then attempts the task.
  The injected text is labelled in the UI/JSON and is not a model generation.
- **Unknown skill:** abstain without substituting coder.

Only checkpoint-1000 of the coder specialist is currently available. It is a
candidate under integration test, not a specialist promoted by held-out quality
evaluation. These disclosed, small tasks cannot establish general code quality.

## Commit and cache rules

1. The adapter's training manifest must match the pinned base ID and immutable
   SHA. The run records adapter-weight and runtime-template hashes.
2. Prefill known task instructions under base weights. Record the absolute
   boundary and clone its clean prefix cache.
3. Generate into a separate working cache. Assert actual enabled/disabled LoRA
   state and active specialist in every policy context.
4. Crop an uncommitted base candidate's cache to the recorded boundary using
   Transformers **4.57.6** absolute-length crop, then require exact prefix K/V.
5. Reject truncated output, failed pure-function tests, or incorrect exact
   replies. Function tests run in a restricted subprocess with a three-second
   timeout, imports/dunder access blocked, and a limited set of builtins and
   methods. This is not an arbitrary-Python security sandbox.
6. Discard specialist cache entries. Re-prefill the exact accepted token IDs
   under base weights from a fresh clone of the preserved base cache.
7. Compare against full-base prefill of the exact same IDs: finite logits,
   max absolute logit delta, next-token argmax, cache length, and four-token
   greedy continuation. Resume from bridge logits, without feeding the last
   patch token twice.
8. Commit only if both verifier and Bridge pass. Otherwise restore both the
   verified DAG artifact and its saved base cache. Later task instructions are
   appended to that committed state.

Absolute logit tolerance was set before the real integration run: **1e-4 for
fp32**, **0.125 for fp16**. Every result carries its tolerance and measured
delta; fp16 has not been validated on this CPU-only machine. Prefix K/V after
crop must be exactly equal independently of logit tolerance.

The specialist deliberately attends to a base-produced verified prefix. This
does not claim equivalence to full-prefix computation under specialist weights.
Keeping Q/K/V projection weights frozen would not prove such equivalence.

## Results and reproduction

Every completed run is saved under `results/cgr_adapter_switching/<run_id>.json`.
The latest is `results/cgr_adapter_switching_latest.json`, displayed automatically
on reopening the UI. The UI can export the complete record, including rejected
attempts, token IDs, task tests, adapter state events, timings, and DAG metadata.

With the local server running:

```powershell
$request = @{ task = 'switch_suite'; mode = 'planned'; max_new_tokens = 160; seed = 42 } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/switching/run' -Method Post -ContentType 'application/json' -Body $request -TimeoutSec 900
```

The structural/verifier tests run with the isolated dependencies:

```powershell
$env:PYTHONPATH = 'D:\Research Paper\.adapter_compare_deps;D:\Research Paper'
& 'D:\Research Paper\.venv\Scripts\python.exe' -m pytest tests\test_adapter_switching.py -q
```

Base diagnostics are conditioned on the same committed prefix; they are not
independent base-only trajectories. Known test prompts and their synthetic
failure controls are not an unseen evaluation split. Fresh-base reference and
continuation diagnostics add overhead, so elapsed time is not a production
speedup measurement. A semantic failure and a Bridge failure are reported
separately.

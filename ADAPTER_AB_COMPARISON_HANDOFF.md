# Handoff: Base Model vs Coder Adapter A/B Studio

**Workspace:** `D:\Research Paper`  
**Prepared:** 2026-10-30  
**Purpose:** Give the next AI enough context to continue the side-by-side inference tool and finish a real comparison without guessing at the model, adapter, or experiment scope.

**Future multi-adapter routing:** The user supplied a two-stage domain selector
and correctness-gate concept as a future goal, not a request to implement it
now. See `pending_experiments/MULTI_ADAPTER_ROUTING_FUTURE.md` for the proposal,
current evidence, limitations, and pending validation. Its loading and scaling
claims are unmeasured. Recent UI and cyber work supersedes older pending-status
notes below; see `notebooks/WEBPAGE_DESIGN_COMPARISON.md` and
`findings/2026-10-30-cross-dataset-routing-generalization.md`.

## Completed follow-up (2026-10-30)

**Webpage baseline requested while UI trains:** The normal Qwen3-0.6B generated
one complete FocusFlow landing page with all 112 LoRA layers disabled. Preview:
`http://127.0.0.1:8765/webpage-test`; direct page: `/webpage-test/base`.
Frozen prompt/settings: `data/validation/webpage_design_v1.json` (seed 42,
greedy, 3072-token cap, same chat format for the future UI adapter).
Result: 1291 tokens, EOS, 313.924 s CPU fp32. Files and screenshots are under
`results/webpage_design_v1/`; `BASELINE_NOTES.md` records omissions: inactive
CTA, no working signup form/confirmation, missing schedule preview and palette.
The model-generated page has not been repaired. UI adapter generation remains
pending. Preserve this baseline and reuse its exact prompt token IDs/settings
for the UI-adapter arm. `POST /api/webpage/base` returns the saved result on
repeat calls; it does not silently regenerate it. See
`notebooks/WEBPAGE_DESIGN_COMPARISON.md`.

**Validation preparation while cyber/UI train:** The user's pasted proposal
is addressed in `ADAPTER_VALIDATION_PROTOCOL.md`; the paper roadmap is
`PAPER_NEXT_STEPS.md`. Generated 20 public variations of one sanitized-card
integration family in `data/validation/cgr_multidomain_v1.json`, plus a pending
candidate template. `scripts/validate_multidomain.py` exercises the new
dependency graph and Python/SQL/static-HTML validators, checks adapter metadata,
and grades supplied outputs. Fixture self-tests use zero model generations;
they do not measure multi-adapter inference or React/browser functionality.
Actual supplied candidates are still pending. The new graph invalidates
dependent artifacts while planning replay for independent suffix artifacts;
it is not yet wired into the live model switching UI. Prior live coder results
remain in their original JSON files. Paper §6 now states the preliminary
evidence and pending experiments and corrects the frozen-K/V equivalence claim.
The highest-priority next diagnostic compares full coder prefill with the
base-prefix-to-coder switch on identical chat tokens. The clamp failure already
used a chat template, and its cause is unknown.

**Later CGR integration follow-up:** The user asked to test adapter switching
inside the CGR architecture. `http://127.0.0.1:8765/switching` now runs the real
checkpoint through the existing `CommitmentDAG`/`Controller`, metadata skill
routing, bounded function tests, prefix-cache rollback, and base-only Bridge
prefill/continuation checks. Saved runs can be selected/exported in the UI.
The five-node planned sequence made four verified policy transitions and
committed 4/5 nodes (wrong coder clamp rejected, even-sum accepted); all
accepted-node cache checks passed. The injected failure repair and
base-pass/no-specialist controls also passed. Only coder is loaded. Details:
`findings/2026-10-30-cgr-adapter-switching-integration.md`,
`notebooks/CGR_ADAPTER_SWITCHING.md`. Core implementation:
`cgr/control/adapter_runtime.py`, `cgr/control/adapter_tasks.py`,
`scripts/adapter_task_verifier.py`, `notebooks/cgr_adapter_switching.html`.
The API uses the same resident model and lock as A/B. Switching finally leaves
adapters disabled; subsequent A/B runs explicitly re-enable coder for their
adapter pass. Twelve focused tests and a real browser-button check passed.

The exact pinned base download finished and a real greedy comparison completed.
The local UI is running at `http://127.0.0.1:8765` and was opened in the user's
browser. A headless Edge check confirmed both generated outputs, metrics, and
the loaded/ready state render correctly; screenshot: `results/adapter_lab_ui.png`.

- Prompt: `Write only Python code for merge_intervals(intervals). Merge overlapping intervals, return a sorted list, and handle an empty list.`
- Shared settings: seed 42, greedy, 192-token cap, identical 37-token chat prompt.
- Base: 192 new tokens, 27.485 s, 6.986 tokens/s; reached token limit during an added example.
- Checkpoint-1000: 96 new tokens, 15.414 s, 6.228 tokens/s; EOS after the function.
- Device: CPU fp32, four threads. These are one prompt's generation timings,
  excluding loading; base ran first. No held-out correctness claim or adapter
  throughput advantage is established by this run.
- Full real outputs and runtime versions: `results/coder_base_vs_adapter_checkpoint_1000.json`.
- Double-click `Start Adapter Lab.cmd` to start/reuse the server and open the UI.
  `scripts/start_adapter_lab.ps1 -NoBrowser` starts/reuses it without opening a browser.
- Dependencies remain isolated in `.adapter_compare_deps`; `.venv` was not downgraded.
- Fixed PEFT's CPU auto-redispatch by passing the same explicit device map to
  both base and adapter loading. Without it, PEFT attempted disk offload.
- The server automatically saves the latest comparison and exposes `/api/latest`;
  the page restores it on load and polls generation stages during interactive runs.
- Background server PID: `results/adapter_lab_server.pid`; logs:
  `results/adapter_lab_server.log`, `results/adapter_lab_server_errors.log`.
- Sandbox-local sockets could not be reached from the host. The successful
  server was launched outside that sandbox through the normal launcher.

The old pending download/session instructions below are historical and superseded
by this completion record. Check live status before restarting the loaded model.

## Goal

Let the user enter one coding prompt and compare actual generations from:

1. The frozen `Qwen/Qwen3-0.6B` base model.
2. The same base model with the local coder LoRA from `checkpoint-1000` enabled.

Both runs must use the same tokenizer/chat format, prompt, seed, and decoding settings. Show the two outputs next to each other, report measured latency/token counts/throughput (and CUDA peak allocation only when CUDA is used), and let the user export the run as JSON.

This A/B studio is ordinary base-vs-adapter generation. It is **not** itself a Bridge Protocol or KV-cache-crop test; those checks are in the separate Bridge notebook.

## Why this work was needed

The supplied `notebooks\cgr_multi_adapter_studio.html` previously simulated inference with hard-coded text and invented values for latency, perplexity, and VRAM. Those numbers could be mistaken for model measurements. It has been replaced with a real local runner and a comparison-focused UI. The new UI deliberately does not claim code correctness or Bridge success; it presents generated text and actual runtime measurements only.

The adapter’s exact base matters. Its config says `Qwen/Qwen3-0.6B`; do not substitute `Qwen/Qwen3-0.6B-Base`. The adapter is a rank-16 LoRA with alpha 32 targeting `q_proj`, `gate_proj`, `up_proj`, and `down_proj`.

## Components created or changed

- `notebooks\cgr_multi_adapter_studio.html` — redesigned responsive A/B page. It provides coder prompt presets, shared generation controls, base/adapter output panels, per-run metrics, connection/error status, copy, and JSON export. It has no CDN dependency and expects to be served by the Python runner.
- `scripts\adapter_compare_server.py` — local Python HTTP server. It exposes `/api/status` and `/api/compare`, serves the HTML at `/`, loads the exact base and local adapter on first comparison, runs the base with the adapter disabled, then runs the adapter with the same inputs/settings. It binds to `127.0.0.1` by default.
- `scripts\requirements-adapter-compare.txt` — comparison dependency list, pinned to Transformers 4.57.6 and PEFT 0.21.2 for compatibility with the notebook/smoke stack.
- `notebooks\ADAPTER_COMPARE_STUDIO.md` — user setup notes. Review its generic `python` commands; in this environment the Windows Store `python` alias failed, while `D:\Research Paper\.venv\Scripts\python.exe` worked.

Default adapter path in the server:

`D:\Research Paper\notebooks\coder_qwen3_0.6b_q_mlp_r16_v1___Coder Adapter\checkpoint-1000`

The server uses the exact base revision recorded in the smoke report and coder run:

`c1899de289a04d12100db370d81485cdf75e47ca`

## Prior evidence and current adapter state

`results\coder_adapter_checkpoint_1000_smoke.json` records a successful local checkpoint-1000 load and a Bridge smoke check: prefix cache crop was bit-identical; base bridge logits were finite and matched a fresh base prefill with max absolute delta `0.0`. This supports using checkpoint-1000 to test the mechanism. Training had reached step 1000 of a configured 2280 at that time; it was not a held-out coder-quality benchmark or a completed final training run. Do not tell the user that 2280 steps are required to test this comparison mechanism.

The current `.venv` reports CPU-only Torch (`2.14.1+cpu`) and Transformers `5.20.2`, but has no PEFT or Accelerate. The checkpoint itself is present and its parent model, rank, and target modules were checked directly.

## Current run status — check before restarting anything

- The exact `Qwen/Qwen3-0.6B` revision was not initially cached. A real greedy A/B call was started using a 128-token topological-sort prompt and seed 42.
- Temporary inference packages were installed under `D:\Research Paper\.adapter_compare_deps`; the shared `.venv` was not changed. The packages include Transformers 4.57.6, PEFT 0.21.2, Accelerate 1.12.0, Hub 0.36.2, Tokenizers 0.22.1, and `hf_xet`.
- The exact base download is going to the normal Hugging Face cache at `C:\Users\abhis\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B`. The large `model.safetensors` blob was about **1.19 GB of 1.503 GB** at the last check and was still increasing.
- The A/B process was still active under unified exec session **47948** at the last poll. Poll it if that session is available. If it is gone, check the cache and report before starting a new run; Hugging Face should resume the partial download.
- `results\coder_base_vs_adapter_checkpoint_1000.json` did **not** exist at the last check. The process writes it only after both generations complete.
- Attempting to bind a local HTTP server from this sandbox was blocked by the environment’s socket permissions. HTML and Python syntax checks passed, but browser-to-server interaction has not yet been verified here.

## Next actions

1. Check whether session 47948 is still running and whether the Hugging Face weight file completed. Do not assume the A/B comparison finished.
2. If it completed, inspect `results\coder_base_vs_adapter_checkpoint_1000.json`; confirm it contains both real outputs, settings, model revision, device, runtime versions, latency, and token counts. Summarize the two outputs without claiming correctness from one prompt.
3. If the run stopped, resume via the runner with the temporary dependency folder first on `PYTHONPATH`, preserving the same model ID/revision and adapter checkpoint. Avoid the cached `Qwen3-0.6B-Base` model because it is the wrong parent.
4. Verify the page by starting the local server outside the restricted socket sandbox if possible. For this workspace, the runner can use the temporary packages with PowerShell:

   ```powershell
   $env:PYTHONPATH = 'D:\Research Paper\.adapter_compare_deps'
   $env:HF_HOME = "$env:USERPROFILE\.cache\huggingface"
   .\.venv\Scripts\python.exe scripts\adapter_compare_server.py
   ```

   The intended executable is `D:\Research Paper\.venv\Scripts\python.exe`. Then open `http://127.0.0.1:8765`.

5. Fix the setup note to show the correct interpreter command for this machine. Do not install into or downgrade the shared `.venv` without deciding that is wanted; the temporary dependency folder currently avoids that change.

## Questions for the user if they affect the next step

- Should this studio run on this computer (currently CPU-only) or should it be adapted to Google Colab/T4? The current implementation is a local server and is not a Colab client.
- Should the comparison continue to default to checkpoint-1000 for the mechanism experiment, or should a later checkpoint be selected after training resumes?
- Should comparison dependencies stay isolated, or may they be installed into the shared `.venv`?

Do not ask these again if the user has already answered them elsewhere in the conversation. Preserve their request for an actual side-by-side comparison and do not replace real generations with mock outputs.

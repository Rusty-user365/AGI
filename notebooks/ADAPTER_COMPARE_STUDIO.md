# Local Base vs Coder Adapter Studio

This page performs real generation with the exact `Qwen/Qwen3-0.6B` checkpoint
used by the coder adapter. It runs the same chat-formatted prompt through the
adapter-disabled base first, then through `checkpoint-1000`. The model stays
resident between runs and the adapter is enabled only for the second pass.

## Start it on Windows

Double-click `D:\Research Paper\Start Adapter Lab.cmd`. It starts a background
runner and opens the studio in your browser. The latest real comparison loads
automatically. Enter a prompt and click **Run base vs adapter** for a new run.
You can copy both outputs or export the full JSON record.

On this machine, the launcher uses `.venv\Scripts\python.exe` with the isolated
`.adapter_compare_deps` packages, four CPU threads, and the already-downloaded
pinned model in the Hugging Face cache. It does not change the shared environment.
To start without opening a browser:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\Research Paper\scripts\start_adapter_lab.ps1" -NoBrowser
```

Open `http://127.0.0.1:8765`. The first comparison may take time while the base
checkpoint is loaded or downloaded. The current machine's smoke report used a
CPU-only PyTorch build; the server automatically selects CUDA when a CUDA-enabled
PyTorch install is available. Check the device shown in the page before
interpreting latency.

The default adapter is
`notebooks\coder_qwen3_0.6b_q_mlp_r16_v1___Coder Adapter\checkpoint-1000`.
To run in a terminal or compare another local checkpoint:

```powershell
$env:PYTHONPATH = 'D:\Research Paper\.adapter_compare_deps'
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
& 'D:\Research Paper\.venv\Scripts\python.exe' 'D:\Research Paper\scripts\adapter_compare_server.py' --adapter-path 'D:\path\to\checkpoint'
```

Use Ctrl+C for a terminal-started server. The background launcher's process ID
is recorded in `results\adapter_lab_server.pid`; its logs are in
`results\adapter_lab_server.log` and `results\adapter_lab_server_errors.log`.
The latest completed comparison is saved automatically in
`results\coder_base_vs_adapter_checkpoint_1000.json` and served by `/api/latest`.
The page exports each run's prompt,
generation settings, output text, model revision, hardware device, latency,
token throughput, and peak allocated VRAM (when running on CUDA) as JSON.

## What this comparison measures

- Same prompt, tokenizer/chat formatting, seed, and generation settings for
  both outputs.
- Sequential base and adapter generation using the same loaded backbone.
- Actual wall-clock generation latency, generated token count, token throughput,
  and CUDA peak allocation when available.

This is an interactive A/B check, not an automated code-correctness benchmark.
For reproducibility, start with greedy decoding (sampling unchecked), use the
same prompt and token limit, and save the JSON exported by the page. The CPU
smoke result is not a GPU/T4 latency estimate.

Generation timings exclude model loading. Base runs first, so a single UI run
is not a warmed, order-balanced performance benchmark. Outputs that hit the
token limit can be incomplete; increase the limit before judging code quality.

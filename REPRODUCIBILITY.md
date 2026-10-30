# Reproducing the experiments

This repository contains code, recorded results, trained router heads, arithmetic
feature artifacts, and the three adapter inference checkpoints used in the
reported coder, cyber, and UI experiments. It is **not a self-contained offline
reproduction package**: base-model weights, benchmark datasets, Python packages,
MiniLM embeddings, and GPU hardware are external. Different hardware and package
builds can change numerical output and latency; exact regeneration has not been
verified from a fresh clone. The saved results can be inspected without rerunning.

## Included adapters and parent model

All three adapter configurations use the chat-capable `Qwen/Qwen3-0.6B`, pinned
to commit `c1899de289a04d12100db370d81485cdf75e47ca`; these are LoRA deltas,
not standalone model weights. They target q_proj, gate_proj, up_proj and down_proj.
The arithmetic experiments instead use the distinct `Qwen/Qwen3-0.6B-Base`.

| Adapter used | In-repository directory | SHA-256 of adapter_model.safetensors |
| --- | --- | --- |
| Coder checkpoint 1000 | `notebooks/coder_qwen3_0.6b_q_mlp_r16_v1___Coder Adapter/checkpoint-1000/` | `cd84e5474cd4bb9667d47de85d0706fceefeec3feb2a506d1e95d3f4da629ba5` |
| Cyber checkpoint 2500, intermediate | `notebooks/cyber 25000/` | `8fb8b80ad3bc0edc5af32f981fde7ed9a32010ce9bf6eb91ae6ab6d52f22bf2a` |
| UI final adapter, step 464 | `notebooks/CGR_UI_Adapter_Training/ui_qwen3_0.6b_q_mlp_r16_v1/final_adapter/` | `156b01b89133e0aae8e1a89cea7e5efbeee34c3bf768c543358ca9e6d105274c` |

Each inference folder contains its adapter config, weights and tokenizer files;
run manifests are kept in the corresponding parent folder (in the cyber folder
itself). Optimizer state and other training checkpoints are not included, so
resuming the original full training run is not supported by this package.

The respective training datasets and revisions are in the run manifests. The
cyber dataset card and the UI dataset card identify Apache 2.0; the coder dataset
card does not clearly declare a license. The base model identifies Apache 2.0.
The project MIT license does not resolve upstream data or model rights: review
individual source terms and provenance before redistribution or commercial use.

- Base: https://huggingface.co/Qwen/Qwen3-0.6B
- Cyber data: https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset
- Coder data: https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca
- UI data: https://huggingface.co/datasets/iamdyeus/ui-instruct-4k

## External benchmark files

The published source datasets are not redistributed here. Download the exact
files below and verify SHA-256 before use; the CyberMetric URL tracks `main`, so
its expected hash is essential. The SecEval and SecQA links use pinned revisions.
Observe the sources' terms. SecQA declares CC BY-NC-SA 4.0; the CyberMetric and
SecEval source pages do not clearly declare an explicit dataset license.

| Local destination | Source | SHA-256 |
| --- | --- | --- |
| `data/benchmarks/cybermetric80/CyberMetric-80-v1.json` | https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-80-v1.json | `ffdda528c5f344871ad0a9ecef5c64ccf9964d68c3047ac0adaac520b2fc7dc6` |
| `data/benchmarks/seceval/questions.json` | https://huggingface.co/datasets/XuanwuAI/SecEval/resolve/205dab7b0888a06f4b53ca7d9c7093e1326683e1/questions.json | `194c3a511104be422a678675a71d44315d0c0392b040745155aeebc321e692e0` |
| `data/benchmarks/secqa/secqa_v2_test.csv` | https://huggingface.co/datasets/zefang-liu/secqa/resolve/d00a07484283be5602e2bae36dbefdaaf555a9fb/data/secqa_v2_test.csv | `b04f92a17b278e9765fa262103fa4127905fbae46d9eaf54f5444f986fc9c722` |

## Environment and commands

The recorded cyber runs used CPU fp32, four PyTorch threads, Transformers
4.57.6, PEFT 0.21.2 and Accelerate 1.12.0; the original environment reported
PyTorch 2.14.1. The current requirements file specifies some lower bounds rather
than a completely locked environment. The arithmetic GPU latency measurements
need the documented GPU, software and data and should not be expected to match
CPU timing. See the original findings and JSON records for each experiment.

Example Windows PowerShell setup (network access required for packages and base
weights; choose a compatible PyTorch build for your device):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r scripts\requirements-adapter-compare.txt
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca')"
.\scripts\start_adapter_lab.ps1
```

The launcher uses `%USERPROFILE%\.cache\huggingface`, enables offline mode and
looks for the pinned model there. The saved webpage comparison is available at
`http://127.0.0.1:8765/webpage-test`. Install additional Chroma and
sentence-transformers dependencies and download `sentence-transformers/all-MiniLM-L6-v2`
for the optional memory experiment. The experience-memory scripts expect the
MiniLM snapshot under `.experience_memory_models/all-MiniLM-L6-v2`.

Entry points include `experiments/cyber_router_crossdataset.py` and
`experiments/cyber_router_live.py`. These guard against overwriting saved output;
use a separate working copy or preserve and move output files before rerunning.
The live runner uses the saved heads and baseline records. Several scripts check
byte-exact hashes, which are preserved for the referenced files via
`.gitattributes`; exact matches can still fail on a different runtime. The
original arithmetic results and scripts are referenced by `PAPER.md` and the
vault findings. The PDF and LaTeX tables are presentation artifacts, not a new
run of the experiments.
## Building the paper PDF

`PAPER.md` is the source; `publication/tables.tex` provides separate LaTeX table
source. The checked-in PDF is rendered from Markdown with KaTeX and headless
Edge, rather than compiled from a complete LaTeX manuscript. To rebuild on
Windows, install Node.js and run `npm install --prefix .publication_deps katex
marked playwright`; set `PAPER_BROWSER` if Edge is not in its standard location,
then run `node publication/build_paper.cjs`. The script also accepts
`PAPER_NODE_MODULES` for an alternative directory containing marked and
playwright. Browser, Node, font and library version differences may alter layout.
The paper currently lists `Research Team` as the author placeholder and requires
actual author/affiliation and submission review before publication.

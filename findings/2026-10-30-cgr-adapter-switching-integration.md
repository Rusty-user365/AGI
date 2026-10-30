# CGR adapter switching: real checkpoint integration (2026-10-30)

**Scope:** local CPU fp32 integration on the chat-capable `Qwen/Qwen3-0.6B`,
revision `c1899de289a04d12100db370d81485cdf75e47ca`, with coder LoRA
`checkpoint-1000`. This is separate from the arithmetic paper's
`Qwen3-0.6B-Base`. These are disclosed smoke/integration tasks, not an unseen
coder-quality evaluation or a trained-router result.

The runtime now connects task metadata, `CommitmentDAG`, `Controller`, actual
PEFT switching, bounded task verifiers, absolute-boundary cache rollback, and
base-only Bridge replay. The local UI is `http://127.0.0.1:8765/switching`.
Setup and protocol: `notebooks/CGR_ADAPTER_SWITCHING.md`.

## Measured runs

| Run | Result | Artifact |
|---|---|---|
| Planned base/coder/base/coder/base | 4/5 nodes committed; two real coder generations; four inspected policy transitions; exact rollback checks; all four accepted-node Bridges passed. 31.13 s sequence time including diagnostics. | `results/cgr_adapter_switching/20261029T213116Z_df30837b.json` |
| Injected addition failure; verifier-triggered coder repair | The injected `return a - b` failed. Controller returned REV, restored the verified ancestor, and the real coder generated a passing addition function. 3/3 nodes committed, one coder call, two policy transitions; all Bridges passed. 11.02 s. | `results/cgr_adapter_switching/20261029T213212Z_4e3f62d0.json` |
| Base-first clamp | Base passed; no coder call was made. 3/3 nodes committed; all Bridges passed. 20.69 s. | `results/cgr_adapter_switching/20261029T213415Z_62762ea9.json` |
| Unknown skill | Abstained with no candidate generation or specialist call. The root base context is still initialized. | `results/cgr_adapter_switching/20261029T213416Z_d93994af.json` |

Run IDs use UTC; these runs occurred on October 30 in Asia/Calcutta.

The failed planned clamp is a real semantic failure, not a parser problem:
coder emitted `def clamp(value, lower, upper): return value`, passing only 2/5
cases. The base diagnostic emitted a correct clamp and passed all five. Planned
mode intentionally tested the selected specialist and rejected its candidate;
the passing baseline was a diagnostic, not an automatic replacement commit.
The coder's even-sum function passed 4/4 cases and was base-bridged before the
next base node. Consequently the planned sequence had base diagnostic passes
on 2/2 coder tasks and specialist passes on 1/2. No specialist quality gain is
established.

The failure-control baseline is explicitly injected, not a base generation. Its
successful repair establishes controller/adapter/Bridge integration, not an
improvement over naturally failing base code.

## Structural evidence

- Every entered policy inspects the actual enabled/disabled flags and active
  adapter name across **112 LoRA layers**. The recorded sequence is base,
  coder, base, coder, base.
- Uncommitted base-tail cache is cropped to the saved absolute task boundary;
  all retained K/V tensors must be bit-identical. The specialist works on a
  separate cache, and its K/V entries are discarded.
- Exact accepted token IDs, including the completion boundary, are prefetched
  with adapters disabled into a clean copy of the base prefix cache.
- Base Bridge logits are compared to a fresh base prefill of the same exact
  token IDs. The fp32 tolerance was fixed at **1e-4 before the runs**. All
  evaluated Bridges had finite logits, matching argmax, correct cache length,
  and matching four-token greedy base continuations. The largest absolute
  logit delta across the three non-abstention runs was **5.20e-5**.
- Resume starts from Bridge output logits; the final patch token is not fed a
  second time. Failed semantic or Bridge checks cannot commit a DAG node.
- The parent training manifest is checked against the loaded base revision.
  New runtime records also include adapter-weight/config and chat-template
  hashes, software versions, cache type, and per-event selection/assertion time.

The first planned/control artifacts precede the addition of config-hash and
cache-type fields; their outputs and results were retained without rewriting.

## Verification

`tests/test_adapter_switching.py` plus the existing controller logic tests:
**12 passed**. Coverage includes wrong-code rejection, truncation, forbidden
operations, verifier timeout, real tiny-Qwen cache rollback, deliberate K/V
corruption, Bridge logit equivalence, and continuation token accounting.

Headless Edge loaded the real saved sequence, rendered its rejected clamp and
accepted nodes, clicked the UI's Run button for unknown-skill abstention, and
selected the saved sequence again. Evidence:
`results/cgr_adapter_switching_ui_check.json`,
`results/cgr_adapter_switching_ui.png`.

## Limits

Only the coder adapter is present. Cyber/full-stack/UI adapters and a learned
Layer-20 router are not loaded. Routing and fault-node selection use declared
skills and the current node's verifier failure, not the arithmetic localizer
transferred to code. This is the metadata-driven operational path proposed in
`ADAPTER_ROUTING_PLAN.md`.

The specialist uses a base-produced verified prefix by deliberate design; these
tests do not establish equivalence to recomputing the whole prefix under
specialist weights. CPU fp32 is measured; fp16's stated tolerance remains
untested. Fresh-base references, prefix snapshots, base diagnostics and
continuation checks add overhead, so sequence timing is not a deployment
speedup claim. Verifier subprocess restrictions are an integration guard, not
a general hostile-code execution sandbox.

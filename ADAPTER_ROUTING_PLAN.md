# CGR Specialist Adapter Routing and Bridge Plan

**Status:** Proposed design. The routing runtime is not implemented or validated yet. This document reviews the two pasted runtime proposals and sets the integration plan for the trained adapter folders.

**2026-10-30 follow-up:** A one-specialist integration runtime now exists in
`cgr/control/adapter_runtime.py`, with the local UI at
`http://127.0.0.1:8765/switching`. Real checkpoint-1000 runs exercised metadata
routing, commitment/rejection, base/coder/base transitions, exact prefix-cache
rollback, base-only Bridge replay, and fresh-base logit/continuation checks.
The planned five-node sequence committed 4/5 nodes, correctly rejecting a wrong
coder clamp; the injected-failure repair control passed. This does not validate
all four specialists, a learned router, held-out quality, or deployment savings.
See `findings/2026-10-30-cgr-adapter-switching-integration.md` and
`notebooks/CGR_ADAPTER_SWITCHING.md`. The original review below describes the
design and earlier repository state.

## 1. Decision

Use a **single pinned base model** with one selected specialist adapter at a time. Let the DAG node's declared skill choose a specialist before generation. If a base-generated node fails its verifier, use the earliest localized failing node and that node's declared skill to select the repair adapter. The verifier decides whether a proposed patch can be committed; a latent probe may be studied as an early-warning or localization aid, but it does not commit output or independently authorize a repair.

Keep each adapter active for one specialist DAG node or one bounded repair attempt. When that node or patch is accepted, disable the adapter and rebuild the accepted text into a cache produced by the base model. Do not carry adapter-produced K/V entries into base continuation or into a different adapter's generation.

## 2. Current project state

- Training notebooks exist for coder, cyber, full-stack/API, and UI adapters. They declare `Qwen/Qwen3-0.6B` and target `q_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- The training notebooks resolve and record a base commit when each run starts. Since runs can start at different times, their manifests must be checked and aligned to one shared commit before they can participate in the same cache-sharing runtime.
- There are no trained LoRA adapter folders in the repository's `checkpoints/` tree yet. The Bridge notebook currently exercises a separate public coder adapter, not the four local training outputs.
- That public coder adapter targets `q_proj`, `k_proj`, `v_proj`, and `o_proj`; its Bridge test avoids reusing its generated cache by re-prefilling the patch under the base model. The locally trained Q/MLP-targeted adapters are a different artifact set and still need their own integration run.
- No Layer-20 routing probe checkpoint or training artifact is present in the repository. The pasted `LatentProbeClassifier` also constructs uninitialized linear layers, so its scores are random until trained and loaded.
- The current Bridge notebook pins Transformers 4.57.6 and verifies crop behavior, base prefill, logits, and continuation on five synthetic examples. It does not test DAG fault localization, adapter routing, actual cyber/full-stack/UI adapters, or task-level repair quality.

The measured localizer results in [the CGR allocation finding](findings/2026-10-29-cgr-as-tool-budget-allocator.md) and [the observer selection finding](findings/2026-10-27-observer-ranks-but-cannot-select.md) are limited to their reported arithmetic tasks. They motivate using a verifier/localizer carefully; they are not evidence that the proposed Layer-20 probe routes coding or security tasks correctly. The localizer finding also reports that its learned gate did not improve recovery on that evaluation, so no learned gate is treated as a proven component here.

## 3. Review of the pasted “corrected runtime”

| Proposal claim | Review | Plan correction |
|---|---|---|
| Crop by `past_key_values.get_seq_length() - len(patch_tokens)` to remove contaminated KV | This arithmetic may return to the cache length from before the adapter loop under the exact one-token-per-loop behavior shown. It does not identify the saved boundary before the earliest failed DAG step. The base `next_token` is used to condition the adapter patch, then discarded by the crop, so the emitted patch and its committed context differ. | Save the token/cache boundary at the last verified step before generation. Crop to that recorded absolute boundary with the API for the pinned Transformers version. Never infer a fault boundary from the number of output tokens. |
| Resume after bridge by setting `input_ids` to the final patch token | The bridge prefill has already placed every patch token in the cache. Feeding the final patch token again in the next forward duplicates it. | Resume from the logits returned by the base bridge prefill: select the next base token from those logits, then proceed with normal autoregression. |
| Check EOS using the pre-patch `next_token` after the adapter run | That token can be EOS even though the adapter emitted a patch, or non-EOS after the patch ended. | Check the actual committed patch and the next token predicted by the bridge/base continuation. |
| The Layer-20 Two-Tier Latent Probe catches the latent divergence at the root fault | The pasted class has random, untrained weights, a hard-coded threshold, and no checkpoint loading. Its code selects `out.hidden_states[-2]`, not an explicitly indexed layer 20. The local Qwen3-0.6B config has 28 layers and hidden size 1024. A latent score does not prove fault localization or domain routing. | Use DAG skill metadata and verifier/localizer evidence as the operational route. If testing a probe, specify the exact layer, train it against verifier-grounded labels, calibrate on train-only data, and measure task-held-out localization/routing and false triggers before allowing it to affect runtime behavior. |
| Freezing `k_proj` and `v_proj` makes cache states mathematically valid across adapter switches | It confirms those projection weights were not directly LoRA-modified. It does **not** make all adapter K/V values equal to base K/V: query or MLP changes can alter activations that later layers use to produce K/V. | Preserve only the base cache for the verified prefix. Discard adapter-generated cache entries. Re-prefill accepted patch tokens under base weights before resuming base generation. A config/parameter check is a compatibility guard, not a cache-equivalence proof. |
| Three specialist domains are enough | The current training set includes coder, cyber, full-stack/API, and UI. | Include all four specialist IDs, plus `base` and `unknown`/abstain. Never silently map unknown tasks to coder. |
| 32 tokens and token ID 198 are suitable universal patch limits/stops | A SQL/schema or UI patch may need more than 32 tokens; a newline can occur inside valid code and is not a reliable patch boundary. | Set per-skill budgets from held-out task lengths and an explicit hard cap. Stop on EOS, a known contract boundary, or a verifier-controlled completion condition. Treat truncation as rejection. |

### Cache API version rule

Transformers 4.57.6 defines `DynamicCache.crop(max_length)` as cropping to an **absolute target length**; that is the version pinned in the current Bridge notebook. Current Transformers cache documentation describes cropping by a signed number of tokens to remove, using a negative number to roll back. The runtime must pin its library versions and have one tested crop wrapper; do not copy the positive absolute-length call into a newer environment without checking that version's API. See the [4.57.6 cache implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/cache_utils.py) and [current cache documentation](https://huggingface.co/docs/transformers/kv_cache).

## 4. Adapter folder and compatibility contract

Treat each `final_adapter/` folder as an inference artifact, not as a self-contained model. Before loading, require and validate:

| Field | Required check |
|---|---|
| `adapter_config.json`, adapter weights | Both exist, load cleanly, and match the manifest; calculate file hashes for experiment records. |
| Base model ID **and immutable commit SHA** | Must match the base already in memory. Matching the repository name alone is insufficient. |
| Tokenizer ID/revision and chat-template hash | Must match the runtime tokenizer and expected message format. |
| PEFT type, rank/alpha, target modules, dtype | Compare requested and observed modules with actual adapter configuration and weights. No unexpected modules. |
| Training run and evaluation summary | Record dataset revisions, split IDs, final/best checkpoint, training metrics, and task-specific held-out results. |
| Runtime compatibility | Pin and record Python, PyTorch, Transformers, PEFT, and cache type/version used for integration. |

Before combining the four folders, choose one exact base commit and ensure every training manifest names that commit. The existing notebooks resolve the revision at run start; if those revisions differ, retrain or explicitly verify compatibility on the common commit. Do not load the adapters onto an arbitrary `Qwen3-0.6B` snapshot. Different target-module sets also require a multi-adapter load/switch smoke test with the pinned PEFT version. PEFT supports named adapters, `set_adapter`, adapter disabling, and active-adapter inspection, but the runtime must verify the installed version and actual target layout. See [Transformers PEFT adapter management](https://huggingface.co/docs/transformers/peft) and the [PEFT model reference](https://huggingface.co/docs/peft/package_reference/peft_model).

## 5. Routing policy: which adapter and when

Each DAG node carries a `required_skill` chosen when the plan is built:

| Skill ID | Adapter | Example node work |
|---|---|---|
| `coder` | coder | Implement or repair general-purpose code; compile/test the result. |
| `cyber_defensive` | cyber | Defensive analysis, secure remediation, threat explanation, or security review within the system's security policy. |
| `fullstack_api_backend_schema` | full-stack/API | SQL, REST endpoints, OpenAPI, JSON schemas, backend contracts, structured outputs. |
| `ui` | UI | UI components, styling, accessibility, and client-side behavior. |
| `base` | no adapter | Planning, ordinary prose, coordination, or tasks with no demonstrated specialist benefit. |
| `unknown` | no automatic specialist | Ambiguous or unsupported request; keep base active, decompose/clarify, or use a separately validated classifier. |

Routing order:

1. **Planned specialist node:** Use the node's explicit `required_skill`. Activate that adapter only if it passes the compatibility and evaluation gates below. Otherwise keep the base model or report the unavailable specialist.
2. **Base node fails:** A deterministic task verifier (tests, parser/schema validation, SQL execution in a sandbox, or task-specific rubric) marks the node failed. The localizer identifies the earliest failing DAG node. Route using that node's `required_skill`.
3. **No reliable node or skill:** Do not infer a specialist from the last generated words, raw entropy, or an unvalidated hidden-state head. Keep the base path, ask the planner to split/relabel the node, or abstain.
4. **Optional probe experiment:** A probe may propose an anomaly position or skill distribution. It cannot override explicit DAG metadata, safety policy, or verifier results until a held-out evaluation shows that it improves the actual routing decision over metadata/base-only and cheap symbolic checks.

Cyber policy checks are independent of adapter routing. Selecting the cyber adapter does not itself make an unsafe request safe or establish that its answer is secure.

## 6. Activation duration and safe switch protocol

**Duration:** One specialist node or one repair patch attempt. Use per-specialist token/compute budgets based on the held-out task-length distribution (record p50/p95 and truncation rate), plus a hard upper bound. End at the declared node/contract boundary or EOS, then run the node verifier. A repair is accepted only when its verifier passes. If it hits the cap or fails verification, discard it and restore the last committed DAG/cache state. Bound retries and total specialist calls per task; after the bound, fall back to base/tool/abstention.

**Switch mechanism:** Load validated adapters under stable names at runtime startup if measured VRAM allows. At a node boundary, call the version-appropriate adapter selection API, then assert the selected active adapter. For base execution, disable adapters and assert none are active. PEFT's normal named-adapter switching leaves inactive adapters resident, so it saves repeated loads but does not save their VRAM; if that does not fit, evaluate on-demand load/offload or compatible LoRA hotswapping and measure the added latency. Do not claim a fixed switch time or memory footprint before measuring. Serialize requests through one mutable model/cache unless the runtime isolates model/cache state per request.

**Bridge sequence for a repair:**

1. Keep the verified token IDs and a cache checkpoint at the last committed base step. Record the absolute token boundary explicitly; do not derive it from output length or decoded text.
2. Roll back the base cache to that boundary if any unverified base tail was generated. Assert the retained K/V tensors equal the saved verified-prefix K/V tensors. For sliding-window caches, verify rollback support before relying on it.
3. Clone or preserve a clean base-prefix cache for the bridge. Generate the specialist candidate using the specialist adapter and a separate working cache. Keep repair instructions transient; do not insert them into the committed conversation unless they are part of the actual output. Track exact generated token IDs.
4. Run the node's verifier on the candidate patch in context. A failed candidate is discarded; it never becomes part of the verified prefix.
5. For an accepted patch, discard the adapter working cache, disable the adapter, and prefill the exact committed patch token IDs into the preserved base-prefix cache with generation disabled. If text must be decoded/re-tokenized, test and log the resulting token sequence; prefer preserving the exact IDs that will be committed.
6. Verify base-prefill logits against a fresh full-base prefill of the same prefix and patch token IDs. Resume by sampling/greedily selecting the next token from the bridge output logits. Do not feed the final patch token again; it is already cached.
7. At the next DAG node, select its adapter from metadata and repeat. If the node uses a different adapter, start from the latest accepted base cache boundary, not the previous adapter's cache.

For a planned specialist node, the same rule applies: the selected adapter may produce the node output, but once accepted, its committed token sequence is base-prefilled before the next base or different-adapter node.

## 7. How success is measured

Use separate gates. A cache test passing does not imply an adapter is useful, and a lower training loss does not imply the Bridge works.

| Gate | Measurements | Pass condition |
|---|---|---|
| Artifact compatibility | Base ID/SHA, tokenizer/template, adapter files/hashes, target modules, load errors, active-adapter identity | All required metadata matches; every folder loads and selects as expected on the same pinned base/runtime. |
| Adapter quality | Paired held-out tasks: base-only versus the matching adapter, with domain-specific verifiers | Report pass-rate difference with confidence intervals and failure categories. Keep an adapter only if it improves its target tasks without unacceptable cross-domain regression. Do not select by train loss alone. |
| Domain checks | Coder: parse/build/unit tests. Cyber: defensive task correctness, policy compliance, unsafe-assistance rate. Full-stack: schema/OpenAPI validation, SQL execution in a sandbox, endpoint/contract tests. UI: typecheck/build, component behavior, accessibility checks. | The relevant verifier passes. Security and correctness checks remain independent. |
| Router quality | Correct skill selection, abstention, false activation, missed specialist opportunities, and repair localization on held-out DAGs | Compare metadata routing, base-only, random routing, and any learned router on the same paired tasks. A probe must improve decision-level utility, not only AUC or representation separability. |
| Bridge correctness | Exact prefix K/V equality after crop; finite logits; max absolute logit difference, argmax agreement, and continuation from bridge cache versus fresh base prefill | Establish dtype/version-specific tolerances in advance; require all structural invariants and report per-case failures. Cosine similarity alone is insufficient. |
| End-to-end benefit | End-task verifier pass rate, accepted patch rate, regressions to verified nodes, false activations, total tokens/calls, switch/patch/bridge latency p50/p95, and peak VRAM | Compare paired held-out tasks against base-only and no-bridge controls. Publish all tradeoffs; do not promise a VRAM ceiling until measured. |

Include an oracle-route arm for diagnosis, but do not report it as deployable performance. Keep all splits grouped by source task to prevent paraphrases or chain variants leaking between train and test. Pre-register any deployment pass thresholds before inspecting final holdout results.

## 8. Implementation sequence

1. **Freeze shared base/runtime:** Choose one Qwen3-0.6B commit SHA and tokenizer/template version. Update run configuration so all adapter training manifests use it.
2. **Finish and inspect artifacts:** Train/save each specialist independently. Check checkpoint resume files, final adapter files, manifests, telemetry, and task-held-out quality. The final adapter folder is the deployable unit.
3. **Build compatibility loader:** Validate metadata and hashes, load the four adapters on the shared base, test named `set_adapter`/disable/active-state behavior and target-module compatibility, and record per-adapter VRAM.
4. **Build a deterministic cache harness:** Test explicit boundary bookkeeping, crop rollback, separate working/bridge caches, exact token accounting, base-prefill logit equivalence, and correct resume token behavior on synthetic chains. Pin Transformers and PEFT versions.
5. **Add DAG routing:** Use `required_skill` metadata and verifier/localizer outputs. Log route reason, failed node, adapter name, cache boundary, patch IDs, verifier verdict, retry count, latency, and peak memory.
6. **Evaluate specialists and routing:** Run paired held-out domain tasks against base-only, random-adapter, metadata-route, and oracle-route controls. Only then consider an online latent probe, with its own trained artifact and evaluation report.
7. **Promote only passing adapters:** Mark each folder `candidate`, `validated`, or `disabled` in a registry. The router only selects `validated` adapters whose compatibility manifest matches the loaded base.

## 9. Runtime log record

Persist one JSONL record per DAG node/repair attempt with: run/task/node IDs; base ID and SHA; tokenizer/template hash; adapter name and folder/hash; route source and reason; verifier and localizer result; verified prefix token length; generated and committed token counts; patch/bridge/base latency; peak VRAM; cache rollback and logit-equivalence metrics; acceptance/rejection reason; retry/fallback action; and software versions. Store prompt/task identifiers or redacted hashes when the raw task contains sensitive content.

This plan intentionally leaves learned Layer-20 routing as an experiment. The first operational router is the DAG's explicit skill label plus verifier/localizer evidence, because those provide an auditable reason to choose a specialist and a concrete criterion for keeping its output.

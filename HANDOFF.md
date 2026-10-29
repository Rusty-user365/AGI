# Handoff — snapshot `v1.1-cgr-path-a-vault-verified-and-level2-ready`

**Date:** 2026-10-28 late night  
**Branch:** `main` (tracked at `origin/main` commit `26f08ef`)  
**Active Working Context:**  
The empirical investigation into step localization, surgical KV rewinds, and out-of-sample generalization is **complete, verified, and locked**. Every finding is backed by deterministic scripts, raw JSON metrics, Wilson 95% CIs, and exact paired hypothesis tests.

---

### Current state (2026-10-29: code pilot stopped, repair loop is a null)

The HumanEval line was built, audited, and stopped at Stage 1 (N=24) under its
pre-registered falsifier. The finding is that the treatment is negative, not
that the sample is small.

| Question | Answer | Evidence |
|---|---|---|
| Does hinted localized repair beat `best_of_4` at matched tokens? | **No.** 16/24 vs 16/24, 2W/2L, p=1.00. Falsifier fired. | `results/code_pilot_stage1.json` |
| Was the apparent 0.518x token win real? | **No.** It was early stopping. A sequential baseline on the same samples costs 0.369x of `best_of_4` and 0.71x of `cgr`. | `results/code_pilot_decompose.json` |
| Does the treatment beat a plain retry loop? | **No, it loses.** `greedy_seq` (greedy, then unhinted samples) 17/24 at 11,930 tok; `cgr` 16/24 at 13,741 tok. | same |
| Where does the cost come from? | Structural: 30-60 extra prompt tokens per hinted attempt, and longer generations (~190 vs ~110 tokens), with no accuracy gain. | same |
| Is the verifier trustworthy? | **Yes.** Stub bodies fail 164/164; all tests call `check()`; localization frame math and reassembly verified. | `experiments/code_audit.py` |

Finding: `findings/2026-10-29-code-pilot-null.md`.
Harness: `experiments/code_loop.py`, `code_bench.py`, `code_audit.py`,
`code_gate.py`, `code_decompose.py`.

**Protocol deviation, disclosed:** the mechanical futility gate did not fire
(2W/2L). The stop is based on the falsifier plus the post-hoc decomposition,
which uses only stored samples and re-runs the verifier.

**Consequence for the architecture:** localization is not the binding
constraint on code yet. Finding the right line does not help if the 0.6B model
cannot write a correct replacement for it. Any future code claim must use
`greedy_seq` as the baseline, not `best_of_4`.

---

### Executive Status Summary (2026-10-28)

| Component | State / Result | Evidence / Artifact |
|---|---|---|
| **Step-4 Tail Bottleneck** | **In-sample only.** Two-Tier ($P_{trf}(4) \ge 0.35 \to 4$, else Linear) lifts Step-4 recall **0.188 to 0.812** and Top-1 to **0.718** — but on the frozen vault it degrades to **0.625 / 0.410**, below Transformer alone (0.677 / 0.500) and the softmax ensemble (0.680 / 0.526). The threshold did not generalize. | `results/path_a_loop_h5.json`, `results/verify_vault_h5.json` |
| **Autonomous Loop Recovery** | **Resolved on the vault.** **22.1%** (73/331) vs full rewrite **13.9%** (46/331), 43W/16L, p=5.84e-4, **1.59x**. The N=110 figure is 29.1% (32/110); the 2.29x sometimes quoted uses a low-variance rewrite draw and is not the headline. | `results/verify_vault_h5.json` |
| **Accept-if-Cleaner Gate** | **Implemented; no measurable recovery effect.** Rejected 23/110 and 81/331 edits where rank-2 degraded a confident step ($lp > -0.25$); those rows contained **0 recoveries in both runs**, and gated == ungated Two-Tier exactly (32/110, same tokens/time). It is a conservative filter: precision among accepted rows is 32/87 = 36.8%, and it skips compute. Fallback-to-top-2 probe yielded 0/23 recoveries. | `results/path_a_loop_h5.json`, `results/verify_vault_h5.json` |
| **Physical KV-Cache Slicing** | **Resolved, two numbers.** Generation slice: 4,498.4 ms to 173.7 ms (**25.9x**, -79.4% tokens). Full loop incl. observer + gate: **2,244 ms vs 4,498.4 ms (2.0x)**; on the vault **1,934.7 vs 3,281.9 ms (1.70x)**. The 25.9x/27.2x figures are generation-slice only. | `results/path_a_loop_h5.json`, `results/verify_vault_h5.json` |
| **Test 1: Bit-Identical Assertion** | **Pass (100%).** Verified on 20 chains: $\max \|\Delta\text{logit}\| = 4.86 \times 10^{-5} < 10^{-4}$; all argmax tokens bit-identical. | `scratch/check_test1_bit_identical.py` |
| **Test 2: Blind Vault Replication** | **Pass (p=5.84e-4).** Frozen CGR evaluated on unseen seed (`seed=20261028`, $N=331$ broken chains): **22.1%** (73/331) vs full rewrite **13.9%** (46/331). | `results/verify_vault_h5.json` |
| **Permutation Control (`random_t`)** | **Decisive (p=7.64e-11).** Uniform random-step rewind collapses to **6.9%** (23/331); CGR wins 57W / 7L, proving localization causally drives repair. | same |
| **Test 3: Long-Horizon ($H=10$)** | **Ceiling Established.** Localizer achieves **52.0% Top-1** ($5.2\times$ over chance); rollout collapses to 0.0% due to compound errors, establishing horizon boundary $H^* \approx 6$–$8$. | `results/verify_h10_scaling.json` |

---

### Superseded: Level-2 Hybrid Specialist Ablation

> **Superseded on 2026-10-28.** The actuator ladder closed this branch:
> `span_digitset` was bit-identical to `rank2` (0/110 disagreements), and no
> deployable branch operator beat rank-2. `oracle_value` (70/110) opened the
> value branch, and the call-allocation curve then showed localization is the
> binding constraint. See
> `findings/2026-10-28-level2-actuator-ladder.md` and
> `findings/2026-10-29-call-allocation-curve.md`. The text below is kept only
> as the record of what was planned.

- **Objective:** Evaluate bidirectional span infilling on the **78 failed holdout chains** from `results/path_a_loop_h5.json` where Level-3's single-token rank-2 exclusion fell short.
- **Script Location:** `experiments/level2_hybrid_test.py`
- **Hardware Protocol (GTX 1050 4GB VRAM Safety):**
  - Level-3 AR (`Qwen3-0.6B-Base`) runs at ~1.2 GB VRAM.
  - To load the Level-2 specialist (`DiffuGPT-S/` bidirectional diffusion or `checkpoints/hf/qwen2.5-1.5b-instruct` infiller proxy at ~2.2 GB), the script **must unload the primary model** (`del model; torch.cuda.empty_cache()`) and verify VRAM drops below 200 MB before loading the specialist.
- **Success Criteria:**
  - 🟢 **Total Success:** Rescues 25–35 of the 78 chains, pushing combined recovery to $>50\%$.
  - 🟡 **Partial Success:** Rescues 5-10 chains; paper frames Level-3 Edge CGR (22.1% vault recovery at 1.7x loop speedup) as primary contribution and Level-2 as theoretical upper-bound ablation.
  - 🔴 **Failure:** If second model causes OOM or syntax formatting hallucinations, drop Level-2 from claims and keep locked Level-3 paper.

---

## Current state (2026-10-27, after the routing gate)

The repair line is closed. On `test` the CGR gate condition is a null against an
in-distribution control (+0.005, quality -0.174 SIG worse), and the repair
corpus cannot test routing at all: parseability and correctness agree 480/480,
so a parser achieves perfect discrimination.

A replacement benchmark was built and the observer was evaluated on it.

| Question | Answer | Evidence |
|---|---|---|
| Can a head rank correct repairs above incorrect ones? | **Yes**, AUC 0.682 [0.573, 0.782] on held-out tasks, permutation control 0.496 | `results/select_eval_arith_s2.json` |
| Does that ranking improve selection over free baselines? | **No.** Head 0.560 vs max-logprob 0.556, 12 wins / 11 losses, p=1.000 | same |
| Is the benchmark property present? | **Yes.** 500 tasks, 100% well-formed, 1096 wrong-but-well-formed holdout candidates | `results/arith_bench_s2.pt` |
| Is an instruct model a better generator? | **No.** Qwen2.5-1.5B-Instruct is 100% well-formed but p1 0.03-0.22, so pass@k tops out at 0.20-0.50 | `findings/2026-10-27-instruct-model-fluent-but-wrong.md` |

**`DAG_Scratchpad` stays unwritten.** The literal gate (AUC > 0.5) is met; the
justification (beat the model's own confidence) is not. A loop reranking by this
observer is indistinguishable from max-logprob, which is free.

New instruments: `experiments/arith_bench.py` (corpus builder, resumable),
`experiments/select_eval.py` (head vs greedy/random/majority/logprob/oracle,
paired sign tests, permutation control), `experiments/wordbench.py`
(notebook-and-pen family). Models on disk: `checkpoints/hf/dual-lm-470m`,
`checkpoints/hf/qwen3-0.6b-base`, `checkpoints/hf/qwen2.5-1.5b-instruct`.

---

## Current state (2026-10-27, late: the localization instrument)

The next question was localization, not selection: can anything point at the
step where the reasoning broke? That needs per-step ground truth, which no
earlier corpus had.

**The box world is closed.** Neither available model can apply a *single*
1-ball move: Qwen3-0.6B-Base 22.5% (2 boxes) and 2.5% (3-4 boxes),
Qwen2.5-1.5B-Instruct 2.5% raw / 15% chat, 40 trials each, all 100% parseable.
With step 1 wrong, `first_error` is 0 everywhere and localization is undefined
rather than low. Evidence:
`findings/2026-10-27-state-tracking-below-floor.md`.

| Question | Answer | Evidence |
|---|---|---|
| Does a pooled observer fail for lack of position? | **Structurally yes**; never tested here, and the test is now possible | below |
| Can the 0.6B track box state at all? | **No**, 22.5% on one move | `scratch/probe_onestep.py` |
| Is there a task with per-step labels it *can* do? | **Yes**, arithmetic chains: p1 0.544 on 2 steps | `results/arith_bench_s2.pt` |

New corpus `experiments/arith_steps.py`: the model emits the value of each
assignment in order (`x -> 5`, `y ->`, ...), self-conditioned, so an error
propagates exactly as in a DAG run and `first_error` is a real divergence label.
Per step it stores `value`, `logprob`, and the mean hidden state of the emitted
number, giving the `[H, d]` un-pooled input a localizer needs. Admission is "the
first line is a bare integer". Generator cross-checked against
`arith_pilot.make_problem`: 0 mismatches in 800 chains, 0 in 300 independently
re-derived intermediate values.

`experiments/localizer.py` compares three arms on the same held-out candidates:
`linear` (per-step, no cross-step information), `transformer` (one encoder layer
over the H rows), and both against **chance, the label prior, always-step-0, and
argmin per-step logprob**. The transformer must beat argmin-logprob, not chance,
or it repeats the selection null. (Superseded: argmin-logprob turned out to be a
constant predictor, so the gate was raised to a position+confidence model. See
the 2026-10-28 section below.)

---

## Current state (2026-10-28: the localizer answers; attention is a null)

The step-level observer question is answered on `arith_steps` h5 (120 train / 50
holdout tasks, 279 / **110** broken complete candidates). The gate was raised
before the run: a hidden-state reader must beat **`poslogprob`** (per-position
bias + per-position logprob slope, 10 parameters, no hidden state), because
position is linearly readable from the state and error labels are not uniform
over steps.

| Question | Answer | Evidence |
|---|---|---|
| Do un-pooled per-step states localize the broken step? | **Yes.** linear+SVD32 0.611 top-1, 0.506 balanced, 114W/23L vs poslogprob | `results/localizer_arith_steps_h5_linear_svd32.json` |
| Is it just the step-index prior? | **No.** Demeaning the per-step mean costs 0.022 and still wins 133W/54L | `..._linear_svd32_demeanpos.json` |
| Does cross-step attention help on average? | **No.** Paired 80W/68L p=0.366; majority vote 0.655 vs 0.627, CI [-0.118, +0.064] | `results/localizer_pair_arith_steps_h5.json` |
| Is the transformer capacity-starved? | **No.** Full width (0.600) >= SVD32 (0.576), so the 200/80 rebuild is unnecessary | `..._transformer.json` |
| Does confidence alone localize? | **No.** poslogprob 0.445 is prior+slope; non-modal 0.141 vs 0.25 chance | same |
| Does `argmin logprob` localize? | **No.** It predicts step 4 in 93/110 cases; balanced 0.293 | same |

Where the best arm fails: per-step recall is 0.92 / 0.74 / 0.77 / **0.19** for
steps 1-4, so a rewind loop built on it cuts to the right place early and to
step 1 late. The transformer's profile is the mirror image on the tail (0.82 /
0.57 / 0.69 / **0.56**): it triples late-step recall while losing early.

Finding: `findings/2026-10-28-localizer-ranks-steps-but-attention-does-not-help.md`.

---

## Current state (2026-10-28 late: Path A — two-tier localizer fixes step-4 in-sample; KV cache sliced)

Path A executed the three fixes proposed above on the 110 holdout chains of `results/arith_steps_h5.pt` with `Qwen3-0.6B-Base`:

| Question | Answer | Evidence |
|---|---|---|
| Can an ensemble cure the step-4 localization collapse? | **In-sample only.** Two-Tier ($P_{trf}(4) \ge 0.35 \to 4$, else Linear) lifts step-4 recall **0.188 to 0.812** and Top-1 to **0.718** — on the split used to pick the threshold. On the frozen vault it falls to **0.625 / 0.410**, below Transformer alone (0.677 / 0.500) and the ensemble (0.680 / 0.526). | `results/path_a_loop_h5.json`, `results/verify_vault_h5.json` |
| Does improved localization translate to higher loop recovery? | **Yes, on the vault.** 22.1% (73/331) vs rewrite 13.9% (46/331), 43W/16L, p=5.84e-4, **1.59x**. The N=110 run is 29.1% (32/110); 2.29x uses a low-variance rewrite draw. | `results/verify_vault_h5.json` |
| Does the "Accept-if-Cleaner" gate protect prefixes? | **No measurable effect on recovery.** Rejected 23 edits where rank-2 degraded a confident step ($lp > -0.25$); the rejected rows contained 0 recoveries in both the N=110 and N=331 runs, and gated == ungated exactly (32/110, same tokens/time). Its value is precision (32/87 = 36.8% among accepted) and skipped compute. | same |
| Does KV-cache slicing (`DynamicCache.crop`) substantiate the systems claim? | **Yes, at the slice level.** Generation: 581.1 to 119.7 tokens (-79.4%), 4,498 ms to 173.7 ms (**25.9x**). Full loop: **2,244 ms (2.0x)**; vault **1,934.7 vs 3,281.9 ms (1.70x)**. Observer inference dominates the residual. | same |

Finding: `findings/2026-10-28-path-a-ensemble-kv-slicing.md`. Script: `experiments/path_a_loop.py`.

---

## Current state (2026-10-28 night: Generalization Vault Audit & Long-Horizon Boundary Complete)

A strict 5-rule Out-of-Sample Stress & Generalization Audit was executed to verify that Level-3 CGR is not overfit to the original $N=110$ holdout set:

| Question | Answer | Evidence |
|---|---|---|
| Does KV-cache slicing introduce numerical drift or token divergence? | **No.** Verified across 20 chains: $\max |\Delta\text{logit}| \le 4.86 \times 10^{-5}$, 100% bit-identical argmax tokens. | `scratch/check_test1_bit_identical.py` |
| Does frozen CGR replicate on an unseen blind vault split ($N=331$)? | **Yes.** CGR achieves **22.1%** (73/331), significantly beating `full_rewrite` (13.9%, 46/331, $43\text{W}/16\text{L}, p = 5.84 \times 10^{-4}$). | `results/verify_vault_h5.json` |
| Does CGR defeat the random-step permutation control? | **Yes.** Slicing to a random step (`random_t_rank2`) collapses to **6.9%** (23/331); CGR wins $57\text{W}/7\text{L}$ (**$p = 7.64 \times 10^{-11}$**), proving localization causally drives recovery. | same |
| Does the KV-cache systems speedup replicate out-of-sample? | **Generation only.** Per-chain generation latency 3,281.9 to 120.7 ms (**27.2x**, -79.3% tokens); the loop is 1,934.7 vs 3,281.9 ms (**1.70x**). | same |
| Does localization generalize to deeper reasoning chains ($H=10$)? | **Yes.** Linear majority probe achieves **52.0% Top-1 localization accuracy** on $N=75$ broken chains ($5.2\times$ over chance $0.100$). | `results/verify_h10_scaling.json` |
| What happens to autonomous recovery at $H=10$? | **Collapses to 0.0% across all arms.** Establishes the unassisted autoregressive capacity ceiling ($H^* \approx 6$–$8$) where rollout error compounding ($(0.45)^6 < 0.01$) demands external tools. Token reduction remains **90.0%** (165.0 vs 1,656.2 tokens). | same |

Finding: `findings/2026-10-28-out-of-sample-and-h10-verification.md`. Harnesses: `experiments/eval_vault.py`, `experiments/eval_h10.py`.

---

## 0. Active branch: DiffuGPT backbone (uncommitted on top of `5345a1e`)

The v0.3 snapshot below is the canvas result. Work since then swaps in
DiffuGPT-S and is **not committed or tagged**. Do not read the v0.3 repair
numbers as DiffuGPT numbers.

| Area | State | Evidence |
|---|---|---|
| DiffuGPT-S backbone + LoRA/plastic integration | working; 64 tests pass | `cgr/model/diffugpt.py`, `tests/test_diffugpt.py` |
| C1 on the DiffuGPT CGR checkpoint | **pass**, valid instrument | `results/exp6_structural_guard_diffugpt.json` |
| Null property on the DiffuGPT CGR checkpoint | **pass**, n=20 | `results/exp1_invariants_diffugpt.json` |
| Editor gate, original 5k checkpoints | **below floor**: CGR 7/40 (0.175), LoRA-only 5/40 (0.125) | `results/check_backbone_*_5k.txt` |
| Diagnosis | objective/evaluation mismatch: whole-block teacher-forced top-1 0.292 vs 0.915 random-mask | `results/diagnose_diffugpt_5k_e10_cgr_test.json` |
| Mixed-mask fine-tune (CGR + LoRA-only) | **running / pending**; 4A+1B from the existing checkpoints | `results/blockft_manifest.json`, `results/train_cgr_blockft.log` |
| Gate after fine-tune | **not yet measured** | — |
| exp8 CGR vs LoRA-only at 5k | **not run**; gated on the 0.20 floor | `experiments/exp8_architecture_ablation.py` |
| Compute/capacity claim | CGR 8,149,661 trainable vs 442,368 (18.4x); 5h23m vs 1h57m | `checkpoints/*/train_summary.json` |

Reproduction for this branch:

```powershell
python scripts/export_corpus.py --preset diffugpt --out data/processed/diffugpt_train.jsonl
powershell -File scripts/run_block_retrain.ps1
powershell -File scripts/eval_blockft.ps1
python experiments/exp8_architecture_ablation.py --split test --n 200
```

The comparison is **CGR system vs LoRA-only task adaptation**, not matched
compute or matched capacity. The gate and a DAG do not participate in either
repair path, so exp8 cannot validate them.

---

## 1. What this snapshot contains

| Area | State | Evidence |
|---|---|---|
| C1: verified-wrong cannot update plastic weights | **Pass, both seeds, valid instrument** | `results/exp6_structural_guard.json`, `_s1.json` |
| C1: verified-wrong cannot write factual memory | **Pass, both seeds** | same |
| C1: rejection is still supervised (not masked away) | **Pass, both seeds** (gate grad 4.894) | same |
| C1 controls | unguarded leaks on A/B/C; verified-right updates | same |
| Metrics/evaluation contracts | 36 tests, negative controls | `tests/`, `pytest -q` |
| Oracle repair taxonomy | one-block local 1.00; cross-block 0.26; global/repeated 1.00; anchor 1.00 | `results/exp4_repair_taxonomy*.json` |
| Counting ontology | reject / reject / accept / defer; bare "five" → REFUSE (4/4 both seeds) | `results/exp2_scope_and_pressure*.json` §E |
| Gate separates defect types | REP 1.00 local, REV 1.00 cross-block; ECE ≈ 0.004 | `results/exp5_gate_triage*.json` |
| Learned editor repairs defects | **Not established** — no method beats no-op on the edited blocks; specificity 0.000 at every budget | `results/exp4_repair_taxonomy*.json`, `results/exp7_editor_floor.json` |
| Triage beats random | **Not established** — p = 0.17, ceiling 1.6 marks | `results/exp5_gate_triage*.json` |
| C2: guard improves held-out resistance | **Unresolved — underpowered, not failed** | `results/exp2_scope_and_pressure*.json` §B |
| Stage B/C discovery beyond imitation | **Not run** | — |

---

## 2. The three corrections that matter most

### 2.1 C1 was testable all along

An earlier assessment said the safety claim was "uninterpretable" because the
editor was at the floor. That conflated two claims. C1 is a property of the
update pipeline:

> a verified-wrong example cannot move the plastic substrate and cannot create
> a factual memory entry, while the gate still receives a supervised rejection
> gradient.

`exp6` tests it with all parameters trainable (a frozen parameter cannot leak),
an unguarded control that must fail, and a verified-right control that must
update. Both seeds pass. This does not depend on generation ability.

### 2.2 The "local repair 0.74" was a metric artifact

Document-level `c_edit` and `quality` average over *all* editable blocks. On a
local defect the untouched siblings already score 1.0 and (k−1)/k, so a
degenerate edit inherits their credit. The model's apparent local
`c_edit = 0.739 / quality = 0.638` was entirely sibling credit; the edited
block was garbage (`reason 1 outside work early work open open ...`).

Fixed by adding `c_target` / `q_target` restricted to the blocks the task
actually edits, plus a no-op baseline on those blocks. With the fix, every
method is at or below no-op:

| Condition | c_edit (doc) | c_target (edited) | no-op c_target |
|---|---|---|---|
| local / one_block | 0.74 | **0.00** | 1.00 |
| cross-block / one_block | 0.00 | 0.00 | 0.00 |

### 2.3 Byte-identical restore is not an ability gate

`exp4` used `informative = infill_exact >= 0.2`, where `infill_exact` is
byte-identical copy. A masked-diffusion infill regenerates the span, so that
number is 0% at *every* budget, and the gate could never open. The operative
precondition is on-thesis output. `exp4` now gates on `infill_on_thesis` and
reports copy fidelity separately.

`exp7` splits the resulting floor into two findings:

| Budget | Steps | Clean-block on-thesis | Specificity | Local q_target | Cross c_target |
|---|---|---|---|---|---|
| 2+1 | 75 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10+5 | 375 | 0.40 | 0.00 | 0.00 | 0.04 |
| 20+10 | 750 | **0.60** | 0.00 | 0.00 | 0.07 |
| 40+20 | 1500 | 0.50 | 0.00 | 0.00 | 0.05 |

- **Clean-block infill is a budget artifact.** The backbone can generate
  on-thesis text; 75 steps is simply too few.
- **Repair is a real failure.** On the edited blocks no budget beats a no-op,
  and specificity never appears.

That is narrower than "the editor cannot edit", and it says exactly which
capability is missing.

---

## 3. What the next experiment must test

The claim to test is **sequential, incremental revision**: modify one span,
propagate the consequence to adjacent spans, leave the rest byte-identical.
The blank canvas never tested it because it could not produce a valid span.

### 3.1 Swap the backbone

- Pre-trained masked diffusion (127M–355M), full-sequence masking,
  bidirectional infilling objective.
- Keep the architecture: same heads, same gate, same evaluator, same memory,
  same C1 guard.
- Do **not** change the repair contract or the metrics in the same commit.
  A capability change and a measurement change in one step is uninterpretable.

### 3.2 Re-run the invariants before the capability

1. **Null property** (`exp1`): empty memory reproduces the base bit-identically.
   If it fails, the memory read path is entangled with the new backbone.
2. **C1** (`exp6`): must still pass. The guard is in the update pipeline, not
   the backbone; a failure here means the swap touched the pipeline.
3. **Oracle gate** (`exp4`): must still pass. If the task is no longer solvable
   with ground truth, the task/budget is wrong, not the model.

Only then run the repair grid.

### 3.3 The repair experiment

Report, on the edited blocks and against no-op:

- `c_target` (does the revised span support T*?)
- `q_target` (is the reason specific to T*?)
- `preservation` and `seam` (are untouched blocks byte-identical, and is the
  seam valid?)
- `blocks_changed` and `tokens_spent` (was the edit local?)
- anchor preservation (locked intro byte-identical)

Pre-register the prediction: a pre-trained backbone clears `c_target` and
`q_target` on local defects, and local one-block repair still fails on
cross-block defects while repeated local/global succeeds at a matched budget.

### 3.4 Report safety and capability separately

C1 and the repair capability are different claims with different evidence.
Do not let a capability failure weaken C1, and do not let C1 stand in for a
capability result. C2 stays unresolved until the editor can produce the
exposure and the effect clears its MDE.

---

## 4. How to reproduce this snapshot

```bash
pip install -r requirements.txt

# invariants and the full unit suite
python -m pytest tests -q                     # 36 passed
python scripts/smoke.py                       # all smoke checks pass

# parameter counts: 172,925 / 1,258,989 / 6,702,429, all params get gradient
python scripts/check_presets.py

# experiments 0-6 at smoke scale, one or both seeds
python experiments/run_all.py --preset smoke --n 40
python scripts/kaggle_run.py --preset smoke --n 40 --seeds 0,1 --skip-train

# the editor-floor curve (retrains 4 budgets, ~10 min CPU; kept out of run_all)
python experiments/exp7_editor_floor.py --preset smoke --n 10 --seed 0

# regenerate the tables (never hand-edit them)
python experiments/aggregate.py
```

Checkpoints are **not** committed (7 MB of binary weights). They live in
`checkpoints/<preset>` and `checkpoints/<preset>_s<seed>`; the smoke checkpoint
is reproduced deterministically by `smoke.py` or `kaggle_run.py`. The floor
checkpoints are `checkpoints/floor_a{a}_b{b}`; `exp7 --reuse` will retrain any
that are missing.

---

## 5. Traps already paid for (do not re-learn these)

1. **A metric that cannot isolate the change it measures will report the
   siblings' credit.** This bit preservation once (positional alignment) and
   repair once (document-level averages). Always report the edited-block score
   and the no-op baseline.
2. **Byte-identical copy is the wrong target for a generative infill.** Use
   task ability (parse + on-thesis) as the gate; report copy fidelity as a
   diagnostic.
3. **Masking everything passes "gradient is zero" while proving the model
   learned nothing.** Verified-wrong must still supervise rejection (check D).
4. **A frozen parameter cannot leak.** Tests of the guard must leave all
   parameters trainable, or the unguarded control is vacuous.
5. **Gradient-mask checks must measure from the same parameter state and the
   same padded width.** `run_update` steps the model; `collate` pads to the
   batch max. Both confounds produced a false 0.3 "leak" in A2.
6. **AdamW weight decay applies to any parameter whose grad is not `None`.**
   Masking leaves the plastic grad as `None`, so it holds; a zero-tensor
   gradient would drift. Exclude plastic params from decay in production.
7. **Console is cp1252.** U+2212 and U+2192 crash prints; avoid them in
   experiment output or reconfigure stdout with `errors="replace"`.
8. **A no-op on a local defect scores c_edit 1.00 and quality (k−1)/k.** Never
   read a document-level local number as a repair without the edited-block
   score.

---

## 6. Where things live

| Path | What |
|---|---|
| `cgr/model/system.py` | `consolidate` factual/policy split; `update_memory` guard; repair modes |
| `cgr/model/memory.py` | write guard; `recall` filters unverified `source=="weights"` |
| `experiments/exp6_structural_guard.py` | C1: four checks, two controls, A2/B2 diagnostics |
| `experiments/exp7_editor_floor.py` | budget curve; clean-block ability vs edited-block repair |
| `experiments/exp4_repair_taxonomy.py` | oracle gate, model grid, `c_target`/`q_target`, no-op |
| `experiments/exp5_gate_triage.py` | per-condition action diversity, head health, triage |
| `tests/test_guard.py` | C1 unit tests (guard, controls, memory, recall filter) |
| `PAPER_REVISION.md` | §1.1 status table, §5.4 metric correction, §5.5 C1, §5.6 floor, §5.7 gate |
| `results/paper_tables.md` | generated tables; source of truth for every reported number |
| `results/*.json` | one per experiment per seed; includes notes and pre-registration |

---

## 7. The one-paragraph summary

> The safety claim is now measured: a verified-wrong example leaves the plastic
> substrate bit-identical and creates no factual memory entry, while the gate
> still learns to reject, and the unguarded control shows the test can detect
> the leak it forbids. The capability claim is not measured, and the earlier
> "floor" was two things: clean-block infill is a budget artifact that clears by
> 375–750 steps, while repair on the edited blocks never beats a no-op and
> specificity never appears. The next experiment is the one that was always
> needed: the same architecture on a pre-trained masked-diffusion backbone, with
> the invariants re-run first and safety and capability reported as separate
> claims.

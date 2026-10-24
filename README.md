# Commitment-Gated Reasoning — reference implementation

A working implementation of the CGR draft, plus the experiment suite that
produces the numbers §5 was missing.

The point of this repository is **not** to confirm the draft. It is to make
each claim in the draft either *checkable* or *falsifiable*, and to report
which is which. Several claims turned out to be design invariants rather than
empirical findings, one motivating example was wrong, and the benchmark is
weaker than the draft assumes. All of that is written down, not hidden.

Eight bugs were found by the harness while building it. Each one changed
either a reported number or the interpretation of a null; they are listed
below with the test that now prevents each from returning.

Read [`PAPER_REVISION.md`](PAPER_REVISION.md) for the revised paper sections.
Read [`results/paper_tables.md`](results/paper_tables.md) for generated tables.

---

## Status

| Component | State |
|---|---|
| Masked-diffusion backbone | working |
| Latent side channel (two-pass steering) | working |
| Seven heads + calibration harness | working |
| Eight-action gate + three overrides | working |
| Exact interpreter with provenance | working |
| Additive memory with null property | working |
| Verifier-gated consolidation | working |
| Experiments 0–7 | working |
| Stage C (RL on verified outcomes) | **not implemented** — specified only |

Everything runs on CPU at the `smoke` preset in about 20 seconds. The `dev`
and `paper` presets are sized for a free Kaggle T4.

---

## Quick start

```bash
pip install -r requirements.txt

# ~20 seconds, CPU: verifies every component and every invariant
python scripts/smoke.py

# ~45 seconds: runs every experiment at smoke scale
python experiments/run_all.py --preset smoke --n 30

# editor-floor curve (retrains at 4 budgets, ~10 min on CPU); exp7 is kept out
# of run_all because the suite's smoke path is meant to stay in seconds
python experiments/exp7_editor_floor.py --preset smoke --n 10

# tables
cat results/paper_tables.md
```

For a real run:

```bash
# measured parameter counts + a check that no parameter is dead weight
python scripts/check_presets.py

# two seeds; the tables grow a `seed` column instead of hiding one run
python experiments/run_all.py --preset smoke --n 30
python experiments/run_all.py --preset smoke --n 30 --seed 1

# Kaggle/Colab: train once per seed, run every ablation at inference
!python scripts/kaggle_run.py --preset dev --n 200 --seeds 0,1,2
```

---

## Layout

```
cgr/
  config.py          presets: smoke / dev / paper
  tokenizer.py       corpus-derived vocabulary (not 32k)
  data/grammar.py    template grammar, exact parsers, consistency check
  data/documents.py  authoritative-thesis tasks, contracts, budgets, oracle
  data/corpus.py     splits + thin wrappers over the corruption operators
  data/facts.py      scope-indexed claims with exact semantics
  model/diffusion.py masked diffusion backbone + plastic adapter
  model/reasoner.py  latent thought-blocks -> steering vector
  model/heads.py     the seven heads
  model/gate.py      eight actions + three overrides
  model/interpreter.py  exact adjudication, provenance, uncertainty
  model/memory.py    additive memory, scope-aware cache
  model/system.py    assembly, answer loop, consolidation
  scaffold.py        the hand-written policy that is the curriculum
  metrics.py         ECE, bootstrap CI, block-aligned seam/preservation
  baselines.py       probes and n-gram models (the validity gate)
  train.py           Stage A / Stage B
experiments/
  exp0_benchmark_validity.py   can trivial baselines solve the task?
  exp1_invariants.py           null property, rollback, masking, isolation
  exp2_scope_and_pressure.py   the 2+2 test, restructured
  exp3_shuffle.py              side-channel causality + positive control
  exp4_repair_taxonomy.py      scope under a matched budget (oracle gate + grid)
  exp5_gate_triage.py          action usage, overrides, triage
  aggregate.py                 results/*.json -> paper_tables.md (per-seed)
scripts/smoke.py               end-to-end verification
scripts/check_presets.py       parameter counts + dead-parameter check
scripts/kaggle_run.py          train-per-seed then run all experiments
tests/test_cgr.py              fast regressions for the bugs below
```

---

## Claim → evidence map

This is the table the draft needed. "Invariant" means true by construction and
verified by test — it belongs in a design section, not in results. "Empirical"
means it can fail, and the experiment is built so that it can.

| Draft claim | Type | Where it is tested | Can it fail? |
|---|---|---|---|
| Empty memory reproduces the base exactly | **Invariant** | `exp1` (100 inputs + negative control) | Only if the guard is broken — the control proves the test has power |
| Rollback is clean | **Invariant** | `exp1` (insert 20, remove 20) | Yes — a stale pointer would show |
| Verified-wrong claims never update weights | **Invariant (C1)** | `exp6` (all params trainable, unguarded + verified-right controls); `exp1` checks the plastic-only path | Yes — the unguarded control leaks on every check, which is what makes the guarded pass meaningful |
| Verified-wrong claims create no factual memory entry | **Invariant (C1)** | `exp6` (write guard + recall filter); `tests/test_guard.py` | Yes — the unguarded control creates the entry |
| Rejection is still learned on verified-wrong | **Invariant (C1)** | `exp6` check D (gate gradient > 1e-8) | Yes — an implementation that masks everything fails this |
| Verified-wrong claims are not *absorbed* (C2) | **Empirical — unresolved** | `exp2` (B, with held-out control) | Yes — measured as NLL diff-in-diff; at smoke scale the unguarded effect is **underpowered**, so C2 is unresolved, not failed |
| Scope-indexed truth is representable | **Empirical** | `exp0` + `exp2` (C) | Yes — and `exp0` shows a linear probe solves it, so the claim must be weakened |
| Counting claims bind to the stated unit | **Controlled demonstration** (4 constructed cases) | `exp2` (E: four claims over four born people + one fetus) | Yes — `2+2=5` and "five born people" rejected, "five entities counting the fetus" accepted, bare "there are five" deferred |
| Side-channel reasoning is causal | **Empirical** | `exp3` (+ positive control) | Yes — CI on paired differences |
| Revision scope should follow defect location and budget | **Empirical** | `exp4` (oracle gate, then defect × method under matched budgets, incl. repeated local) | Yes — and the oracle gate must pass before the grid is interpreted |
| The learned editor repairs defects | **Empirical — not established** | `exp4` (model grid) + `exp7` (budget curve) | Yes — on the edited blocks no method beats a no-op; specificity is 0.000 at every budget |
| The learned gate separates local from cross-block repair | **Empirical** | `exp5` (per-condition action distribution) | Yes — REP 1.00 on local, REV 1.00 on cross-block; bare claims 0.80 |
| Triage beats sequential | **Empirical** | `exp5` | Yes — and at smoke scale it does not; the metric is also ceiling-bound |
| All eight actions are used | **Empirical** | `exp5` histogram | Yes — two are never offered, one is offered but unlearned |

**C1 and C2 are different claims.** C1 is the structural guarantee (verified-wrong
cannot update the substrate or write factual memory, while rejection is still
supervised). It does not depend on the editor and is **supported** by `exp6`.
C2 is the behavioural benefit (the guard improves held-out resistance to false
information). It requires the editor to produce the exposure and is
**unresolved** because the effect is below the minimum detectable effect at this
scale. Do not write C2 up as a failure of selective plasticity.

---

## Four things this repository changes about the draft

**1. Invariants are separated from results.** The null property and gradient
masking are properties of the code. Reporting them as findings invites the
obvious reviewer objection. They are now verified by test, with negative
controls, in their own section.

**2. The motivating example is replaced.** The draft's pregnancy example does
not work (2 males + 2 females is four people, and a pregnancy is not a person)
and, worse, it requires world knowledge the exact interpreter does not have.
The replacement is `2 + 2 = 10`, false in decimal and true in base 4, and
`2 + 2 = 5`, true under "round up to the nearest 5". Both are exactly
adjudicable, so the interpreter is load-bearing instead of decorative.

**3. The benchmark is gated before GPU time is spent.** `exp0` runs linear
probes and n-gram baselines against every label. It currently reports that
scope classification is 100% solvable from surface features — which means the
scope head cannot be described as representing scope. That is a finding about
the benchmark, and it is cheaper to discover in ten minutes than after a week
of training.

**4. The label pipeline is part of the result.** The most damaging bug found
here was not in the model: non-adjudicable claims were labelled as resolved,
which taught the value head to endorse everything and made one of the draft's
headline diagnostics meaningless. A paper that reports a head's behaviour
without reporting its label distribution and variance is reporting an artefact
it has not excluded. The generated tables therefore carry head-health numbers
next to the results they qualify.

---

## Bugs the harness caught

These were found while building the suite, and they are the reason the
validity gate exists. Each has a regression test or a check script, so the
bug cannot silently return.

- **Empirical claims were "adjudicated false".** `adjudicate` returned
  `correct=False` for non-adjudicable claims instead of `None`. Since the
  scaffolding labels a claim "resolved" whenever `correct is not None`, every
  empirical claim was labelled `value=1.0, source=derived` — which is exactly
  what made the value head learn to endorse everything. Fixed in
  `model/interpreter.py`; `test_empirical_is_not_adjudicated_false` pins it.
- **Scope-unsafe memory cache.** `verified_hit` matched on bag-of-words
  similarity, so a bare `2 + 2 = 5` retrieved the cached answer for
  `rounding up to the nearest 5, 2 + 2 = 5`. That is exactly the
  default-scope leakage the architecture is supposed to prevent. Fixed by
  requiring a scope match (`model/memory.py`).
- **The side channel was decorative.** The first implementation never passed
  the steering vector to the backbone, so §3.2 was vacuous and `exp3` would
  have reported a null for structural reasons. Fixed with a two-pass scheme
  (`model/system.py:encode`).
- **The plastic substrate could not move decisions.** It sat on the output
  projection, so consolidation could change token probabilities but not any
  head output — making the sycophancy test unmeasurable. Fixed by making it a
  low-rank adapter on the hidden state (`model/diffusion.py`).
- **Two dead parameters.** A `log_sigma` scale (never used, the value head is
  MSE) and a zero-initialised `to_vocab` projection in memory (never called)
  were counted in the model size and received no gradient. Removed;
  `scripts/check_presets.py` now fails if any parameter is unreachable.
- **`DELIB` was reported dead when it was never offered.** `exp5` built its
  eval set from marker-present claims only, and `DELIB` is the scaffolding
  action for *marker-stripped* claims. The histogram therefore said "DELIB
  never selected" for a curriculum reason, not a modelling one. Fixed by
  including both marker conditions; `test_oracle_action_covers_delib_and_def`
  pins the curriculum contract.
- **Triage was scored against the model's own value head.** The first version
  defined "solvable" as `preds["value"] > 0.5`, so a saturated head made every
  ordering score exactly the budget and triage looked tied with random for a
  reason unrelated to `tau`. Now scored against labels, with a permutation
  test, a normalised gain, and an explicit statement of the metric's ceiling.
- **Seeds silently collapsed.** `kaggle_run.py` trained seeds 1 and 2 into
  suffixed checkpoint directories, but every experiment loaded
  `checkpoints/<preset>`. A three-seed run would have reported the same model
  three times. Fixed with `ckpt_dir(preset, seed)`, per-seed result files, and
  a `seed` column in the generated tables.

---

## Honest limitations

1. **Smoke-scale numbers are not results.** The checked-in
   `results/paper_tables.md` comes from the `smoke` preset (172.9k parameters,
   400 documents, 3 epochs, seeds 0 and 1). It proves the pipeline and the
   seed plumbing; it is not evidence.
2. **The editor floor is two findings, not one.** (i) *Clean-block infill is a
   budget artefact.* Byte-identical restore is 0% at every budget because a
   masked-diffusion infill regenerates rather than copies; on-thesis infill
   rises from 0% at 75 optimizer steps to 60% at 750 (`exp7`). The earlier
   version of this limitation used byte-identical restore as its sanity check
   and would have called a budget artefact an architecture limit. (ii) *Repair
   is a real failure.* Scored on the blocks the task actually edits, no method
   beats a no-op at any budget, and specificity is 0.000 throughout. The
   deterministic oracle gate passes, so the task is solvable and the failure is
   the editor's. `exp4` sets `informative = 0` and suppresses the model grid
   rather than interpreting it.
3. **C2 is unresolved, not failed.** The value head collapses at one seed and
   not another (mean 0.78, sd 0.015 at seed 0; mean 0.33, sd 0.14 at seed 1),
   which is why head health is reported per seed. The sycophancy measure is the
   NLL diff-in-diff, and after the curriculum rebuild it is within its minimum
   detectable effect at both seeds (+0.013 / -0.001). That is a **power**
   limitation: it is not evidence that the guard works, and it is not evidence
   that it does not. C1 (the structural guarantee) is established separately in
   `exp6` and does not depend on this.
4. **Dead actions, two different kinds.** `REORD` and `UPD` are never offered
   by the scaffolding, so they have no training signal (effective action space
   is six). `DEF` is offered and never predicted by seed 0, which is a gate
   learning failure. Both are reported; neither is a property of the
   architecture.
5. **Triage is underpowered by construction.** With a 93% solvable mix and a
   20% budget, the metric can move by at most 1.6 marks. The reported result
   is a null with p=0.17, and the design, not just the model, is why. The 100%
   normalized gain is a ceiling artefact.
6. **Stage C is not implemented.** The paper says it is specified but not run.
   That remains true.
7. **The interpreter's parser is templated.** It is given a structured spec,
   not free text. A real system needs a parser, and parser errors would be a
   new failure mode.

---

## Compute

| Preset | Params (measured) | Docs | Time (CPU) | Time (T4) |
|---|---|---|---|---|
| smoke | 172,925 | 400 | ~20 s | — |
| dev | 1,258,989 | 4,000 | ~15 min | ~1 min |
| paper | 6,702,429 | 30,000 | not run | ~1–2 h |

Counts come from `scripts/check_presets.py`, which constructs each preset,
forwards a real batch, backprops, and fails if any trainable parameter receives
no gradient. The draft's "~18M parameters, 28 GPU-hours" was not achievable
with its own budget (8 ablations × 3 seeds × 12 h of training exceeds 27 h).
The revised budget is in `PAPER_REVISION.md`.

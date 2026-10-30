> **2026-10-30 evidence correction:** Historical 25.9x/27.2x cached-generation
> and 2.0x/1.70x full-loop speedup interpretations are withdrawn. The cache
> timer measured prefix prefill with no repaired-suffix decode; repair timers
> excluded required controller work. Token/FLOP counts were estimates.
> Recovery counts remain descriptive conditional-repair observations, with
> candidate clustering unaccounted for. Use `PAPER.md` and
> `findings/2026-10-30-paper-review.md` for the corrected interpretation.

# Paper revision memo

Concrete replacement text for the CGR draft. Everything here is either a
deletion, a re-scoping, or a number with a script behind it.

---

## 0. Summary of what changed

| # | Change | Reason |
|---|---|---|
| 1 | Null property and gradient masking moved out of §5 into a **Design invariants** section | They are properties of the code. Testing them is a unit test, not a finding. |
| 2 | The motivating example replaced | The pregnancy example is arithmetically wrong and not adjudicable by the interpreter. |
| 3 | New **Experiment 0: benchmark validity** | A linear probe solves scope at 100%. Without this gate the paper reports a null it cannot interpret. |
| 4 | The "verified-wrong does not update" claim split in two | For adjudicable claims the *interpreter*, not the mask, does the work. The mask matters only for non-adjudicable claims. |
| 5 | Shuffle ablation gains a **positive control** | A null shuffle result is uninterpretable without evidence the channel *can* carry information. |
| 6 | Repair benchmark rebuilt around an **authoritative, locked thesis** | Without an explicit T\*, "revert the intro" is a valid local repair and the non-local condition is undecidable. Defects are now different in kind (quality vs coherence), corruptions are validated on the produced document, and methods are compared under a matched budget including repeated local repair. |
| 7 | Compute budget rewritten | The original (8 ablations × 3 seeds × 12 h) does not fit in 27 GPU-hours. Most ablations are inference-time and need no retraining. |
| 8 | ARC-AGI-2 appendix cut | Unrelated to every claim in the paper, and it consumes the compute budget. |
| 9 | Label pipeline audited as part of the method | Non-adjudicable claims were being labelled as resolved, which taught the value head to endorse everything. Head-health numbers now sit next to the results they qualify. |
| 10 | Multi-seed reporting made structural | Seeds previously collapsed onto one checkpoint. Results are now per seed and the tables carry a `seed` column. |

---

## 1. Claim audit

The draft's §5 conflates three kinds of statement. Split them.

**Kind A — invariants (true by construction).** Report in a design section,
verify with property-based tests, and include a negative control so the test
is shown to have power.

- `M = ∅ ⇒ output ≡ f(x; W0)` bit-identically.
- Rollback: insert then remove returns to bit-identity.
- `verified_wrong ⇒ ∇φL = 0`, **and** no factual memory entry is created, **and**
  the gate still receives a supervised rejection gradient. This is C1, tested
  end-to-end in exp6 with an unguarded control that must fail and a
  verified-right control that must pass (§5.5).
- The reasoner has no vocabulary projection.

**Kind B — empirical claims (can fail).** These get confidence intervals,
seeds, and a pre-registered prediction.

- Non-adjudicable claims are absorbed under unguarded consolidation. At this
  scale the positive control is **underpowered**, not absent: the CI includes
  zero and the MDE is reported. C2 is unresolved, not failed.
- The side channel is causally used.
- Revision scope should follow the location and dependency structure of a
  defect, *given a budget*: one block suffices for a local quality defect,
  while a cross-block coherence defect requires either a global revision or
  several local edits. The claim is conditional, not "local repair can never
  fix a non-local defect".
- Triage beats sequential ordering.
- All eight actions are used.

**Kind C — claims to delete.** They cannot be supported at this scale, or they
are true by definition:

- "Scope-indexed truth is representable." A linear probe solves the task, so
  this measures the marker token.
- "Edit modes can be selected by a gate." Two of the eight actions are never
  selected by the scaffolding, so the gate selects among six.
- "Memory can be additive and null-safe." Null-safety is an invariant; the
  interesting question is whether memory *helps*, which is not measured.

### 1.1 Status after the structural test

The table separates the structural guarantee (C1) from the behavioural benefit
(C2). They are different claims with different evidence; conflating them is
what made an earlier assessment say the moat was untestable.

| Claim | Status | Evidence |
|---|---|---|
| Metrics and evaluation contracts behave as intended | **Supported** | 36 tests, incl. negative controls |
| Oracle gate solves the task; taxonomy discriminates | **Supported** | exp4 oracle grid, 2 seeds |
| Single-pass local repair is insufficient for cross-block defects; repeated local repair succeeds at matched budget | **Supported** | exp4 oracle grid, n=40 |
| Counting ontology: reject / reject / accept / defer, bare "there are five" → REFUSE | **Controlled demonstration** (4 constructed cases) | exp2 §E, both seeds |
| **C1: verified-wrong cannot update the plastic substrate or write factual memory, while rejection is still supervised** | **Supported** | exp6 four checks + unguarded and verified-right controls |
| Learned editor repairs defects better than doing nothing | **Not established** | exp7: target-isolated c_target/q_target never beat no-op; specificity 0.000 at every budget |
| Learned gate separates local from cross-block repair | **Supported at smoke scale** | exp5: REP 1.00 on local, REV 1.00 on cross-block; ECE ≈ 0.004 |
| Triage beats random ordering | **Not established** | exp5: p = 0.17, ceiling 1.6 marks |
| **C2: the guard improves held-out resistance to false information** | **Unresolved — underpowered** | exp2 NLL diff-in-diff within MDE; editor cannot produce the exposure |

Two rows deserve emphasis. C1 is a structural guarantee about the update
pipeline and does not depend on the editor; it is the claim a prompt cannot
replicate. The editor row is *not* "the editor is at the floor": clean-block
infill rises to 60% with training, but on the blocks the task actually edits no
method beats a no-op, and specificity never appears. That is a narrower and
more precise failure (§5.6).

---

## 2. The motivating example

### Delete

> `2 + 2 = 5, where 2 males and 2 females are counted, and one female is pregnant`

Two problems. First, it is arithmetically wrong: 2 males + 2 females is four
people, and a pregnancy is not a person, so the claim is simply false and the
"scope" does no work. Second, adjudicating it needs world knowledge the exact
interpreter does not have, which undercuts the binding-evidence claim in §3.5.

### Replace with

> `2 + 2 = 10` is false in decimal and true in base 4.
>
> `2 + 2 = 5` is false in decimal and true when the operation is "round up to
> the nearest multiple of 5".

Both are exactly adjudicable, so the interpreter is load-bearing. The base-4
case is especially clean: the numeral `10` is well formed in both scopes, and
the *only* thing that changes the truth value is the scope. That is the
phenomenon the paper is about, with no confound.

### Suggested introduction paragraph

> Consider `2 + 2 = 10`. In decimal it is false; in base 4 it is true. A model
> that answers without first establishing the scope has no way to be right
> for the right reason. Current systems resolve this by guessing from surface
> form, which works until the marker is absent — and then the model either
> applies a silent default or contradicts itself across turns. The failure is
> not arithmetic. It is that the system never represented *which question was
> being asked*, and therefore had nothing to check its answer against.

---

## 3. Revised abstract

> Current autoregressive transformers conflate generation with commitment: an
> emitted token *is* the output, with no revision operator, no record of prior
> belief, and no distinction between what the model considered and what it
> endorsed. We present Commitment-Gated Reasoning (CGR), in which reasoning
> runs in a latent side channel that can steer but never become the artifact,
> seven heads predict the quantities a gate needs, an exact interpreter makes
> executed values binding, and only verified outcomes with positive advantage
> update a small plastic substrate. We report results on a controlled
> synthetic benchmark, and we report them in three tiers. **Invariants** —
> null-safety, rollback, and gradient masking — are verified by property-based
> tests with negative controls. **Empirical claims** are tested with paired
> bootstrap confidence intervals and pre-registered predictions, and are
> preceded by a benchmark-validity gate that runs linear probes and n-gram
> baselines against every label. **Negative results** are reported with the
> same prominence as positive ones: on our benchmark, scope classification is
> solved by a linear probe, so scope representation is not established at this
> scale; and for exactly adjudicable claims the interpreter, not the gradient
> mask, is what prevents absorption of false claims. We claim mechanism, not
> capability, and we specify the conditions under which each claim would fail.

---

## 4. Revised §4 (experimental setup)

Four changes, each of which fixes a confound.

**4.1 Vocabulary.** Not 32k. Build from the corpus (183 tokens in our run).
A 32k embedding table on a corpus with a few hundred types is 90% padding and
makes the parameter count meaningless.

**4.2 Splits.** Hold out *template combinations*, not just documents. Report
in-distribution and held-out numbers separately. The draft's 30k documents
from 12 templates gives ~2,500 exposures per template; at smoke scale we
already measure an 11% exact-duplicate rate between test and train.

**Backbone hardening update (not part of the v0.3 canvas results above).** The
current DiffuGPT generator has 5,000 training documents, eight intro, eight
conclusion, and eight body templates, six thesis labels, two of which are
isolated in an `unseen` split, and a `cross_block_partial` defect whose wrong
thesis occupies a strict majority of editable blocks. This is controlled,
exactly parsed template variation; it is not open-ended paraphrasing.
`exp8_architecture_ablation.py`
compares a jointly trained CGR reasoner condition against a LoRA-only
masked-denoising baseline on the same unseen tasks. The current repair path
does not use the gate or a DAG, so this comparison cannot support a claim
about either component. The 1-epoch smoke result is a pipeline check only;
both systems scored at the parse/repair floor and a full matched run is still
needed.

**Repair-floor diagnosis (5k checkpoints).** The first 5k CGR run cleared the
denoising sanity check (synthetic top-1 0.915) but scored only 7/40 (0.175)
parseable-and-on-thesis on the ordinary test split, below the pre-registered
0.20 floor. A per-block diagnostic (`scripts/diagnose_repair.py`, n=8) found no
mask leaks or empty spans; 6/8 outputs were unparseable because they blended
templates (e.g. `point 3 : the observation that ... supports ... in this
reason .`), while the thesis phrase and a pool reason were present in 7/8
outputs. A teacher-forced probe that masks the whole target block and predicts
it in one pass reached only 0.292 top-1 (0/8 exact blocks), versus 0.915 on
15% random token masks. The training objective masked scattered tokens
(15-85% of the document); the benchmark masks one contiguous block. The model
therefore learned local denoising but not the template-selection operation the
gate scores.

The targeted change is a mixed masking objective: with probability 0.5 an
example masks one whole block (a run between separator tokens), otherwise it
keeps the original random-token mask. Claims, which have no separator, keep
the token mask. Both the CGR model and the LoRA-only control are being
fine-tuned from their existing checkpoints with the same mixture (4 additional
Stage A + 1 Stage B epoch, LoRA rank 8, learning rate, seed and corpus
unchanged). This is one factor: the mask distribution. The result is not yet
in; no claim about clearing the floor or unseen-thesis repair is made here.

The realised mixture is not 50/50 over all training examples. Claim episodes
carry no separator token, so they always keep the token mask; documents and
repair episodes (8,750 of 13,750 examples per epoch) are block-capable, and
half of those receive a block mask. The realised fraction of block-masked
examples is therefore about 32%, with 68% token-masked. That is deliberate:
masking an entire short claim would be unconditional generation, a different
task from the one the benchmark scores.

**4.3 Stochasticity.** In a deterministic template world, `V`, `o`, and `τ`
are deterministic functions of the same features and the seven heads collapse
into each other. The draft's own health check ("inter-head correlation must
not exceed 0.95") will fail. Either make the world stochastic (outcome noise,
variable completion time, noisy counts) or cut to four heads. We chose
stochasticity, because expected-value-of-computation claims are vacuous
without stochastic cost and benefit.

**4.4 Baselines.** Add, before training: majority class, order-2 n-gram, and a
ridge probe on hashed bigrams for every head. If a probe matches the head, the
head is not doing the work the paper says it is.

---

## 5. Revised §5 (results structure)

Six tables, in this order. Table 0 is a gate: if it fails, the rest are
reported as exploratory.

### 5.0 Benchmark validity

| Task | Linear probe | Majority | Verdict |
|---|---|---|---|
| scope (marker present) | **1.000** | 0.200 | surface-solvable |
| scope (marker stripped) | 0.733 | 0.200 | cue mostly absent; above chance |
| epistemic | 0.967 | 0.767 | surface-solvable |

> **Finding.** Scope classification is solved by a linear probe on hashed
> bigrams. The scope head therefore cannot be described as representing
> scope; at this scale it learns the marker token. We report the scope
> experiments as a test of *marker-conditioned behaviour*, not of scope
> representation, and we identify marker-free scope inference as future work.
> The 0.733 on marker-stripped claims is the interesting number: it is well
> above the 0.200 majority baseline, so the residual cue is *lexical* (the
> claim body differs by family), not semantic scope inference.

### 5.1 Design invariants (not results)

| Invariant | Verified | Negative control |
|---|---|---|
| null property (100 inputs) | yes | non-empty store changes output |
| rollback (insert 20, remove 20) | yes | — |
| gradient masking (50 repeats) | 50/50 unchanged | verified-right moves substrate |
| side-channel isolation | no vocab projection | — |

### 5.2 Scope, pressure, and what the mask is for

| Condition | Metric | Value |
|---|---|---|
| wrong adjudicable claims, before update | not accepted | 1.000 |
| wrong adjudicable claims, after unguarded update | not accepted | 1.000 |
| empirical claims, guarded | ΔNLL | 0.000 |
| empirical claims, unguarded | ΔNLL (diff-in-diff) | +0.013 (s0) / −0.001 (s1) |
| scoped-true | accept / scope stored | 1.000 / 1.000 |
| bare-false | reject / assumption flagged | 1.000 / 1.000 |
| empirical | defer rate | 1.000 |
| counting ontology (E) | interpreter matches spec | 1.000 |
| counting ontology (E) | system matches spec | 1.000 |

> **Counting ontology (E), measured separately from arithmetic.** One fixed
> scenario: four born people plus exactly one fetus. The four claims and their
> required dispositions are `2 + 2 = 5` → reject (ordinary arithmetic gives
> four, and the rejection is flagged as resting on the default decimal scope);
> "there are five born people" → reject (four); "there are five entities,
> counting the fetus" → accept; "there are five" with no counting rule →
> defer. The last case is the one that matters: the interpreter REFUSES rather
> than assuming "born people", so the undefined unit is handled explicitly and
> the scope-confusion matrix is reported per claim. This tests scope binding,
> not arithmetic whose truth changes with context.

> **Finding (negative, about the architecture).** The substrate moved, and the
> decisions did not: for exactly adjudicable claims the interpreter outranks
> the weights. Gradient masking is therefore not what prevents absorption of
> false arithmetic claims — the interpreter is. The mask is load-bearing only
> for non-adjudicable claims, which is the only place sycophancy can occur.
>
> **Head-health caveat, measured.** On empirical claims the value head is a
> near-constant at seed 0 (mean 0.78, sd 0.015) and therefore "endorses" all of
> them: 0.5 sits below the constant. Endorsement rate is not evidence about
> sycophancy and is not reported as such.
>
> **The unguarded-absorption check fails at smoke scale.** The pre-registered
> measure is the diff-in-diff on sequence NLL, treated vs held-out control.
> After the curriculum rebuild it is +0.013 at seed 0 and −0.001 at seed 1:
> the treated claims are not absorbed more than the controls. The check is
> recorded as failed rather than reframed, and the effect is reported as a
> floor: at 173k parameters and 20 consolidation steps the substrate is too
> small (or the step budget too low) to demonstrate absorption. This is the
> second failed positive-control-style result in the suite, alongside the
> oracle side-channel control in §5.3.
>
> This is a stronger and more useful claim than the original, and it is a
> *reduction* in what the mask is credited with.

### 5.3 Side-channel causality

Five conditions, including the control the draft lacked:

| Condition | Loss | Δ vs full |
|---|---|---|
| full | 3.507 | 0 |
| shuffled | 3.696 | +0.188 |
| constant | 3.656 | +0.149 |
| none | 4.692 | +1.185 |
| **oracle (clean-doc steering)** | 3.523 | +0.016 |

> **Finding, and a failed control.** The channel is causally used: full vs
> shuffled is +0.186, 95% CI [0.116, 0.264], MDE 0.110, so the CI excludes
> zero. But `shuffled ≈ constant` (+0.188 vs +0.149): what matters is the
> *presence* of a steering vector, not its content. And the positive control
> **failed** — the oracle vector did not beat `full` (+0.016), so at this scale
> the channel is not shown to be able to carry information the input lacks.
> That makes the shuffle null uninterpretable in the way the draft would have
> used it: we cannot say "reasoning is decorative", only "we did not
> demonstrate that it is not". Report the MDE and the failed control together;
> a control that fails is a result about the control.

### 5.4 Revision scope under a matched budget

**The task contract is explicit and is part of the data:**

> The introduction is the authoritative, immutable thesis T\*. All editable
> blocks must be revised to support it.

This closes the loophole in the previous design. A document whose intro says B
and whose bodies say A is inconsistent only *relative to an intended thesis*.
Without an authoritative intro, "revert the intro to A" is a valid one-block
repair, and global coherence cannot distinguish it from the intended solution.
T\* is now given per task, and the intro is `locked`: editing it is a contract
violation, reported as `anchor_preserved = 0`, not scored as an improvement.
The intro's quality is therefore *preserved*, never presented as an outcome of
local repair.

**The two defects are different in kind.** They are not two severities of the
same corruption:

| | local | cross-block |
|---|---|---|
| operation | one body loses specificity, proposition kept | every editable block retargeted to one wrong thesis W |
| coherence C(D) | 1.0 | 0.0 over editable blocks |
| defect | quality | coherence |
| one-block repair | sufficient | insufficient by construction |

Every block carries metadata (`text`, `thesis_id`, `role`, `quality_target`,
`locked`), and every corruption is validated by re-parsing the document it
actually produced. A `requires_neighbor` flag is never trusted.

**Metric.** C(D) is the mean over required blocks of 1[thesis(b) = T\*]. It is a
*synthetic label-consistency score*, not a semantic coherence measure. `c_all`
includes the locked intro, so a no-op on a cross-block task floors at
1/|required|; `c_edit` restricts to editable blocks and is the metric that can
move. Reported alongside: anchor preservation, local block quality, blocks
changed, tokens spent, and NLL-based improvement/preservation/seam. Polished
text is not success.

`preservation` and `seam` are computed **block-aligned**: the untouched prefix
and suffix are compared from their own ends, and each edited span is located in
its own sequence. A generative rewrite that changes a block's length therefore
is not scored as damage to every token after it. The earlier positional version
did exactly that, which is why the metric was changed before any number was
reported; a negative control (damage the intro) confirms the metric still
falls.

**Protocol, frozen before measurement.** (1) Oracle solvability gate: a
deterministic editor with ground truth must reach the target under each
method's budget, or no model number is interpretable. (2) Model grid over
defect × method. (3) Gate-selected condition: does the trained gate pick a
useful strategy, or merely the one that edits more text? `global` and
`repeated_local` receive identical block and token allowances; `one_block`
receives one block. `repeated_local` uses a ground-truth detector, which
favours local repair, so a loss is not a detection artifact.

**Oracle gate (measured, deterministic).** At smoke scale, 40 tasks per
condition (one document skipped by the context filter, reported in the run):

| Defect | Method | Anchor | c_edit | c_all | Quality | Blocks | Tokens |
|---|---|---|---|---|---|---|---|
| local | one_block | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.4 |
| local | global | 1.00 | 1.00 | 1.00 | 1.00 | 4.03 | 55.7 |
| local | repeated_local | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 14.4 |
| cross-block | one_block | 1.00 | **0.26** | 0.41 | 0.36 | 1.00 | 14.5 |
| cross-block | global | 1.00 | **1.00** | 1.00 | 1.00 | 4.03 | 55.7 |
| cross-block | repeated_local | 1.00 | **1.00** | 1.00 | 1.00 | 4.03 | 55.7 |

> **The task is solvable and genuinely non-local.** A one-block oracle repair
> leaves c_edit = 0.26 on cross-block tasks and 1.00 on local tasks. The
> anchor is preserved in every condition. Because repeated local repair also
> reaches 1.00 at the matched budget, the *only* defensible claim is the
> conditional one: scope should follow the defect's dependency structure and
> the budget, and repeated local repair is a legitimate strategy whose cost is
> the number of coordinated calls, not an impossibility.

**Model grid.** The first version of this paragraph said the backbone "restores
a masked clean block exactly 0% of the time, so the model grid is at the
floor." That was measuring the wrong thing, and the correction matters.

Byte-identical restore is **copy fidelity**, and a masked-diffusion infill
regenerates the span rather than copying it, so the number is 0% at *every*
training budget. Gating on it made an undertrained preset look like an
architecture failure. The operative precondition is **task ability**: does the
generated block parse and support the document's thesis? On clean-block infill
that rises from 0% at 75 optimizer steps to 60% at 750 (§5.6), so the smoke
preset's 0% is a budget artefact, not a capacity limit.

The second correction is to the *scoring*. Document-level `c_edit` and
`quality` average over all editable blocks, so on a local defect the two clean
siblings contribute 1.0 and (k−1)/k and a degenerate edit inherits their
credit. That is the same class of bug as the earlier preservation artefact: a
metric that cannot isolate the change it claims to measure. Every model row now
also reports `c_target`/`q_target`, restricted to the blocks the task actually
edited, plus the no-op baseline on those same blocks.

With both corrections, the smoke-preset grid reads:

| Condition | c_edit (doc) | c_target (edited) | q_target | no-op c_target | no-op q_target |
|---|---|---|---|---|---|
| local / one_block | 0.74 | **0.00** | **0.00** | 1.00 | 0.00 |
| local / global | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| cross-block / one_block | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cross-block / global | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

> **The document-level local number is not a repair.** A no-op already scores
> c_edit 1.00 on a local defect, and the model's 0.74 comes from the siblings it
> did not touch while breaking the edited block (c_target 0.00). On the blocks
> the task actually edits, no method beats doing nothing. The pre-registered
> interaction is therefore not evaluated; the run records `informative = 0` and
> suppresses the scope claim rather than reporting a null the design could not
> have detected. The oracle gate above is what makes that interpretable: the
> task is solvable, so the failure is the editor's, not the benchmark's.

**What the learned-policy comparison does and does not cover.** The
`gate_selected` row is a learned policy over scope: the gate chooses REP (one
block) or REV (all editable blocks), and its outcome is compared against the
two fixed policies. A learned controller for *repeated local* repair (when to
stop, which block next) is Stage C in the design and is specified but not
implemented here. The repeated-local rows therefore use a ground-truth
detector, which is the conservative choice: it gives local repair the best
possible detector, so a loss cannot be blamed on detection.

### 5.5 Structural guard (C1): pre-registration

Written **before** the test was run. C1 is a structural claim about the update
pipeline and does not depend on the editor: *a verified-wrong example cannot
produce a parameter update to the plastic substrate and cannot create a factual
memory entry.* C2 (the behavioural benefit) is a separate claim, measured by the
NLL diff-in-diff in §5.2, and is unresolved because the editor is at the floor.
Conflating C1 with C2 is what made the previous assessment too pessimistic.

**Three operations, deliberately distinct.** A test that masks all three would
pass "gradient is zero" while proving the model learned nothing at all:

| Operation | On verified-wrong | Check |
|---|---|---|
| store as factual memory | prohibited | C |
| reinforce the false answer as correct | masked | A, B |
| learn to verify/reject what the verifier flagged | **supervised** | D |

**Pre-registered pass criteria.**

| Check | Pass condition | Tolerance |
|---|---|---|
| A gradient masking | max abs gradient on `plastic_a`/`plastic_b` is **exactly 0.0** | 0.0; a value in (0, 1e-8) is **ambiguous**, not pass |
| B parameter immutability | `torch.equal` on both plastic tensors before/after | bit-identical |
| C memory write prohibition | no entry created; `update_memory(verified=False)` returns -1 | exact |
| D rejection learning | gradient norm on gate parameters **> 1e-8** | 1e-8 |

**Controls that validate the test itself.**

| Control | Required behaviour |
|---|---|
| unguarded (same example, no mask) | A **fails** (gradient non-zero), B **fails** (parameters drift), C **fails** (entry created) |
| verified-right (same pipeline, verified-correct) | gradient flows, parameters change, entry created |

If the unguarded control also shows zero gradient, the test is invalid: the
gradient was not reaching the parameters, and A's pass means nothing. The
unguarded control is a validity check on the instrument, not a comparison.

**Pre-registered outcomes.**

- **Pass**: A exactly 0, B bit-identical, C no entry, D gate gradient > 1e-8,
  and both controls behave as above.
- **Fail**: any leakage in A, B or C; or D zero when it should be non-zero.
- **Ambiguous**: a gradient in (0, 1e-8). Reported as ambiguous, not rounded to
  pass.

**Two implementation facts checked before the run.** (1) `consolidate` dropped
every non-verified example *before* backward, so D had no path at all; the
factual term and the policy term are now separated, with the policy term
computed on detached features so it cannot reach the substrate. (2)
`SparseMemory.add` had no write guard, and `recall` did not filter unverified
entries, so a verified-wrong claim could be stored and read back; unverified
writes now require an explicit opt-in that only the null-property test uses.

**Results (seed 0, n = 16 adjudicated-wrong and 16 adjudicated-right claims).**
The run is `experiments/exp6_structural_guard.py`; every number below is in
`results/exp6_structural_guard.json`.

| Check | Measured | Pre-registered verdict |
|---|---|---|
| A gradient masking | max abs plastic gradient **0.0** | pass |
| B parameter immutability | both plastic tensors `torch.equal` before/after | pass |
| C memory write prohibition | `update_memory(verified=False)` returned −1; 0 entries created | pass |
| D rejection learning | gate gradient norm **4.894** (> 1e-8); policy term supervised | pass |
| unguarded control | gradient 0.275, parameters drifted, memory entry created | A, B, C all fail as required |
| verified-right control | gradient 0.212, parameters changed, entry created | passes as required |

**Verdict: C1 passes with a valid instrument.** The unguarded control is the
part that makes this a test rather than a tautology: with the same examples and
no verifier, every guard check fails. The verified-right control rules out the
opposite failure, a pipeline that cannot update at all.

Two diagnostics were run but **not** pre-registered, and are reported as
diagnostics rather than pass/fail:

- **A2, mixed-batch inertness.** A mixed batch with the wrong claims marked
  unverified gives a plastic gradient **identical** to the verified-right
  examples alone (max difference 0.0), with row alignment and padding width
  matched. The first version of this diagnostic reported a 0.3 difference; it
  was wrong twice over — it marked the wrong claims verified, and it compared
  gradients measured after one of the runs had already stepped the model.
- **B2, weight decay.** Under AdamW, a *zero-tensor* gradient still moves a
  parameter (weight decay applies whenever `grad is not None`), while a
  `None` gradient does not. Masking leaves the plastic gradient as `None`, so
  the guard holds on this path. A production optimizer should still exclude
  the plastic parameters from decay rather than rely on that detail.

C1 is a structural guarantee about the update pipeline. It says nothing about
whether the guard improves held-out behaviour; that is C2, and it remains
unresolved for the statistical-power reason in §5.2. The moat is the first
claim, and it is now measured rather than asserted.

### 5.6 Editor floor: budget, not capacity (and not a repair)

The smoke preset runs 75 optimizer steps. The original floor report treated
that as an editor failure. Two measurements show it was partly a budget
artefact and partly a real failure, and they are different:

| Budget | Opt. steps | Clean-block on-thesis | Specificity | Local c_target | Local q_target | Cross c_target |
|---|---|---|---|---|---|---|
| 2+1 | 75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10+5 | 375 | 0.40 | 0.00 | 0.20 | 0.00 | 0.04 |
| 20+10 | 750 | **0.60** | 0.00 | 0.00 | 0.00 | 0.07 |
| 40+20 | 1500 | 0.50 | 0.00 | 0.30 | 0.00 | 0.05 |

`local_*` and `cross_*` are scored on the edited blocks; the no-op baselines
are c_target 1.00 / q_target 0.00 for local and 0.00 / 0.00 for cross-block.
The run is `experiments/exp7_editor_floor.py`; `results/exp7_editor_floor.json`
is the source.

> **Two findings, not one.** (i) Clean-block infill is a **budget** artefact:
> on-thesis output rises from 0% to 60% and then plateaus. The backbone is not
> incapable of generating a block. (ii) Repair is a **real** failure: on the
> blocks the task edits, no budget beats a no-op, and specificity is 0.000 at
> every budget. The editor can restore label consistency on a clean document
> and cannot restore the specificity that defines a local defect, nor revise a
> cross-block document against its locked intro.
>
> This is narrower than "the editor cannot edit", and it is more useful: it
> says exactly which capability is missing. Model-dependent claims (the grid,
> the gate-selected row, C2) remain blocked, but the reason is specificity and
> cross-block revision, not text generation.

A caveat on the curve: 20+10 scores higher than 40+20 on clean-block infill.
That is one seed and 10 documents, so the non-monotonicity is not interpreted;
the robust statement is that the floor clears by 375–750 steps and specificity
never does.

### 5.7 Gate usage and triage

Numbers below are seed 0; `results/paper_tables.md` carries every seed.

| Action | Model | Scaffolding |
|---|---|---|
| ANS | 0.333 | 0.333 |
| VER | 0.200 | 0.133 |
| REP | 0.167 | 0.167 |
| REV | 0.167 | 0.167 |
| DELIB | 0.133 | 0.133 |
| DEF | **0.000** | 0.067 |
| REORD | **0.000** | **0.000** |
| UPD | **0.000** | **0.000** |

> **Finding.** Three different things are true here and the draft merges them.
> (i) `REORD` and `UPD` are never *offered* by the scaffolding, so they have no
> training signal; the effective action space is six, and those two should be
> removed from the diagram or given a task. (ii) `DEF` *is* offered (6.7% of
> the eval set) and is never predicted: that is a gate learning failure, not a
> curriculum gap. (iii) The gate reproduces the scaffolding at 0.933 accuracy,
> so the learned policy is close but not equivalent. An earlier version of
> this experiment built its eval set from marker-present claims only, which
> made `DELIB` look dead when it had never been offered a single example;
> the eval set now includes marker-stripped claims.
>
> **The gate does separate the defect types.** Overall accuracy hides this, so
> the run now reports the action distribution *within* each input condition.
> On 20 local repairs the gate chooses REP 1.00 (oracle REP 1.00); on 20
> cross-block repairs it chooses REV 1.00 (oracle REV 1.00). The failure is
> concentrated on bare claims, where accuracy is 0.80 and the gate never
> predicts DEF. This is imitation of the scaffolding policy, not independent
> discovery, but it is enough for the controller to pick a revision scope on
> the evidence rather than always editing more text.
>
> **Head calibration is reported and is good at this scale.** ECE is 0.0039
> for scope and 0.0025 for epistemic; scope accuracy 0.867, epistemic 0.925.
> The value head is not saturated (mean 0.890, sd 0.122), which is the
> head-health check the earlier run lacked.
>
> **Triage is still a null, and the metric is bounded.** Triage scores 24 of
> an attainable 24 against a random mean of 22.4 (permutation p = 0.170, gain
> +1.60 of a possible +1.60). With a 93% solvable base rate and a 20% budget,
> the *ceiling* on this metric is 1.6 marks, so the design could not have
> detected a modest effect; the 100% normalized gain is a ceiling artefact,
> not a success. The "sequential" baseline also scores 24 only because the
> eval set is concatenated docs-then-claims; it is an artefact of arrival
> order and is not evidence about any policy. A usable triage test needs a
> balanced solvable/unsolvable mix and a pre-registered permutation criterion.

---

## 6. Revised compute budget

The original budget was not achievable: 8 ablations × 3 seeds × 12 h of
training exceeds the 27 GPU-hours claimed. The fix is that **most ablations
are inference-time** and need no retraining.

| Stage | Runs | Time each | GPU-hours |
|---|---|---|---|
| Data generation | 1 | CPU | 0 |
| Backbone + Stage A + Stage B | 3 seeds | 2 h | 6 |
| Exp 0 (validity) | 1 | CPU | 0 |
| Exp 1 (invariants) | 3 | CPU | 0 |
| Exp 2 (scope/pressure) | 3 | 20 min | 1 |
| Exp 3 (shuffle, 5 conditions) | 3 | 20 min | 1 |
| Exp 4 (repair grid) | 3 | 30 min | 1.5 |
| Exp 5 (gate/triage) | 3 | 10 min | 0.5 |
| **Total** | | | **~10** |

Ten GPU-hours, three seeds, all ablations at inference. That fits the free
Kaggle quota with room for a second sweep.

Seeds are per-checkpoint: seed 0 uses `checkpoints/<preset>`, seed *k* uses
`checkpoints/<preset>_s<k>`, and results are written per seed. The generated
tables grow a `seed` column when more than one seed is present, so a
three-seed claim is visible rather than asserted. `scripts/kaggle_run.py
--seeds 0,1,2` trains each seed once and then runs every inference-time
ablation against that seed's checkpoint.

---

## 7. New section: threats to validity

Add before Limitations. This is where the honest paper lives.

1. **Surface-solvability.** Scope and epistemic labels are predictable from
   surface form (probes at 1.00 and 0.95). Any claim about internal
   representation is unsupported; only claims about behaviour are.
2. **Repair floor, split into two findings.** (i) Clean-block infill is a
   *budget* artefact: byte-identical restore is 0% at every budget because a
   masked-diffusion infill regenerates rather than copies, while on-thesis
   infill rises from 0% at 75 steps to 60% at 750. (ii) Repair is a *real*
   failure: scored on the blocks the task actually edits, no budget beats a
   no-op, and specificity is 0.000 at every budget (§5.6). The model grid and
   the gate-selected row are therefore reported as a floor, and the oracle
   gate shows the task is solvable. The earlier version of this threat used
   byte-identical restore as the sanity check and would have called a budget
   artefact an architecture limit.
3. **Head collapse.** On empirical claims the value head is a near-constant
   (mean 0.68, sd 0.018 at smoke scale), so its endorsement rate is a
   threshold artefact. NLL diff-in-diff is used instead. At larger scale the
   number to report is the head's standard deviation, not its accuracy.
4. **Dead actions, two kinds.** `REORD` and `UPD` are never offered by the
   curriculum (no training signal); `DEF` is offered and never predicted
   (learning failure). Both are reported; neither is presented as a property
   of the architecture.
5. **Label noise.** `V`, `o`, `τ` carry seeded noise by design. Head
   calibration numbers include it; the noise scale is reported.
6. **Templated parsing.** The interpreter receives a structured spec. A real
   system needs a parser, and parser error is a new failure mode not measured
   here.
7. **Single benchmark.** Every result is a statement about one generator. The
   generator is in the repository so the claim can be checked, not so it can
   be generalized.

---

## 8. What to cut

- **Appendix A.8 (ARC-AGI-2).** Unrelated to every claim, and it competes for
  the same GPU quota.
- **Stage C.** Keep the specification, drop the implication that it was run.
- **§6.1 items 1 and 2 as written.** Rewrite per §5.2 above; the mask is not
  what does the work for adjudicable claims.
- **"~18M parameters".** The measured count is **6.70M** at the `paper` preset
  (172.9k at `smoke`, 1.26M at `dev`), from `system.n_params()` and
  `scripts/check_presets.py`. The number depends on the corpus-derived
  vocabulary, so report the measured count, not the design target. The check
  script also fails if any parameter receives no gradient, which is how two
  dead parameters (a `log_sigma` scale and a `to_vocab` projection) were found
  and removed.
- **Any 2026 citation not personally verified.** Fabricated references are
  unrecoverable. Check LaDiR, DiffuReason, Engram, and STEM before submission.

---

## 9. Pre-registration block

Commit this before running the `paper` preset. It goes in the repository, not
in the paper.

```yaml
preset: paper
seeds: [0, 1, 2]
primary_metric: paired bootstrap mean difference, 95% CI

predictions:
  exp0.scope_probe:
    expect: ">= 0.90"          # and if so, scope claims are behavioural only
  exp2.interpreter_dominance:
    expect: "1.00 before and after"
    falsified_if: "< 1.00"
  exp2.unguarded_absorption:
    expect: "NLL diff-in-diff < 0"
    falsified_if: ">= 0"
  exp3.full_vs_shuffled:
    expect: "CI excludes 0, full better"
    falsified_if: "CI includes 0"   # report null + MDE, not 'decorative'
  exp3.oracle_control:
    expect: "oracle < full"
    required_for: "interpreting exp3 at all"
  exp4.interaction:
    expect: "> 0"
    precondition: "infill_exact >= 0.2"
  exp5.triage:
    expect: "permutation p < 0.05 AND normalized gain >= 0.2"
    falsified_if: "p >= 0.05 or gain < 20% of attainable"
    note: "bootstrap CI on the random mean is not the test; it is too permissive
           when the solvable/unsolvable mix bounds the metric"

decision_rules:
  - "If exp0 fails, all downstream claims are reported as exploratory."
  - "If the infill precondition fails, exp4 reports a floor, not a conclusion."
  - "Every null result is reported with its minimum detectable effect."
```

> **Status at smoke scale (recorded, not re-specified).** The oracle positive
> control in Exp 3 failed (Δ = +0.016 vs `full`), so the shuffle comparison is
> uninterpretable and is reported as such. Exp 5's triage null has p = 0.65.
> Both were predicted to succeed. Recording the failures here is what keeps the
> pre-registration from becoming a description of the results.

---

## 10. The strongest defensible version of this paper

If you want one sentence for the cover letter:

> We built a small, fully reproducible testbed in which generation and
> commitment are separate mechanisms, and we report which of the architecture's
> claimed benefits survive contact with measurement — including the ones that
> do not.

That is a publishable workshop paper. It is honest, the code runs in twenty
seconds, and every number traces to a JSON file. The alternative — six tables
of predicted values with `TO BE MEASURED` filled in by a single seed — is not.

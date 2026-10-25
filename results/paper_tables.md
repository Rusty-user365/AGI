# Generated results tables

Every number below is read from `results/*.json`. Do not edit.

A `seed` column appears when more than one seed is present. Seeds present: 0, 1.

## Table 0 — Benchmark validity (must pass before anything else)

| seed | task | probe | majority |
|---|---|---|---|
| 0 | scope (marker present) | 1.000 | 0.2 |
| 0 | scope (marker stripped) | 0.725 | 0.2 |
| 0 | epistemic | 0.95 | 0.775 |
| 1 | scope (marker present) | 1.000 | 0.2 |
| 1 | scope (marker stripped) | 0.725 | 0.2 |
| 1 | epistemic | 0.95 | 0.775 |

- [s0] exact duplicate rate test→train: 0.11
- [s1] exact duplicate rate test→train: 0.11
- [s0] Scope is surface-solvable: a linear probe reaches 1.00. The scope head cannot be claimed to represent scope; at best it learns the marker token.
- [s0] 11.0% of test documents are exact duplicates of training documents. Deduplicate before reporting repair results.
- [s0] Model scope head is -0.800 relative to the probe; a negative gap means the architecture is worse than a linear probe.
- [s0] Epistemic status is also surface-solvable (probe 0.95). The head is recovering template form, not epistemic state.
- [s1] Scope is surface-solvable: a linear probe reaches 1.00. The scope head cannot be claimed to represent scope; at best it learns the marker token.
- [s1] 11.0% of test documents are exact duplicates of training documents. Deduplicate before reporting repair results.
- [s1] Model scope head is -0.300 relative to the probe; a negative gap means the architecture is worse than a linear probe.
- [s1] Epistemic status is also surface-solvable (probe 0.95). The head is recovering template form, not epistemic state.

## Table 0b — Editor gate (one-block infill)

Gate: on-thesis rate >= 0.20. `test` and `unseen` are separate runs and must not be pooled. Wilson 95% intervals.

_(not run)_

## Table 1 — Design invariants (not empirical claims)

| seed | invariant | result |
|---|---|---|
| 0 | null property (M=empty) | yes |
| 0 | negative control detects leak | yes |
| 0 | rollback restores bit-identity | yes |
| 0 | memory has an effect when populated | yes |
| 0 | gradient masking (50 repeats) | yes |
| 0 | masking has power | yes |
| 0 | side channel isolated from vocab | yes |
| 1 | null property (M=empty) | yes |
| 1 | negative control detects leak | yes |
| 1 | rollback restores bit-identity | yes |
| 1 | memory has an effect when populated | yes |
| 1 | gradient masking (50 repeats) | yes |
| 1 | masking has power | yes |
| 1 | side channel isolated from vocab | yes |


## Table 2 — Scope, pressure, and what the mask is for

| seed | condition | metric | value |
|---|---|---|---|
| 0 | A wrong, before update | not accepted | 1.000 |
| 0 | A wrong, after update | not accepted | 1.000 |
| 0 | B guarded | delta NLL | 0.000 |
| 0 | B unguarded | NLL diff-in-diff | 0.01101 |
| 0 | B unguarded | value diff-in-diff | -0.003645 |
| 0 | C scoped-true | accept rate | 1.000 |
| 0 | C scoped-true | scope stored | 1.000 |
| 0 | C bare-false | reject rate | 1.000 |
| 0 | C bare-false | assumed flag | 1.000 |
| 0 | B head health | value mean | 0.7841 |
| 0 | B head health | value sd | 0.01474 |
| 0 | B head health | endorse rate | 1.000 |
| 0 | D empirical | defer rate | 1.000 |
| 0 | E counting ontology | interpreter matches spec | 1.000 |
| 0 | E counting ontology | system matches spec | 1.000 |
| 0 | E undefined unit | defers | yes |
| 1 | A wrong, before update | not accepted | 1.000 |
| 1 | A wrong, after update | not accepted | 1.000 |
| 1 | B guarded | delta NLL | 0.000 |
| 1 | B unguarded | NLL diff-in-diff | -0.002449 |
| 1 | B unguarded | value diff-in-diff | -0.002843 |
| 1 | C scoped-true | accept rate | 1.000 |
| 1 | C scoped-true | scope stored | 1.000 |
| 1 | C bare-false | reject rate | 1.000 |
| 1 | C bare-false | assumed flag | 1.000 |
| 1 | B head health | value mean | 0.3335 |
| 1 | B head health | value sd | 0.1402 |
| 1 | B head health | endorse rate | 0.125 |
| 1 | D empirical | defer rate | 1.000 |
| 1 | E counting ontology | interpreter matches spec | 1.000 |
| 1 | E counting ontology | system matches spec | 1.000 |
| 1 | E undefined unit | defers | yes |

- [s0] A is a negative result about the architecture: because the exact interpreter outranks the weights, gradient masking is not what prevents absorption of adjudicable false claims. The interpreter is.
- [s0] The mask is load-bearing only for non-adjudicable claims (B), which is where sycophancy can actually occur.
- [s0] B uses 20 consolidation steps at lr=0.2. The effect size is a function of both; report them.
- [s0] UNINTERPRETABLE, NOT FAILED: unguarded absorption is +0.0110 nats (95% CI [-0.0354, +0.0576], MDE 0.0699 at 80% power). The CI includes zero, so this design cannot tell absorption from no absorption. It is NOT evidence that the guard is unnecessary: the guard's invariant (verified-wrong => no gradient) is established separately in exp1, and what fails here is the positive control that would show the guard is load-bearing. The confound is statistical power at this substrate size and step budget, not the editor: consolidation is gradient descent on the plastic adapter and is scored on claim NLL, so infill ability is not on its causal path. Raise n, steps or rank before drawing any conclusion about selective plasticity.
- [s0] The value head has collapsed to a near-constant on empirical claims (mean 0.78, sd 0.0147). It endorses 100% of them because 0.5 sits below the constant, not because it judges them. Endorsement rate is therefore not a usable sycophancy measure at this scale; use the NLL diff-in-diff. This is a head-health finding, not a result.
- [s0] The counting ontology tests scope binding, not arithmetic whose truth changes with context: 2+2=4 in ordinary arithmetic, and the only accepted 'five' is the claim whose counting unit is stated. 'there are five' with no rule is deferred, not guessed.
- [s1] A is a negative result about the architecture: because the exact interpreter outranks the weights, gradient masking is not what prevents absorption of adjudicable false claims. The interpreter is.
- [s1] The mask is load-bearing only for non-adjudicable claims (B), which is where sycophancy can actually occur.
- [s1] B uses 20 consolidation steps at lr=0.2. The effect size is a function of both; report them.
- [s1] UNINTERPRETABLE, NOT FAILED: unguarded absorption is -0.0024 nats (95% CI [-0.0533, +0.0471], MDE 0.0691 at 80% power). The CI includes zero, so this design cannot tell absorption from no absorption. It is NOT evidence that the guard is unnecessary: the guard's invariant (verified-wrong => no gradient) is established separately in exp1, and what fails here is the positive control that would show the guard is load-bearing. The confound is statistical power at this substrate size and step budget, not the editor: consolidation is gradient descent on the plastic adapter and is scored on claim NLL, so infill ability is not on its causal path. Raise n, steps or rank before drawing any conclusion about selective plasticity.
- [s1] The counting ontology tests scope binding, not arithmetic whose truth changes with context: 2+2=4 in ordinary arithmetic, and the only accepted 'five' is the claim whose counting unit is stated. 'there are five' with no rule is deferred, not guessed.

## Table 3 — Side-channel causality

| seed | condition | loss | vs full | repair consistency |
|---|---|---|---|---|
| 0 | full | 3.386 | 0.000 | 0.000 |
| 0 | shuffled | 3.64 | 0.255 | 0.000 |
| 0 | constant | 3.492 | 0.1064 | 0.000 |
| 0 | none | 4.907 | 1.522 | 0.000 |
| 0 | oracle | 3.484 | 0.09846 | 0.000 |
| 1 | full | 3.105 | 0.000 | 0.000 |
| 1 | shuffled | 3.25 | 0.1454 | 0.000 |
| 1 | constant | 3.259 | 0.1549 | 0.000 |
| 1 | none | 3.842 | 0.7371 | 0.000 |
| 1 | oracle | 3.153 | 0.0488 | 0.000 |

- [s0] full vs shuffled: 0.2474, 95% CI [0.1492, 0.3582], MDE 0.1535
- [s0] positive control carries information: NO
- [s1] full vs shuffled: 0.1459, 95% CI [0.106, 0.189], MDE 0.05942
- [s1] positive control carries information: NO
- [s0] The positive control did not beat `full`: the steering vector could not carry information the input lacked. A null shuffle result is therefore uninterpretable and must not be reported as 'reasoning is decorative'.
- [s0] shuffled ~ constant: content is irrelevant, only the presence of a steering vector matters.
- [s1] The positive control did not beat `full`: the steering vector could not carry information the input lacked. A null shuffle result is therefore uninterpretable and must not be reported as 'reasoning is decorative'.

## Table 4 — Revision scope under a matched budget

| seed | condition | anchor | c_edit | c_all | c_target | q_target | quality | blocks_changed | tokens_spent | consistent | improvement | preservation | seam | n | n_quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | local/one_block | 1.000 | 0.7388 | 0.7946 | 0.000 | 0.000 | 0.6375 | 1.000 | 16.73 | 0.000 | 0.6619 | 1.000 | 0.0139 | 40 |  |
| 0 | local/global | 1.000 | 0.000 | 0.2054 | 0.000 | 0.000 | 0.000 | 4.025 | 57.95 | 0.000 | 0.8605 | 1.000 | 0.04413 | 40 |  |
| 0 | local/repeated_local | 1.000 | 0.000 | 0.2054 | 0.000 | 0.000 | 0.000 | 4.025 | 57.95 | 0.000 | 0.8237 | 1.000 | -0.003872 | 40 |  |
| 0 | cross_block/one_block | 1.000 | 0.000 | 0.2054 | 0.000 | 0.000 | 0.000 | 1.000 | 14.5 | 0.000 | -0.2038 | 1.000 | 0.03548 | 40 |  |
| 0 | cross_block/global | 1.000 | 0.000 | 0.2054 | 0.000 | 0.000 | 0.000 | 4.025 | 55.65 | 0.000 | 0.0287 | 1.000 | -0.007327 | 40 |  |
| 0 | cross_block/repeated_local | 1.000 | 0.000 | 0.2054 | 0.000 | 0.000 | 0.000 | 4.025 | 55.65 | 0.000 | 0.02151 | 1.000 | -0.006225 | 40 |  |
| 0 | local/gate_selected | 1.000 | 0.475 | 0.5821 | 0.000 | 0.000 | 0.4042 | 2.125 | 32.17 | 0.000 | 0.714 | 1.000 | 0.0004237 | 40 |  |
| 0 | cross_block/gate_selected | 1.000 | 0.000 | 0.2054 | 0.000 | 0.000 | 0.000 | 4.025 | 55.65 | 0.000 | 0.0287 | 1.000 | -0.007327 | 40 |  |
| 1 | local/one_block | 1.000 | 0.7325 | 0.7904 | 0.000 | 0.000 | 0.6271 | 1.000 | 13.95 | 0.000 | 0.2735 | 1.000 | -0.02506 | 40 |  |
| 1 | local/global | 1.000 | 0.000 | 0.2096 | 0.000 | 0.000 | 0.000 | 3.9 | 47.45 | 0.000 | 0.1023 | 1.000 | 0.0155 | 40 |  |
| 1 | local/repeated_local | 1.000 | 0.000 | 0.2096 | 0.000 | 0.000 | 0.000 | 3.9 | 47.8 | 0.000 | 0.2676 | 1.000 | -0.01214 | 40 |  |
| 1 | cross_block/one_block | 1.000 | 0.000 | 0.2096 | 0.000 | 0.000 | 0.000 | 1.000 | 12.18 | 0.000 | -0.289 | 1.000 | 0.002835 | 40 |  |
| 1 | cross_block/global | 1.000 | 0.000 | 0.2096 | 0.000 | 0.000 | 0.000 | 3.9 | 45.83 | 0.000 | -0.1824 | 1.000 | 0.0343 | 40 |  |
| 1 | cross_block/repeated_local | 1.000 | 0.000 | 0.2096 | 0.000 | 0.000 | 0.000 | 3.9 | 46.67 | 0.000 | 0.01894 | 1.000 | 0.04927 | 40 |  |
| 1 | local/gate_selected | 1.000 | 0.7838 | 0.8308 | 0.2 | 0.000 | 0.6271 | 0.8 | 11.38 | 0.2 | 0.2507 | 1.000 | -0.02717 | 40 |  |
| 1 | cross_block/gate_selected | 1.000 | 0.000 | 0.2096 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 40 |  |

- [s0] split: test
- [s0] oracle solvability gate: yes
- [s0] infill sanity (exact block restore): 0.000
- [s0] informative (infill >= 0.2): NO
- [s0] local/global_minus_one_block: c_edit diff -0.7388 (95% CI [-0.7563, -0.7204])
- [s0] cross_block/global_minus_one_block: c_edit diff 0.000 (95% CI [0.000, 0.000])
- [s0] cross_block/repeated_local_minus_global: c_edit diff 0.000 (95% CI [0.000, 0.000])
- [s0] cross_block/repeated_local_minus_one_block: c_edit diff 0.000 (95% CI [0.000, 0.000])
- [s0] local/repeated_local_minus_global: c_edit diff 0.000 (95% CI [0.000, 0.000])
- [s0] scope interaction: 0.7388; prediction holds: NO

- [s1] split: test
- [s1] oracle solvability gate: yes
- [s1] infill sanity (exact block restore): 0.000
- [s1] informative (infill >= 0.2): NO
- [s1] local/global_minus_one_block: c_edit diff -0.7325 (95% CI [-0.7492, -0.7158])
- [s1] cross_block/global_minus_one_block: c_edit diff 0.000 (95% CI [0.000, 0.000])
- [s1] cross_block/repeated_local_minus_global: c_edit diff 0.000 (95% CI [0.000, 0.000])
- [s1] cross_block/repeated_local_minus_one_block: c_edit diff 0.000 (95% CI [0.000, 0.000])
- [s1] local/repeated_local_minus_global: c_edit diff 0.000 (95% CI [0.000, 0.000])
- [s1] scope interaction: 0.7325; prediction holds: NO

- [s0] Context filter: 1 of 41 scanned documents were skipped because the clean, local or cross-block variant exceeds the model's 96-token context. The filter is applied to all three variants together, so the defect conditions stay paired. Kept: {'local': 40, 'cross_block': 40}. This is a selection effect and is reported.
- [s0] Infill sanity: the backbone produces a parseable, on-thesis block only 0% of the time (parses 0%; no-op baseline parses 100% and is on-thesis 100%; byte-identical copy 0%). Below the 0.2 ability gate, every repair number is at the floor; the scope comparison is NOT decidable with this editor. The oracle gate above shows the task itself is solvable, so this is a model failure, not a task failure. Report the floor and do not interpret the grid or the gate-selected row.
- [s0] On the edited blocks, no method beats doing nothing (c_target and q_target are at or below the no-op baseline in every condition). The document-level c_edit/quality numbers therefore come from blocks the method left untouched, not from a repair. Do not read the model grid as an editor result.
- [s1] Infill sanity: the backbone produces a parseable, on-thesis block only 0% of the time (parses 0%; no-op baseline parses 100% and is on-thesis 100%; byte-identical copy 0%). Below the 0.2 ability gate, every repair number is at the floor; the scope comparison is NOT decidable with this editor. The oracle gate above shows the task itself is solvable, so this is a model failure, not a task failure. Report the floor and do not interpret the grid or the gate-selected row.
- [s1] On the edited blocks, no method beats doing nothing (c_target and q_target are at or below the no-op baseline in every condition). The document-level c_edit/quality numbers therefore come from blocks the method left untouched, not from a repair. Do not read the model grid as an editor result.

## Table 5 — Gate usage, overrides, triage

| seed | action | model | oracle |
|---|---|---|---|
| 0 | ANS | 0.3333 | 0.3333 |
| 1 | ANS | 0.3333 | 0.3333 |
| 0 | DEF | 0.000 | 0.06667 |
| 1 | DEF | 0.06667 | 0.06667 |
| 0 | DELIB | 0.1333 | 0.1333 |
| 1 | DELIB | 0.1333 | 0.1333 |
| 0 | REP | 0.1667 | 0.1667 |
| 1 | REP | 0.1583 | 0.1667 |
| 0 | REV | 0.1667 | 0.1667 |
| 1 | REV | 0.175 | 0.1667 |
| 0 | VER | 0.2 | 0.1333 |
| 1 | VER | 0.1333 | 0.1333 |

- [s0] gate accuracy vs scaffolding: 0.9333

- [s0] overrides: {'surprise': 0.0, 'risk_ladder': 0.0, 'none': 1.0}

- [s0] cost per correct: model 3.828, oracle 4.086

- [s0] gate action diversity **within** each condition (a mixed eval set can look diverse only because the oracle is diverse):

| condition | n | acc | model_top | model_top_share | model_actions | oracle_actions |
|---|---|---|---|---|---|---|
| clean doc | 40 | 1.000 | ANS | 1.000 | 1 | 1 |
| claim | 40 | 0.8 | VER | 0.6 | 2 | 3 |
| repair: local | 20 | 1.000 | REP | 1.000 | 1 | 1 |
| repair: cross_block | 20 | 1.000 | REV | 1.000 | 1 | 1 |

- [s0] gate separates local from cross-block: yes

| ordering | marks |
|---|---|
| sequential | 24 |
| triage | 24 |
| random | 22.4 |
| hardest first | 24 |

- [s1] gate accuracy vs scaffolding: 0.975

- [s1] overrides: {'surprise': 0.0, 'risk_ladder': 0.0, 'none': 1.0}

- [s1] cost per correct: model 3.407, oracle 3.363

- [s1] gate action diversity **within** each condition (a mixed eval set can look diverse only because the oracle is diverse):

| condition | n | acc | model_top | model_top_share | model_actions | oracle_actions |
|---|---|---|---|---|---|---|
| clean doc | 40 | 1.000 | ANS | 1.000 | 1 | 1 |
| claim | 40 | 1.000 | DELIB | 0.4 | 3 | 3 |
| repair: local | 20 | 0.9 | REP | 0.9 | 2 | 1 |
| repair: cross_block | 20 | 0.95 | REV | 0.95 | 2 | 1 |

- [s1] gate separates local from cross-block: yes

| ordering | marks |
|---|---|
| sequential | 24 |
| triage | 24 |
| random | 22.41 |
| hardest first | 16 |

- [s0] Never offered by the curriculum: ['REORD', 'UPD']. These have no training signal; report the effective action space, not the nominal eight.
- [s0] Offered but never predicted: ['DEF']. This is a gate learning failure, not a curriculum gap.
- [s0] Triage does not establish an advantage over random (triage 24, random mean 22.4, CI [22.3, 22.5]; gain +1.60 of an attainable 1.60 = 100%; permutation p=0.170; budget 24 of 120, solvable base rate 0.93). Report the null, and note the ceiling: with this mix the metric can only move by 1.6 marks.
- [s0] 'Sequential' scores the maximum (24) because the eval set is concatenated docs-then-claims; it is an artefact of arrival order, not evidence about any policy. Compare triage to random only.
- [s1] Never offered by the curriculum: ['REORD', 'UPD']. These have no training signal; report the effective action space, not the nominal eight.
- [s1] Triage does not establish an advantage over random (triage 24, random mean 22.4, CI [22.3, 22.5]; gain +1.59 of an attainable 1.59 = 100%; permutation p=0.196; budget 24 of 120, solvable base rate 0.93). Report the null, and note the ceiling: with this mix the metric can only move by 1.6 marks.
- [s1] 'Sequential' scores the maximum (24) because the eval set is concatenated docs-then-claims; it is an artefact of arrival order, not evidence about any policy. Compare triage to random only.

## Table 6 — Structural guard (C1), independent of the editor

C1 is the structural claim: a verified-wrong example cannot move the plastic substrate and cannot create a factual memory entry. It does not depend on the learned editor. C2 (does the guard improve held-out behaviour?) is a separate claim and is unresolved.

| seed | check | verdict |
|---|---|---|
| 0 | A | pass |
| 0 | B | pass |
| 0 | D | pass |
| 0 | C | pass |
| 0 | control: unguarded leaks | yes |
| 0 | control: verified-right updates | yes |
| 1 | A | pass |
| 1 | B | pass |
| 1 | D | pass |
| 1 | C | pass |
| 1 | control: unguarded leaks | yes |
| 1 | control: verified-right updates | yes |

- [s0] C1 pass: yes; ambiguous: NO; instrument valid: yes

- [s0] diagnostic A2 (not pre-registered): mixed batch vs verified-right-only max gradient diff 0.000 (holds: yes)

- [s0] diagnostic B2 (not pre-registered): AdamW weight decay moves the parameter when grad is a zero tensor: yes; when grad is None: NO

- [s1] C1 pass: yes; ambiguous: NO; instrument valid: yes

- [s1] diagnostic A2 (not pre-registered): mixed batch vs verified-right-only max gradient diff 0.000 (holds: yes)

- [s1] diagnostic B2 (not pre-registered): AdamW weight decay moves the parameter when grad is a zero tensor: yes; when grad is None: NO

- [s0] C1 passes with a valid instrument: verified-wrong examples leave the plastic substrate bit-identical and create no memory entry, while the gate still receives a rejection gradient. This is a structural guarantee about the pipeline and is independent of whether the editor can generate good repairs.
- [s0] B2 (not pre-registered): under AdamW with weight decay, a *zero-tensor* gradient still moves the parameter, because decay is applied whenever grad is not None. Masking leaves the plastic grad as None, which AdamW skips, so the guard holds on this path — but a production optimizer should exclude the plastic parameters from weight decay rather than rely on that.
- [s0] C2 (does the guard improve held-out behaviour?) is a different claim and is unresolved: exp2's NLL diff-in-diff is within its minimum detectable effect at this scale. C1 does not depend on it.
- [s1] C1 passes with a valid instrument: verified-wrong examples leave the plastic substrate bit-identical and create no memory entry, while the gate still receives a rejection gradient. This is a structural guarantee about the pipeline and is independent of whether the editor can generate good repairs.
- [s1] B2 (not pre-registered): under AdamW with weight decay, a *zero-tensor* gradient still moves the parameter, because decay is applied whenever grad is not None. Masking leaves the plastic grad as None, which AdamW skips, so the guard holds on this path — but a production optimizer should exclude the plastic parameters from weight decay rather than rely on that.
- [s1] C2 (does the guard improve held-out behaviour?) is a different claim and is unresolved: exp2's NLL diff-in-diff is within its minimum detectable effect at this scale. C1 does not depend on it.

## Table 7 — Editor floor: training budget vs capacity

Two questions were conflated by the earlier byte-identical gate. `on_thesis` is clean-block infill ability; `local_*`, `cross_*` and `partial_*` are scored on the blocks the task actually edits, with the no-op baseline shown. Copy fidelity (`exact`) is a diagnostic only.

**Seed 0 only, n=10 tasks per defect.** This is a budget curve, not a multi-seed result; the non-monotonicity between 750 and 1500 steps is not interpreted.

| budget | opt_steps | exact | on_thesis | specific | local_c | local_q | local_noop_c | local_noop_q | cross_c | cross_q | cross_noop_c | partial_c | partial_q | partial_noop_c |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2+1 | 75 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | n/a | n/a |
| 10+5 | 375 | 0.000 | 0.4 | 0.000 | 0.2 | 0.000 | 1.000 | 0.000 | 0.04 | 0.000 | 0.000 | n/a | n/a | n/a |
| 20+10 | 750 | 0.000 | 0.6 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.07333 | 0.000 | 0.000 | n/a | n/a | n/a |
| 40+20 | 1500 | 0.000 | 0.5 | 0.000 | 0.3 | 0.000 | 1.000 | 0.000 | 0.05 | 0.000 | 0.000 | n/a | n/a | n/a |

- split: test
- ability rises with budget: yes; specificity rises: NO; local repair rises: NO; cross-block repair rises: NO; partial cross-block repair rises: n/a

- BUDGET, NOT CAPACITY: on-thesis infill rises to 60% at 20+10 (750 optimizer steps), from 0% at 2+1 (75 steps). The smoke preset's 75 steps are below the ability floor; the earlier 0.000 was a property of the budget, not of the architecture.
- COHERENCE WITHOUT SPECIFICITY: the editor produces parseable, on-thesis blocks but never a specific reason (quality 0.000 at every budget). This is a narrower and more useful result than 'the editor cannot edit': it can restore label consistency, and it cannot restore the specificity that defines the local defect. A local-repair claim needs the specificity number, not c_edit.
- On the actual local repair task, quality stays at the no-op baseline (0.000): the model does not repair specificity even when it can infill a clean block. Do not read c_edit on local defects as a repair result; a no-op already scores 1.000 there.
- On the actual cross-block task, c_edit stays at its no-op floor: clean-block infill ability does not transfer to revising every editable block against a locked intro. The editor is above the infill floor and still below the repair floor.
- Byte-identical restore is reported as a diagnostic only. A masked-diffusion infill regenerates the span, so exact copy is ~0 by construction and cannot be the precondition for interpreting the repair grid. exp4 now gates on on-thesis output and reports both.

## Table 8 — CGR vs LoRA-only on unseen theses

Both conditions use the same backbone, tokenizer, tasks, decoder, edit scopes and token/block budgets. The CGR condition has 8.15M trainable parameters and the LoRA-only control has 442k; this is a matched-task comparison, **not** a matched-compute or matched-capacity one. The gate and a DAG do not participate in either repair path, so this table cannot validate them.

_(not run)_

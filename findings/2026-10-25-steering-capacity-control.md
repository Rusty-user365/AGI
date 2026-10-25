# The CGR advantage does not survive a capacity-matched control (2026-10-25)

## Question

Does the CGR checkpoint repair better than a control that received the same
training, the same parameter budget and the same losses, differing only in
whether `cond` carries information about the document?

This is the pre-registered decision run. Both arms were resumed from the 0.380
checkpoint with FiLM live, trained 2 Stage-A epochs each, 22,324,637 trainable
parameters, identical corpus, seed and schedule.

- `cgr` arm — `diffugpt_5k_e10_cgr_fixup_reasoner`, evaluated with the reasoner
  path (`cond = f(document)`)
- `baseline` arm — `diffugpt_5k_e10_cgr_fixup_constant`, evaluated reasoner-free
  (`cond = 0`)

Artifact: `results/exp8_fixup_steering_test.json`. split=`test`, n=200, seed=0,
steps=8, 52 min. Produced by the code revision before `--cgr-cond-mode` was
added, so the JSON has no `cond_mode` field; the modes above are the ones that
revision hard-coded.

## The pre-registered condition: null

| `local / one_block` | CGR | constant control | paired diff | 95% CI | sig |
|---|---|---|---|---|---|
| thesis restored | 0.385 | 0.350 | **+0.035** | **[-0.045, +0.115]** | **no** |
| quality target | n/a | n/a | -0.050 | [-0.105, +0.005] | no |
| edit quality change | -0.504 | -0.349 | **-0.156** | **[-0.240, -0.065]** | **yes, CGR worse** |

The decision rule required the paired interval on `c_target` to exclude zero on
`test`, with the same sign on `unseen`. It does not exclude zero. **The steering
claim fails**, and per the rule no `unseen` confirmation is required to say so.

The one significant difference on this condition is a **quality loss**: the
steered arm damages the document more (-0.156) while restoring the thesis no
more reliably than the control.

## Why this overturns the previous headline

The earlier paired run (`exp8_architecture_ablation_test.json`) compared the CGR
checkpoint against the LoRA-only checkpoint, which has **442,368** trainable
parameters against CGR's **8,149,661**, and reported +0.135 [0.060, 0.210] on
this same condition. That number was real but it was not attributable to the
conditioning path.

| `local/one_block`, `c_target` | value |
|---|---|
| LoRA-only checkpoint (0.44M trainable) | 0.245 |
| constant-conditioning control (22.3M trainable) | **0.350** |
| CGR checkpoint (22.3M trainable) | 0.385 |
| paired CGR - LoRA-only | +0.135 [0.060, 0.210], significant |
| paired CGR - constant control | **+0.035 [-0.045, 0.115], not significant** |

Raising the baseline's parameter budget from 0.44M to 22.3M and giving it the
same training moved it from 0.245 to 0.350. The CGR arm moved from 0.380 to
0.385. **Essentially all of the apparent CGR advantage was capacity, not
conditioning.**

This is the exact overclaim the two-arm design existed to prevent, and it would
have been invisible without the control.

## What the steering path does do: a trade, on other conditions

Steering is not inert. It changes behaviour measurably, but off the
pre-registered condition:

| Condition | `c_target` diff | 95% CI | `improvement` diff | 95% CI |
|---|---|---|---|---|
| cross_block / global | **+0.058** | [+0.022, +0.092] | **-0.102** | [-0.147, -0.057] |
| cross_block_partial / global | **+0.037** | [+0.002, +0.075] | **-0.116** | [-0.161, -0.072] |
| cross_block / one_block | **-0.014** | [-0.023, -0.004] | -0.042 | [-0.086, +0.002] |
| cross_block_partial / one_block | **-0.019** | [-0.034, -0.005] | +0.007 | [-0.037, +0.053] |

On global rewrites the steered arm restores the thesis more often (+0.058,
+0.037) and damages quality more (-0.102, -0.116). On the near-chance
single-block cross-block conditions it is slightly worse on `c_target`
(-0.014, -0.019, magnitudes small enough to be practically irrelevant).

So the honest description is: **per-document conditioning buys thesis
restoration on large rewrites and pays for it in document quality.** It does not
improve the gate condition, and it does not beat a capacity-matched control
there.

These four conditions are **exploratory**, not pre-registered, and `test` is now
a selection set. A directional confirmation on `unseen` would be needed before
any of them is stated as a finding.

## Addendum (same day): the control in its own distribution

The constant arm was re-evaluated with `--cond-mode learned`, which feeds the
`const_cond` it was trained with instead of `cond=0`:

| Split | constant, `cond=0` | constant, `cond=const_cond` | CGR steered |
|---|---|---|---|
| test | 0.350 | **0.380** | 0.385 |
| unseen | 0.210 | **0.155** (below floor) | 0.290 |

On `test` the fair control is **0.380**, so CGR's 0.385 is **+0.005**
(marginal, unpaired; the paired run against this control has not been done). The
retraction above is strengthened: the control had been handicapped by
out-of-distribution evaluation, and once that is corrected the decision split
shows no difference at all.

On `unseen` the control scores 0.155 in its own mode, below the 0.20 floor and
worse than no conditioning at all. It is degenerate on that split and cannot
serve as a comparator, so the `unseen` positive is uninterpretable rather than
disproven. See
[unseen-confirmation](2026-10-26-unseen-confirmation.md).

## What this does not support

- No claim that the CGR system repairs better than LoRA-only task adaptation.
  The +0.135 was capacity.
- No claim that steering helps on the gate condition. The interval includes 0.
- No claim that steering is useless in general: it measurably changes behaviour,
  with a consistent quality cost. That is a real trade, not a null.
- No capacity- or compute-matched superiority claim in either direction: the
  arms are matched here, but only to each other.
- No Gate/DAG claim, and no unseen-thesis claim from this run.

## Consequence for the repo

The headline in `2026-10-25-exp8-paired-test.md` and the "CGR mixed-mask 0.380"
framing in `2026-10-25-editor-gate-n200.md` describe a difference that is now
explained by parameter budget. Both entries are annotated rather than rewritten,
per the convention in `findings/README.md`.

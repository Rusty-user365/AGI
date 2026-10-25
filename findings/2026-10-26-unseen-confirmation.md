# Unseen confirmation: the global trade replicates, the gate condition does not (2026-10-26)

## Predictions, written before the run

For `cross_block/global` and `cross_block_partial/global` on `unseen`:

1. `c_target` higher for the steered arm
2. `improvement` lower for the steered arm

## Result: both predictions held on both conditions

Artifact: `results/exp8_fixup_steering_unseen.json`, split=`unseen`, n=200,
seed=0, steps=8. CGR arm `cond=reasoner`, baseline arm `cond=none`.

| Condition | `c_target` diff | 95% CI | `improvement` diff | 95% CI |
|---|---|---|---|---|
| cross_block / global | **+0.050** | [+0.014, +0.086] | **-0.391** | [-0.439, -0.344] |
| cross_block_partial / global | **+0.076** | [+0.033, +0.120] | **-0.407** | [-0.453, -0.363] |

The `test` values were +0.058 and +0.037 with quality costs -0.102 and -0.116.
Same sign, same ordering, on held-out theses. **The trade is real**: per-document
conditioning buys thesis restoration on large rewrites and pays for it in
document quality. Four predictions, four hits.

## The gate condition is split-dependent

`local / one_block`, the pre-registered condition:

| Split | `c_target` diff | 95% CI | `improvement` diff | 95% CI |
|---|---|---|---|---|
| test (decision set) | +0.035 | [-0.045, +0.115] | **-0.156** | [-0.240, -0.065] |
| unseen (confirmation) | **+0.080** | **[+0.015, +0.140]** | -0.025 | [-0.095, +0.040] |

The point estimates are the same sign; only the `test` interval includes zero.
On `unseen` the effect is significant and, unlike `test`, carries no significant
quality cost.

Harness check: exp8's arm means match the standalone gate runs exactly
(0.290 vs 58/200, 0.210 vs 42/200), so this is not a scoring discrepancy.

## This makes my retraction too strong, and that has to be said plainly

Yesterday's entry concluded "the CGR advantage was capacity, not the
conditioning path" and "the steering claim fails". Both sentences need
qualifying:

- **The LoRA-capacity part stands.** On `test` the constant-conditioning
  control at 22.3M scored 0.350 against LoRA-only's 0.245 at 0.44M, and the
  paired CGR-minus-LoRA +0.135 collapsed to +0.035 against the control. Raising
  the baseline's budget accounts for that gap.
- **"Steering adds nothing measurable" does not stand.** On `unseen` the same
  paired comparison is +0.080 with an interval excluding zero.
- **"The steering claim fails" is true only in the procedural sense** that the
  pre-registered decision statistic was the `test` interval and it included
  zero. A rule that fails is a failed rule; it does not become a discovery
  because a different split came out positive afterwards.

The accurate summary is **inconclusive and split-dependent**, not null and not
established. Two splits, same condition, one null and one positive. Pooling
them was forbidden in advance and stays forbidden.

## Why the control is degenerate on `unseen` — resolved

`local / one_block`, `c_target`:

| Split | LoRA-only (0.44M) | constant, `cond=0` | constant, `cond=const_cond` | CGR steered |
|---|---|---|---|---|
| test | 0.245 | 0.350 | **0.380** | 0.385 |
| unseen | 0.250 | 0.210 | **0.155** | 0.290 |

The constant arm evaluated in its own distribution (`--cond-mode learned`,
`results/gate_fixup_constant_*_learned.json`):

- **test: 76/200 = 0.380** [0.316, 0.449] — better than its free-mode 0.350
- **unseen: 31/200 = 0.155** [0.111, 0.212] — worse than its free-mode 0.210,
  and **below the pre-registered 0.20 floor**

Two consequences, and they point in opposite directions:

**On `test` the steering effect disappears.** CGR steered 0.385 against a
fairly-evaluated control at 0.380 is **+0.005**. These are two separate gate
runs, so this is a marginal comparison, not a paired interval; the paired run
against the in-distribution control (`exp8 --baseline-cond-mode learned`) has
not been executed. But a 0.005 gap between marginal rates at n=200 cannot
support an effect regardless of pairing. The +0.035 in the paired comparison
was measured against a control handicapped by out-of-distribution evaluation.
Give the control its own conditioning and the decision split shows no
difference at all.

**On `unseen` the control is not a valid comparator.** It scores 0.155 in its
own mode — below the floor the whole project is gated on — while scoring 0.210
with no conditioning at all. A control whose own conditioning makes it worse
than no conditioning is misfiring, not merely underperforming. Comparing
against 0.155 produces +0.135 and comparing against 0.210 produces +0.080;
neither number can be attributed to steering, because the baseline is a
degenerate model on that split.

So the `unseen` positive does not survive. It is not that the effect was
disproven there; it is that the comparison has no valid denominator.

## Where this leaves the claim

- **Decision split (`test`): null, and now more clearly so.** Against the
  in-distribution control, +0.005. The earlier retraction was right for the
  wrong reason: not that capacity explains everything, but that the control had
  been evaluated outside its training distribution.
- **Confirmation split (`unseen`): uninterpretable.** The control is
  degenerate, so the split cannot adjudicate.
- **The global-scope trade replicated directionally, but against a free-mode
  control.** On `test` that control was itself handicapped (0.350 vs 0.380), so
  the trade as measured is if anything overstated there. Settling it requires
  exp8 with `--baseline-cond-mode learned`.

## Methodological finding, which is the durable one

The same control, on the same split, moves the answer by 0.030 (`test`) and
0.055 (`unseen`) depending only on whether it is evaluated in the distribution
it was trained in. A capacity-matched control is not matched until its
evaluation mode is matched too. This is worth more than the steering result
itself.

## What this does not support

- No pooled or averaged estimate across splits.
- No pre-registered claim: the decision set was `test`, and it was null.
- No architecture-versus-LoRA claim; that comparison remains confounded by
  parameter count in both directions.
- No Gate/DAG claim.
- Nothing about the 470M or 1.7B backbone plans; those are untested here.

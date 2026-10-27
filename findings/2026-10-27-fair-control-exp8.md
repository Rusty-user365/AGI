# Fair-control exp8: the gate null holds, one global-scope trade survives (2026-10-27)

Artifact: `results/exp8_fixup_steering_test_faircontrol.json`
Run: `--split test --n 200 --steps 8 --cgr-cond-mode reasoner --baseline-cond-mode learned`

## The question this run answers

The previous exp8 compared the reasoner arm against the constant arm evaluated
with `cond=0`. That control was out of distribution: it was trained with a
learned constant vector and scored 0.380 when fed its own vector versus 0.350
when fed zeros. So the +0.035 steering effect on the gate condition was measured
against a baseline that had been handicapped by evaluation mode.

This run feeds the control the `const_cond` it was trained with. Same
checkpoints, same schedule, same corpus, same seed. The only thing that changes
is that the baseline is no longer evaluated out of distribution.

## Measurement

`local/one_block` is the pre-registered gate condition.

| Condition | Metric | reasoner | constant | diff [95% CI] |
|---|---|---|---|---|
| local/one_block | thesis restored | 0.385 | 0.380 | **+0.005 [-0.070, +0.080]** |
| local/one_block | quality target | n/a | n/a | **-0.065 [-0.120, -0.010] SIG** |
| local/one_block | edit quality change | -0.504 | -0.331 | **-0.174 [-0.253, -0.089] SIG** |
| local/global | thesis restored | 0.275 | 0.370 | -0.095 [-0.175, -0.010] SIG |
| cross_block/global | thesis restored | 0.573 | 0.522 | **+0.051 [+0.019, +0.084] SIG** |
| cross_block/global | quality target | n/a | n/a | +0.048 [+0.019, +0.078] SIG |
| cross_block/global | edit quality change | -0.193 | -0.123 | -0.069 [-0.107, -0.032] SIG |
| cross_block_partial/global | thesis restored | 0.592 | 0.559 | +0.033 [-0.001, +0.068] NS |
| cross_block_partial/global | edit quality change | -0.165 | -0.082 | -0.083 [-0.121, -0.044] SIG |
| cross_block/one_block | thesis restored | 0.033 | 0.042 | -0.009 [-0.020, +0.000] NS |

Headline from `report_exp8.py`: `c_target` differs significantly in 2 of 6
conditions, one favouring each arm.

## What this settles

**1. The gate condition is null, and now for a cleaner reason.** +0.005
[-0.070, +0.080] with the control in its own distribution. The earlier +0.035
was the control's evaluation mode, not steering. This is the third independent
statement of the same null (0.385 vs 0.350 free, 0.385 vs 0.380 learned, and the
capacity-matched control at 0.350), and it is the one that removes the last
objection to it.

**2. Steering is not free even where it is null.** On the gate condition the
reasoner arm degrades edit quality by 0.174 [-0.253, -0.089] and misses the
quality target by 0.065. Equal accuracy, worse text. A null on the primary
metric is not a null overall.

**3. One of the two global-scope trades survives the fair control.**
`cross_block/global` keeps +0.051 [+0.019, +0.084] thesis restoration against a
properly evaluated control, at a cost of -0.069 quality. `cross_block_partial/global`
falls to +0.033 [-0.001, +0.068], not significant. Against the free-mode control
these were +0.058 and +0.037; the weaker of the two does not survive.

**4. The side channel can hurt.** `local/global` is -0.095 [-0.175, -0.010] for
the reasoner arm. This is the first condition where steering is significantly
*worse* than the matched control on the primary metric, and it should be
reported alongside the positive one rather than omitted.

**5. The evaluation-mode effect replicates exactly as predicted.** The control
moves 0.350 to 0.380 on the gate condition from evaluation mode alone. That was
the prediction from the earlier finding; it is now confirmed on the same
artifact family.

## What this does not support

- **No steering benefit on the gate condition.** It is null on accuracy and
  negative on quality.
- **No unseen confirmation of the surviving global-scope trade.** The constant
  arm scores 0.155 on `unseen` in its own mode, below the 0.20 floor. A
  comparison against a degenerate control cannot adjudicate anything, so the
  `cross_block/global` result stands on `test` only.
- **And `test` is the selection split.** It chose the 0.380 checkpoint. A
  test-only significant result is therefore selection-contaminated, and this is
  the specific reason it cannot be promoted to a claim.
- **No architecture claim over LoRA-only task adaptation.** Unchanged.
- **Nothing about any backbone other than this 138M one.**

## Consequence for the next experiment

The fair-control design cannot close the `cross_block/global` trade, because the
control it needs is degenerate on the confirmation split. Closing it needs a
control that is non-degenerate on `unseen`, which means training a control with
the same schedule on the unseen distribution rather than reusing this one. That
is a new training run, not a new evaluation flag.

Recorded here so the trade is not mistaken for established while the
confirmation is missing.

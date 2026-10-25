# Fix-up arms: gate matrix with a live side channel (2026-10-25)

## Question

Two arms were resumed from the 0.380 checkpoint with the FiLM side channel
unfrozen, identical in schedule, data, parameter count and losses, differing
only in where `cond` comes from:

- `reasoner` — the per-document reasoner vector (the CGR claim)
- `constant` — one learned vector for the whole batch (`cond_mode=learned`),
  carrying no information about the input

Both arms trained 2 Stage-A epochs, 1,719 steps each, 22,324,637 trainable
parameters. The `constant` arm is what makes a difference attributable to
steering instead of to the extra training.

## Training

| | reasoner | constant |
|---|---|---|
| A0 loss | 1.0286 | 1.0859 |
| A1 loss | 0.9122 | 0.9508 |
| wall clock | 4,025 s | 3,797 s |
| `film_l2` | 10.25 | 8.00 |
| `cond_probe_delta` | 15.09 | 8.52 |
| `side_channel_live` | true | true |

Both arms moved FiLM off its zero init and both have a causally live `cond`
path at training time, so no gate difference can be explained by one arm having
a working wire and the other not.

Caveat on the control: `reasoner_l2_delta` is 2.74 in the constant arm, so the
reasoner module still moved there, driven by the other heads. The control is
"`cond` carries no per-document information", not "reasoner disabled".

## Gate, n=200, steps=8, seed=0

Gate definition unchanged: one-block infill on `local`, `on_thesis >= 0.20`.

| Arm | Split | Path | `on_thesis` | 95% Wilson CI | Floor |
|---|---|---|---|---|---|
| constant | test | reasoner-free | 70/200 = 0.350 | [0.287, 0.418] | pass |
| constant | unseen | reasoner-free | 42/200 = 0.210 | [0.159, 0.272] | pass |
| reasoner | test | reasoner-free | 67/200 = 0.335 | [0.273, 0.403] | pass |
| reasoner | test | reasoner | **77/200 = 0.385** | [0.320, 0.454] | pass |
| reasoner | unseen | reasoner-free | 46/200 = 0.230 | [0.177, 0.293] | pass |
| reasoner | unseen | reasoner | **58/200 = 0.290** | [0.232, 0.356] | pass |

All six cells clear the floor. This is the first time that has been true, and
the first time the `unseen` split clears it at all.

`cond_probe_delta` at evaluation was non-zero in every run (6.11 to 32.06), so
the conditioning path was live during the measurements, not just during
training.

## Three readings, in order of how much they support

### 1. Within-checkpoint steering is positive on both splits

Same weights, only the conditioning differs:

| Split | reasoner-free | reasoner | difference |
|---|---|---|---|
| test | 0.335 | 0.385 | **+0.050** |
| unseen | 0.230 | 0.290 | **+0.060** |

Same sign on both splits, which is the pattern the pre-registered decision rule
asks for. **These are unpaired point comparisons at n=200 and the Wilson
intervals overlap heavily.** They are not evidence of an effect on their own.
The paired test is `results/exp8_fixup_steering_test.json`.

### 2. Cross-arm, reasoner-free, the arms are indistinguishable

This is the check that the extra two epochs did not simply make a better model:

| Split | reasoner arm | constant arm | difference |
|---|---|---|---|
| test | 0.335 | 0.350 | -0.015 |
| unseen | 0.230 | 0.210 | +0.020 |

Both within noise of each other, with opposite signs across splits. Training
the reasoner path did not improve the reasoner-free path relative to a control
that was trained identically but saw no per-document signal.

### 3. The `unseen` crossover is gone

Before the fix-up the headline checkpoint scored 0.380 on `test` and **0.190 on
`unseen`** — a crossover against the LoRA-only control's 0.245/0.250. After the
fix-up the reasoner arm scores 0.230 reasoner-free and 0.290 steered on
`unseen`, both above the floor.

Two epochs of training with a live conditioning path closed a gap that the
inert-channel checkpoint could not. That is consistent with the side channel
mattering, and it is also consistent with two more epochs of training on the
`unseen` split's thesis distribution. The reasoner-free control rose on `unseen`
too (0.190 to 0.230), so most of the movement is training, not steering. Only
the +0.060 steered-vs-free difference is attributable to the channel, and that
difference is not yet tested paired.

## A confound found in the control, and what it changes

`encode()` returns `cond=None` for `reasoning_mode="none"`, and `build_features`
then substitutes a **zero** vector. So the reasoner-free path feeds FiLM
`cond=0`, not the checkpoint's `const_cond`.

That means the `constant` arm, which was trained with `cond = const_cond`, was
being evaluated with a different conditioning vector than it trained on. The
control was out of distribution. The pre-registered rule named
`fixup(reasoner) - fixup(constant)` as the decision statistic, and taken
literally it compared an in-distribution arm against an out-of-distribution
one.

How much this matters is bounded by the training diagnostic: `cond_abs_mean`
ended at **0.0106** for the constant arm against **0.0923** for the reasoner
arm. `const_cond` shrank toward zero during training, so the constant arm's
in-distribution conditioning is already close to the zero vector it was
evaluated with. The OOD gap is therefore small for this particular control,
which is why the run in flight is still worth reporting — but it is not zero,
and it is not something to leave implicit.

Fix: `repair()` and `run_model()` now take `reasoning_mode` (`none` |
`reasoner` | `learned`), `check_backbone.py` gained `--cond-mode`, and exp8
gained `--cgr-cond-mode` / `--baseline-cond-mode` and records both in its JSON.
`learned` feeds the checkpoint's own `const_cond`, which is the only
in-distribution path for a `cond_mode=learned` checkpoint. The reasoner-free
path keeps its old behaviour exactly (no extra backbone pass).

So there are now three distinct comparisons, and they answer different
questions:

| Comparison | Question |
|---|---|
| both reasoner-free | did training with an informative `cond` improve the no-cond path? |
| reasoner-steered vs constant-free | does the deployed CGR path beat the control as previously evaluated? (confounded, small) |
| reasoner-steered vs constant-learned | does per-document information in `cond` help, with both arms in-distribution? |

The third is the one the claim needs.

## What this does not support

- No significance claim. Nothing here is paired; every interval is a marginal
  Wilson interval on a single rate.
- No capacity or compute claim: 22.3M trainable vs the LoRA-only arm's 0.44M,
  4,025 s vs 4,141 s.
- `test` is now a selection set — the 0.380 checkpoint was chosen using it — so
  any conclusion must be confirmed on `unseen`.
- No Gate/DAG claim; the gate is a label, not a repair path.

## Next

Paired exp8 on `test`, `fixup_reasoner` (reasoner path) vs `fixup_constant`
(reasoner-free), then `unseen` only if `test` shows an effect. Decision rule
unchanged: the steering claim requires the paired CI to exclude 0 on `test`
with the same sign on `unseen`.

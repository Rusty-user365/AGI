# Arithmetic is out of support for the 470M, but the task family is sound (2026-10-27)

Artifacts: `experiments/arith_pilot.py`, `results/_smoke_arith.json`,
`results/_smoke_arith2.json`, `results/_smoke_arith3.json` (ladder probes,
fp16 and fp32)

## Why this pilot exists

The repair benchmark cannot test routing, because it contains no
wrong-but-well-formed candidate: 148/148 parseable answers were correct and
332/332 unparseable ones were wrong, so a parser achieved perfect discrimination
([benchmark-has-no-plausible-wrong-answers](2026-10-27-benchmark-has-no-plausible-wrong-answers.md)).

An arithmetic/state-tracking chain has the missing property by construction.
`17 * 3 = 51` and `17 * 3 = 48` are equally well formed and only one is right.
The pilot asks two questions at once:

1. Does the generator ever produce a correct answer? (If not, selection cannot
   help, because rejection sampling cannot find an answer outside the support.)
2. Does the generator ever produce a wrong-but-well-formed answer? (If not, the
   task is as useless as the repair corpus.)

## The instrument

A ladder of chained assignments, `steps` from 0 (pure copy) to 4, with small
non-negative values and a two-example prefix ending in `Answer:`.

Per rung: `p1_greedy`, `p1_sampled`, `pass@k`, `wellformed_rate`, and a count of
`wrong_wellformed` answers. Acceptance was fixed before the run: a rung is
usable when `p1 in [0.15, 0.50]`, `pass@k >= p1 + 0.15`, and
`wrong_wellformed > 0`.

## Result on ltg/dual-lm-470m

| steps | p1 greedy | p1 sampled | pass@8 | wellformed | wrong-but-wellformed |
|---|---|---|---|---|---|
| 0 (copy) | 0.000 | 0.250 | 0.000 | 1.000 | 8 |
| 1 | 0.000 | 0.000 | 0.125 | 0.875 | 6 |

**Usable rungs: none.** And the failure is not the task's:

- `wellformed_rate` is 1.000 at the copy rung and 0.875 at one step, so the
  model emits integers reliably.
- `wrong_wellformed` is 8 and 6, so **the task family does have the property the
  repair corpus lacks.** A wrong answer here is well formed, and a parser cannot
  distinguish it.

The generator simply never produces the right answer. Direct probes confirm why:

| prompt | top-1 token | logit | second |
|---|---|---|---|
| `3 + 4 =` | `1` | -2.62 | `6` -2.79 |
| `7 - 6 =` | `1` | -2.25 | `3` -2.53 |
| `9 + 9 =` | `1` | -1.77 | `2` -3.77 |
| `2 * 4 =` | `1` | -2.15 | `2` -2.61 |

Every distribution is nearly flat and top-1 is `1` almost everywhere. The
apparently constant `10` output in the first smoke run was not a bug: the model
picks `1`, then `0`, on every prompt. The same output appears in fp32 and fp16,
so it is model behaviour, not a precision artifact.

The model is not broken: `The capital of France is` predicts `Paris` at -2.38.
It is a 470M model trained on 32B tokens of web text with no arithmetic in
distribution. Its own card reports ARC-Easy 28.6 normalized, which is barely off
chance on a 4-way task, and this is consistent with that.

## What this settles

**The constraint is now measured rather than asserted.** Rejection sampling
needs the answer inside the support. For this backbone on this task family it is
not, so best-of-8 cannot reach it and no observer can be evaluated. `pass@8` of
0.125 at one step is not headroom; it is noise around zero.

**The benchmark design is validated even though the run failed.** The
`wrong_wellformed` counts prove the family supplies what the repair corpus
cannot. That is the piece worth keeping.

## Two harness defects found and fixed

1. **`token_type_ids`**: the remote tokenizer returns it, the custom model does
   not accept it, and `generate` raises rather than ignoring it. Stripped before
   every call.
2. **Prompt reuse in the sampler**: the first version regenerated problems with
   a different RNG seed for the k-sample loop, so sample s was scored against a
   different item's answer. Now the same prompts are reused and only decoding
   differs. This would have produced a meaningless `pass@k`.

## What this does not say

- **Nothing about the 470M after fine-tuning.** The model has never seen this
  format in training. A LoRA run on generated chains is the obvious next
  attempt, and it is a capability question, not an architecture one.
- **Nothing about other backbones.** Qwen3-0.6B has not been run on this pilot.
  If it lands inside the window, the benchmark is viable and the 470M is simply
  the wrong generator; if it does not, the task family needs to be easier than
  arithmetic.
- **Nothing about routing.** No observer was trained here, and none could be:
  with `p1` at zero there is nothing to select.

## Next step

Either raise the generator until `p1` enters the window (fine-tune on generated
chains), or find a generator that is already competent. The pilot is the
instrument that decides, and it now works: it reports `p1`, `pass@k`, the
well-formedness rate and the wrong-but-well-formed count in one run.

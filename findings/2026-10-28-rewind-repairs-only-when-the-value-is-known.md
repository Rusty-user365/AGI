# Rewinding repairs the chain only when the value is known; the hindsight constraint is inert (2026-10-28)

Artifacts: `experiments/oracle_rewind.py`, `results/oracle_rewind_h5.json`,
`experiments/loop_rewind.py`, `results/loop_rewind_h5.json`

## Why this experiment exists

Localization is a precondition, not a fix. Replaying greedily from the same
prefix deterministically re-emits the same wrong value, so a rewind loop needs a
perturbation that changes what the model emits at the broken step.

This measures the repair mechanism with the **oracle** step label, so the answer
does not depend on the localizer. If forcing the true value at the true step
does not recover the chain, no observer can rescue the architecture and the
localizer question stops mattering.

Every arm starts from the verified prefix `truth[:t]` and records whether the
whole chain ends correct, how many steps it generated, and whether it re-emitted
the value it was supposed to replace.

| arm | mechanism |
|---|---|
| `full_rewrite` | regenerate all 5 steps at T=0.8, the "let me redo it" baseline |
| `rewind_greedy` | re-decode step `t` greedily from the true prefix (the no-op) |
| `hindsight` | inject `# constraint: <name> is not <bad>` before step `t`, greedy |
| `hindsight_neutral` | inject `# constraint: none` — the control for the line itself |
| `rank2` | forbid the bad value's first digit at step `t`, greedy argmax of the rest |
| `resample` | re-decode step `t` at T=0.8, greedy afterwards |
| `oracle_force` | set step `t` to the true value, greedy afterwards |

## Result

110 broken holdout chains, Qwen3-0.6B-Base, 5 steps.

| arm | recovered | rate | 95% CI | steps generated | re-emitted bad value | paired vs rewrite |
|---|---|---|---|---|---|---|
| `full_rewrite` | 20/110 | 0.182 | [0.121, 0.264] | 5.00 | — | — |
| `rewind_greedy` | 17/110 | 0.155 | [0.099, 0.234] | 2.71 | 0.618 | 6W/9L, p=0.61 |
| `resample` | 22/110 | 0.200 | [0.136, 0.284] | 2.71 | 0.445 | 15W/13L, p=0.85 |
| `hindsight` | 30/110 | 0.273 | [0.198, 0.363] | 2.71 | 0.136 | 23W/13L, p=0.13 |
| `hindsight_neutral` | 30/110 | 0.273 | [0.198, 0.363] | 2.71 | 0.500 | 20W/10L, p=0.10 |
| **`rank2`** | **35/110** | **0.318** | **[0.239, 0.410]** | 2.71 | 0.000 | **22W/7L, p=0.008** |
| `oracle_force` | 94/110 | 0.855 | [0.777, 0.908] | 1.71 | — | — |

## The no-op is real

`rewind_greedy` re-emits the same wrong value in **61.8%** of cases and recovers
less often than a full rewrite (0.155 vs 0.182). For greedy-origin trajectories
it must reproduce exactly, and it does; for sampled ones it usually does.

That is the mechanism problem stated numerically: cutting back to the right
place and replaying deterministically is not a repair. Any rewind loop needs an
actual perturbation, which is why the remaining arms exist.

## The hindsight constraint does nothing

This is the result that changes the architecture.

`hindsight` and `hindsight_neutral` recover **30 of 110 either way**, and paired
against each other they are **15W/15L, p=1.000**. Injecting the line
"`# constraint: e is not 7`" is exactly as good as injecting "`# constraint:
none`". The text carries no information the model can use.

The constraint does change *what* gets emitted — the bad value reappears 13.6%
of the time with the real constraint against 50.0% with the neutral line — but
that suppression does not convert into a correct chain. The model avoids the
named value and lands on another wrong one.

So the "Hindsight Constraint" mechanism, as specified, is a prompt perturbation
with no measured advantage over a placebo prompt perturbation. It cannot be the
repair primitive.

## Rank-2 token exclusion is the only arm that beats a rewrite

`rank2` recovers 0.318 against 0.182 for a full rewrite, **22W/7L, p=0.008**,
and it is the only arm whose paired test clears. By construction it can never
re-emit the excluded value (0.000), so the mask is doing exactly what it claims.

It is still small: 1.75x on a base rate of 0.182, and it costs the same 2.71
forwards as the other rewind arms.

## The aggregate hides four different regimes

Recovery by the true broken step. `n` is the number of chains with that
`first_error`.

| t | n | full rewrite | greedy | hindsight | neutral | rank2 | resample | force truth |
|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.667 | 1.000 |
| 1 | 39 | 0.154 | 0.077 | **0.359** | 0.179 | 0.154 | 0.179 | 0.667 |
| 2 | 23 | 0.304 | **0.435** | 0.130 | 0.261 | **0.435** | 0.304 | 0.957 |
| 3 | 13 | 0.077 | 0.000 | 0.077 | **0.308** | **0.308** | 0.077 | 0.846 |
| 4 | 32 | 0.188 | 0.125 | 0.375 | 0.406 | **0.469** | 0.156 | 1.000 |

Three things are visible only here.

**The hindsight line works at step 1 and fails at step 2.** 0.359 against 0.154
at `t=1`, but 0.130 against 0.304 at `t=2`. With n=39 and n=23 that is not a
reliable difference in either direction, but it is enough to say the arm has no
consistent effect, which matches the paired null against the placebo.

**`t=4` flatters every rewind arm.** A late error needs one regenerated step, so
`rank2` gets 15/32 there. The same arm gets 6/39 at `t=1` — identical to the
full rewrite. The headline 0.318 is substantially a late-step effect.

**`oracle_force` is trivially 1.000 at `t=4`** because there is nothing left to
generate. Excluding that row, the ceiling is 62/78 = **0.795**, not 0.855.

## What the ceiling means

Forcing the correct value at the correct step recovers **79.5%** of chains
(excluding the trivial last-step case) against **17.9%** for regenerating from
scratch, while generating 1.71 forwards instead of 5.00.

That is the strongest evidence in this project that step-level repair has real
value: **the fix is worth localizing for.** It also locates the remaining loss
precisely. Even with a perfect step and a perfect value, one chain in five still
fails, because the downstream steps have to be re-derived and the model gets
them wrong. At `t=1` the ceiling is only 0.667.

So the architecture's upper bound is not 1.0 minus localization error. It is
0.795 times localization error, and the localizer is 0.611 top-1 / 0.506
balanced with a step-4 recall of 0.19.

## Compute

Rewind arms generate 2.71 forwards against 5.00 for a rewrite. **This is not a
KV-cache measurement**: the harness has no cache, every step is a fresh forward
over the whole prefix, so the saving is `H - t` forwards and nothing else. Cache
truncation remains untested, and no claim about it can be made from this run.

## What this does not say

- **Nothing about a real localizer driving the loop** in this file; that is the
  next section.
- **Nothing about other perturbation families.** One mask (rank-2 first digit),
  one constraint phrasing, one temperature. A different perturbation could
  behave differently.
- **Nothing about longer chains.** `H = 5`. The late-step advantage of `rank2`
  and the triviality of `t = H-1` both scale with `H`, so the aggregate rates
  are not transferable.
- **Nothing about `first_error = 0`.** Only 3 chains, so the `t=0` column is
  noise.

## Reproduction

```
python experiments/oracle_rewind.py --data results/arith_steps_h5.pt
```

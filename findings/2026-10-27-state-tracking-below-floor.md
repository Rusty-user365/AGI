# State tracking is below the floor of every model on this card (2026-10-27)

Artifacts: `experiments/statetrack.py`, `experiments/steplocal.py`,
`results/statetrack_h3.json`, `results/statetrack_h10.json`,
`results/statetrack_h20.json`, `results/_probe_b2_h3.pt`,
`results/_probe_b2_h5.pt`, `scratch/probe_onestep.py`,
`scratch/probe_onestep_instruct.py`

## What was being tested

The one falsifiable version of the long-horizon argument: if accuracy falls with
horizon *while* the model's own log-probability stays high on its wrong answers,
then confidence stops being a correctness proxy and a separate observer has room
again. The box world was built to measure that, and to supply the per-step
ground truth that localization needs.

## Result: the generator cannot track state at all

Four boxes, exact match on four integers, Qwen3-0.6B-Base:

| horizon | split | p1 greedy | p1 sampled | pass@4 | clean |
|---|---|---|---|---|---|
| 3 | train | 0.000 | 0.000 | 0.000 | 1.000 |
| 3 | holdout | 0.050 | 0.000 | 0.000 | 1.000 |
| 10 | train | 0.000 | 0.000 | 0.000 | 0.970 |
| 10 | holdout | 0.000 | 0.000 | 0.000 | 0.990 |
| 20 | train | 0.000 | 0.000 | 0.000 | 0.980 |
| 20 | holdout | 0.000 | 0.000 | 0.000 | 0.980 |
| 30 | — | — | — | — | killed (no signal at 3, 10, 20) |

Reducing the answer to two integers and the move to one ball did not help:

| horizon | split | p1 greedy | p1 sampled | pass@3 |
|---|---|---|---|---|
| 3 | train | 0.000 | 0.000 | 0.000 |
| 3 | holdout | 0.000 | 0.091 | 0.250 |
| 5 | train | 0.000 | 0.042 | 0.125 |
| 5 | holdout | 0.250 | 0.000 | 0.000 |

The failures are near-misses, not format errors. Admission was fixed to "the
first line is N bare integers" and clean rate rose to 1.000, at which point the
completions were unambiguously wrong: truth `(3,3,1,1)` came back as `(3,3,1,2)`,
`(3,3,2,2)`, `(3,3,0,2)`.

## The decisive measurement: one move

Chaining cannot rescue a task whose first step is wrong, so the single
transition was measured directly — state in, one move, state out, 40 trials per
config, greedy:

| model | boxes | max move | exact | parseable |
|---|---|---|---|---|
| Qwen3-0.6B-Base | 2 | 1 | **0.225** | 1.000 |
| Qwen3-0.6B-Base | 2 | 2 | 0.125 | 1.000 |
| Qwen3-0.6B-Base | 3 | 1 | 0.025 | 1.000 |
| Qwen3-0.6B-Base | 4 | 1 | 0.025 | 1.000 |
| Qwen2.5-1.5B-Instruct (raw few-shot) | 2 | 1 | 0.025 | 1.000 |
| Qwen2.5-1.5B-Instruct (chat template) | 2 | 1 | 0.150 | 0.750 |

The 0.6B errors are not a consistent wrong rule that could be corrected; they are
scattered. `(6,3)` + "move 1 from box 1 to box 2" gave `(5,2)` (both boxes
decremented), `(3,2)` + "move 1 from box 2 to box 1" gave `(3,1)` (source only),
`(0,6)` + the same move gave `(1,6)` (destination only). There is no partial
credit structure to exploit.

## Why this closes the line rather than delaying it

Localization needs a trajectory that is mostly right and diverges at a
*particular* step. `first_error` must vary across tasks for the label to carry
information. With a single move at 22.5% (2 boxes) and 2.5% (3+ boxes), the first
step is wrong in the large majority of trajectories, so `first_error = 0`
everywhere and any localizer scores the same as "always answer step 1".

That is not a tuning problem. Both models available on this card are below the
floor of the task, and the larger one is worse, not better: 1.5B-Instruct scores
0.150 where 0.6B-Base scores 0.225. More parameters did not buy state tracking
here.

## What this does *not* say

It does not say long-horizon reasoning fails for Transformers. It says the box
world is not measurable with these generators, so the horizon-vs-confidence
question (`logprob_auc` per horizon) is **unanswered**, not answered negatively.
The sweep that would have answered it produced no correct candidate at any
horizon, so the AUC is undefined rather than low.

It also does not touch the localization question itself. That question moved to
arithmetic chains, where the same 0.6B scored p1 0.544 on 2 steps
(`results/arith_bench_s2.pt`), and where per-step ground truth is available from
the generator: `experiments/arith_steps.py`.

## Cost note

The 4-box sweep at horizon 30 was still running when it was killed; its child
process survived the first kill attempt and held 2.9 GB of the 4 GB card,
stalling a later run for 15 minutes. The lesson recorded in `HANDOFF.md`:
`Stop-Process` filters that match nothing fail silently, so verify GPU memory is
0 MiB after any kill.

# Code pilot: does localized repair beat best-of-k at matched tokens? (2026-10-29)

**Status:** pre-registration. No results yet.

**Artifacts to be produced:** `experiments/code_loop.py`,
`results/code_pilot_rows.jsonl`, `results/code_pilot_h5.json`

## Why this is the first real test of the architecture

Every previous domain had a verifier that was exact and free. That made the
localizer's value measurable but economically empty: a calculator, or MRV, or a
legality check ends the argument. Code is different. The verifier is exact but
**expensive** -- a test run costs ~150 ms here and seconds for real suites -- and
no classical heuristic orders the search over programs. That is the only
configuration in which the CGR claim has a product.

The claim, stated once:

> Given a failing test, re-proposing only the localized region with the failing
> assertion in context solves more problems per generated token than redrawing
> whole functions.

This is a compute-matched claim. It cannot be "we solve more HumanEval", because
with enough samples best-of-k solves more by definition.

## Contamination, stated up front

HumanEval is public and almost certainly in the pretraining data of any modern
model. Absolute pass rates are therefore **not** a capability claim. The
comparison between arms is still valid because every arm sees the same problems
and the analysis is paired. No result here will be reported as "the model can
code"; only as "arm A beats arm B on the same 164 problems at matched tokens".

## Corpus and split

All 164 HumanEval problems. There is no train/test split inside the benchmark,
so there is no tuning on it: every hyperparameter below is fixed before the run.
The corpus is cached at `data/humaneval.jsonl.gz` (44 KB) and the verifier gate
is already passed (canonical 164/164, corrupted 0/164).

## Verifier

`run_tests` in `experiments/code_bench.py`: a fresh temp directory, an isolated
interpreter (`-I`), a 5 s timeout, and the tail of stderr as feedback. It is not
a security sandbox; the timeout is the guard, and model code runs with the
user's filesystem permissions. That is acceptable for a local research harness
and is stated rather than assumed.

`failure_kind` separates the two failure modes, which must not be pooled:

- **exception** -- the traceback points at the faulty line. Localization is free.
- **assert** -- the failure is raised inside the test, so the traceback does not
  point at the candidate's line. This is the learned localizer's actual job.

## Arms

Model: Qwen3-0.6B base, fp32, ~2.3 GB, as in the arithmetic runs.

| Arm | Budget | What it does |
|---|---|---|
| `greedy` | 1 generation | one body, tests decide |
| `best_of_4` | 4 generations | four full bodies, tests select the first that passes |
| `cgr_4` | 1 + up to 3 repairs | test, localize, re-prompt from the faulty line with the failing assertion, re-test |

`cgr_4` re-prefills rather than copying a KV cache. The semantics are the same
branch (shared prefix, regenerated suffix); the cache fork is a latency
optimization and is not part of this claim.

Localization for the pilot, in order:

1. exception: the last traceback frame that names `prog.py`, giving the line;
2. assert: the model's own last body line that assigns the return value, i.e.
   the final statement, because HumanEval failures are almost always in the
   returned value;
3. otherwise: the last line of the body.

Step 2 is deliberately simple and pre-registered. If the pilot shows the
assert-failure arm is where the loss lives, the learned localizer is the next
build; it is not smuggled in here.

## Metric

Primary: **problems solved at matched total generated tokens.** Tokens are
counted, not estimated: prompt tokens plus generated tokens, summed per arm.

Secondary:

- test runs per solved problem;
- pass rate by failure kind (`assert` vs `exception`);
- how often localization pointed at the line that, when rewritten, produced a
  pass (the "hit rate" of the localizer).

Paired exact McNemar on the same 164 problems for every arm comparison. Wilson
CIs on each pass rate. No bare point estimates.

## Pre-registered readings

- `cgr_4` vs `best_of_4` at matched tokens, paired McNemar.
- `best_of_4` vs `greedy`, to confirm the baseline is real and not a strawman.
- If `best_of_4` does not beat `greedy`, the sampling budget is too small for
  the comparison to mean anything and the pilot is inconclusive, not negative.

**Falsifier:** if `cgr_4` does not beat `best_of_4` at matched tokens, the loop
is decoration in code and there is no CGR claim to make. That result will be
reported as the finding.

## What is not in this pilot

- No static judge, no threshold. The judge's threshold failed to transfer twice
  (`tau = 0.35` on the vault, 0.95 precision on the stateless judge), so it is
  not being reintroduced until the loop itself shows a gain.
- No learned localizer for assert failures. That is the follow-up, and only if
  the pilot shows localization is the binding constraint.
- No second model, no diffusion, no infilling. A 1.5B checkpoint does not fit in
  4 GB next to the resident model.

## Cost

~164 problems x ~5 generations x ~200 tokens. At the arithmetic run's measured
rate this is roughly 1.5-2 hours on the 1050, plus 164 x 5 x 148 ms of test
time (~2 minutes). A 10-problem smoke run comes first and produces no claims.

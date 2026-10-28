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
| `cgr_4` | 1 generation + repairs until `best_of_4`'s token budget is spent (max 8) | test, localize, re-prompt from the faulty region with the failing assertion, re-test |

`cgr_4` re-prefills rather than copying a KV cache. The semantics are the same
branch (shared prefix, regenerated suffix); the cache fork is a latency
optimization and is not part of this claim.

Each repair is capped at `max_new = best_of_4_tokens - spent - repair_prompt`,
so the arm cannot exceed the baseline's generated-token budget. Prompt tokens
are charged on every re-prefill.

Localization for the pilot, in order:

1. exception: the last traceback frame that names `prog.py`, giving the line,
   then walk back to the enclosing 4-space statement so the cut keeps syntax;
2. assert: **cut = 0**, i.e. rewrite the whole body with the failing assertion
   in context;
3. otherwise: cut = 0.

### Revision made before the run: assert localization changed from "last `return`" to cut = 0

The original rule cut at the last `return` line, on the reasoning that HumanEval
failures are usually in the returned value. A two-problem smoke plus a direct
diagnostic of `HumanEval/0` falsified that:

- the model wrote `for i in range(len(numbers) - 1): if abs(a[i]-a[i+1]) < t`,
  i.e. it compared only adjacent pairs; the fix belongs *inside the loop*;
- cutting at `return False` leaves the buggy loop above the cut, so every repair
  could only rewrite the final statement and the bug was structurally
  unreachable -- the arm could not win regardless of model quality;
- the same diagnostic showed the model duplicating the function when the
  completion is not truncated; the loop's `truncate` already handles that, the
  diagnostic bypassed it.

The change was made after the 2-problem smoke and before the 164-problem run,
and is recorded here rather than silently applied. It makes the repair arm
*stronger* (full-body rewrite) and therefore does not manufacture a win for it:
cut = 0 is the same action `best_of_4` takes, except conditioned on the failing
assertion.

If the assert arm is still where the loss lives, the learned localizer is the
next build; it is not smuggled in here.

### What the two failure kinds therefore compare

The revision changes the meaning of the by-kind split, so it is stated before
the run rather than after:

- **exception failures** are the only place this pilot tests *localization*:
  the traceback names the line, the cut keeps the code above it, and the repair
  rewrites one statement.
- **assert failures** are a test of *conditioning*: cut = 0 plus the failing
  assertion is a hint-conditioned full resample. It answers "does executable
  feedback beat selection at matched tokens", not "does localization work".

Both are part of the CGR claim, but they are different claims and will be
reported as different rows. A win on asserts is a win for executable feedback;
it is not evidence for the localizer. A win on exceptions is evidence for
localization, on the subset where localization is free.

### Second revision before the run: the hint moved above the prompt

The first smoke put the failing assertion in an in-body comment:

```
def has_close_elements(...):
    """..."""
    for i in range(...):
        ...
    # FAILED: assert candidate(...) == True
```

On `HumanEval/0` the model responded with `    # 1.0, 2.2` and then a new
top-level `def`, which `truncate` cut. The repair arm was losing to a formatting
artifact: a comment inside the body reads as "function finished" to a base
model, so it restarts instead of continuing.

The repair prompt is now a comment block **above** the original prompt:

```
# The function below failed a test. Rewrite the body so it passes.
# Failing test: assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True

from typing import List
...
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """..."""
```

The continuation point is then exactly where a normal completion starts. A
direct diagnostic confirmed the model now writes bodies rather than restarting
(`results/_diag_repair_format.py`). Reassembly of prompt + prefix + canonical
suffix passed 60/60 across 20 problems at cuts 0, 1, and last.

This change was also made before the run. It does not favour the repair arm on
substance: the hint is the treatment, and the fix only removes a formatting
penalty that was making the treatment unreadable.

### Cost, corrected after the format fix

Full-body repairs cost more than the prefix repairs the earlier estimate
assumed. Measured: ~96 s and ~186 s for the two smoke problems, ~140 s/problem.
So Stage 1 (24 problems) is ~1 h and the full 164 is ~6 h. That is why the
futility gate matters: it buys the same decision for one hour instead of six.

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

## Staged run and futility rule (added before the run)

A full 164-problem run costs ~4 h on the 1050. The harness audit passed (see
below), but a correct harness can still be pointed at an effect too small for
the sample to see. So the run is staged, and the stopping rule is fixed now:

- **Stage 1:** the first 24 problems, written to the same resumable rows file.
- **Futility stop:** stop if either
  1. `cgr` has **zero** paired wins against `best_of`, or
  2. `cgr` trails `best_of` by **3 or more** problems outright.

  If the true per-problem win rate were 10%, the chance of zero wins in 24 is
  0.9^24 = 0.08, so the gate is sensitive to a moderate effect and near-certain
  to a null. Rule 2 catches an arm that is clearly worse even if it wins the odd
  problem. On a stop, report "no detectable signal at N=24; stopped for
  futility" with the discordant counts.
- **Stage 2:** if at least one win appears, continue to 164 with `--resume`.

This is a single look, declared in advance. The primary analysis remains the
paired McNemar on all 164; a futility stop is reported as its own result, not as
a failure of the loop and not as evidence for it.

## Harness audit (run before the long run, no GPU)

`experiments/code_audit.py`, logged to `results/_audit_code.log`:

- **Test sensitivity:** stub bodies `pass`, `return 0`, `return []` fail
  **164/164** each, and **164/164** tests contain a `check()` call. The tests
  discriminate.
- **Localization math:** a synthetic exception raised inside a loop is cut at
  the enclosing `for` statement (line 1 of the body), not at the raising line,
  and the `prog.py` frame offsets resolve inside the candidate (frame 15,
  expected 15).
- **Truncation:** indented comments and nested `def`s survive; only column-0
  `#`, `print(`, `class`, `def` cut, which is where the body had already ended.
- **Repair assembly:** cuts 0, 1, and last all reassemble into code that passes
  the official test.

Two accounting fixes were made as a result:

1. Tokens are counted from the generation length actually produced, not from the
   re-tokenized, truncated text. The old count undercounted a completion that
   ran into a stop sequence, which would have let the repair arm exceed its
   budget while appearing matched.
2. An exception frame that resolves outside the candidate body (raised by the
   test harness) now falls back to cut = 0 instead of a clamped index.

Per-problem seeding (`--seed`, default 1234) makes a resumed run regenerate a
problem exactly as an uninterrupted run would.

## What is not in this pilot

- No static judge, no threshold. The judge's threshold failed to transfer twice
  (`tau = 0.35` on the vault, 0.95 precision on the stateless judge), so it is
  not being reintroduced until the loop itself shows a gain.
- No learned localizer for assert failures. That is the follow-up, and only if
  the pilot shows localization is the binding constraint.
- No second model, no diffusion, no infilling. A 1.5B checkpoint does not fit in
  4 GB next to the resident model.

## Cost

Measured after the format fix: ~140 s/problem (96 s and 186 s on the two smoke
problems), so ~1 h for Stage 1 and ~6 h for the full 164 on the 1050. Test time
is ~150 ms per run and is not the bottleneck; generation is. Stage 1 exists
precisely so a null costs one hour instead of six.

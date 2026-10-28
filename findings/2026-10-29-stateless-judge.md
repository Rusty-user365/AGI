# The stateless judge does not clear its own threshold (2026-10-29)

**Status:** complete. Verdict: a stateless neural judge over hidden states cannot
gate commits at the precision the proposal requires, on this corpus. The idea
survives only where the verifier is expensive and there are many candidates per
problem.

**Artifacts:** `experiments/global_judge.py`, `results/global_judge_h5.json`,
`results/global_judge_center_h5.json`

## The proposal being tested

A judge that reviews the artifact every now and then, blind to the writer's
reasoning, and commits only above a threshold -- the exam model, where the
writer also anticipates the marker.

Three design choices make it safe, and all three are encoded here:

1. **The label is the verifier's verdict**, not human taste. A judge trained on
   "what an evaluator might like" can be gamed; a judge trained on "what the
   verifier will say" cannot move the target. The remaining risk is
   distribution shift, which is measurable.
2. **Blind.** The judge reads the hidden state at the prefix and the step
   logprobs. It never reads generated commentary, so it cannot be persuaded by
   the writer's own narrative. We already know prose carries no corrective
   signal (30/110 vs 30/110 placebo).
3. **The threshold is chosen on train only**, and its transfer to holdout is
   reported as a first-class number, because `tau = 0.35` for the two-tier rule
   was tuned in-sample and degraded on the frozen vault.

## Setup

480 train chains (201 correct, 279 wrong) and 200 holdout chains (90 correct,
110 wrong), 5 steps each. For prefix length p = 1..4, predict whether the
finished chain is correct. Features: last-step state and mean state projected to
32 SVD dimensions each, plus last/mean/min logprob. Logistic regression, weight
decay 1e-3.

`tau` is the lowest threshold whose commits have **>= 0.95 precision on train**.
A false commit accepts a wrong chain; a false reject costs one more verifier
call. The asymmetry is why precision, not balanced accuracy, is the target.

## Results

Uncentered (absolute judge):

| p | judge AUC | train AUC | best logprob AUC | tau | holdout precision | holdout recall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.478 | 0.816 | 0.674 | 2.03 | 0.750 | 0.067 |
| 2 | 0.614 | 0.931 | 0.387 | 1.01 | 0.552 | 0.411 |
| 3 | 0.673 | 0.938 | 0.415 | 0.51 | 0.571 | 0.444 |
| 4 | 0.570 | 0.977 | 0.362 | 0.09 | 0.423 | 0.367 |

Centered within task (relative judge -- removes task difficulty):

| p | judge AUC | train AUC | holdout precision | commits |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.530 | 0.508 | -- | 0 |
| 2 | 0.526 | 0.573 | -- | 0 |
| 3 | 0.649 | 0.613 | 0.000 | 1 |
| 4 | 0.612 | 0.637 | 0.000 | 8 |

## Three findings

**1. The absolute judge learns task difficulty, not chain quality.** Train AUC
reaches 0.98 while holdout sits at 0.54-0.67. Each task has five candidates, so
the state encodes task identity and the judge memorises "this task is easy".
Centering each task's mean removes exactly that: train AUC collapses to
0.51-0.64, which confirms the diagnosis. But holdout does not improve -- the
within-task signal is itself weak (0.53-0.65).

**2. The threshold does not transfer, again.** A threshold with 0.95 precision
on train lands at 0.42-0.75 precision on holdout uncentered, and commits nothing
(or only errors) centered. This is the second time a tuned threshold failed
out-of-sample: `tau = 0.35` for the two-tier rule degraded on the vault, and the
gate's accept rule was a no-op on recovery. The pattern is consistent enough to
state as a rule: **thresholds tuned on a few hundred rows do not transfer.**

**3. Confidence inverts.** Raw logprob predicts correctness at p=1 (AUC 0.674)
and *anti*-predicts at p=2-4 (0.362-0.415). Later steps are more confident when
they are wrong. A threshold on confidence is therefore worse than useless after
the first step, and a judge that learns to invert it is fitting a corpus
artefact, not a stable signal.

## Structural limit

Only **18 of 50 holdout tasks are mixed** (some candidates correct, some wrong).
12 are all-correct and 20 are all-wrong. On unanimous tasks a *relative* judge
has nothing to rank by construction, so it can only ever act on 36% of tasks --
and on those its AUC is ~0.65.

## Verdict

**The stateless neural judge fails its own operating requirement.** It cannot
reach 95% precision out-of-sample at any prefix, its threshold does not
transfer, and its confidence signal inverts.

This is consistent with the gate result: a thresholded accept rule on a learned
score was a no-op on recovery (gated == ungated exactly), bought only precision
and skipped compute. The judge is a screening device, and on this corpus there
is nothing to screen -- the verifier is `eval_lines`, which is exact and free.

## Where the idea survives

The exam analogy is precise about *why* it failed. A marker does not predict
what an evaluator might like in the abstract; the marker works **against a
marking scheme**. That scheme is the spec. An absolute taste score across
different exams learns which exam is hard, not which answer is good -- which is
exactly what the train/holdout gap shows.

So the judge is viable only when all three hold:

1. **The verifier is expensive.** In code a test run costs seconds; in
   arithmetic it costs microseconds. The judge's value is proportional to the
   verifier's cost, and it is zero here.
2. **There are many candidates per problem.** Five is too few, and often
   unanimous. Twenty candidates with per-candidate test outcomes would give a
   relative judge something to rank.
3. **The judgment is spec-anchored or relative, never absolute across tasks.**
   "Which of these candidates passes the spec" is answerable; "is this good"
   is not, out of distribution.

That points at code, with static analysis (types, AST, lint) as the cheap judge
and the test run as the expensive verifier: run the tests only when the static
signal clears a threshold, and measure test runs saved at matched pass rate.
That judge is symbolic and needs no second model, which also keeps it inside
4 GB.

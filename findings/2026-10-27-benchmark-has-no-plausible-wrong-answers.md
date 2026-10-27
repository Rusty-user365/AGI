# The repair benchmark has no wrong-but-well-formed answers (2026-10-27)

Artifacts: `results/observer_features_test2.pt`,
`results/observer_head_train_parseable.json`,
`results/observer_head_train_permuted.json`

## What was being tested

The shipped seven heads score within-task AUC 0.425-0.550 on ranking repairs
([observer-cannot-rank](2026-10-27-observer-cannot-rank.md)). That is a fact
about those heads, not about the task: the oracle gap was 0.409 (0.308 base
rate to 0.717 best-of-4), so there was a large amount of signal for a selector
to find.

A predictor was then trained directly on the correctness label from cached
features, split by task, and tested on held-out tasks. Result: the
full-document feature the heads consume fails (0.589 mean over five splits,
2/5 passing) but a **block-level** feature succeeds (0.680 mean, 0.612-0.724,
4/5 passing), and a label-permutation control collapses to 0.503. So the
signal was real and the pipeline was not leaking.

The obvious objection was written down before the control ran: a block-only
feature has no document context, so it cannot know which thesis is
authoritative. A high AUC might only mean "this text is not garbled".

## The control, and what it found

Candidates were re-dumped with a `parseable` flag (`parse_block` returns a
thesis id). Across 480 candidates:

| | correct | wrong |
|---|---|---|
| **parseable** | 148 | **0** |
| **unparseable** | **0** | 332 |

Agreement between "parses as a block naming a thesis" and "names the
authoritative thesis" is **1.000**. There is not one candidate in 480 that is
well formed and wrong.

Consequence: the held-out AUC of 0.684 is not measuring correctness
discrimination. It is measuring **well-formedness**, and on this corpus those
are the same event. Restricting evaluation to parseable candidates leaves zero
discordant tasks, so the within-task AUC is undefined — not 0.5, undefined.
There is no pair of (correct, incorrect) candidates to rank.

## Why this ends the scratchpad line on this corpus

**1. A parser dominates every neural observer, perfectly and for free.**
`parse_block` already achieves the maximum possible discrimination on this
task, at zero parameters and zero forward passes. The block-level head reaches
0.684; the shipped heads reach chance. Any rejection loop built here would be a
regex wearing a neural network.

**2. The 0.409 oracle gap is sampling luck, not reasoning.** "At least one of
four samples parsed" is a statement about the decoder's failure rate, not about
the model's ability to tell right from wrong. Rejection sampling with a
validity check captures it; no observer is needed and none would add anything.

**3. The task cannot falsify the DAG's central claim.** The DAG exists to
discard a candidate that is *wrong but plausible*. This benchmark contains no
such candidate. A positive result here would be attributable to a one-line
parser, and a negative result would say nothing about routing.

## What this does not say

- **It does not say the seven heads are badly trained.** It says they cannot
  detect even well-formedness from the full-document feature, which is a
  dilution problem, and that the block-level feature fixes that. Both remain
  true.
- **It does not say observers cannot work.** It says this corpus cannot test
  them. The claim needs a task where a well-formed answer can be wrong.
- **It does not invalidate the 0.680.** As a well-formedness detector it is
  real and permutation-controlled. It is simply not the thing the DAG needs.

## What the next benchmark must satisfy

This is now a measured requirement rather than a preference:

1. **A wrong candidate must be well formed.** Arithmetic chains, unit
   conversions and state tracking qualify: `17 * 3 = 51` and `17 * 3 = 48` are
   equally parseable and only one is right. A corrupted-thesis block does not
   qualify.
2. **Exact-match scoring**, so correctness is not a judgement call.
3. **Enough headroom for selection to matter**: the single-pass rate must be
   well above chance and well below the oracle, and that must be measured on
   the actual backbone rather than assumed.
4. **The validity filter must be available to every arm**, including
   best-of-n. Otherwise the observer arm gets a parser the baseline does not,
   and the comparison measures the parser.

## Status

The observer gate is **not cleared**, and the reason is stronger than a null:
the benchmark cannot distinguish correctness from parseability. `DAG_Scratchpad`
stays unwritten. Building it requires a benchmark with wrong-but-well-formed
candidates, which does not exist in this repository yet.

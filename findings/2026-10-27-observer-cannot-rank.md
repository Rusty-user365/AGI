# The observer heads cannot rank repairs, so the DAG stays unwritten (2026-10-27)

Artifact: `results/observer_diagnostic_test.json`
Run: `--n 120 --candidates 4 --steps 8 --cond-mode reasoner --temperature 0.8 --split test`

## The question

A rejection loop only helps if the observer's score correlates with being right.
The pre-registered gate for writing `DAG_Scratchpad` was: does any of the seven
heads rank a correct candidate above an incorrect one, better than chance?

## Measurement

120 tasks, 4 candidates each, 480 pairs. The candidate generator was verified to
produce distinct drafts (3.84 distinct of 4; no task had a single candidate).

| | |
|---|---|
| base rate (mean candidate correct) | 0.308 |
| **oracle top-1** (best candidate per task) | **0.717** |
| discordant tasks (both classes present) | 82/120 (0.68) |
| within-task pairs | 280 |
| length alone, pooled AUC | 0.495 |
| length alone, within-task AUC | 0.500 |

The oracle gap is 0.409. **Selection on this task is worth a great deal, and
there is no length confound.** So this is not a task ceiling and not a metric
artifact.

| head | pooled AUC | within-task AUC [95% CI] | len-ctl | top1 | gain |
|---|---|---|---|---|---|
| value | 0.483 | 0.529 [0.446, 0.610] | 0.529 | 0.308 | +0.000 |
| outcome_max | 0.476 | 0.471 [0.394, 0.551] | 0.471 | 0.292 | -0.017 |
| tau | 0.509 | 0.457 [0.372, 0.538] | 0.457 | 0.275 | -0.033 |
| disagreement | 0.492 | 0.425 [0.347, 0.507] | 0.425 | 0.267 | -0.042 |
| epistemic_conf | 0.475 | 0.550 [0.480, 0.625] | 0.550 | 0.342 | +0.033 |
| scope_conf | 0.507 | 0.505 [0.438, 0.567] | 0.505 | 0.342 | +0.033 |
| source_conf | 0.500 | 0.500 [0.500, 0.500] | 0.500 | 0.350 | +0.042 |

**Every confidence interval includes 0.5, in both directions.** The strongest
signal is `epistemic_conf` at 0.550 [0.480, 0.625] and `disagreement` at 0.425
[0.347, 0.507]; neither excludes chance. The length control changes nothing,
because there was no length confound to remove.

Calibration is equally empty. Brier is 0.204 for every head against a base rate
of 0.308, where a constant predictor of the base rate scores 0.213. The Platt fit
has learned to output approximately the base rate for every input. ECE 0.066 is
the spread of that constant, not calibration of a signal.

Gate operating points, held-out half, base rate 0.308:

| head | precision at 50% coverage | precision at 30% coverage |
|---|---|---|
| value | 0.256 | 0.360 |
| outcome_max | 0.231 | 0.280 |
| tau | 0.289 | 0.360 |
| disagreement | 0.314 | 0.200 |
| epistemic_conf | 0.205 | 0.276 |
| scope_conf | 0.308 | 0.333 |
| source_conf | 0.275 | 0.275 |

Accepting the top half of candidates yields precision no better than accepting
all of them. `gate_usable` and `rerank_usable` are False for all seven heads.

Between-task variance share (ICC) is 0.56 to 0.77 for every head. A fixed
threshold Gate would therefore be dominated by prompt difficulty — the exact
failure mode predicted before the run. It is moot here because there is no
signal to threshold, but it means even a head with pooled AUC 0.6 would need to
be a tournament, not a Gate.

## Decision

**The pre-registered gate fails. No `DAG_Scratchpad` code is written.**

The numbers say something sharper than "the loop would not help". They say the
loop is worth 0.409 in principle (0.308 to 0.717) and the observer captures
none of it (best significant-looking gain +0.042, from a head whose AUC is
exactly 0.500 and whose "gain" is the artifact of breaking ties by candidate
order).

## What this does not say

These heads were trained on synthetic episode features — `value`, `outcome`,
`tau`, `epistemic`, `scope`, `source`, `disagreement` over scaffold episodes with
outcome noise. They were never trained on `(draft, correct)` pairs. An AUC of
0.5 here means **these heads as trained do not transfer to judging repairs**. It
does not mean no observer can, and the 0.409 oracle gap is the reason to care
about the difference.

## The one experiment this licenses

Train a head on correctness labels, not on the scaffold objective. The backbone
is frozen, so the features are computable once and cached:

1. Run N candidates per task, record `cat([pooled, cond, clock])` and the
   correctness label. 480 rows is one diagnostic run.
2. Fit a small MLP or logistic model on the cached 1544-wide features, split by
   task so no task appears in both halves.
3. Report within-task AUC on the held-out tasks, against the same 0.5 test.

This is an hour of CPU work and it is the only thing that could justify the DAG.
If a head trained directly on the selection objective still cannot beat 0.5, the
observer idea is dead for this backbone and the answer is to stop, not to build
more machinery.

## Incident: the decoder was deterministic and every candidate was identical

The first version of the diagnostic generated four candidates by varying `seed`
and got byte-identical text four times. `repair()` defaulted to
`temperature=0.0` with a deterministic `left_to_right` commit order, so the seed
had no effect (`cgr/model/diffugpt.py:641`).

Consequences that were caught before they mattered:

- The diagnostic would have reported within-task AUC 0.5 and I would have
  attributed it to the heads rather than to the generator.
- **Best-of-n on a greedy decoder is 1-of-n.** Any control arm written that way
  would have looked compute-matched while sampling one candidate N times.

`repair()` now takes `temperature` and `order` and passes both to `infill`.
Defaults are unchanged, so no existing result moves; 71/71 tests pass.

# Hidden states localize the broken step; cross-step attention does not beat a linear read (2026-10-28)

Artifacts: `experiments/arith_steps.py`, `experiments/localizer.py`,
`experiments/localizer_pair.py`, `results/arith_steps_h5.pt`,
`results/localizer_arith_steps_h5_*.json`, `results/localizer_pair_h5.json`

## Why this experiment exists

The observer that ranks whole candidates is closed: it beats random and ties
the model's own confidence
([observer-ranks-but-cannot-select](2026-10-27-observer-ranks-but-cannot-select.md)).
Ranking a finished candidate cannot support a rewind, because a rewind needs to
know **which step** to cut back to.

The obvious reason pooling fails was never tested. A mean over steps destroys
position before the first layer, so a pooled reader cannot point at a step even
in principle. This reads the un-pooled per-step state `[H, d]` instead: pooling
happens within a step (over that step's emitted tokens) and never across steps.

## The corpus

Self-conditioned arithmetic chains, `h = 5`, Qwen3-0.6B-Base, 120 train / 50
holdout tasks, 4 candidates per task (1 greedy + 3 samples at T=0.8, top-p
0.95). Self-conditioning is what makes the label real: step `i` sees the
model's own earlier values, so one wrong value propagates and `first_error` is
a genuine divergence point, not a decoding accident.

| split | broken and complete candidates | first_error 0 / 1 / 2 / 3 / 4 |
|---|---|---|
| train | 279 | 3 / 124 / 83 / 32 / 37 |
| holdout | **110** | 3 / 39 / 23 / 13 / 32 |

Only complete chains that actually broke are usable, because the label is
undefined for a correct chain. Splits are used as recorded and never pooled.

## The gate, and why it is not `argmin logprob`

The original gate was "beat `argmin` per-step log-probability". That gate is
too weak, and the data says so twice.

Hidden states carry position: RoPE and residual-norm growth make the step index
linearly readable. Error labels are not uniform over steps either. A reader
over `[H, d]` can therefore beat `argmin logprob` and the label prior
*separately* by learning "which step is usually wrong" plus "how confident was
the model", and never extract an arithmetic signal at all.

The baseline that closes that hole is `poslogprob`: `logits[t] = bias[t] +
w[t] * logprob[t]`, ten parameters, no hidden state. It is trained on the same
rows with the same seed. **A hidden-state reader earns its place only by
beating `poslogprob` on paired candidates.**

## Result

Holdout, n = 110 broken candidates, 5 seeds. `bal` is mean per-class recall;
`non-modal` is accuracy on the 71 candidates whose true step is not the modal
one.

| arm | top-1 | 95% CI | bal | non-modal | vs poslogprob |
|---|---|---|---|---|---|
| **linear + SVD32** | **0.611** | [0.516, 0.695] | **0.506** | 0.443 | 114W/23L, p<1e-14 |
| transformer, full width | 0.600 | [0.507, 0.687] | 0.474 | 0.507 | 137W/52L, p<1e-9 |
| linear + SVD32 + demean-pos | 0.589 | [0.497, 0.678] | 0.494 | 0.510 | 133W/54L, p<1e-8 |
| transformer + SVD32 | 0.576 | [0.479, 0.661] | 0.475 | 0.485 | 136W/64L, p<1e-6 |
| transformer + SVD32 + demean-pos | 0.535 | [0.444, 0.627] | 0.467 | 0.473 | 127W/78L, p=0.001 |
| poslogprob (10 params, no state) | 0.445 | — | 0.287 | 0.141 | — |
| argmin logprob | 0.400 | — | 0.293 | 0.549 | — |
| label prior (always step 1) | 0.355 | — | — | — | — |
| chance | 0.200 | — | 0.200 | 0.250 | — |

Two things are true at once.

**The hidden state localizes.** Every arm beats `poslogprob` on paired
candidates, and the best arm beats it by 0.166 in top-1 and 0.22 in balanced
accuracy. The label prior is 0.355 and the modal step holds 39 of 110
candidates, so 0.611 is not the prior resurfacing.

**Confidence alone does not.** `poslogprob` reaches 0.445 by learning the step
prior, and its non-modal accuracy is **0.141 — below the 0.25 chance rate among
the four non-modal classes.** It has no location information; it is a prior
plus a confidence slope.

## `argmin logprob` is a constant predictor

Its prediction histogram over 110 candidates is `[0, 5, 9, 3, 93]`: it blames
step 4 in 93 cases. Step 4 is the last step and the highest-entropy one
(conditional step accuracy 0.74 against 0.90 at step 3), so minimum confidence
almost always lands there.

That explains its odd profile. Its non-modal accuracy, 0.549, looks better than
every learned arm — because the non-modal set is 32/71 step-4 candidates and it
always answers step 4. Its balanced accuracy, 0.293, exposes the constant.

This is a reporting trap worth naming: **non-modal accuracy can be gamed by a
constant predictor whenever one non-modal class dominates.** Balanced accuracy
cannot, which is why it is the column that decides here. The honest comparison
on the metric that cannot be gamed is 0.506 (linear) against 0.293 and 0.287
for the two free baselines.

## The hidden state is not redundant with confidence

The selection head was killed by one test: adding log-probability to its feature
vector changed nothing, so the head was re-reading confidence
([observer-ranks-but-cannot-select](2026-10-27-observer-ranks-but-cannot-select.md)).
The same test runs here, in the other direction.

| arm | top-1 | balanced |
|---|---|---|
| linear + SVD32 | 0.611 | 0.506 |
| linear + SVD32 + step logprob | 0.624 | 0.503 |

Adding confidence is worth +0.013 top-1 and **nothing** in balanced accuracy.
That is the opposite of redundancy: the per-step state already contains what
confidence contains, and confidence alone reaches only 0.445. The state is the
richer signal, not a restatement of the logprob.

## The position confound, controlled

`--demean-position` subtracts the per-step mean feature fitted on train, so the
step index cannot be read off the residual-norm shift. Both arms survive:

| arm | top-1 | vs poslogprob |
|---|---|---|
| linear + demean | 0.589 (from 0.611) | 133W/54L, p<1e-8 |
| transformer + demean | 0.535 (from 0.576) | 127W/78L, p=0.001 |

Demeaning costs 0.022 for linear and 0.041 for the transformer, and both still
beat the ten-parameter confound model on paired candidates. The effect is not
step-index leakage.

## Attention does not clear its gate

The gate for the encoder layer was that it beat linear by a margin surviving a
paired test. It does not. Paired on the same seed and the same split
(`experiments/localizer_pair.py`, full width, 5 seeds):

| seed | linear | transformer | paired |
|---|---|---|---|
| 0 | 0.627 | 0.618 | 13W/12L, p=1.00 |
| 1 | 0.600 | 0.582 | 15W/13L, p=0.85 |
| 2 | 0.645 | 0.627 | 16W/14L, p=0.86 |
| 3 | 0.645 | 0.582 | 21W/14L, p=0.31 |
| 4 | 0.591 | 0.591 | 15W/15L, p=1.00 |

Pooled 80W/68L, p=0.366. Majority vote over the 5 seeds:

| | top-1 | balanced |
|---|---|---|
| linear | 0.627 | 0.524 |
| transformer | 0.655 | 0.528 |
| difference | -0.027 | 95% bootstrap [-0.118, +0.064] |

The two comparisons disagree in sign, and both intervals cross zero. That is a
null, not a win in either direction: **one encoder layer over the H rows buys no
reliable accuracy.** Its seed spread at matched capacity (0.473–0.673) against
linear (0.591–0.618) also means any single-seed result in either direction would
be noise.

**Capacity is not the explanation.** The earlier concern was that 279
trajectories cannot support a 1024→64 trainable projection. The full-width
transformer (trainable 1024→64, ~65k parameters) scores 0.600 against 0.576 for
the frozen-SVD version. Removing the projection entirely does not rescue it, so
the 200/80 rebuild is not needed to settle this question.

### But the two arms fail in different places

Per-step recall, majority vote, same candidates:

| step | n | linear | transformer |
|---|---|---|---|
| 0 | 3 | 0.000 | 0.000 |
| 1 | 39 | **0.923** | 0.821 |
| 2 | 23 | **0.739** | 0.565 |
| 3 | 13 | **0.769** | 0.692 |
| 4 | 32 | 0.188 | **0.562** |

The transformer is worse on the early steps and **triples late-step recall**
(0.19 → 0.56, 6/32 → 18/32). Attention is doing something real; it is just not
doing anything that improves the average. This is consistent with what
cross-step attention should be able to see — a state that contradicts the
previous step is easier to detect late, when there is more context to
contradict — and it is the one result here that argues against closing the
transformer line completely.

Two things follow, and neither is measured yet: a linear+transformer vote might
beat both (their errors are not the same errors), and the step-4 deficit is
where that would show up. Neither is in this report.

## Where the best arm still fails

Step 1 is nearly solved (0.923) and step 4 is not (0.188). The 32 late errors
are confused with step 1 in 20 cases: for an error at the end of the chain the
state says "something is off", and the cheapest explanation is the modal step.
So the usable summary is not "0.627". It is **0.92 / 0.74 / 0.77 / 0.19 by
step**, and any rewind loop inherits exactly that profile: it cuts back to the
right place early in the chain and to step 1 late.

This is also why the transformer's step-4 recall matters more than its average.
A rewind that always lands on step 1 regenerates the whole chain — that is a
full rewrite with extra bookkeeping, not a repair.

## What this settles

**Step-level localization from un-pooled hidden states is real, and it is not
the label prior.** A 32-dim linear read of the per-step state scores 0.611
top-1 and 0.506 balanced against 0.445 / 0.287 for the strongest
position+confidence model and 0.400 / 0.293 for `argmin logprob`. The paired
counts are 114W/23L.

**Attention is not the mechanism, but it is not inert either.** One encoder
layer over the H rows does not beat the independent per-step read on average
(pooled 80W/68L, p=0.366; majority-vote difference -0.027, CI [-0.118, +0.064]).
It does triple late-step recall (0.19 → 0.56) while losing on steps 1–3, so
cross-step context carries a real late-chain signal that the linear arm cannot
see. Neither fact justifies building on attention yet, and neither justifies
discarding it.

**The pooled null was a pooling artifact, as suspected.** The same backbone
that could not select between whole candidates can point at a step when it is
allowed to see position.

## What this does not say

- **Nothing about repair.** Localization is a precondition, not a fix. Replaying
  greedily from the same prefix deterministically re-emits the same wrong value;
  whether a rewind repairs anything is the next experiment
  (`experiments/oracle_rewind.py`), and it is run with the *oracle* label so the
  answer does not depend on this localizer.
- **No compute claim.** This harness has no KV cache: every step is a fresh
  forward over the whole prefix. "Rewind to step t" saves `H - t` forwards and
  nothing more; cache truncation is untested.
- **One task family, one backbone, h = 5.** Arithmetic chains on
  Qwen3-0.6B-Base. Nothing here transfers to the 470M Dual-LM, to longer chains,
  or to non-arithmetic reasoning without a rerun.
- **The null is bounded at this size.** One layer, width 64, 4 heads, 279
  trajectories. A deeper block, more data, or a different pooling of the step's
  tokens is untested. The paired test bounds the claim here, not in general —
  and the late-step result is a reason to keep that door open.
- **`p` values pooled over seeds are stability checks, not independent tests.**
  Every seed sees the same 110 candidates. The candidate-level interval in
  `localizer_pair_h5.json` is the one that reflects sampling error.

## Reproduction

```
python experiments/arith_steps.py --steps 5 --n-train 120 --n-holdout 50 --k 3 --resume
python experiments/localizer.py --data results/arith_steps_h5.pt --arch linear --frozen-proj 32 --seeds 5
python experiments/localizer.py --data results/arith_steps_h5.pt --arch linear --frozen-proj 32 --linear-logprob --seeds 5
python experiments/localizer.py --data results/arith_steps_h5.pt --arch transformer --seeds 5
python experiments/localizer.py --data results/arith_steps_h5.pt --arch transformer --frozen-proj 32 --demean-position --seeds 5
python experiments/localizer_pair.py --data results/arith_steps_h5.pt --seeds 5
```

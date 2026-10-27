# The observer ranks repairs but cannot select better than the model's own confidence (2026-10-27)

Artifacts: `experiments/arith_bench.py`, `experiments/select_eval.py`,
`results/arith_bench_s2.pt`, `results/select_eval_arith_s2.json`,
`results/select_eval_arith_s2_perm.json`, `results/select_eval_arith_s2_rank.json`

## Why this experiment exists

The repair corpus could not test routing: parseability and correctness agreed
in 480/480 candidates, so a one-line parser achieved perfect discrimination and
no observer could be distinguished from a regex
([benchmark-has-no-plausible-wrong-answers](2026-10-27-benchmark-has-no-plausible-wrong-answers.md)).

This builds the benchmark that corpus could not be. Two-step assignment chains
(`a = 7 / b = a + 5 / b = ?`) admit a wrong answer that is well formed: `12`
and `11` are equally parseable, and only one is right. A validity filter of
"contains an integer" therefore removes nothing, and correctness has to be
judged arithmetically.

The gate was fixed in advance: **AUC > 0.5 on held-out tasks, with the CI
excluding 0.5, permutation-controlled, and at least 30 discordant tasks.**
A second requirement was added before the run, because this project has twice
been fooled by a comparison against the wrong baseline: the observer must also
beat the free statistics that already exist — the model's own token
log-probability and self-consistency. A control loop that only beats random is
a reranker for something the model already knows.

## The corpus

Qwen3-0.6B-Base, 500 tasks (250 train / 250 holdout, disjoint by seed
construction), 9 candidates per task (1 greedy + 8 samples at T=0.8, top-p
0.95), 4500 candidates.

| split | p1 greedy | p1 sampled | pass@8 | well formed | wrong but well formed |
|---|---|---|---|---|---|
| train | 0.520 | 0.485 | 0.748 | 1.000 | 1150 |
| holdout | 0.544 | 0.509 | 0.776 | 1.000 | 1096 |

This is the window the design needs: single-pass accuracy is well above chance
and well below the oracle, and **1096 holdout candidates are wrong and well
formed**. The property the repair corpus lacked is now present in bulk.

## The ranking signal is real

Observer: logistic regression on a 32-dim SVD of the mean-pooled final hidden
state over the completion tokens, fitted on train tasks only.

| | within-task AUC | 95% CI | discordant tasks |
|---|---|---|---|
| train | 0.663 | [0.552, 0.768] | 126 |
| **holdout** | **0.682** | **[0.573, 0.782]** | 121 |
| holdout, labels permuted | 0.496 | [0.457, 0.536] | 249 |

The CI excludes 0.5 and the permutation control collapses to chance, so the
signal is not leakage and not a split artifact. **The literal gate is met.**

## The signal does not become a decision

Selection accuracy on the 250 holdout tasks. Every arm sees the same candidates
and the same admission rule.

| arm | accuracy | 95% CI |
|---|---|---|
| greedy (1 pass) | 0.544 | [0.482, 0.605] |
| random pick (expected) | 0.513 | — |
| self-consistency | 0.540 | [0.478, 0.601] |
| max log-probability | 0.556 | [0.494, 0.616] |
| **observer head** | **0.560** | [0.498, 0.620] |
| oracle (any correct) | 0.780 | [0.725, 0.827] |

The head leads logprob by one task in 250. Paired sign tests over the same
tasks:

| head vs | wins | losses | p |
|---|---|---|---|
| logprob | 12 | 11 | **1.000** |
| self-consistency | 14 | 9 | 0.405 |
| greedy | 15 | 11 | 0.557 |

That is a wash, not a win.

## Six ways of giving the head a better shot, all null

Because this project has previously mistaken a weak feature for a dead idea,
the head was re-fitted five more ways before the conclusion was drawn.

| variant | holdout AUC | head acc | head vs logprob (p) |
|---|---|---|---|
| 8 components | 0.579 | 0.520 | 12W 21L (0.163) |
| 16 components | 0.694 | 0.564 | 13W 11L (0.839) |
| 32 components | 0.682 | 0.560 | 12W 11L (1.000) |
| 64 components | 0.660 | 0.560 | 19W 18L (1.000) |
| 128 components | 0.688 | 0.568 | 19W 16L (0.736) |
| 32 + logprob as a feature | 0.681 | 0.560 | 12W 11L (1.000) |
| ListNet ranking loss | 0.522 | 0.508 | 19W 31L (0.119) |

Two of these matter most.

**Adding the logprob to the feature vector changes nothing.** If the hidden
state carried information the model's own confidence does not, this variant
should have beaten logprob-alone. It ties it exactly, which says the hidden
state is redundant with confidence *for this decision*.

**The listwise objective overfits.** Train AUC 0.684, holdout 0.522. Optimising
the comparison directly, which is the objective a router actually needs, did
worse than pointwise BCE at this data size.

## Why 0.682 AUC produces no accuracy gain

AUC pools every (correct, incorrect) pair within a task. Most of those pairs
are easy: one candidate is right and another is grossly wrong. The head orders
those correctly and the AUC rises.

Top-1 selection is decided somewhere else — on tasks where several candidates
are plausible, and specifically on the 22 points between the head's 0.560 and
the oracle's 0.780. There the head has no edge over confidence.

So the metric and the decision disagree, and the decision is what the DAG
needs. Reporting the 0.682 as a success would have been the exact error this
project keeps finding: a real signal measured on the wrong question.

## What this settles

**A control loop built on this observer would be indistinguishable from
max-logprob.** Log-probability is free — it is computed during generation
anyway. Routing on the head instead adds a model, a feature cache, a latency
budget and a new failure mode, for 0.004 accuracy and p=1.00.

**The gate as literally written is cleared, and the loop is still not
justified.** The rule was "AUC > 0.5 before any `DAG_Scratchpad` code". That
condition is satisfied. The condition that actually matters — beat the free
baseline — is not. `DAG_Scratchpad` stays unwritten, and this time the reason is
a measured tie rather than a missing benchmark.

**If a best-of-N harness is built anyway**, it must be scored against
max-logprob and self-consistency, not against random. Against random the head
looks like a +0.047 win; against confidence it is a tie.

## What this does not say

- **Not that observers cannot work.** One backbone (0.6B base), one task family
  (two-step chains), one pooling (mean over completion tokens), one head family
  (linear on an SVD). The ranking signal is real, so a richer feature could
  change the decision result.
- **Not that the 0.682 is wrong.** It is correct and permutation-controlled. It
  is simply not the quantity the routing decision depends on.
- **Nothing about CGR.** This observer reads Qwen3-0.6B-Base hidden states. The
  shipped CGR heads were measured against the repair corpus and are closed
  ([observer-cannot-rank](2026-10-27-observer-cannot-rank.md)).
- **Nothing about training the observer jointly** with the generator, which was
  not attempted and is not a Python-level control loop.

## Reproduction

```
python experiments/arith_bench.py --n-train 250 --n-holdout 250 --steps 2 --k 8
python experiments/select_eval.py --corpus results/arith_bench_s2.pt
python experiments/select_eval.py --corpus results/arith_bench_s2.pt --permute
python experiments/select_eval.py --corpus results/arith_bench_s2.pt --loss rank
```

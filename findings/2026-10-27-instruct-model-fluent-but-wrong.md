# An instruct model gives fluent-but-wrong answers, but is too inaccurate to select from (2026-10-27)

Artifacts: `experiments/wordbench.py`, `results/_smoke_word.pt`,
`results/_smoke_word_l0.pt`, `results/_smoke_word_easy.pt`

## What was being tested

The 470M Dual-LM is at chance on arithmetic: `9 + 9 =` puts top-1 on the token
`1` with nearly flat logits
([arithmetic-out-of-support](2026-10-27-arithmetic-out-of-support.md)). That is
consistent with a base model that has no arithmetic in distribution and no
instruction to do any.

The replacement hypothesis: an **instruction-tuned** model will attempt the
problem, and a small one will attempt it *fluently and wrongly*. Fluent-but-wrong
is the exact property the repair corpus lacked, because there every wrong
candidate was also unparseable.

## The family

Notebook-and-pen money problems, reconstructed as a parametrised set (the
original prompt was not in this repository):

> A notebook costs $3 and a pen costs $2. I buy 4 notebooks and 5 pens.
> I already own 2 erasers. How much change do I get from $50?

The distractor ("2 erasers") is numeric and irrelevant, so a model that adds
everything produces a *plausible* wrong answer rather than a malformed one.
Answers are single integers, so scoring is exact match and not a judgement call.

**The admission rule is the scoring rule**: a candidate counts only if its
completion is a bare integer. Every admitted candidate is therefore well formed
by construction, and the same filter is applied to every arm downstream. This
is what makes the corpus immune to the failure that killed the repair benchmark.

Generator correctness was checked independently: 600 generated problems
re-derived from their own text, 0 mismatches.

## Result: the property is present, in full

Qwen2.5-1.5B-Instruct, fp16, 3.09 GB weights, 2685 MiB resident on a 4 GB card.

Across three difficulty settings, **every single completion was a bare
integer** — clean rate 1.000 in all six split/setting combinations. The model
understands the format completely. It simply computes the wrong number.

| setting | split | p1 greedy | p1 sampled | pass@k | wrong but admitted |
|---|---|---|---|---|---|
| change from \$50, distractor, k=4 | train | 0.250 | 0.000 | 0.000 | 16 |
| | holdout | 0.000 | 0.125 | 0.500 | 7 |
| total cost, distractor, k=6 | train | 0.100 | 0.100 | 0.200 | 54 |
| | holdout | 0.200 | 0.067 | 0.200 | 28 |
| total cost, no distractor, k=6 | train | 0.200 | 0.217 | 0.300 | 47 |
| | holdout | 0.000 | 0.033 | 0.200 | 29 |

Verbatim examples, all bare integers, all wrong:

```
truth 29  ->  12   (logprob -1.19)
truth 29  ->  17   (logprob -1.37)
truth 29  ->   9   (logprob -3.74)
truth 21  ->  20   (logprob -1.28)
truth 21  ->  28   (logprob -1.75)
```

**So the hypothesis holds: a small instruct model answers a perturbed word
problem fluently, in the right format, and wrongly.** No parser can separate
`29` from `12`.

## Result: and it is too inaccurate to select from

The same numbers kill the setting for its intended purpose.

At p1 between 0.03 and 0.22, `pass@k` sits at 0.20–0.50. That means **at most
half the tasks have a correct candidate anywhere in the sample set** — and in
the hardest setting, one task in five. A selector cannot demonstrate anything
on tasks where no correct answer exists, and the oracle is too low for headroom
to be measurable.

Removing the distractor and dropping the change step — the easiest variant the
family admits — moved holdout `p1_sampled` from 0.125 to 0.033. Difficulty is
not the knob here; the model is simply weak at multi-step money arithmetic.

This is the same failure as the 470M, for the opposite reason: the 470M was
accurate at nothing because it never tried, this model tries and is wrong.

## The uncomfortable comparison

The **0.6B base model** on two-step assignment chains reaches p1 0.50 and
pass@8 0.78 with 100% well-formed answers
([observer-ranks-but-cannot-select](2026-10-27-observer-ranks-but-cannot-select.md)).

So the instruct pivot produced a generator that is *worse* for this benchmark
than the smaller base model it replaced. Instruction tuning made the model
answer in the right format; it did not make it better at the arithmetic.

That is worth stating plainly because the premise of the pivot was "we cannot
test the DAG on a model that can't reason." The 0.6B base model could do enough
two-step state tracking to supply a usable window. The 1.5B instruct model
could not do two-item money arithmetic well enough to supply one.

## Throughput

Measured, not estimated: 0.019 task/s at k=4 and 12 new tokens, i.e. **53 s per
task**, about 0.86 s per forward-equivalent. A 500-task corpus is 4–7 hours
depending on `k` and `max_new`, because cost scales with `max_new * (1 + k)`.
The 1.5B model is near the edge of what a 4 GB Pascal card does quickly.

## What this settles

- **The family is validated.** It supplies fluent, well-formed, wrong answers
  in bulk, which is the property the repair corpus could not supply.
- **Qwen2.5-1.5B-Instruct is not a usable generator for it.** p1 too low, oracle
  too low, headroom absent.
- **The usable benchmark already exists** on Qwen3-0.6B-Base with two-step
  chains, and the observer was evaluated on it.

## What this does not say

- **Not that instruct models cannot do this.** One 1.5B model, three settings.
  A 3B or 7B instruct model would very likely land in the window, and neither
  fits in 4 GB at any precision.
- **Not that the family is at fault.** The generator was checked against an
  independent re-derivation; the model is wrong, not the problems.
- **Nothing about the observer.** No observer was trained here, because with
  `pass@6` at 0.20 there is almost nothing to select.

## Harness notes

- Qwen2.5 ships `use_sliding_window`, which `sdpa` does not implement; the
  warning is real but inert here, because prompts are ~150 tokens against a
  32768-token window.
- The tokenizer emits `token_type_ids`, which `generate` rejects. Stripped.
- `generate` inherits `temperature`/`top_p`/`top_k` from the model's
  `generation_config` and warns when `do_sample=False`. Inert, but noisy.

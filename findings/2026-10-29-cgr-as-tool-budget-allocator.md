# Standing framing: CGR is a tool-budget allocator, not a reasoner (2026-10-29)

**Status:** framing note. No new measurements. Every number below is bound to the
artifact that produced it, so the framing can be checked rather than remembered.

**Why this note exists.** The code null and the closed actuator branch are easy to
misread as "CGR is useless". They are not. They say the *payload* was wrong, not
the allocator. This note exists so that reading is never re-derived from memory,
and so that "CGR did not beat a retry loop" is never rewritten as "CGR is
unnecessary".

## The reframing in one sentence

CGR is an operating-system component for a frozen model: a **detector** decides
*when* an external tool is worth calling, a **slicer** decides *where* to resume,
and the tool supplies the **payload**. It is a memory controller and an interrupt
handler, not a thinker.

That distinction is what the ledger actually supports. Asked to think, a 0.6B
model hits its capacity ceiling and every arm converges to the same score. Asked
to route, the same model produces a 2.8x accuracy-per-call advantage over random
allocation, because routing is not a capability the model lacks.

## Ledger: claim -> artifact -> measured number

| claim | artifact | number |
|---|---|---|
| Rewind with no payload changes nothing | `results/level2_ladder_h5.json` | `rank2` 32/110; `span_digitset` 32/110, **0W/0L**, p=1.00 |
| Verbal correction is inert | `results/level2_ladder_h5.json` (constraint arm) | 30/110 vs 30/110 placebo, 15W/15L, p=1.000 |
| A correct value at the localized step is the gain | `results/level2_ladder_h5.json` `oracle_value` | **70/110**, 38W/0L, **p=7.28e-12** |
| The gain is concentrated where localization is right | `level2_ladder_h5.json` `split_by_localization` | **68 of 79** (`t_hat == t*`); the 31 mis-localized chains give 0 |
| The localizer is the router | `level2_ladder_h5.json` `localizer` | two-tier top-1 **0.718** (linear 0.618) |
| Localized calls beat random calls per call | `results/interp_calls_h5.json` | 1 call: **70 vs 25**; 2: **94 vs 52**; 3: **100 vs 66** |
| ...and the paired tests hold | `interp_calls_h5.json` `paired` | k1 46W/1L p=6.8e-13; k2 50W/8L p=1.6e-08; k3 41W/7L p=6.2e-07 |
| 3 localized calls capture most of the 5-call headroom | `interp_calls_h5.json` | (100-32)/(110-32) = **87.2%** of the 78-chain headroom |
| The gate is precision, not recovery | `results/path_a_loop_h5.json` | gated == ungated exactly (32/110, identical ms); precision **32/87 = 36.8%** |
| Slicing is a latency device | `findings/2026-10-28-path-a-ensemble-kv-slicing.md` | generation 2,732.2 -> **173.7 ms** (25.9x); loop 2,244 vs 4,498.4 (**2.0x**); vault **1.70x** |
| Code: the payload, not the allocator, failed | `results/code_pilot_decompose.json`, `results/code_mutate.json` | `cgr` 16/24 vs `greedy_seq` 17/24, 1W/2L p=1.00; **0/11** failures single-mutation local |

## Corrections to the framing, so the paper does not overclaim

Three numbers in circulation are close but not exact. Use these forms:

1. **"120.7 ms" and "26x" are generation-slice numbers.** The recorded generation
   latency is **173.7 ms per chain against 2,732.2 ms (25.9x)**. The end-to-end
   loop is **2,244 ms vs 4,498.4 ms (2.0x)**, and **1.70x** on the vault. Quoting
   26x as a loop speedup is the error already corrected once in this repo; it
   must not come back.
2. **"86% of headroom" is 87.2% at k=3**, computed as (100-32)/(110-32). Its
   denominator is `k5_all` = 110, which is a 5-call *ceiling*, not a deployable
   policy. Say "87% of the 5-call headroom", not "87% of achievable recovery".
3. **"3-4x fewer tool calls" must be stated as accuracy-per-call, not as
   substitution.** The paired test for "one localized call beats three random
   calls" is **25W/21L, p=0.66** -- it is false. What is measured is 70 vs 25 at
   one call and 100 vs 66 at three. The advantage is a rate that shrinks with
   budget, not a constant multiplier.

## The claim the paper can make

> Localizing the failing step and spending the verifier there recovers 87% of a
> five-call budget's headroom with three calls, against 66 chains for three
> random calls, while a resampling loop with no localization stalls at 32.

And the companion, equally important:

> The allocator is not a capability. With no payload -- no calculator, no
> compiler, no correct value -- localization routes to nothing and the loop ties
> resampling exactly. The payload, not the router, is the binding constraint.

## What would falsify the framing

- If a stronger model closes the localized-vs-random gap, allocation stops
  mattering and the framing collapses to a latency optimization.
- If verifier calls become free, there is nothing to allocate.
- If code repair succeeds without localization, the "router" half is not needed
  in that domain.

## Live branch

The diffusion infill run (`findings/2026-10-29-code-diffusion-preregistration.md`)
is a **payload** experiment, not an allocator experiment. Its question is whether
a span constrained by code on both sides is a better payload than left-to-right
continuation. The code null it follows is about the payload, and must never be
cited as evidence against the allocator.

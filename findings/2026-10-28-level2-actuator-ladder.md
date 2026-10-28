# Level-2 actuator ladder: is the bottleneck the token or the value? (2026-10-28)

**Status:** complete. Verdict: actuator branch closed, value branch opened.

**Artifacts:** `experiments/level2_ladder.py`, `results/level2_ladder_h5.json`,
`results/level2_ladder_rows.jsonl`

## Why this exists

A proposal to escalate to an off-the-shelf diffusion/infilling specialist
changed four things at once relative to the current loop: model identity,
attention directionality, actuator granularity, and escalation policy. Any
outcome would have been uninterpretable, including a positive one. This ladder
changes one thing at a time, on the model already resident, before any second
checkpoint is loaded.

It also tests the premise directly. The proposal's implicit claim is that a
bidirectional model can "see the relationships" and therefore compute a value
the causal model gets wrong. That is an information claim, and we have already
run it: injecting the correct constraint as text (`# constraint: x is not 7`)
scored 30/110 against 30/110 for a neutral placebo line, 15W/15L, p = 1.000
(`results/oracle_rewind_h5.json`). More correct context did nothing. The
intervention that did work was physical: banning the token, 22W/7L, p = 0.008.

## Why bidirectionality has no information to add on this corpus

`build_prompt` (`experiments/arith_steps.py:128`) places the entire definition
block and all prefix values in the left context:

```
a = 4
b = a + 3
c = b * 2
...
a -> 4
b -> 7
c ->
```

The value at step *t* is a deterministic function of what is already visible on
the left. The only right context that exists is the model's own steps *t+1..H*,
and those were computed from the value at *t*. Conditioning on them cannot
reveal the error; it reinforces it, because the continuation is consistent with
the mistake rather than contradicting it. A bidirectional reader of a corrupted
chain is a reader of a self-consistent wrong story.

There is therefore no independent right-side constraint in this benchmark. That
is the condition under which infilling helps (a Sudoku row summing to 45, a
program that must compile), and it is absent here.

## Arms

All arms are ungated and use the frozen two-tier localizer's `t_hat`, so they
are paired on the same 110 rows as `results/path_a_loop_h5.json`. The gate is
excluded because it is a no-op on recovery: gated and ungated two-tier arms are
identical at 32/110 with identical token and time totals.

| Arm | What changes | What it isolates |
|---|---|---|
| `rank2` | none (baseline) | must reproduce 32/110 |
| `span_digitset` | bans every digit of the bad value | the proposal's "mask the whole span" actuator |
| `resample_neq` | samples, rejects only the exact bad value, up to 4 draws | the loosest possible actuator upgrade |
| `resample_neq_best4` | 4 draws, drop the bad value, keep the highest step logprob | sampling width with no ground-truth selection |
| `oracle_value` | injects `truth[t_hat]` | the ceiling reachable *through this localizer* |
| `resample_at_tstar` | 4 unmasked draws at the true step, success if any recovers | bound on any actuator that only changes the value at `t*` |

`span_digitset` is expected to be mechanically handicapped: for a two-digit bad
value it also bans the truth whenever the digits overlap, so it can lose to
`rank2` for reasons unrelated to actuator quality. That is why the loose
variant exists.

## Pre-registered decision rule

Level 2 is worth building **iff**

1. the best actuator beats `rank2` on paired rows at exact McNemar p < 0.05, **and**
2. `oracle_value - rank2 >= 15` chains.

Otherwise Level 2 closes as measured, not as failed-to-load.

Rule 2 is the load-bearing one. `oracle_value` hands the loop the correct value
at the step the localizer actually chose. Infilling is strictly weaker than
knowing the value, so no infiller can exceed that arm. If the arm does not clear
`rank2` by much, the actuator branch has no headroom regardless of which
checkpoint is swapped in.

## Results (N=110, 27 min, peak VRAM 2,294 MB)

`rank2` reproduces `path_a_loop_h5.json` exactly at 32/110, so the ladder is
anchored to the committed loop.

| Arm | Recovered | Rate | 95% CI | Same value | Paired vs `rank2` |
|---|:---:|:---:|:---:|:---:|:---:|
| `rank2` | 32/110 | 0.291 | [0.214, 0.382] | 0.00 | — |
| `span_digitset` | 32/110 | 0.291 | [0.214, 0.382] | 0.00 | 0W / 0L, p = 1.00 |
| `resample_neq` | 25/110 | 0.227 | [0.159, 0.314] | 0.28 | 9W / 16L, p = 0.23 |
| `resample_neq_best4` | 32/110 | 0.291 | [0.214, 0.382] | 0.27 | 10W / 10L, p = 1.00 |
| `oracle_value` *(ceiling)* | **70/110** | **0.636** | [0.543, 0.720] | 0.16 | 38W / 0L, p = 7.3e-12 |
| best-of-4 at `t*`, oracle judge *(ceiling)* | 46/110 | 0.418 | [0.330, 0.512] | — | 14W / 0L, p = 0.029 |

1. **The strict span mask is not a stronger actuator; it is the same actuator.**
   `span_digitset` and `rank2` differ on **0 of 110 rows**. Once the first digit
   is banned, the model's next-best token already differs, so banning the
   remaining digits changes nothing. The proposal's "mask the entire span" idea
   is a no-op in greedy decoding on this corpus.

2. **Sampling does not beat masking.** Rejecting only the exact bad value loses
   a net 7 chains (9 gained, 16 lost); four draws selected by step logprob are a
   pure wash (10 gained, 10 lost, 20 rows flip). Rank-2 keeps the model's ranked
   preference; sampling discards it. `resample_neq` re-emits the bad value on
   28% of rows even after four attempts.

3. **Value injection is monotone.** 38 chains gained, 0 lost. It recovers 68 of
   the 79 correctly localized rows and 2 of the 31 mis-localized ones (where the
   injected value re-anchors the chain).

4. **The headroom is value-level, and the binding constraint is now
   localization.** With perfect localization the same injection reaches ~94/110
   (`oracle_force`, 0.855). The 70-vs-94 gap is 31 mis-localized rows, not a weak
   actuator. Improving the localizer is now worth more than improving the ban.

5. **More samples are not a substitute for knowing.** Four draws at the true step
   with a perfect judge reach 46, still 24 below knowing the value.

## Verdict

**The actuator branch closes.** No deployable actuator beats `rank2`, so the
pre-registered rule fails on its first condition. The `31.8%` oracle-localized
ceiling is an actuator-family limit, not a tuning problem: subtractive methods
are exhausted.

**The value branch opens.** `oracle_value` is not an oracle in the deployment
sense: the definition block is in the prompt, so a subprocess can compute the
step's value without any ground-truth leak. That makes 70/110 (0.636) a
deployable number for a calculator actuator, capped at ~94/110 by localization
error.

This also closes the diffusion proposal on this benchmark. An infiller is a
value *guesser* with strictly less information than the calculator. It cannot
beat 70, and it would be competing with a tool that already computes the answer.

## Correction to the pre-registered rule

The rule as first coded searched `arms[1:]` for the "best actuator", which
included `oracle_value`, and therefore fired `build_level2` on the ceiling arm.
`oracle_value` and `resample_at_tstar` are ceilings, not actuators. The script
now excludes them (`deployable_actuators` in the payload), and the corrected
verdict is `close_level2_actuator_branch`. Disclosed because a rule that can
select its own ceiling is not a decision rule.

## What this cannot claim

- No diffusion or infilling checkpoint was run. This measures the ceiling such
  a checkpoint would have to beat, not the checkpoint itself.
- Tuning holdout ($N=110$) only. The vault ($N=331$) has no per-row log for the
  loop, so the same ladder there needs its own logging pass.
- No latency or compute claim. The ladder is an accuracy ablation; model swap
  cost is not measured and would only make the loop slower.

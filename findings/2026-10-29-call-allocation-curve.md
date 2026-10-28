# Call-allocation curve: what is "we know what to put where" worth? (2026-10-29)

**Status:** complete. Verdict: localization is a 2-4x verifier-call multiplier and
the advantage is a rate, not a constant. The pre-registered falsifier did not
fire.

**Artifacts:** `experiments/interp_calls.py`, `results/interp_calls_h5.json`,
`results/interp_calls_rows.jsonl`

## Why this exists

The actuator ladder closed the subtractive branch. No deployable ban beats
rank-2 at 32/110, while injecting the correct value at the localized step
reaches 70/110 (`results/level2_ladder_h5.json`). The headroom is therefore
value-level, and the residue is localization error: 68 of 79 correctly localized
rows recover, 2 of 31 mislocalized ones do. With perfect localization the same
injection reaches ~94/110 (`oracle_force`, `results/oracle_rewind_h5.json`).

That makes the interpreter the weapon and the localizer the aiming system. This
experiment prices the aiming system.

## The claim this can and cannot support

If the interpreter is run at every step, the chain is correct by construction
and the model is decorative. Accuracy is then not the interesting number. The
interesting number is **how few interpreter calls are needed**, so the claim is:

> localized calls reach a given accuracy with fewer interpreter calls than
> randomly placed calls.

This is a compute-matched claim and it is the only honest form of the CGR claim
in a domain with an exact verifier. It cannot support "we solve more puzzles",
because `k5_all` solves everything by construction.

## The interpreter is deployable, not an oracle

Values are computed by `eval_lines` from the definition block in the prompt,
i.e. exactly what a Python subprocess would return:

```
a = 4          ->  a = 4
b = a + 3      ->  b = 7
c = b * 2      ->  c = 14
```

No ground truth is read. The generator's stored `truth` is used only as an
assertion that the evaluator is correct: the script aborts if the subprocess
evaluator disagrees with `truth` on any holdout task, and it did not.

The only ground-truth use is in the `k1_oracleloc` arm, which is labelled a
ceiling and is not a deployable arm.

## Arms (N=110, paired on the same rows as `path_a_loop_h5.json`)

| Arm | Interpreter calls | Placement | What it isolates |
|---|:---:|---|---|
| `k0_rank2` | 0 | — | current CGR; must reproduce 32/110 |
| `k1_loc` | 1 | `t_hat` | must reproduce 70/110 (`oracle_value`) |
| `k2_loc` | 2 | `t_hat` + runner-up | does a second call buy much? |
| `k3_loc` | 3 | top-3 localizer steps | saturation point |
| `k1_rand` | 1 | uniform random | matched-compute control at k=1 |
| `k2_rand` | 2 | uniform random | matched-compute control at k=2 |
| `k3_rand` | 3 | uniform random | matched-compute control at k=3 |
| `k1_oracleloc` | 1 | true error step | perfect binding (ceiling) |
| `k5_all` | 5 | every step | plumbing sanity check; must be 110/110 |

Localized ranking: the two-tier argmax is rank 1, so `k1_loc` is anchored to the
committed loop. Remaining steps are ordered by descending mean ensemble
probability (mean of the linear and transformer softmax over 5 seeds).

Random placement is seeded per row (`random.Random(90000 + j)`), so it is
reproducible and does not resample across a `--resume`.

## Pre-registered readings

Three anchors must reproduce exactly, or the arms are not comparable:

1. `k0_rank2` = 32/110 (`level2_ladder_h5.json` arm `rank2`)
2. `k1_loc` = 70/110 (`level2_ladder_h5.json` arm `oracle_value`)
3. `k1_oracleloc` = 94/110 (`oracle_rewind_h5.json` arm `oracle_force`)

The script asserts all three and writes them into the payload as `anchors`.

Anchor 3 is the one that pins the replay semantics. `oracle_force` reuses the
chain's original prefix up to the error step, injects the value, and regenerates
only downstream. Any harness that instead regenerates steps `0..t-1` is running
a different experiment.

## Correction made during the first run: the prefix bug

The first full run failed its anchor: `k1_loc` = **72/110**, not 70/110. The
cause was a bug in `fill_with`, not in the localizer or the interpreter.

`fill_with` walked the chain from step 0 and **regenerated every step before the
first injection**, instead of reusing the values the chain already emitted.
Greedy regeneration does not reproduce the stored prefix, so two rows took a
different path downstream. Both anchors `k0_rank2` and the localizer were
unaffected; only the replay arms were.

Fixed semantics, now asserted:

```
seq = emitted[:first_injected]        # original prefix, untouched
for i in first_injected .. steps-1:
    if i injected: seq.append(truth[i])
    else:          seq.append(greedy(prompt(seq, i)))
```

With this, `k1_loc` is by construction the ladder's `oracle_value` and
`k1_oracleloc` is `oracle_force`. The failing rows were deleted and the run was
restarted from scratch; no row from the buggy run is reported. This is the
reason the anchors exist: without them, 72/110 would have looked like a fresh
draw rather than a harness error.

## Primary comparison, paired exact McNemar on the same rows

- `k1_loc` vs `k1_rand` — the headline. **If this is not significant, the
  localizer is decoration and the correct policy is to run the interpreter
  everywhere.**
- `k2_loc` vs `k2_rand`, `k3_loc` vs `k3_rand` — does the advantage persist or
  decay as calls increase?
- `k1_loc` vs `k0_rank2` — the value of one interpreter call over the best
  subtractive actuator (expected 38W/0L, already measured).

Secondary, descriptive:

- `ok/call` for each arm, to price the aiming system.
- The call count at which random placement first matches `k1_loc`'s accuracy.
- `k1_oracleloc` minus `k1_loc`: the remaining value of perfect binding, i.e.
  how much localization error still costs at k=1.

## What each outcome would mean

- **`k1_loc` >> `k1_rand` and the gap persists at k=2,3:** localization is the
  asset. The next investment is the localizer, not the actuator.
- **`k1_loc` >> `k1_rand` at k=1 but the gap closes by k=3:** localization buys
  a constant, not a rate. Report it as a constant and stop.
- **`k1_loc` ≈ `k1_rand`:** the localizer is decoration on this corpus. The
  honest policy is `k5_all`, and CGR has no claim in verifier domains.
- **`k1_oracleloc` ≈ `k1_loc`:** localization error no longer costs anything at
  k=1, and the 70-vs-94 gap is not about placement.

## Results (N=110, 946 s, peak VRAM 2,328 MB)

**All three anchors reproduced exactly**: `k0_rank2` 32/110, `k1_loc` 70/110,
`k1_oracleloc` 94/110. The arms are therefore on the same rows as the committed
ladder and rewind runs.

| Arm | Recovered | Rate | 95% CI | Calls | ok/call |
|---|:---:|:---:|:---:|:---:|:---:|
| `k0_rank2` | 32/110 | 0.291 | [0.214, 0.382] | 1 | 0.291 |
| `k1_loc` | 70/110 | 0.636 | [0.543, 0.720] | 1 | **0.636** |
| `k2_loc` | 94/110 | 0.855 | [0.777, 0.908] | 2 | 0.427 |
| `k3_loc` | 100/110 | 0.909 | [0.841, 0.950] | 3 | 0.303 |
| `k1_rand` | 25/110 | 0.227 | [0.159, 0.314] | 1 | 0.227 |
| `k2_rand` | 52/110 | 0.473 | [0.382, 0.565] | 2 | 0.236 |
| `k3_rand` | 66/110 | 0.600 | [0.507, 0.687] | 3 | 0.200 |
| `k1_oracleloc` *(ceiling)* | 94/110 | 0.855 | [0.777, 0.908] | 1 | 0.855 |
| `k5_all` *(sanity)* | 110/110 | 1.000 | [0.966, 1.000] | 5 | 0.200 |

### Paired at matched k (pre-registered)

| Comparison | Wins / losses | p |
|---|---|---|
| `k1_loc` vs `k1_rand` | **46W / 1L** | **6.8e-13** |
| `k2_loc` vs `k2_rand` | 50W / 8L | 1.6e-8 |
| `k3_loc` vs `k3_rand` | 41W / 7L | 6.2e-7 |
| `k1_loc` vs `k0_rank2` | 38W / 0L | 7.3e-12 |

The advantage **does not decay with k**. It is a rate, not a constant, which is
the pre-registered "localization is the asset" branch.

### Across k (exploratory, not pre-registered)

| Comparison | Wins / losses | p | Reading |
|---|---|---|---|
| `k1_loc` vs `k2_rand` | 34W / 16L | 0.015 | one aimed call beats two random calls |
| `k1_loc` vs `k3_rand` | 25W / 21L | **0.66** | **tie**: same accuracy, one third of the calls |
| `k2_loc` vs `k3_rand` | 38W / 10L | 6.2e-5 | two aimed calls beat three random calls |
| `k2_loc` vs `k1_oracleloc` | 13W / 13L | 1.00 | two aimed calls equal one perfectly-bound call |

**The "one call beats three random calls" claim is false as stated.** `k1_loc`
has the higher point estimate (0.636 vs 0.600) but the paired difference is a
tie (25W/21L, p = 0.66). The defensible claim is *same accuracy at one third of
the verifier calls*, not a win. It is recorded this way because the
pre-registered comparison was at matched k, and the cross-k comparison was not.

### Where the remaining error is

- Minimal localized k: **70 at k=1, +24 at k=2, +6 at k=3, 10 never.** The arms
  are nested by construction, so this is a genuine curve.
- The random arms are **independent draws**, not nested (`rng.sample` per k), so
  there is no minimal-k curve for random; only the per-k totals are meaningful.
- Rank of the true error step in the localizer's order: **79 rank 1, 16 rank 2,
  6 rank 3, 6 rank 4, 3 rank 5.** Top-2 covers 95/110 (86.4%); top-3 covers
  101/110 (91.8%).
- On the 31 mislocalized rows: `k1_loc` 2, `k2_loc` **18**, `k3_loc` 22,
  `k1_oracleloc` 26. **A second localized call recovers 18 of the 31 rows the
  first call misses**, so localization error is insurable, not fatal.
- `k3_loc` still fails on 10 rows: **9 of them have the true step at rank 4 or
  5**, and only 1 is correctly localized. The residual is the localizer's
  ranking depth, not the actuator and not the generator.

## Verdict

**The aiming system is worth 2-4x in verifier calls, and its advantage is a
rate.**

1. At the 70% level, one localized call suffices; random placement has not
   reached 70 after three calls (66/110), so it needs **at least four**.
2. At the 94% level, two localized calls beat three random calls (38W/10L,
   p = 6.2e-5).
3. Two localized calls match **perfect binding** (94 vs 94, p = 1.00) — but on
   different rows (81 shared, 13 each way). Localization error can be bought
   back with a second call rather than a better localizer.
4. Per-call efficiency is 2.8x random at k=1 (0.636 vs 0.227) and still 1.5x at
   k=3 (0.303 vs 0.200).

The pre-registered falsifier did not fire: localized placement is not
decoration. The policy this supports is **check the top two localized steps**,
which reaches 94/110 — the same ceiling as perfect localization at one call —
and leaves the last 10 rows to localizer ranking depth.

This is the transferable number for the target domains. In code the verifier is
a test run, which costs seconds rather than microseconds, so a 2-4x reduction in
verifier calls is the product. In Sudoku the verifier is free, which is why
Sudoku cannot carry this claim (`2026-10-29-sudoku-closed-as-target.md`).

## Limits, stated up front

- This is arithmetic, where a calculator solves the benchmark. The result is a
  statement about call allocation under a tool, not about reasoning.
- One model (Qwen3-0.6B fp32), one localizer, one corpus, one seed per random
  placement. Random placement is a control, not an optimum.
- The claim transfers to Sudoku and code only as a hypothesis: there the
  interpreter is a legality checker or a test runner, and "which step to check
  first" is the same question, but it has not been measured.

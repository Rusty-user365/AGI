# Code pilot: the repair loop loses to a plain retry loop (2026-10-29)

**Status:** stopped at Stage 1 (N=24) under the pre-registered falsifier. The
full 164 was not run. Pre-registration:
`findings/2026-10-29-code-pilot-preregistration.md`.

**One sentence:** at equal accuracy, hinted whole-body repair costs 15% more
generated tokens than a plain greedy-then-resample loop, and solves one fewer
problem; the apparent "half the tokens" win over `best_of_4` was early stopping,
not repair.

## Results

All five arms on the same 24 problems, same model (Qwen3-0.6B base, fp32), same
5 s verifier. Wilson 95% CIs. Peak VRAM 2,648 MB. Run time 1,228 s.

| arm | what it does | solved | 95% CI | tokens | tok/solve | tests |
|---|---|---:|---|---:|---:|---:|
| `greedy` | one body | 13/24 | [0.351, 0.721] | 4,974 | 383 | 44 |
| `best_of_4` | 4 samples, tests select | 16/24 | [0.467, 0.820] | 26,508 | 1,657 | 63 |
| `sequential` | the same 4 samples, stop at first pass | 16/24 | [0.467, 0.820] | 9,774 | 611 | 63 |
| `greedy_seq` | greedy, then unhinted samples, early stop | **17/24** | [0.508, 0.851] | **11,930** | 702 | 62 |
| `cgr` | greedy, then hinted localized repairs | 16/24 | [0.467, 0.820] | 13,741 | 859 | 61 |

Paired exact McNemar:

| comparison | result |
|---|---|
| `cgr` vs `best_of_4` | 2W/2L, p = 1.00 |
| `cgr` vs `sequential` | 2W/2L, p = 1.00 |
| `cgr` vs `greedy_seq` | 1W/2L, p = 1.00 |
| `best_of_4` vs `greedy` | 4W/1L, p = 0.375 |
| `cgr` vs `greedy` | 3W/0L, p = 0.250 |

## The 0.518x token result was early stopping

Against `best_of_4`, `cgr` looks twice as efficient: 13,741 vs 26,508 tokens at
identical accuracy. That comparison is invalid as a repair claim, because
`best_of_4` always pays for four samples while `cgr` stops after one when greedy
already passes.

`sequential` is the fair baseline: the same four samples, tested one at a time,
stopping at the first pass. It solves the same 16 problems for 9,774 tokens,
**0.369x** of `best_of_4` and **0.71x** of `cgr`. Early stopping, available to
any retry loop, accounts for the entire apparent win.

Against `sequential`, `cgr` costs **1.406x** the tokens at equal accuracy.

## The ablation that decides it

`greedy_seq` is the harness ablation: identical greedy first attempt, identical
early stop, but unhinted whole-body samples instead of hinted localized repairs.
It isolates the treatment (the failing assertion in the prompt, and the cut).

- `greedy_seq`: **17/24 at 11,930 tokens**
- `cgr`: **16/24 at 13,741 tokens**

The treatment is negative on both axes. `cgr` pays **1.152x** the tokens and
loses one problem.

On the 11 problems where greedy failed:

| arm | solved | tokens |
|---|---:|---:|
| `cgr` | 3/11 | 11,330 |
| `sequential` | 4/11 | 6,956 |

`cgr` is **1.63x** more expensive on exactly the subset the loop exists to fix.

Why the cost is structural, not noise: every hinted repair pays a 30-60 token
prompt surcharge for the hint block, and the model generates longer bodies after
the hint (~190 vs ~110 tokens) without producing more correct ones. More tokens
per attempt and no accuracy gain cannot reverse with a larger N.

## By failure kind (underpowered, reported as such)

Split by the *greedy* failure kind, so the groups are descriptive, not
randomised:

| greedy failure | n | greedy | best_of_4 | cgr |
|---|---:|---:|---:|---:|
| `assert` | 8 | 0 | 2 | 3 |
| `exception` | 3 | 0 | 2 | 0 |
| solved by greedy | 13 | 13 | 12 | 13 |

Two readings, both weak at this N:

- **`assert` (n=8):** `cgr` 3 vs `best_of_4` 2. This is the hint-conditioned
  arm (cut = 0), the only place the treatment looks positive, +1 problem. Not
  significant and not evidence for localization, because cut = 0 does not
  localize.
- **`exception` (n=3):** `cgr` 0 vs `best_of_4` 2. The localized cut did not
  help, but three problems cannot distinguish "localization is harmful" from
  "the model cannot fix the statement it was shown".
- **greedy-solved (n=13):** `cgr` 13 vs `best_of_4` 12. Greedy is stronger than
  a single T=0.8 sample, so `cgr` inherits a better first attempt. That is the
  greedy-first effect, not repair.

Localization first-repair hit rate: **2/11** repaired problems passed on the
first repair; mean repairs 1.54.

## The falsifier fired

Pre-registered: *"if `cgr_4` does not beat `best_of_4` at matched tokens, the
loop is decoration in code and there is no CGR claim to make."*

At matched budget, `cgr` 16/24 vs `best_of_4` 16/24. It does not beat it. The
falsifier fired, and the decomposition shows the loss is worse than a tie
against a fair baseline.

## Protocol deviation, disclosed

The mechanical futility gate (`stop if wins == 0 or cgr <= best_of - 3`) **did
not fire**: the paired result was 2W/2L with a 0 difference. The stop is based
on the falsifier plus the `sequential` / `greedy_seq` decomposition.

The decomposition was built after Stage 1, so it is a post-hoc analysis. It uses
only the stored samples and re-runs the verifier; it generates nothing new. Its
token comparison is structural (hint surcharge plus longer generations), not a
sample-size artifact, which is why it is a reason to stop rather than a reason
to collect more rows.

A reader who insists on the mechanical gate should read this as: Stage 1 is a
pre-registered falsifier stop, not a gate stop.

## What this does not claim

- **Not "the model cannot code."** HumanEval is public and almost certainly in
  pretraining; the 13/24 greedy rate is not a capability claim.
- **Not "localization is useless."** The exception subset is n=3, and the assert
  arm used cut = 0, so localization was barely exercised.
- **Not "CGR is dead in every domain."** This is a 0.6B base model doing
  full-function synthesis, where the repair *content* is wrong, not merely
  mislocated. Arithmetic had a calculator as the actuator; code has no
  equivalent.
- **No claim** about the static judge, the gate, a DAG, or compute-matched
  capacity. None of them participated.

## What it does establish

1. **Executable feedback in the prompt is not free and, for this model, not
   helpful.** It costs tokens and does not buy accuracy. This is consistent with
   the arithmetic result (prose feedback inert, 30/110 vs 30/110) and the judge
   result (confidence inverts after step 1): the binding constraint is the
   model's ability to *act* on feedback, not the availability of feedback.
2. **The repair loop's apparent efficiency was early stopping.** Any retry loop
   has it. Future code claims must use `greedy_seq` or `sequential` as the
   baseline, never `best_of_4` alone.
3. **Localization is not the binding constraint on this domain yet.** Finding
   the right line does not help if the model cannot write a correct replacement
   for it. This inverts the arithmetic conclusion, where localization was worth
   2-4x in verifier calls.

## If this line continues

Change the treatment, not the sample size:

- a model that can use a concrete failing input (the 0.6B cannot, at least not
  from one assert line);
- a localizer that changes *what* is rewritten (the assert arm used cut = 0, so
  no localization happened there; the exception arm localized and still lost);
- `greedy_seq` as the baseline in any future code claim;
- no static judge until a treatment beats `greedy_seq`, because a cheaper screen
  cannot rescue a treatment that loses at equal verifier calls.

## Artifacts

| Path | What |
|---|---|
| `experiments/code_loop.py` | the five-arm harness; resumable, seeded, budget-capped |
| `experiments/code_bench.py` | corpus, truncation, sandboxed verifier, `FAILED_LINE` |
| `experiments/code_audit.py` | pre-run audit: test sensitivity, localization, truncation, assembly |
| `experiments/code_gate.py` | the pre-registered futility gate |
| `experiments/code_decompose.py` | `sequential` and `greedy_seq` baselines from stored samples |
| `results/code_pilot_rows.jsonl` | 24 rows, full bodies retained |
| `results/code_pilot_stage1.json` | arm summary, paired tests, by-kind split |
| `results/code_pilot_decompose.json` | decomposition, per-problem detail |
| `results/code_pilot_stage1.log` | per-problem trace |

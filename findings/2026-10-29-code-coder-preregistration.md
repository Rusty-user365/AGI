# Code pilot with a specialist coder: does a model that can code use the hint? (2026-10-29)

**Status:** pre-registration. No results yet.

**Artifacts to be produced:** `results/code_coder_rows.jsonl`,
`results/code_coder.json`, `results/code_coder_decompose.json`

## Why this run exists

The first code pilot (`findings/2026-10-29-code-pilot-null.md`) found that
hinted repair lost to a plain retry loop. That result is **model-bound**: the
resident model was `Qwen3-0.6B-Base`, which cannot write a correct replacement
even when shown the failing line. The pilot therefore tested whether a model
that cannot code can use executable feedback, not whether the architecture works
with a model that can.

This run swaps in a coding specialist. The architecture's claim was always
conditional on an actuator that can act; this is the actuator.

## Model and memory

`Qwen/Qwen2.5-Coder-1.5B-Instruct`, fp16, ~3.0 GB of weights on a 4 GB GTX 1050.
It replaces the 0.6B for the whole loop; the two are never resident together.

The 0.6B localizer is **not** used, for three reasons:

1. it does not fit alongside a 1.5B model (1.2 + 3.0 GB > 4 GB);
2. it was trained on Qwen3 hidden states and would not transfer to Qwen2.5;
3. this pilot's localization is heuristic anyway: traceback for exceptions,
   cut = 0 for assertion failures.

Sequential offload (load the coder, run, unload, load the localizer) is a
systems optimization, not part of this claim, and is not done here.

## Arms

Same harness, same verifier, same 24-problem stage-1 set, plus the full 164 if
stage 1 passes its gate.

| Arm | What it does |
|---|---|
| `greedy` | one body, tests decide |
| `best_of_4` | four samples, tests select |
| `greedy_seq` | greedy, then unhinted samples, stop at first pass (derived offline from stored samples) |
| `cgr` | greedy, then hinted repairs, stop at first pass, budget-capped at `best_of_4` |

`sequential` (the same four samples tested one at a time) is also derived
offline. It is reported but is not the primary baseline.

## Sampling parameters, stated because they are inherited

The loop sets `temperature=0.8`, `top_p=0.95`, and `do_sample` per arm, but it
does **not** override two values in the coder's `generation_config.json`:

- `top_k = 20`
- `repetition_penalty = 1.1`

They apply to both sampling and greedy decoding, and to every arm, so the
arm comparison is internally valid. They are recorded because they differ from
the 0.6B base run (which had no such config), so absolute rates are not compared
across models. The model card recommends these values for Qwen2.5-Coder, which
is why they were left in place rather than neutralised after the run started.

## Primary comparison

**`cgr` vs `greedy_seq` at matched generated tokens**, paired exact McNemar, on
all problems run.

`greedy_seq` is the honest baseline from the first pilot: it has the same greedy
first attempt and the same early stopping, and differs only in that its retries
are unhinted. Whatever `cgr` gains over it is the treatment.

`best_of_4` is reported but is not the primary baseline, because it always pays
for four samples and therefore loses on tokens for reasons unrelated to repair.

## Prediction, stated before the run

- **Absolute pass rates rise for every arm.** A 1.5B coder should solve far more
  than the 0.6B base. That is expected and is **not** evidence for the
  architecture.
- **The gap is the question.** If the coder can use a concrete failing input,
  `cgr` should beat `greedy_seq` on the `assert` subset. If it solves most
  problems greedily, the hard subset shrinks and the comparison loses power; in
  that case the run is inconclusive, not negative.
- **The hint surcharge remains.** Each hinted attempt still pays the extra
  prompt tokens. For `cgr` to win, the hint must buy more accuracy than that
  surcharge costs.

**Falsifier:** if `cgr` does not beat `greedy_seq` at matched tokens on the
subset where greedy failed, the hint adds nothing even for a coding specialist,
and the repair loop is not the product. Report that as the finding.

## Metric

Problems solved at matched generated tokens; tokens counted from actual
generation length (prompt charged on every attempt); test runs per solve; split
by greedy failure kind (`assert`, `exception`, solved-by-greedy); localization
first-repair hit rate. Wilson CIs and paired McNemar throughout.

## Contamination, unchanged

HumanEval is public and likely in pretraining. Absolute pass rates are not a
capability claim. Every comparison is paired on the same problems.

## What this run cannot show

- It cannot show that localization is the binding constraint, because the
  `assert` arm uses cut = 0 and does not localize.
- It cannot isolate the specialist from the prompt format: the coder is an
  instruct model and may prefer a chat template. The same raw-completion format
  is used for every arm, so the arm comparison is fair, but the absolute rate
  may be below the model's chat-mode ability.
- It cannot support a compute-matched or capacity-matched claim against
  LoRA-only adaptation. None of that is measured here.

## Cost

Measured on the 1050 with the 1.5B class in fp16 before launch:

- weights resident: **2,965 MB** peak (fits, with the display on integrated
  graphics);
- load time: 190 s;
- generation: **9.4 tok/s**, so a 192-token body is ~20 s.

At ~5 generations per problem this is ~100 s/problem, i.e. **~40 min for 24
problems** and **~4.5 h for 164**. Stage 1 is therefore the first 24, and the
remaining 140 are appended with `--resume` only if the hard subset is large
enough to compare on.

A signal from the feasibility run: the 1.5B instruct model solved
`HumanEval/0` correctly on the first greedy attempt (nested loop over all
pairs). If the coder behaves similarly, most problems will be solved greedily
and the hard subset will be small. That is expected, and it is why the primary
analysis is conditional on greedy failure rather than on the full set.

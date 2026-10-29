# Code repair with a code-trained diffusion model: pre-registration (2026-10-29)

**Status:** pre-registration. No results yet. This is a **deviation** from the AR
HumanEval line, not a continuation of it.

**Artifacts to be produced:** `results/diffusion/<arm>.jsonl`,
`results/diffusion_summary.json`

## Why this run exists, and why the earlier AR line does not cover it

The Level-2 actuator ladder closed diffusion **on arithmetic**, and said so
explicitly (`findings/2026-10-28-level2-actuator-ladder.md`):

> "There is therefore no independent right-side constraint in this benchmark.
> That is the condition under which infilling helps (a Sudoku row summing to 45,
> **a program that must compile**), and it is absent here."

Code is the named excluded case. The AR HumanEval pilots then ran as if the
closure applied everywhere. It did not. This run tests the case the closure
carved out.

The first code pilot's null is also model-bound: `Qwen3-0.6B-Base` cannot write
the fix even when shown the failing line, and **0/11** of its failures are one
operator/literal flip from passing (`results/code_mutate.json`). Both facts are
about the actuator, not about directionality.

## What is actually different about a diffusion repair

Not information. The failing test can be put in the prompt for both models. The
differences are:

1. **Revision.** An AR model commits each token; a masked-diffusion model
   rewrites the whole span at every denoising step and can change its mind.
2. **Two-sided constraint.** The span is generated between fixed left context
   (the prompt) and fixed right context (the rest of the body and the failing
   test). An AR continuation only ever sees what is to its left.

So the hypothesis under test is: **given the same broken program and the same
failing test, does a model that must produce a span consistent with both sides,
and can revise it, repair more often per forward pass than one that continues
left-to-right?**

If yes, that is the first evidence for the infilling branch. If no, the branch
closes on the one domain where the record said it should open.

## Models

| role | checkpoint | params | precision |
|---|---|---|---|
| diffusion (D) | `dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1` | 0.5B | fp16 |
| AR twin (A) | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 0.5B | fp16 |

D is an MDLM fine-tune **of A**, on `opc-sft-stage1/2`. Same base, same
tokenizer, same width. That is the capacity match: the only differences are the
objective and the mask direction. A 1.5B AR coder is available as a capability
ceiling but is not part of the main comparison.

D's custom `modeling_qwen2.py` is loaded with `trust_remote_code=True` in a
**separate venv** (`transformers==4.57.0`, reusing the system torch). The
checkpoint's own code is not patched. Rationale: the file imports
`TransformersKwargs` from `transformers.utils`, which does not exist in the
system 4.51.3, and `create_causal_mask` from `transformers.masking_utils`.
Patching custom model code to fit an older transformers is exactly the kind of
silent behaviour change this project has been burned by before.

**Verified from source, before any run:** the active mask path in
`modeling_qwen2.py` (lines 85–106) builds a padding-only
`_prepare_4d_attention_mask`; the causal version is commented out. The model is
bidirectional. This will also be verified **empirically**: perturb a token at
position *j* and confirm the hidden state at position *i < j* changes. If it
does not, the checkpoint is not bidirectional and the run stops there.

## The repair set, and why it is paired

The set is **problems where A's greedy attempt fails**. A's failing body is then
the *shared input* to every repair arm.

This removes the confound that would otherwise dominate: D may simply be a worse
coder than A, because the MDLM fine-tune can damage the base. If each arm
started from its own model's first attempt, a low D score would be
indistinguishable from "diffusion is worse at coding". Starting every repair arm
from the same broken program isolates the **repair mechanism**.

`ar_greedy` and `mdlm_full` are still run on all problems, as a damage check on
the fine-tune, and are reported separately. They are not the primary comparison.

## Arms

| arm | model | what it does |
|---|---|---|
| `ar_greedy` | A | one body, tests decide |
| `mdlm_full` | D | full body by iterative unmasking (block sampler from the model card), tests decide |
| `ar_hinted` | A | rewrite the body with the failing test as a prompt comment (cut = 0 for asserts, traceback cut for exceptions) |
| `mdlm_infill` | D | mask the localized span of A's failing body, keep left context **and** right context (rest of body + failing test), denoise the span |
| `mdlm_infill_nohint` | D | identical, but the failing test is **removed** from the right context (placebo control) |

`mdlm_infill_nohint` exists because the arithmetic chapter established that the
hint must be shown to matter. If `mdlm_infill` and `mdlm_infill_nohint` score the
same, the test text is inert in diffusion too, and any gain is not from the
failure information.

## Localization, stated because it is weak

- **Exception failures:** the traceback frame gives the statement. Mask exactly
  that statement.
- **Assertion failures:** the traceback lands in the test, not the candidate, so
  there is no free localization. Mask the **whole body**. This is not infill; it
  is the weak branch, and it will be reported as a separate subset.

The same rule is applied to `ar_hinted`, so the localization quality is matched
across arms. A difference between arms therefore cannot come from one arm
getting a better cut.

## Matched compute, defined precisely

A forward pass is the unit. KV-cached AR generation of *L* tokens costs *L*
forward passes; an MDLM run of *S* denoising steps costs *S*.

For every `mdlm_infill` attempt, **S is set to the number of tokens
`ar_hinted` generated for the corresponding attempt on the same problem**. That
makes the two arms equal in forward passes per attempt, per problem, exactly.

`mdlm_full` is matched against `ar_greedy` the same way, using A's greedy length.

Reported alongside: wall clock, peak VRAM, verifier calls, and *S* itself, so
the reader can see what was traded.

## Primary comparison

Paired exact McNemar, on the repair set:

**`mdlm_infill` vs `ar_hinted`**, final pass rate, at matched forward passes.

Secondary: `mdlm_infill` vs `mdlm_infill_nohint` (does the hint matter?),
`mdlm_full` vs `ar_greedy` (did the fine-tune damage the coder?).

Wilson CIs on every rate. No pooling with the 0.6B pilot, the 1.5B coder run, or
any arithmetic result.

## Predictions, stated before the run

1. `mdlm_full` will be **worse** than `ar_greedy`. A 0.5B model fine-tuned to a
   diffusion objective for a short run usually loses unconditional generation
   quality. If it is not worse, that is itself worth reporting.
2. The hint will matter: `mdlm_infill` > `mdlm_infill_nohint`, because the
   failing assertion is the only place the expected output appears.
3. The interesting one: on the **exception** subset, where the span is genuinely
   localized, `mdlm_infill` may beat `ar_hinted` at matched forward passes,
   because it can revise the statement while keeping the correct code after it.
   On the **assert** subset, where the whole body is masked, there is no
   two-sided constraint to exploit and I expect a tie or a loss.

## Falsifiers

- `mdlm_infill` does not beat `ar_hinted` at matched forward passes on the
  exception subset → **the two-sided constraint buys nothing on code**, and the
  diffusion branch closes on the domain the record named for it.
- `mdlm_infill` == `mdlm_infill_nohint` → the failure information is inert in
  diffusion as it was in arithmetic prose.
- The bidirectionality check fails → the checkpoint is mislabeled; stop.
- `mdlm_full` solves fewer than half as many problems as `ar_greedy` → the
  fine-tune damaged the model; the repair comparison is **inconclusive**, not
  negative, and must be reported that way.

## What this run cannot show

- It cannot support a capacity-matched claim against LoRA-only adaptation. No
  adaptation is trained here.
- It cannot separate "diffusion objective" from "fine-tune on opc-sft": D differs
  from A in both. The paired repair set controls the *starting program*, not the
  weight provenance.
- It cannot show that localization is the binding constraint: the assert branch
  does not localize.
- Absolute pass rates are not a capability claim. HumanEval is public and almost
  certainly in pretraining.

## Cost

To be filled from a single-problem smoke (load time, seconds per denoising step,
peak VRAM) before the stage-1 launch.

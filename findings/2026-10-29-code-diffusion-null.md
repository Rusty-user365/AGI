# Code-trained diffusion repair at 0.5B: the payload is not usable (2026-10-29)

**Status:** complete, negative on the checkpoint, **not** a test of the
hypothesis. The run is a **deviation** from the AR HumanEval line, not a
continuation of it.

**Artifacts:** `experiments/code_diffusion.py`,
`results/diffusion/`, `results/diffusion_t0/`,
`results/diffusion_v0_walkback/`, `results/diffusion_v1_fullsplice/`,
`results/ar05b/`

Pre-registration: `findings/2026-10-29-code-diffusion-preregistration.md`.

## What was asked

Given the **same** broken body and the **same** failing test, does a
bidirectional model that fills a span between fixed left and right context
repair more problems per forward pass than an autoregressive model that
continues left to right?

The pre-registration narrowed this deliberately. The model card's sampler
commits tokens and never revises them, so "revision" is not part of the claim.
The claim is only the **two-sided constraint**.

## Setup

| role | checkpoint | params |
|---|---|---|
| diffusion | `dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1` | 0.5B, MDLM |
| AR twin | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 0.5B, same base |

Both run on the same 24 HumanEval problems. Every repair arm starts from the
**AR twin's failing body**, so the comparison is about the repair mechanism and
not about the fine-tune's effect on unconditional coding.

Forward passes are the matched unit: a KV-cached AR generation of *L* tokens
costs *L* passes, an MDLM run of *S* steps costs *S*. The diffusion arm receives
the AR arm's total budget and may spend it over up to 8 attempts.

## Results

**The AR twin, for scale** (24 problems): `greedy` 19/24, `best_of_4` 21/24,
hinted repair (`cgr`) 22/24. Repair set: **5 problems** where greedy failed.

| arm | repaired | forward passes | attempts |
|---|---|---|---|
| AR hinted (`cgr`) | **3/5** | 3,915 | 5 |
| MDLM infill, T=0.8 | **0/5** | 891 | 35 |
| MDLM infill, T=0.0 | **0/5** | 930 | 35 |

**The decisive observation is not the rate, it is the syntax.** Of the final
bodies, 4/5 (T=0.8) and 3/5 (T=0.0) are **not valid Python**: unmatched `]`,
`[` never closed, indentation that does not match any outer level. The AR twin
from the same base produces valid Python on 19/24.

Examples from the stored bodies:

- `if_pal_palindrome[string[i]:]:` where the body had `if is_palindrome(string[i:]):`
- `[[[[string[: i] for i in range(len(string))]` where the body had one `[`
- `   _int_list = list(...)` where the body had four-space indentation

**And the mode is the bug.** On `HumanEval/5` the diffusion arm spent 8 attempts
and 216 passes and returned a body **byte-identical** to the failing one. Its
most likely output is the AR model's greedy output, which is the bug. A model
that reproduces the failure with high confidence cannot repair it by being given
more attempts.

**The denoiser itself works.** In `--check`, masking the inner loop of
`HumanEval/0`'s canonical body and denoising 32 steps recovered the original text
exactly, and the result passed the test. The mechanism is functional when the
surrounding code is correct. It fails when the surrounding code is the thing that
is wrong.

## What this does and does not show

**Shows:**
- The checkpoint is genuinely bidirectional (perturbing the last token moves the
  first token's hidden state by 3.85; the 470M Dual-LM failed this same test).
- It fits and runs on a 4 GB card: 1,885 MB at load, 2,313 MB peak.
- At 0.5B, this MDLM fine-tune does **not** preserve syntax-level code
  generation well enough to fill a span in broken code.

**Does not show:**
- That the two-sided constraint is worthless. The hypothesis is **untested**,
  because the payload cannot produce valid Python. This is the same error the
  project already made once: the 0.6B AR null tested whether a model that cannot
  code can use feedback. This run tests whether a model that cannot produce valid
  syntax can repair. Neither is a test of the architecture.
- Anything about capacity-matched comparison to LoRA adaptation.
- That diffusion is worse than AR for code in general. The AR twin is the same
  base **without** the diffusion fine-tune, so the gap here is the fine-tune, not
  the objective.

## Three harness bugs, all disclosed

The first three runs all scored 0/5. Each was preceded by a real defect, and all
runs are kept rather than overwritten:

1. `results/diffusion_v0_walkback/` — the span reused the AR harness's walk-back
   to a 4-space statement, so a traceback inside a top-level `for` masked to the
   end of the function. Almost every "infill" was a full-body rewrite with no
   right-hand context.
2. `results/diffusion_v1_fullsplice/` — the budget was capped per attempt instead
   of in total, leaving most of the AR arm's passes unspent (`nfe` 84 against a
   190 budget). The arms were not matched.
3. `results/diffusion_v2/` (the `results/diffusion/` rows before the splice fix) —
   when the span started at character 0, a masked token could begin inside the
   prompt, rewriting prompt text and misaligning the slice by one character.
   Verified first that contiguous token-range decoding is exact for byte-level
   BPE (0 failures in 100 subranges), then spliced by covered character range.

After the third fix the configuration was frozen and the result stood. The
temperature-0.0 run is a control for the sampler, not a fourth attempt at a
positive: it fails identically, so the failure is not the sampling temperature.

## Why the branch is blocked, not closed

The diffusion branch needs a code model that can write valid syntax in a masked
span. At 0.5B this checkpoint cannot. The obvious candidate is a larger code
diffusion model — `apple/DiffuCoder-7B-*` exists — but 7B does not fit in 4 GB in
any precision that this card can run.

So the honest state is: **the two-sided-constraint hypothesis remains untested
for code, and cannot be tested on this hardware with an available checkpoint.**
Closing it would require claiming that a model which emits unmatched brackets is
evidence about bidirectionality, which is exactly the mistake this project has
already paid for once.

## Cost

- load: 37–129 s (the 1,036 s in the pre-registration was disk contention);
- 253 ms/step full generation, 189 ms/step infill;
- 0.5B fp32, 2,313 MB peak.

## Next

1. Do **not** extend the AR twin to 164 for this comparison: it would cost ~2 h
   to add power to a rate whose numerator is zero for a reason that is not
   statistical.
2. If a ≥1.5B code diffusion checkpoint becomes available on hardware that can
   hold it, the pre-registration is already written and the harness already
   exists; only the checkpoint changes.
3. The CGR allocator framing is unaffected: this was a payload experiment, not an
   allocator experiment. See
   `findings/2026-10-29-cgr-as-tool-budget-allocator.md`.

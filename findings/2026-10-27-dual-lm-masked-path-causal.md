# Dual-LM 470M: the masked path is causal on this stack, and a 15-line patch fixes it (2026-10-27)

Artifact: `scripts/check_dual_lm.py`
Model: `ltg/dual-lm-470m` main branch, `checkpoints/hf/dual-lm-470m`, 1.87 GB F32,
468,700,160 parameters.

## The claim being tested

The card says one set of weights serves both objectives, switching on the
attention mask: `DLMLlamaForCausalLM` is causal, `DLMLlamaForMaskedLM` is full
bidirectional. The second half is what a span-repair (`REP`) mechanism depends
on, because a masked position can only use right-hand context if attention is
bidirectional.

The masked class enforces it with `config.is_causal = False` in `__init__`.
Whether transformers 4.51.3 honours that flag is not documented and not
obvious, and the failure mode is silent: the model loads, runs, and produces
plausible text while being causal.

## Measurement

Probe: change a token strictly *after* the position whose prediction is read,
and measure the change in logits there. A past-token probe is run as a control,
so a null cannot be blamed on a broken probe.

| | past token 3 | future token 15 |
|---|---|---|
| causal class | moves, 5.55 | does not move, 0.000 |
| masked class, stock | moves, 5.55 | **does not move, 0.000** |
| masked class, patched | moves, 6.04 | **moves, 2.27** |

`model.config.is_causal` is correctly `False` on the masked model, and it still
does not see the future. The control shows the probe can detect dependence.

Cause, `transformers/models/llama/modeling_llama.py`, 4.51.3:
`LlamaModel._update_causal_mask` calls
`_prepare_4d_causal_attention_mask_with_cache_position` unconditionally. It
never reads `config.is_causal`, and this version has no 4-D attention-mask
passthrough, so the usual escape hatch is unavailable too.

## The patch

Replace `LlamaModel._update_causal_mask` with a builder that returns a
full-attention additive mask (zeros, minus a large negative on padding). Applied
in `load_class(..., patch=True)`; roughly fifteen lines, no upgrade, no new
dependency.

Verified with the patch: future-token dependence appears (2.27), and masking
"Paris" in *"The capital of France is Paris and the river through it is the
Seine."* still predicts `Paris`. The patch restores the capability without
breaking the model.

## What this means for the plan

- **The 470M is usable as an autoregressive model as shipped.** That half works
  and is unpatched.
- **Bidirectional span repair requires the patch.** Without it, `REP` is
  autoregressive regeneration wearing a mask, and the "surgical diffusion repair
  uses fewer tokens than AR regeneration" metric has no mechanism behind it.
- **`--patch` must be on for every infill measurement on this backbone**, and
  any result produced without it should be discarded rather than compared.

## What this does not say

- **Nothing about infill quality.** The probe tests that information flows, not
  that the model fills spans well. That needs a span-repair evaluation on real
  text, which does not exist yet.
- **Nothing about the other 49 branches.** Only `main` was downloaded. The
  `alpha=0.750_n-reps=32` and `alpha=0.500_n-reps=16` branches are untested and
  may behave differently, though the mask bug is in transformers, not in the
  checkpoint, so they will need the same patch.
- **Nothing about training.** The patch makes inference bidirectional; it says
  nothing about whether LoRA fine-tuning on this card is viable.

## Two corrections to the earlier plan

**Branch names use `=`**: `alpha=0.750_n-reps=32`, not
`alpha_0.750_n-reps=32`. The card's prose uses underscores; the refs do not.
`revision="alpha_0.750_n-reps=32"` returns 404.

**`main` ships F32**, 1.87 GB, not the 0.94 GB fp16 that was quoted earlier.
fp16 is a cast at load. On this card that saves memory and buys no speed.

The card's own capability numbers, which are the binding constraint on any
benchmark built for this model: ARC-Easy 28.6, ARC-Challenge 5.7, PIQA 40.9,
HellaSwag 31.1, all normalized so 0 is chance. On a 4-way task, 28.6 means the
model is barely above guessing. Any task used to evaluate a scratchpad must be
one where this model is well above chance, which rules out world-knowledge QA
and makes in-context rule tracking the only plausible family.

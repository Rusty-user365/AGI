# Closing the implicit-conditioning chapter (2026-10-26)

## What is established

| Finding | Evidence |
|---|---|
| The pre-registered gate is cleared | test 0.385 steered / 0.335 free, unseen 0.290 / 0.230, all n=200, all above 0.20 |
| Mixed masking lifts block infill for both arms | LoRA-only 0.145 to 0.245, CGR 0.215 to 0.380 |
| The FiLM side channel was dead in every earlier checkpoint | `film_max_abs` exactly 0.0, reasoner path byte-identical |
| The channel is now live and causally connected | `cond_probe_delta` 0.0 to 15.09, `film_l2` 0.0 to 10.25 |
| The +0.135 advantage over LoRA-only was parameter budget | matched 22.3M control 0.350 vs LoRA 0.245; CGR 0.385 |
| Against a fairly-evaluated matched control, steering is null on the decision split | CGR 0.385 vs control 0.380, +0.005 |
| The global-scope trade replicated directionally | +0.050 and +0.076 `c_target`, -0.391 and -0.407 quality, 4/4 predictions on unseen |
| Control evaluation mode changes the answer | 0.030 on test, 0.055 on unseen, from evaluation mode alone |

## What is not established

- **No steering benefit on the gate condition.** The decision split is null and
  the confirmation split has a degenerate control.
- **No architecture claim over LoRA-only task adaptation.** That comparison is
  confounded by parameter count in both directions.
- **No confirmation of the global-scope trade against a fair control.** It was
  measured against a free-mode control that is demonstrably weak on `test` and
  degenerate on `unseen`.
- **No compute- or capacity-matched superiority claim** anywhere.
- **Nothing about larger backbones.** No 470M or 1.7B model has been run.

## Why the next step is explicit control flow, not more FiLM

The FiLM channel is one vector per document, injected into every block. It is a
bias, not a decision. It cannot express "this draft is wrong, discard it and try
again", because inside a single forward pass there is no draft, no rejection and
no second attempt. Anything that requires iteration *with a decision* has to
live outside the model, in a loop.

That argument stands on its own and does not depend on any result above. It is
also the reason the two-arm experiment was worth running: it showed the channel
is not where the leverage is.

## Hardware reality for the next backbone

Measured on this machine: GTX 1050, **4096 MiB, compute capability 6.1
(Pascal)**, 7.87 GB RAM, torch 2.5.0+cu121, transformers 4.51.3,
`bitsandbytes`/`peft`/`accelerate`/`trl` all absent.

| Backbone | Precision | Weights | Trainable here? |
|---|---|---|---|
| Dual-LM 470M | fp16 | 0.94 GB | yes, with plain LoRA |
| Dual-LM 470M | 4-bit | ~0.25 GB | needs bitsandbytes on sm_61: unsupported |
| TiDAR 1.7B | fp16 | 3.44 GB | no — 84% of the card before activations |
| TiDAR 1.7B | 4-bit | ~0.9 GB | same bitsandbytes blocker |

So the only trainable option on this machine is the 470M model in fp16 with
plain LoRA. The 1.7B model is an inference-only or rented-hardware target.

## What a scratchpad experiment would have to get right

Recorded now because each of these is an easy way to produce a meaningless
positive:

1. **The observer must score text.** The seven heads take
   `cat([pooled, cond, clock])` of width `d_model*2+8` (1544 for this config),
   not a string. Scoring a candidate draft means running the backbone over it
   and pooling, not calling a head on raw text.
2. **Rejection sampling needs the answer in the support.** If the model cannot
   generate the correct continuation at any temperature, re-sampling will not
   find it. Best-of-n only helps when the target is reachable.
3. **The loop needs a bound.** Any retry loop without a step budget and a
   termination proof will spin on exactly the problems it is meant to solve.
4. **The baseline is best-of-n with matched compute**, not a frontier model. A
   small model allowed to sample twenty times is not being compared fairly
   against a large model sampling once, and the claim "GPT-4 gets this wrong"
   is both false and unfalsifiable as stated.
5. **The gate thresholds must be calibrated.** A head output compared to a
   fixed `0.8` is a threshold on an uncalibrated logit, not a decision rule.

## Status

The implicit-conditioning chapter is closed. The gate is cleared, the side
channel is proven live, and no result in this repo supports claiming that
conditioning improved repair over a matched control. Next work is explicit
routing with a matched best-of-n baseline, on a backbone that fits the card.

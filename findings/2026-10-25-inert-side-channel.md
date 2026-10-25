# The FiLM side channel was frozen at zero in every CGR checkpoint (2026-10-25)

## What was observed

The deployed CGR editor (`CGRSystem.repair(..., use_reasoner=True)`) produced
byte-identical blocks to the reasoner-free path.

- Gate, `test`, n=200: reasoner-free 76/200, `--use-reasoner` 76/200.
- Direct diff, 10 documents: **10/10 target blocks identical**.

## Root cause

`DiffuGPTBackbone.freeze_base()` froze every parameter and then re-enabled an
allowlist:

```python
for p in self.parameters():
    p.requires_grad_(False)
for p in self.lora_parameters():
    p.requires_grad_(True)
self.plastic_a.requires_grad_(True)
self.plastic_b.requires_grad_(True)
```

The per-layer FiLM maps were never re-enabled. They kept their zero
initialisation, so `film(cond) == 0` exactly and `cond` could not affect a
single logit. The reasoner computed a steering vector, multiplied it by a zero
matrix, and discarded it.

Verified numerically on both checkpoints:

| Checkpoint | `film.weight` max abs | `film.bias` max abs |
|---|---|---|
| `diffugpt_5k_e10_cgr` | 0.000e+00 | 0.000e+00 |
| `diffugpt_5k_e10_cgr_blockft` | 0.000e+00 | 0.000e+00 |

## Why nothing caught it

1. A zero side channel is a *valid identity at load time*, and there is a test
   asserting exactly that. The property was designed in and then never
   re-checked after training.
2. The load-time check in `check_backbone.py` used `cond = zeros`, which is
   zero for any FiLM weights, live or dead. It could not detect the failure.
3. `test_freeze_base_leaves_only_adapters_trainable` asserted that the
   trainable set was *only* LoRA and plastic, i.e. the test encoded the bug as
   the contract.
4. The loss is unaffected: the model trains, converges, and reports plausible
   numbers. Only a probe with a non-zero `cond` reveals it.

## Consequence for the claims

At inference, CGR and the LoRA-only control are the same function class: same
frozen 138M base, same LoRA shapes, same plastic adapter, differing only in the
adapter weights those losses produced. The measured +0.135 on `test` is an
**auxiliary-supervision** effect (reasoner/heads/gate losses shaping the shared
adapter), not a conditioning or architecture effect.

## Fix applied

- `DiffuGPTBackbone.side_channel_parameters()` returns the FiLM maps and the
  constant, and `freeze_base()` re-enables them by name.
- `assert_side_channel_live()` raises if the maps are frozen or the constant is
  exactly zero; `train.py` calls it before the first step for the CGR
  objective.
- The first backward pass asserts that `film.weight.grad` is non-zero, so a
  detached or zeroed `cond` fails at step 0 instead of after three hours.
- `check_backbone.py` now probes with a **non-zero** `cond` and reports
  `film max|w|` and `cond probe delta` in the JSON; a dead channel prints a
  warning.
- `--cond-mode {reasoner,learned,none}` and a learned `const_cond` parameter
  provide a capacity-matched control: identical FiLM parameters, driven by one
  learned vector for the whole batch, so the conditioning carries no
  information about the input.
- Tests added: freeze keeps the side channel trainable; the side channel
  receives gradient; the assertion rejects a dead channel; the learned control
  is input-independent and does reach the representation; `const_cond` moves
  after two optimizer steps.

## A subtlety worth recording

Because `film.weight` starts at zero, `dL/d(cond)` is zero on the **first**
step. `const_cond` therefore cannot move until FiLM has moved once. This is a
one-step delay, not a dead path, but any test of the control must run at least
two steps. The same applies to the reasoner: its gradient through the
denoising loss is zero at step 1.

## What this does not support

- It does not show that a live side channel would help. That is an open
  experiment, and it needs the constant-conditioning control trained on the
  identical schedule; otherwise any gain is confounded with extra training.

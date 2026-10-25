# Mixed block/token masking as the repair objective (2026-10-25)

## Question

Training masked scattered tokens; the gate masks one contiguous block. Does
matching the training mask to the evaluation operation raise block infill?

## Design

One mask per example, drawn once before the forward pass:

| Example type | Mask |
|---|---|
| Document / repair episode (has separators) | 50% one contiguous block between separators; 50% scattered tokens at a ratio drawn from 15-85% |
| Claim episode (no separator) | scattered tokens only |

Realized mixture per epoch: 13,750 episodes = 5,000 docs + 5,000 claims + 3,750
repairs. Only 8,750 have block separators, so the realized split is about
**32% block / 68% token**, not 50/50. Masking a whole short claim would be
unconditional generation, a different task from the one the gate scores.

Both arms received the identical mixture. Fine-tune schedule: 4 Stage-A + 1
Stage-B epoch from the existing 5k checkpoints, LoRA rank 8, same LR, seed and
corpus. Config and corpus hash in `results/blockft_manifest.json`.

## Runs

| Arm | Wall time | Stage-A losses | Stage-B loss | Trainable |
|---|---|---|---|---|
| CGR | 9,898 s (2 h 45 m) | 1.2564 / 1.1334 / 1.0937 / 1.0481 | 1.0106 | 8,149,661 |
| LoRA-only | 4,141 s (1 h 09 m) | 1.3698 / 1.2369 / 1.1645 / 1.1177 | 1.0800 | 442,368 |

Losses are not comparable to the earlier token-only 0.735: a whole 15-20 token
block is much harder to predict than a scattered word.

## Result

Gate at n=200 (see [editor-gate-n200](2026-10-25-editor-gate-n200.md)):

- LoRA-only: 0.145 -> 0.245 on `test`
- CGR: 0.215 -> 0.380 on `test`

## Interpretation

Mixed masking raises block infill for both arms. That is the supported claim.
The CGR-vs-control difference is +0.135 unpaired (z ~ 2.9, p ~ 0.004) on
`test`, but the CGR checkpoint's conditioning path is inert
([inert-side-channel](2026-10-25-inert-side-channel.md)), so that gap is an
auxiliary-supervision effect, not an architecture effect.

## What this does not support

- No claim that the objective change repaired the editor *because* of CGR.
- No compute-matched claim: 9,898 s vs 4,141 s, 18.4x trainable parameters.
- No paired significance claim yet; that is what exp8 measures.

# exp8: paired CGR vs LoRA-only repair comparison on `test` (2026-10-25)

## Question

On the same 200 documents, does the CGR checkpoint repair better than the
LoRA-only checkpoint, and does the difference hold across defect types and
editing scopes?

## Design

`experiments/exp8_architecture_ablation.py`, split=`test`, n=200, seed=0,
steps=8. Six conditions: 3 defect types x 2 scopes (`one_block`, `global`).
Paired bootstrap 95% intervals on the per-document difference. Artifact:
`results/exp8_architecture_ablation_test.json`. Runtime 55 min.

## Results

`c_target` = repaired document names the authoritative thesis.
`improvement` = quality change attributable to the edit (negative = the edit
made the document worse).

| Condition | c_target CGR | c_target LoRA | paired diff | 95% CI | sig |
|---|---|---|---|---|---|
| local / one_block | 0.380 | 0.245 | **+0.135** | **[0.060, 0.210]** | **yes** |
| local / global | 0.305 | 0.315 | -0.010 | [-0.075, 0.060] | no |
| cross_block / one_block | 0.036 | 0.044 | -0.008 | [-0.023, 0.007] | no |
| cross_block / global | 0.451 | 0.462 | -0.011 | [-0.051, 0.031] | no |
| cross_block_partial / one_block | 0.060 | 0.076 | -0.015 | [-0.043, 0.013] | no |
| cross_block_partial / global | 0.480 | 0.471 | +0.009 | [-0.037, 0.055] | no |

`improvement` differs significantly in four conditions, and in every one the
CGR arm is **worse**:

| Condition | improvement diff | 95% CI |
|---|---|---|
| local / one_block | -0.203 | [-0.280, -0.122] |
| local / global | -0.105 | [-0.151, -0.058] |
| cross_block / global | -0.075 | [-0.121, -0.025] |
| cross_block_partial / global | -0.057 | [-0.104, -0.011] |

## Interpretation

1. The gate result is confirmed paired: the only CGR advantage is
   **local / one_block**, +0.135 [0.060, 0.210], and the interval excludes 0.

2. That advantage does **not** generalize. On the other five conditions the two
   models are indistinguishable on `c_target`, and where they differ
   significantly on `improvement`, CGR is the worse editor. So the honest claim
   is narrow: "the CGR-trained checkpoint is more likely to restore the thesis
   on a single-block local defect", not "CGR repairs better".

3. The two metrics point in opposite directions on `local / one_block`: CGR
   names the right thesis more often but degrades document quality more
   (-0.203). Restoring the thesis id and preserving quality are not the same
   operation, and the gate only scores the first.

4. `cross_block` and `cross_block_partial` are near chance under `one_block`
   (0.036-0.076) for both models. The high `global` numbers (0.45-0.48) come
   from rewriting more of the document, and the negative `improvement` values
   say that rewriting is not repairing.

## What this does not support

- No architecture claim. The conditioning path is inert in this checkpoint
  ([inert-side-channel](2026-10-25-inert-side-channel.md)), so the difference
  is auxiliary supervision.
- No claim of compute-matched or capacity-matched superiority: 8,149,661 vs
  442,368 trainable parameters, 9,898 s vs 4,141 s.
- No unseen-thesis claim. The `unseen` run is separate and is not pooled here.
- No Gate/DAG claim; the gate is not part of this comparison.

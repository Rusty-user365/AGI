# Findings

Human-readable record of what each experiment measured, what broke, and what
was corrected. Raw artifacts stay in `results/` (one JSON per run, plus console
logs); this folder is the narrative that makes them interpretable later.

Convention: one file per experiment or incident, named
`YYYY-MM-DD-<slug>.md`. Each entry states the question, the measurement, the
interpretation, and what it does **not** support. Corrections to earlier claims
are recorded in the same entry rather than quietly deleted.

| Date | Entry | Outcome |
|---|---|---|
| 2026-10-25 | [editor-gate-n200](2026-10-25-editor-gate-n200.md) | Pre-registered gate cleared (0.380); the earlier 0.175 "failure" was a small-sample artifact |
| 2026-10-25 | [inert-side-channel](2026-10-25-inert-side-channel.md) | The FiLM side channel was frozen at zero in every CGR checkpoint; reasoner path is a no-op |
| 2026-10-25 | [mixed-mask-objective](2026-10-25-mixed-mask-objective.md) | Mixed block/token masking raises block infill for both arms; realized mixture is 32/68 |
| 2026-10-25 | [exp8-paired-test](2026-10-25-exp8-paired-test.md) | Paired +0.135 on local/one_block only; CGR is worse on quality in four conditions; no generalisation |
| 2026-10-25 | [fixup-steering-gate](2026-10-25-fixup-steering-gate.md) | Both arms live; all six gate cells clear 0.20; steering is +0.050 test / +0.060 unseen but unpaired |
| 2026-10-25 | [steering-capacity-control](2026-10-25-steering-capacity-control.md) | **The +0.135 CGR advantage was capacity**: against a matched 22.3M control it is +0.035 [-0.045, 0.115], not significant |

Open gaps, tracked here so they are not forgotten:

- The gate is cleared, but no capacity-matched result shows CGR beating a
  control on the gate condition. The earlier +0.135 is retracted to a capacity
  effect.
- Steering does show a measurable trade on `global` rewrites (+0.058 / +0.037
  `c_target`, -0.102 / -0.116 quality). These are exploratory and `test` is a
  selection set; they need a directional `unseen` confirmation.
- Nothing yet distinguishes "CGR as an architecture" from "more trainable
  parameters". The two-arm design showed the difference is not the side
  channel; whether the extra adapters themselves matter is untested.

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

Open gaps, tracked here so they are not forgotten:

- No checkpoint has a live side channel yet. Every architecture statement in
  the repo currently describes the adapters, not the conditioning path.
- exp8 on `unseen` has not been run. The `unseen` gate result (0.190, below
  floor) makes it a confirmation run, not a discovery run.
- The two-arm fix-up (reasoner vs learned-constant, both resuming from the
  0.380 checkpoint) has not been started.

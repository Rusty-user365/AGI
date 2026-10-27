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
| 2026-10-26 | [unseen-confirmation](2026-10-26-unseen-confirmation.md) | Global-scope trade replicated 4/4 on unseen; gate condition null on test (+0.005 vs a fair control) and uninterpretable on unseen (control below floor) |
| 2026-10-26 | [chapter-close](2026-10-26-chapter-close.md) | What is and is not established; why the next step is explicit control flow, not more FiLM |
| 2026-10-27 | [fair-control-exp8](2026-10-27-fair-control-exp8.md) | Gate condition null at +0.005 against an in-distribution control, with a quality cost; one of two global-scope trades survives on test only |
| 2026-10-27 | [observer-cannot-rank](2026-10-27-observer-cannot-rank.md) | No head ranks repairs above chance (all CIs include 0.5); oracle gap is 0.409, so the DAG stays unwritten |
| 2026-10-27 | [dual-lm-masked-path-causal](2026-10-27-dual-lm-masked-path-causal.md) | Dual-LM 470M's masked path is causal under transformers 4.51.3; a 15-line mask patch restores bidirectional infill |
| 2026-10-27 | [benchmark-has-no-plausible-wrong-answers](2026-10-27-benchmark-has-no-plausible-wrong-answers.md) | Parseability and correctness agree in 480/480 candidates, so a retrained head's 0.684 is a well-formedness detector and a parser dominates it |
| 2026-10-27 | [arithmetic-out-of-support](2026-10-27-arithmetic-out-of-support.md) | The 470M is at chance on arithmetic (top-1 token `1` on `9 + 9 =`), but the task family does produce wrong-but-well-formed answers, so the benchmark design is validated and the generator is not |
| 2026-10-27 | [instruct-model-fluent-but-wrong](2026-10-27-instruct-model-fluent-but-wrong.md) | Qwen2.5-1.5B-Instruct answers money problems as bare integers 100% of the time and is wrong most of the time, but p1 0.03-0.22 leaves too little headroom to select from |
| 2026-10-27 | [observer-ranks-but-cannot-select](2026-10-27-observer-ranks-but-cannot-select.md) | On a 500-task corpus with 1096 wrong-but-well-formed holdout candidates, a head reaches AUC 0.682 (permutation 0.496) yet ties max-logprob 12W-11L, p=1.00, so the routing loop is unjustified |

Open gaps, tracked here so they are not forgotten:

- **No steering claim survives.** On the decision split a fairly-evaluated
  capacity-matched control matches CGR (0.380 vs 0.385). On `unseen` that
  control is degenerate (0.155, below floor), so the split cannot adjudicate.
- The `global`-scope trade replicated directionally on both splits, but both
  measurements used a free-mode control that is demonstrably handicapped on
  `test`. Confirming it needs `exp8 --baseline-cond-mode learned`.
- Nothing yet distinguishes "CGR as an architecture" from "more trainable
  parameters". The two-arm design showed the difference is not the side
  channel; whether the extra adapters themselves matter is untested.
- **Durable methodological finding:** the same control moves the answer by 0.030
  (`test`) and 0.055 (`unseen`) purely from evaluation mode. A capacity-matched
  control is not matched until its evaluation mode is matched too.

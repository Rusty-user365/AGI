# Future work: routing among multiple specialist adapters

**Status: proposed, not implemented or validated.** The user provided this as
context for a future goal and did not ask to start the implementation, training,
paper revision, or publication. A future assistant should discuss or undertake
that work when the user requests it.

## Objective

Extend the present base-versus-cyber routing experiments to a system that can
choose among several specialized adapters, such as cyber, coding, math, and UI,
while retaining the option to use the base model. The user envisions a small
router and selective adapter loading for resource-constrained hardware.

The proposed two-stage design is:

1. **Domain or skill selection:** use prompt features and a multiclass head to
   identify candidate specialties. A proposed starting point is a linear head
   with softmax output over domain labels.
2. **Correctness or intervention gate:** use base-model answer/step features to
   estimate whether an intervention is warranted; a binary head is one possible
   design. If the gate indicates risk and a relevant specialist is available,
   route to that adapter; otherwise use the base or abstain.

The suggested thresholds (domain score greater than 0.60 and correctness score
below 0.50) are **hypotheses**, not validated or calibrated settings. Scope and
route decisions must be evaluated on separate data; do not use test answers to
choose a route, threshold, or adapter. Domain classification is not the same as
predicting that an adapter will improve an answer. Consider an explicit benefit
predictor or an abstention option, and count harmful as well as useful switches.

## What already exists

- The earlier arithmetic work used linear and cross-step Transformer localizers;
  its two-tier threshold and softmax ensemble are distinct methods. See
  `PAPER.md` and `experiments/localizer.py` for the actual architecture and
  its held-out limitations.
- For the cyber experiment, binary heads trained on 200 SecEval questions
  routed CyberMetric-80 to 57/80, versus 54/80 for each standalone model, with
  three useful and zero harmful switches in that sample. A live test reproduced
  these results and checked fresh full-prompt cyber prefill on nine switches.
  See `findings/2026-10-30-cross-dataset-routing-generalization.md` and
  `results/cyber_router_live_cybermetric80_v3.json`.
- This was single-question base/cyber routing, not multi-adapter routing, a
  multi-step DAG, general safety proof, a universal uncertainty signal, or a
  demonstration of a calibrated classifier. Its public evaluation data had
  already been examined.
- The existing coder integration in `cgr/control/adapter_runtime.py` exercises
  cache rollback and base-prefill bridge checks in a limited workflow. It does
  not implement the proposed multiclass domain classifier.

## Future implementation and evaluation

- Assemble domain-labelled prompts and independently verified outputs, with
  source-grouped training, validation and blind test splits. The user's
  illustrative proposal was 100 prompts each for math, code, cyber, UI and
  general tasks; dataset size and balance need empirical justification.
- Train and evaluate the domain head and correctness gate separately, including
  calibration, abstention and cross-domain regressions. Compare against base
  only, always-specialist, metadata routing, random routing, uncertainty-only
  and oracle routing. Measure end-task accuracy and useful/harmful switches.
- Verify model revision, tokenizer and adapter compatibility. When using a full
  specialist prefill, discard the base generation cache and prefill the full
  prompt under the chosen adapter. For a multi-step repair, separately verify
  rollback to the accepted boundary and the base-only bridge before resuming;
  full prefill is not itself a DAG or bridge validation.
- Measure actual adapter load/switch latency, persistent and peak CPU/GPU memory,
  model and KV-cache footprint, generation and verification costs, and p50/p95
  end-to-end latency on specified hardware. Compare resident and on-demand
  loading where feasible. Adapter files in this project are approximately
  27.6 MB in weight bytes; different adapters may differ in size. Claims such
  as a 10 ms load time, 50 experts on a 4 GB GPU, or arbitrary scaling are
  unverified and must not be presented as results.
- If later requested, prepare a code skeleton and paper text clearly marked as
  proposed future work; do not claim an established mixture-of-experts or edge
  deployment without experiments that support it.

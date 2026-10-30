# Paper revision and evidence plan — 2026-10-30

The current paper has a measured arithmetic contribution and a preliminary
adapter extension. Keep their models, hardware, data and evidential strength
explicitly separate. The training adapters use chat-capable Qwen3-0.6B; the
locked arithmetic experiments use Qwen3-0.6B-Base.

## What can be written now

- Retain the vault headline: 73/331 versus 46/331 recovery, with 1.70x full-loop
  speedup on the reported hardware. Do not replace it with generation-slice
  latency or interpret CPU prototype timing as the old GPU result.
- Update §6 to report the local coder integration and its limits: four observed
  policy transitions, four accepted nodes out of five, failed clamp rejected,
  even-sum accepted, and successful base Bridge checks. The injected addition
  repair is explicitly a synthetic control. This is mechanism evidence.
- Correct §6.1: unchanged K/V projection weights do not guarantee unchanged
  higher-layer K/V activations under Q/MLP adapters. Preserving independent
  artifacts also does not preserve their post-repair cache entries.
- Explain that the current graph/verifier package passes known fixture tests,
  while cyber/UI inference, natural multi-domain recovery and planner routing
  remain unmeasured. Do not turn the 20 parameter variations into N=20
  independent benchmark tasks.

## Next experiments, in order

1. **Coder prefix ablation:** full-adapter prefill versus switching from a
   base-produced prefix, using identical prompts/tokens/checkpoints. The known
   clamp failure has no established causal explanation.
2. **Artifact gates when training finishes:** exact parent/template/config,
   finite weights, named adapter switching, and restoration to base.
3. **Fixed-graph multi-domain milestone:** failure discovered late, root
   contract rechecked, bounded specialist patch verified, descendants rebuilt,
   independent artifacts replayed, and final task verified. Add frontend
   build/browser tests matching the UI adapter's training contract.
4. **Fair comparative study:** localized base repair, specialist full-prefill,
   specialist Bridge, and base sequential retry. Count all prompts, attempts,
   replays and tools. Use independent task groups and retain negative results.
5. **Automatic planner:** a separate extension once the supplied-graph path
   works; measure decomposition, routing, and error attribution rather than
   reporting an untrained hidden-state classifier as operational.

The executable preparation and prospective design are in
`ADAPTER_VALIDATION_PROTOCOL.md`. The independent evaluation dataset and
multi-adapter inference runner remain future work; the new file grades supplied
artifacts and tests the dependency/validator contracts.

## Submission audit still needed

| Draft claim | Required revision/evidence |
|---|---|
| §2.4 says capacity is ruled out by asymmetric recalls | Asymmetry alone does not identify attention's causal benefit. Add a matched per-step MLP/attention ablation or frame this as a hypothesis. |
| §4.3 formerly called a 4.86e-5 logit difference below fp32 epsilon | Corrected in this revision to the 1e-4 comparison tolerance and identical argmax sequences. |
| H*=6–8 presented as an established horizon | Measurements at H=5 and H=10 do not locate a transition at 6–8. Measure intermediate horizons or label the range a conjecture. |
| “All outputs and weights deterministic and tracked” | Audit versioned artifacts, untracked files, stochastic runs and absent original predictions. Provide immutable manifests and exact reproduction limits. |
| Gate saves compute despite identical measured aggregate timing/tokens | Separate implemented skip behavior from an empirically demonstrated cost reduction. |
| Novelty / related-work positioning | Complete a primary-source literature comparison for localization, rollback, adapter routing and cache reuse before claiming novelty. |

These are an audit queue, not newly completed experiments. Do not silently
rewrite locked raw measurements. If multi-domain trials are positive, promote
§6 to an evaluated extension with its own results and ablations. If negative or
inconclusive, the arithmetic result remains the paper's measured core and the
extension reports the observed limits.

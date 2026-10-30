# Paper review: source-code and evidence audit

Reviewed 2026-10-30. Scope: `PAPER.md`, relevant arithmetic implementation and
saved JSON, live cyber routing metrics, LaTeX tables, and reproducibility guide.
No model experiments were run and no historical results were overwritten.
This is a substantive review, not PDF layout validation or a complete audit of
every experiment. Line references refer to the source at review time.

## Overall assessment

The recorded arithmetic recovery counts and live cyber routing totals agree
with the paper. However, the headline cache speedup and full-loop timing claims
are not supported by the implementation cited for them. Correct these before
sharing the manuscript as publication-ready. The conservative abstract and
adapter limitations are useful, but several older sections still overstate
what was demonstrated.

## 1. Critical: the 25.9x / 27.2x measurement is not cached repair generation

**Paper:** contributions 4–5; Sections 4.2–4.3; conclusion; systems LaTeX table.

**Evidence:** `experiments/path_a_loop.py:269–278` constructs a fresh prefix
cache, calls `crop` with that cache's existing length, and stops the timer.
There is no suffix decoding inside this block. `experiments/eval_vault.py:377–385`
does the same. Cropping a prefix cache to its current length does not test a
rewind from a longer failed trajectory. The computed token counter adds
`(steps - t_hat) * 2` without actually generating those tokens in the timed block.

The recovery outputs are generated separately using `gen_step` and
`finish_chain`, which pass fresh prompts to `model.generate` and do not pass
the cropped cache (`experiments/oracle_rewind.py:93–117`). Thus the current
records do not show that the 73/331 recoveries were produced by the timed
cache-reuse path.

**Correction:** withdraw the interpretation of 25.9x/27.2x as generation speedup.
Describe these historical values as incomplete prefix-prefill measurements,
unsuitable for an equal-work latency comparison. A replacement benchmark must
reuse a real longer cache, truncate it, generate and verify the repaired suffix,
and compare equivalent output work. The new notebook's numerical cache check
is separate evidence; it does not retroactively validate these timings.

## 2. Critical: 1.70x / 2.0x are not measured end-to-end loop speedups

**Paper:** abstract; Section 4.2, especially its explanation of observer overhead.

**Evidence:** `path_a_loop.py:245–256` times repair generation, then reuses that
time across gated and ungated arms. Gate readouts occur later at lines 281–299.
Localizer work and stored feature acquisition are outside the arm timer.
Likewise, `eval_vault.py:339–348` stops its timer before gate readouts at
355–356; line 372 adds only that earlier time to the result.

Consequently the difference between the purported generation-slice and loop
times is not evidence of ten observer passes plus a gate. The identical gated
and ungated timing is shared accounting, not proof that gating costs nothing.
The gate runs after the candidate has already been generated, so the paper's
claim that these rejected candidates saved generation compute is unsupported.

**Correction:** label the saved ratios as repair-generation timing ratios under
the existing implementation, excluding localization and gate costs. Remove the
full-loop claim from the abstract. A complete timer must cover the required
features, localization, repair, gate, and verification with explicit boundaries.

## 3. High: token and FLOP savings are estimated, not measured

**Evidence:** rewrite counters add `prompt_length + 2` per step; repair counters
also assume two generated tokens (`path_a_loop.py:202,253,275` and
`eval_vault.py:300,326,349,382`). These are not model-forward instrumentation.
At H=10, `eval_h10.py:230–233` assigns a cached-prefix token estimate although
the preceding repair uses fresh-prompt generation. There is no crop operation
in that H=10 repair path.

**Correction:** remove the claim that measured cache truncation reduced H=10
forward work by 90%. Identify formula-derived estimates as such. Rename
Table 4's “FLOPs / Token Reduction” column: token counts are not FLOP counts,
especially when comparing prefill and decoding at different context lengths.

## 4. High: the localizer architecture is described incorrectly

**Paper:** Section 2.2 and the “single linear hyperplane” explanation in 2.4.

`LinearStep` is a two-layer per-step MLP with GELU and dropout, not a linear
probe (`experiments/localizer.py:124–144`). The arithmetic loop applies SVD-32
to this MLP but gives full-width standardized features to the Transformer:
`path_a_loop.py:91–92` and `eval_vault.py:76–77` pass `proj` versus `None`.
Both therefore do not use the same frozen SVD-32 representation.

**Correction:** use “independent per-step MLP (historical code name LinearStep)”;
document each arm's actual input dimension and parameter count. Check whether
the separate diagnostics used to build Table 1 used the same configuration as
the loop before claiming they describe an identical architecture.

## 5. High: attention's causal advantage is not isolated

Section 2.4 says the evidence “rules out” capacity and explains the differences
through residual-stream drift. Per-step error asymmetry cannot establish this
mechanism. Input dimension, parameter count, positional embeddings and cross-step
attention differ; no matched ablation is presented here.

**Correction:** frame relational attention and drift as hypotheses. A useful
future control is a width/parameter/input-matched per-step MLP, with explicit
position controls and an attention-disabled counterpart. The earlier FiLM
capacity-control experiment does not control this different comparison.

## 6. High: recovery given failure is presented as autonomous detection

`load_candidates` explicitly keeps only complete chains whose ground-truth
`first_error` is non-null (`experiments/localizer.py:68–85`). The experiment
therefore measures where to intervene conditional on a known-broken chain.
It does not measure whether the arithmetic controller can detect that an
arbitrary chain needs intervention, or how often it damages a correct chain.

**Correction:** state this conditioning prominently in the abstract and method.
Report malformed/incomplete chains and originally correct chains separately.
A future mixed evaluation should measure detection false positives, damage to
correct chains, and total final accuracy. The binary cyber router is a separate
experiment and does not fill this arithmetic detection gap.

## 7. High: statistical units and uncertainty need clarification

The paper describes 560 candidates from 140 vault tasks. Multiple completions
of one task share the task, so the 331 broken chains are not automatically 331
independent experimental units. Candidate-level Wilson intervals and paired
tests do not account for this clustering.

**Correction:** retain the descriptive counts but qualify inferential claims.
Use task-level paired outcomes or a task-cluster bootstrap, retaining task IDs
and per-arm outcomes. The published vault JSON stores aggregate wins/losses,
not the row-level recovery arrays, so a cluster-aware calculation cannot be
reconstructed from that file alone. Do not claim that re-fitting localizers
reconstructs the historical recovery records.

For the cyber comparison, 3 beneficial versus 0 harmful switches gives a
two-sided exact paired binomial/McNemar p-value of 0.25. The 57/80 observation
is real in the saved record, but is not statistically compelling superiority
on its own. Keep the paper's existing exploratory and zero-observed-harm caveats.

## 8. High: no evidence establishes a universal H*=6–8 boundary

Sections 4.4 and 5 assert a 6–8-step horizon and that tools or larger models are
strictly necessary. Results at H=5 and H=10 do not identify a threshold between
them, an impossibility result, or a model-capacity cause. The manuscript's own
Section 5.1 correctly acknowledges alternative explanations.

**Correction:** retain “0/75 recoveries under the tested H=10 policies.” Remove
the inferred threshold and necessity claim. Distinguish an illustrative product
of conditional probabilities from measured conditional per-step accuracy.

## 9. Medium: an observed oracle baseline is not a theoretical ceiling

Section 4.1 calls 35/110 the theoretical rank-2 repair ceiling and 32/35 the
fraction of that ceiling captured. Correct localization with one particular
decoding intervention is an empirical reference, not a proven upper bound over
all interventions or earlier rewinds.

**Correction:** say “91.4% of the observed oracle-localized rank-2 recovery count
on the tuning sample” (32/35 = 91.4286%; 91.5% uses rounded percentages).
Avoid an upper-bound interpretation and distinguish this tuning result from
the vault. Likewise, masking one digit need not select the globally second-best
complete answer; describe the token-level exclusion precisely.

## 10. Medium: negative controls are overgeneralized

Section 1 uses “falsified,” “failed completely,” and “zero marginal value” for
non-significant comparisons. A failure to reject equality does not prove
equivalence or invalidate whole classes of methods. The reported +0.004 is not
literally zero, and p=1.000 is not an equivalence test.

**Correction:** say “no demonstrated advantage in this experiment,” report
effect uncertainty, and limit conclusions to the specific pooling, prompt and
training conditions. The introduction already models this cautious wording.

## 11. High: reproducibility statement contradicts the evidence

The final section says all weights, outputs and logs are deterministic and
tracked. The errata explicitly state original localizer predictions were not
saved and the later fit differs; `REPRODUCIBILITY.md` says fresh-clone exact
regeneration has not been verified. In addition, the random-step control seeds
Python RNG using `hash((split, idx, cand))` (`eval_vault.py:284`); the string hash
depends on interpreter hash randomization unless pinned.

**Correction:** replace the blanket assertion with an artifact inventory
distinguishing available files, missing historical outputs, external assets,
and hardware-dependent results. Use stable seeds for future runs and save
per-row predictions/checkpoints. Do not silently replace old results with a
new run. Keep original-fit Top-1 separate from re-fit recall/precision, including
in the contributions summary where the provenance is currently easy to miss.

## 12. Improvements to positioning and presentation

- Add a real Related Work section. The current references mainly cover datasets
  and model sources. Compare memory-based feedback to
  [Reflexion](https://arxiv.org/abs/2303.11366), and adapter-serving mechanics to
  [S-LoRA](https://arxiv.org/abs/2311.03285). These primary papers address relevant
  components; they are not evidence that this project's full combination was
  previously demonstrated. Also distinguish intrinsic self-correction from
  externally verified feedback using the
  [DeepMind self-correction study](https://deepmind.google/research/publications/48252/).
- Add one compact algorithm and a state diagram distinguishing failure
  detection, localization, candidate repair, verification, and cache rebuilding.
  Mark implemented paths versus proposed integration explicitly.
- Add one methods table covering data splits, unique task counts, candidates,
  hidden-state extraction, model revisions, head configurations, seeds, decoding
  policy, hardware, and timing boundaries. Define CGR at first use.
- Move historical narrative and detailed errata to an appendix, while keeping
  material limitations in the main text. Separate confirmatory evaluation from
  tuning-set development and hand-authored adapter examples.
- Synchronize `publication/tables.tex`, rendered paper, findings and abstract
  after correcting claims; the misleading timing interpretation occurs in
  multiple artifacts.

## What the inspected records do support

- `results/verify_vault_h5.json`: 73/331 recovered for the recorded gated repair
  arm, 46/331 for rewrite, and 23/331 for random-step repair. This audit does not
  establish that those repairs used a persistent cropped cache.
- `results/cyber_router_live_cybermetric80_v3.json`: base/cyber/router/oracle
  54/54/57/61 out of 80; nine switches, three useful and zero harmful; 80 base
  outputs and decisions match; nine cyber outputs match and nine empty-cache
  prefills recorded.
- The recently saved notebook smoke artifacts support a separate CPU cache
  numerical comparison and factorial base/coder checks. They are not historical
  arithmetic timing or ARC competition evidence.

## Recommended revision order

1. Correct the abstract and all cache/timing/token claims using findings 1–3.
2. Correct architecture, detection scope, and causal explanations (4–6, 8–10).
3. Qualify statistics and document missing evidence (7, 11).
4. Add methods and related work, then rebuild presentation artifacts.
5. If stronger systems claims are retained, run a new explicitly instrumented
   equal-work benchmark and save per-example outcomes. Do not relabel the old
   prefix-prefill measurements as repaired-generation results.

## Corrections applied after this review

The author requested corrections for both the general research and ARC routes.
`PAPER.md`, `publication/tables.tex`, the rendered PDF, ARC plan/notebook,
reproducibility guide and current handoff now carry the corrected interpretation.
Archived drafts/findings have prominent supersession notices. Original numerical
result JSON was preserved byte-for-byte; hashes and caveats are recorded in
`results/measurement_audit_20261030.json`.

The arithmetic scripts now label incomplete timing, omit misleading speedup
summaries, default to new output paths, and reject overwrites. Vault reruns use
stable random-control seeds and retain row IDs/recovery arrays. These are
reporting/reproducibility changes, not a new persistent-cache repair benchmark.
The ARC notebook no longer treats a directory name as content verification.

Validation: Python parsing/compilation, live CLI overwrite rejection before
experiment execution, and model-free notebook scaffold checks passed. Records
are in `results/arc_scaffold_audit_checks/`. The corrected 21-page PDF was rebuilt
and visually checked. No new model experiment or ARC benchmark score was produced.

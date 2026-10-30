# ARC Prize 2026: Kaggle plan for CGR

**Status: development plan and runnable local scaffold, not a submitted or
competition-validated entry.** Prepared 2026-10-30, revised 2026-10-30 after an
audit of the notebook and the licence position. Default target for this separate
notebook is **ARC-AGI-2** (static grids), because it has a simpler input/output
interface than ARC-AGI-3. This is a working assumption, not a venue or track
choice imposed on the author. Do not substitute a mechanism demo for an actual
ARC task-solving competition submission.

## Official requirements and dates

- The [ARC Prize overview](https://arcprize.org/competitions/2026) lists code
  submissions due **November 2, 2026**, and papers due **November 8, 2026**.
  Verify the Kaggle deadline in the competition's own time zone before relying
  on it; the two pages can disagree by a day.
- The [Paper Prize](https://arcprize.org/competitions/2026/paper) requires a
  linked ARC-AGI-2 or ARC-AGI-3 Kaggle code submission that *confirms the paper
  describes a real, working entry*. A high score is not a minimum eligibility
  condition, but leaderboard accuracy **is** one of six rubric categories, each
  scored 0-5: accuracy, universality, progress, theory, completeness, novelty.
  The title and the Outstanding Papers pool (above 4.5 average) are separate
  awards. A paper about unrelated arithmetic and cyber demonstrations alone does
  not establish an ARC-AGI method.
- [ARC-AGI-2 Kaggle](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/)
  requires notebook submissions, **12 hours** CPU and GPU runtime, **no internet
  at evaluation**, freely available external data including pretrained models,
  and a file named exactly `submission.json`.
  [arcprize.org](https://arcprize.org/competitions/2026/arc-agi-2) states the
  scoring rule: predict exactly **2 outputs per test input grid**, and a task
  scores 1 if *either* attempt matches exactly.
- **Licence (resolved).** The general rules require authored code and methods to
  be released under "a permissive public domain license (eg. CC0 or MIT-0)", and
  third-party components under at least an open licence that allows public
  sharing (for example Apache-2.0 or GPLv3). The ARC artifacts in this repository
  are now released under **MIT-0** in `LICENSE-MIT-0.txt`; the rest of the
  repository stays MIT. See the inventory below before attaching anything else.

## Licence inventory

| Component | Licence | Ships in a public ARC submission? |
|---|---|---|
| Notebook, this plan, `results/arc_kaggle_scaffold_smoke/` (authored) | **MIT-0** | yes |
| PyTorch | BSD-3-Clause | yes, as a dependency |
| transformers | Apache-2.0 | yes, as a dependency |
| peft | Apache-2.0 | yes, as a dependency |
| `Qwen/Qwen3-0.6B` weights | Apache-2.0 | yes, as a model input |
| Coder LoRA adapter | derivative of `Qwen/Qwen3-0.6B`; **training-data provenance unresolved** | **no**, until the corpus and its licence are documented |

Excluding the adapter is a deliberate restriction, not an oversight. Redistributing
a derivative whose training data cannot be licensed is the one licensing problem a
permissive code licence does not fix. The adapter diagnostic in the notebook is
gated behind `CGR_ADAPTER_LICENCE_ACK=1` for that reason, and the ARC prediction
path does not use it.

`REPRODUCIBILITY.md` already records that the project MIT licence does not resolve
upstream data or model rights. That remains true for the rest of the repository and
for the SecEval/CyberMetric corpora; it does not block the ARC artifacts, which are
self-contained and use only Apache-2.0/BSD dependencies.

## Unmeasured ARC accuracy

The solver considers seven fixed global transforms. This is a limited hypothesis
class; it is not the paper's learned repair mechanism. Transpose and quarter-turn
rotations swap height and width for rectangular grids. Exact agreement with
training examples is not a bound on test accuracy: the solver also emits partial-fit
fallback predictions, and training fit does not guarantee test correctness.

No ARC score is established by the synthetic smoke checks. Measure the public
evaluation split and report the actual result, including zero if observed. Do
not forecast an exact score or describe symbolic predictions as CGR evidence.

## What exists and what does not

- Existing arithmetic recovery records and coder/cyber experiments are
  not ARC-AGI evaluation. Historical arithmetic timing interpretations are withdrawn. The tested arithmetic localizer was trained on
  arithmetic states from `Qwen3-0.6B-Base`; a trained ARC error localizer has not
  been demonstrated. The paper's adapter experiments use chat-capable
  `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca` and
  different data. Do not reuse their results as ARC scores.
- `notebooks/ARC_PRIZE_2026_CGR_KAGGLE.ipynb` is a **development scaffold**. It
  ranks seven fixed transforms by how many public training pairs each reproduces,
  then assigns the best-fitting transform to `attempt_1` and the best remaining
  *distinct* hypothesis to `attempt_2`. That is **candidate filtering with a
  ranked fallback** — it is not a retry, not the paper's learned localizer, and
  not DAG repair. An earlier revision of this plan called it a "verifier-driven
  candidate retry"; nothing retried, and the phrase is withdrawn.
- Optional live diagnostics exist for KV-cache crop coherence and coder-adapter
  behaviour. They are **not wired into ARC predictions**, and the notebook says so
  in its first cell.
- Historical results are references, not executable fallbacks, and must never
  masquerade as freshly generated ARC outputs. The solver must create real
  predictions from each competition input, including unseen test tasks.

## Assets required inside the offline Kaggle image

All of these must be present before evaluation, because the run has no network:

- `transformers==4.57.6` — the version the cache-crop API was validated against.
  The notebook now warns instead of hard-failing on a mismatch and checks for
  `DynamicCache.crop` at runtime, but the pinned version is still the only one
  with measured evidence.
- `peft` for the adapter diagnostic (Apache-2.0).
- The `Qwen/Qwen3-0.6B` snapshot at `c1899de289a04d12100db370d81485cdf75e47ca`,
  attached as a Kaggle dataset. Snapshot directories are named by commit hash;
  the notebook reads a revision hint from the directory name, so renaming the attached
  folder warns and records `revision_path_matches` separately. `revision_verified` remains false without content verification.
- The competition data mounts under `/kaggle/input`. The notebook auto-detects
  competition mode from that path and globs for `*test*challenges*.json`.

## Build the actual ARC entry

1. **Acquire and verify assets:** join ARC-AGI-2, accept rules, attach competition
   data and permitted public model artifacts. Pin versions, record checksums and
   licences, confirm the checkpoint hash appears in `verification_results.json`,
   and verify the accelerator and memory budget on the real machine.
2. **Establish a correct baseline:** run against the public evaluation split,
   validate grids, task IDs and both predictions, then record accuracy and wall
   clock. Never read test outputs. Keep the local demo artifact separate from
   `/kaggle/working/submission.json`.
3. **Build a genuine ARC-specific mechanism:** define a task-level hypothesis
   representation, a public training-example verifier, committed state, failed-step
   localization and replay semantics. Compare the deterministic baseline against
   CGR-style repair under identical budgets. Do not describe verifier feedback as
   a trained hidden-state probe.
4. **Integrate the intended model component:** attach the immutable checkpoint and
   validated runtime. A cache test must compare cropped-cache continuation to a
   fresh recomputation and report measured tolerances and timings. Either train an
   ARC-specific localizer on separated task families, or disclose the verifier as
   the simpler mechanism it is. Add a specialist adapter only if it is compatible
   with the same base, **licence-clear**, verified beneficial on ARC, and present
   in the actual prediction path. Factorial and cyber wins do not transfer.
5. **Record complete evidence:** model revisions, seeds, split, candidate
   decisions, verifier results, token counts, cache boundary, base-recomputation
   comparison, timing including controller and verification, memory, final
   predictions and paired baseline outcomes. Keep every failure case.
6. **Kaggle dry run:** run with internet disabled and the exact attached assets,
   generate `submission.json`, reconcile task IDs against
   `sample_submission.json`, and confirm the artifact is accepted. No submission
   is performed by this plan; scores and private execution cannot be claimed until
   the author runs and submits the notebook.
7. **Paper-track material:** only after a valid code submission exists, write an
   ARC-specific paper describing the actual entry and its measured evaluation,
   with references, leaderboard or public-evaluation results, and a link to the
   submitted notebook. Distinguish prior unrelated experiments and do not claim a
   measured speedup without benchmarking it.

## Suggested sequence under the deadline

Validate the contest account, open-source terms and notebook mechanics; run public
tasks and record the baseline. Then integrate the smallest genuine ARC repair
mechanism and a verifier with fair no-repair controls. Reserve time for the offline
Kaggle run, missing inputs and output validation. Prepare the paper only around
results the working entry demonstrates. If these gates cannot be met, report a
draft rather than claiming eligibility.

Completion before the deadline is uncertain. Prioritize the baseline and offline validation; do not claim an integrated method until it exists and is evaluated.

## Local scaffold checks (2026-10-30)

Evidence: `results/arc_kaggle_scaffold_smoke/verification_results.json` (answers
and checks), `notebook_run.log` (cell-by-cell transcript), `verify_scaffold.py`
(harness) and `README.md` (reproduction, including the `PYTHONPATH` that supplies
`transformers 4.57.6` and `peft` from `.adapter_compare_deps`; the `.venv` alone
carries `transformers 5.20.2`). These are local CPU checks. They are not a
Kaggle execution, not an ARC score, and not a timing measurement.

Verified in this run:

- **Submission shape.** One entry per test input, exactly `attempt_1` and
  `attempt_2`, every grid validated as rectangular integers 0-9 with sides 1-30.
- **Distinct second attempt.** `attempt_2` is now the best remaining distinct
  hypothesis rather than a copy of `attempt_1`. The previous revision duplicated
  it whenever only one candidate passed; the synthetic demo produced
  `attempt_1 == attempt_2 == [[3,2,1],[6,5,4]]`. A 1x1 grid still admits only one
  distinct grid across all seven transforms, and that case is recorded as
  `distinct_attempts: [False]` rather than silently duplicated.
- **Partial-fit fallback.** When no transform fits, ranking by training-example
  pass count still supplies a distinct second attempt instead of discarding the
  ranking and emitting identity twice.
- **Error containment.** A malformed task now degrades to identity in competition
  mode and is logged, instead of aborting the whole submission; strict mode still
  re-raises so the local harness cannot hide a bug.
- **Placeholder file.** A complete, valid `submission.json` is written *before*
  solving, so an exception cannot leave the run with no artifact. A missing file
  scores 0 with no diagnostic, which is the failure mode this removes.
- **Sample reconciliation.** When `sample_submission.json` is present, task IDs are
  compared and missing/extra IDs are printed.
- **Invalid grids.** Four cases rejected: ragged rows, value 10, boolean `True`,
  empty grid.
- **Synthetic demo.** The horizontal-flip fixture selects `flip_h` and now emits a
  distinct `rot180` second attempt.
- **Cache coherence.** Cropped-cache continuation matched a fresh recomputation:
  max absolute logit delta `1.8596649169921875e-05`, argmax match, crop boundary 8
  of 17 prompt tokens, CPU float32, `torch 2.14.1+cpu`, `transformers 4.57.6`. This
  independently reproduces the `1.86e-5` figure previously recorded only in prose.
- **Factorial diagnostic.** Base answer failed all three cases (it returns 0 for
  every input); coder answer passed all three. One prompt, one task, three cases.

Known limits of this evidence:

- The notebook still has **no saved cell outputs**. The transcript is the record;
  a Kaggle commit would store outputs and is still required.
- Older `revision_verified` values were derived from a directory name and are not cryptographic attestations. The corrected notebook keeps this flag false and records path agreement separately.
- The fp16 tolerance `0.125` has **never been exercised** — every run so far is CPU
  float32 — and is reported with `tolerance_validated: false`. It is a diagnostic,
  not a gate.
- The factorial verifier executes model output in a restricted subprocess with a
  3-second timeout; that is containment, not a security sandbox.
- The adapter diagnostic loads a derivative whose data provenance is unresolved;
  it is enabled locally only, behind an explicit acknowledgement flag.

## Remaining work before any eligibility claim

- [ ] Mount ARC-AGI-2 public evaluation data and record the baseline accuracy
      (no score assumed).
- [ ] Run the notebook inside Kaggle with internet disabled and confirm the
      committed version produces `submission.json` without manual edits.
- [ ] Replace or supplement the seven transforms with a mechanism whose predictions
      depend on the paper's actual components.
- [ ] Produce an ARC-specific paper; the current `PAPER.md` is about arithmetic
      KV-cache repair and specialist routing and cannot serve as the linkage.

## Shared evidence correction (2026-10-30)

The general research paper and ARC route share one evidence standard. Historical
arithmetic 25.9x/27.2x cache and 2.0x/1.70x full-loop speedup interpretations
are withdrawn. Those timers omitted required work; the ARC notebook's CPU
cache-coherence check does not validate them. The ARC entry has no demonstrated
learned localizer, adapter contribution, or cache speedup on ARC tasks.
See `findings/2026-10-30-paper-review.md`. A directory name can indicate an
expected model revision but cannot cryptographically verify the weights.

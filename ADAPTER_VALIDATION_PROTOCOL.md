# Adapter validation package and next experiments

Prepared 2026-10-30 while the cyber and UI adapters are training. This protocol
is prospective for the new multi-domain experiments. Previously observed coder
results are development evidence, not retrospectively preregistered outcomes.

## Available now

- `data/validation/cgr_multidomain_v1.json`: 20 public sanitized-card scenarios,
  deterministic task prompts, boundary values, untrusted display names, and a
  20-node dependency graph. These are variations of **one task family**, not
  20 independent evaluation tasks. The SHA-256 is recorded by the runner.
- `cgr/control/dependency_graph.py`: plan validation, ancestor-constrained
  fault candidates, descendant invalidation, stale-parent rejection, atomic
  candidate rejection, and a separate cache suffix replay plan.
- `cgr/control/multidomain_validation.py`: executable clamp checks; SQLite
  parameterization/behavior checks; static HTML label, value, status, and
  escaping checks. Positive controls and deliberately wrong candidates exercise
  the verifiers. Static HTML checks do **not** validate React/Next.js behavior,
  keyboard interaction, visual quality, or full accessibility conformance.
- `scripts/validate_multidomain.py`: freezes fixtures, exercises the graph and
  validators, checks adapter metadata, and grades supplied generated artifacts.
  Missing candidates are `not_run`, never successful model results.

Run:

```powershell
& 'D:\Research Paper\.venv\Scripts\python.exe' scripts\validate_multidomain.py --self-test --adapter 'coder=D:\Research Paper\notebooks\coder_qwen3_0.6b_q_mlp_r16_v1___Coder Adapter\checkpoint-1000'
& 'D:\Research Paper\.venv\Scripts\python.exe' -m pytest tests\test_multidomain_validation.py -q
```

Output: `results/multidomain_validation_readiness.json`. The current fixture
self-test passes all 20 scenarios with **zero model generations**. Descendant
recomputation uses fixture artifacts; synthetic token boundaries validate the
replay plan, not actual model KV replay. Five new
unit tests pass. The coder's base ID/revision, LoRA configuration and weight
presence pass metadata checks; metadata alone does not promote an adapter.

Copy `data/validation/candidate_template.json` to a results file and populate
each scenario's `coder` Python source, `cyber_defensive` JSON query contract,
and `ui` HTML with actual model outputs. Grade with:

```powershell
& 'D:\Research Paper\.venv\Scripts\python.exe' scripts\validate_multidomain.py --candidates results\candidate_outputs.json
```

Retain raw outputs, model/checkpoint hashes, exact prompts/token IDs, generation
settings, route decisions, verifier feedback, tokens and timings in the
inference record. The grading command cannot establish their provenance.

## Corrections to the pasted proposal

1. The observed clamp failure used a chat template. A bare-signature/chat
   mismatch is not an established explanation. Neither is specialization
   forgetting or a preference for elaborate code: the actual wrong function
   simply returned `value`. Keep this regression in the diagnostic set.
2. The specialist currently reads a base-produced prefix cache. Its output
   should be compared with full-prefix specialist prefill before attributing
   the failure to adapter quality. This is distinct from the measured return
   to base through the Bridge.
3. Preserved independent artifacts after a repaired node still require their
   **later KV entries to be rebuilt** if their original token positions attended
   to the old prefix. Crop preserves only the unchanged prefix. Replay preserved
   artifact tokens; regenerate dependent artifacts. Semantic independence does
   not imply attention-cache independence.
4. A variable definition or an exception site supplies a suspect, not proof of
   a causal fault. Multiple upstream causes must remain ambiguous. Binary
   rollback search needs a justified monotone diagnostic predicate; arbitrary
   program correctness is not monotone under prefix truncation.
5. Planned specialist nodes remain a valid comparison arm. Restricting routing
   to verifier-triggered repair is a policy to evaluate, not an established
   architectural law. Do not remove simple regression tasks after seeing losses.

## Gate 0: diagnose coder switching before adding specialists

Freeze identical full chat token sequences for the known clamp/even-sum cases
and fresh development prompts. Use the same checkpoint, seed, greedy decode,
and 256-token cap for four arms:

| Arm | Prefix computation | Continuation policy |
|---|---|---|
| Base | Full base prefill | Base |
| Full specialist | Full coder prefill | Coder |
| Switched specialist | Base prefix; last header token re-evaluated under coder | Coder |
| Full specialist then Bridge | Full coder prefill and verified patch; discard its cache | Base replay and continuation |

Record function tests, truncation, actual LoRA state, first-token logit changes,
prefix length, and generation cost. If full coder succeeds but switched coder
fails, investigate context/cache strategy; do not describe that as payload
incapacity. If both fail, the checkpoint/prompt remains the candidate cause.
This is a diagnostic, not a significance test on two known tasks.

## Gate 1: register each arriving adapter

Check immutable parent SHA, training manifest, tokenizer revision and chat
template hash, target modules/rank/alpha, weight hashes and finite tensors.
Check exact named adapter selection, disabling, and base-logit restoration.
Test each ordered pair base/coder/cyber/UI on separate working caches, including
unknown/unavailable routes. Record model loading, resident memory and swap time.
Do not invent missing manifests or silently use a different parent revision.

The metadata CLI accepts `--adapter cyber_defensive=<folder>` and
`--adapter ui=<folder>` when the folders arrive. Loading, tokenizer equivalence,
and full switch/Bridge checks remain separate gates. Candidate status persists
until held-out domain quality is measured.

## Gate 2: end-to-end integration milestone

First use declared node skills and a fixed graph. At step 20, invoke independent
contracts on the failing node and its ancestors; report all unresolved upstream
suspects. The fixture deliberately injects an error at step 6 and only reveals
it through a later boundary check. Injection labels are withheld from the
attribution function. Known fixture repairs are harness controls, never model
recoveries.

For real inference: generate a bounded patch with the selected specialist;
verify before commit; invalidate its descendants; preserve independent
artifacts; crop to the saved pre-step-6 token boundary; base-replay the exact
accepted patch; then regenerate dependent artifacts and replay preserved
suffix artifacts in dependency order. Check final task behavior again. A
patched helper alone is not end-task success.

Use the public static-HTML/SQL suite first, then add a **separate pinned frontend
fixture** matching the UI training stack. Require build/typecheck, browser
input/change behavior, accessible names, keyboard behavior and escaped hostile
display strings. Framework versions must come from the trained artifact's task
contract. The current package does not claim those browser/component gates pass.

Do not add an automatic planner until this fixed-graph milestone works. Test a
planner separately for schema validity, cycles, missing dependencies, unavailable
skills, and agreement with task contracts before it controls model execution.

## Gate 3: prospective comparative study

Construct and freeze **60 independent multi-part tasks** after a separate
development set: 20 coder-led, 20 defensive-security-led and 20 UI-led. Group by
source/project/template; allocate a whole group to one split. Check overlap
against training data where available. These task artifacts are still to be
created; the 20 public fixtures above are not their replacement. Freeze hashes,
verifiers, adapter checkpoints, tolerances and analysis before model evaluation.

Each task is evaluated under all arms:

| Arm | Purpose |
|---|---|
| Base one-pass | Initial difficulty / failure denominator |
| Base sequential retry | Required practical comparator; same early stopping |
| Localized base repair | Isolate localization from specialist choice |
| Localized specialist, full context recompute | Isolate specialist benefit from cache reuse |
| Localized specialist + Bridge | Proposed system |
| Oracle route/root, diagnostic only | Attribution/routing headroom; not deployable |

Do not use polluted specialist-cache continuation as a production baseline.
An explicitly labelled contaminated-cache negative control can test whether
coherence checks are sensitive to the violation they prohibit.

Initial per-attempt caps: coder 256, cyber 256, UI 512 new tokens. At most two
extra repair calls per task, at most 4,096 generated tokens and 32,768 forwarded
token positions across production operations. Include prompts, replay and
discarded attempts in accounting; token-cap breaches terminate and count as
failure. Select different caps only on development tasks before freezing this
study. A fixed 32-token patch is not assumed sufficient. Report early stopping
and actual budget consumed; also compare accuracy-versus-budget curves.

First attempt is greedy; retry sampling uses temperature 0.8/top-p 0.9 with
task-derived fixed seeds, shared across arms where applicable. Randomize arm
order within task; load once and warm each policy separately. Report cold load
and resident memory separately from warm inference. Numerical reference
prefills are diagnostic operations: report their cost separately and do not
mix diagnostic timing with production latency.

Primary outcome: final task verifier pass rate. Secondary: naturally failing
base-task recovery, regressions to verified artifacts, attribution accuracy,
abstentions, accepted patches, calls/tokens, wall time p50/p95, and peak memory.
Never mix synthetic injections into natural-failure recovery denominators.
Use exact paired tests on discordant **task** outcomes and paired task-level
bootstrap intervals; related variants are clustered. Predeclare the primary
comparison (Bridge versus base sequential retry); label other comparisons
secondary and account for multiplicity if making confirmatory claims.

Any reproducible cache corruption blocks that runtime path. A quality gain
requires a positive paired 95% interval against the primary comparator; a
speed claim also requires measured wall-clock savings under comparable quality
and budgets. An inconclusive result does not prove equivalence. A negative
specialist result is retained. These thresholds are for reporting evidence,
not a promise that N=60 has adequate power for a small effect; assess required
sample size from development discordance before freezing the final study.

> **2026-10-30 evidence correction:** Historical 25.9x/27.2x cached-generation
> and 2.0x/1.70x full-loop speedup interpretations are withdrawn. The cache
> timer measured prefix prefill with no repaired-suffix decode; repair timers
> excluded required controller work. Token/FLOP counts were estimates.
> Recovery counts remain descriptive conditional-repair observations, with
> candidate clustering unaccounted for. Use `PAPER.md` and
> `findings/2026-10-30-paper-review.md` for the corrected interpretation.

# Commitment-Gated Reasoning: Draft Sections 1–3

_Working draft. Claims about capability remain provisional until the listed
experiments pass; the project is a controlled synthetic reference system._

## 1. Introduction

Large language models can produce intermediate text and final answers in one
generation stream. That makes it difficult for a downstream system to tell
which statements were tentative, which were checked, and which were committed
as the answer. Autoregressive decoding also provides no native operation for
revising an earlier span after later evidence arrives. An application can add
search, retries, tools, or explicit editing, but those controls are usually
external to the model's token prediction objective. Untrusted text creates a
related control problem: an instruction embedded in a document can compete
with the instruction the application intended the model to follow.

We study **Commitment-Gated Reasoning (CGR)**, a prototype architecture that
separates a latent reasoning signal from the text artifact, predicts the
epistemic and control quantities used by an action gate, and routes supported
claims through an exact interpreter before commitment. Its safety claim C1 is
narrow and structural: a verified-wrong example is prevented from updating
the plastic substrate or creating a factual memory entry, while the gate can
still receive a rejection-training signal. C1 is not a general prompt-injection
defense, nor does it establish that the learned editor produces useful repairs.

The implementation and current evidence constrain the paper's claims. The
interpreter handles structured, templated claim specifications; it is not a
free-text theorem prover. C1 and the empty-memory invariant have passing
DiffuGPT experiments. The original DiffuGPT repair checkpoints remain below
the predeclared 0.20 clean-infill floor, and the full architecture ablation
has not been run. The new checkpoint controller is currently a CPU-only
prototype tested with callbacks; the demo can pass the learned gate proposal
to it, but it is not an end-to-end loop inside `CGRSystem`.
We therefore use this paper to specify a falsifiable control mechanism, not
to claim broad reasoning, coding, or injection-resistance capability.

Our contributions are:

1. A reference implementation that separates a latent reasoner, task heads,
   an action classifier, an exact interpreter, and guarded consolidation.
2. A structural test of C1 with unguarded-leak and verified-right controls.
3. A benchmark and reporting pipeline that distinguishes design invariants
   from empirical results and gates repair comparisons on editor ability.
4. A lightweight checkpoint/backtracking controller and deterministic logic,
   safety, and injection dataset generators for subsequent control experiments.

## 2. Related Work

### 2.1 Latent diffusion for reasoning

LaDiR encodes textual reasoning steps into blocks of continuous thought tokens
and applies latent diffusion to iteratively refine those blocks. It studies
reasoning trajectory generation, diversity, and test-time compute for an
existing language model [1]. CGR shares the separation of latent reasoning
from ordinary text and the use of iterative denoising. Its research question
is different: CGR makes action selection, exact adjudication, and the
conditions for updating persistent state explicit. In the current code,
however, the latent reasoner and diffusion editor do not form a verified
DAG-search system. The controller in `cgr/control/commitment.py` is a separate
mockable prototype, so the distinction is an architectural target rather
than a demonstrated end-to-end advantage over LaDiR.

The name **DiffuReason** is used by a 2026 sequential-recommendation method.
That work combines thinking tokens, diffusion refinement, and group-relative
policy optimization for recommendation ranking [2]. Its task, objective, and
evaluation are not directly comparable to this repository's synthetic
claim-adjudication and text-repair tasks. We cite it as adjacent work on
think-then-diffuse designs, not as a baseline or evidence of priority over it.

### 2.2 Verbal reflection and persistent memory

Reflexion converts feedback into verbal reflections and stores them in an
episodic memory buffer for later agent trials [3]. CGR instead distinguishes
retrievable experience memory from a small plastic parameter substrate and
places a verifier gate on consolidation. Both approaches use feedback and
persistent state, but their memory contents and update mechanisms differ.
The present experiments establish structural update constraints; they do not
show that CGR's memory improves task performance over a strong Reflexion
baseline.

### 2.3 Preference optimization

Direct Preference Optimization (DPO) derives a preference-learning objective
that directly fine-tunes a policy from chosen/rejected response pairs without
the conventional separately trained reward model and online policy
optimization loop [4]. DPO is a training-time alignment method. CGR's gate
is a run-time controller over proposed actions, and its C1 mechanism governs
whether verified examples may update persistent state. DPO is therefore a
complementary alignment approach, not a system-level substitute for
checkpointing, external verification, or rollback. A future study could
compare a DPO-tuned generator under the same verifier and control policy.

### 2.4 Scope of comparison

The comparisons above motivate design choices; they do not imply that the
current experiments outperform those methods. Our benchmark uses deterministic
templates and exact parsers, while the cited methods use different datasets,
tasks, model scales, and objectives. A fair empirical comparison would require
matched tasks, compute, backbone, and evaluation protocol. None is claimed in
this draft.

## 3. Methodology: Architecture

### 3.1 System overview

CGR receives an input, produces a shared representation, estimates task and
epistemic quantities, and proposes an action. The text artifact is generated
or edited by a masked-diffusion backbone. For the claim tasks implemented
here, a structured specification is evaluated by an exact interpreter. A
verified outcome can be returned or considered for consolidation; an
unsupported claim is deferred, and a verified-wrong claim is excluded from
the plastic and factual-memory update paths.

The current implementation uses a pre-trained DiffuGPT-S backbone with a
GPT-2 tokenizer in the DiffuGPT preset. The latent reasoner produces a
conditioning signal for the backbone through a two-pass path. The seven heads
and action gate consume shared features that combine pooled token features,
reasoner conditioning, and a task clock. This is a small experimental system,
not a production agent with arbitrary tool use.

### 3.2 Latent reasoning channel

The reasoner maps the input representation to a sequence of latent thought
blocks and iteratively refines them. A projection yields a conditioning vector
for the artifact model. The channel is intended to influence generation
without exposing its latent vectors as user-facing text. The implementation
tests whether the conditioning path affects outputs, but the existing
side-channel experiment's positive control did not establish that the channel
carried information absent from the input. Thus causal use of useful
reasoning content remains an empirical question.

### 3.3 Seven prediction heads

The shared head module estimates seven quantities used for analysis and
routing:

| Head | Meaning in this implementation |
|---|---|
| Value | Probability-like estimate of a successful outcome |
| Outcome | Vector of predicted outcome features |
| `tau` | Estimated completion or reasoning time |
| Epistemic status | Known, computable, check later, or unknown |
| Scope | One of the controlled scope families or no explicit scope |
| Source | Weights, experience, or derived evidence source |
| Disagreement | Estimated disagreement/uncertainty signal |

These heads are supervised by labels emitted from the hand-written scaffold.
Their presence does not by itself show that the model has calibrated
uncertainty or learned general epistemic reasoning; calibration and label
health are reported in the experiments.

### 3.4 Action gate and controller

The learned gate scores eight actions: `ANS` (answer), `DELIB` (deliberate),
`VER` (verify), `REP` (local repair), `REORD` (reorder), `REV` (global
revision), `DEF` (defer), and `UPD` (update). During training, it imitates the
scaffolding policy and then uses a Gumbel-softmax policy with a KL leash. The
curriculum does not offer all eight actions: existing experiment notes report
that `REORD` and `UPD` lack training examples. The nominal action count must
not be reported as eight learned capabilities.

The standalone `CommitmentDAG` stores candidate artifacts as nodes with
parent links, verification status, and child links. Its controller defers
when the estimated completion time exceeds a deadline and routes a high
surprise signal to `REV`, restoring the nearest verified ancestor. A verifier
rejection also returns to a verified ancestor. These routes are covered by
mock tests. The demo can pass the model's proposed action to this controller,
while surprise and deadline signals are user-set test inputs. The controller
is not part of `CGRSystem`'s forward/repair loop, so its tests establish
control-code behavior only, not learned model behavior.

### 3.5 Exact interpretation and commitment

The interpreter operates on structured claim specifications whose arithmetic
and scope semantics are explicitly represented. It can accept, reject, defer,
or refuse according to those semantics. In `answer_claim`, refusal and
unresolved scope handling precede the learned gate, so the interpreter is
load-bearing for the current controlled claim task. This is not an open-ended
natural-language verifier; a parser that maps arbitrary text to a correct
structured specification remains future work.

### 3.6 C1 guarded consolidation

Let \(\phi\) denote the plastic substrate and \(M\) the factual experience
memory. For an example adjudicated as verified-wrong, the update guard enforces

\[
\nabla_{\phi} L = 0 \quad\text{and}\quad \Delta M_{\mathrm{factual}}=0,
\]

while leaving the gate's rejection signal trainable. The implementation uses
the absence of a plastic-parameter gradient so AdamW skips those parameters on
this path; the paper records that optimizer weight decay makes this detail
material. Exp6 checks parameter identity, memory writes, gate gradients, an
unguarded leak control, and a verified-right update control. The result
supports this structural invariant for the tested pipeline. It does not prove
that all false statements can be recognized, that prompt injection is
prevented, or that held-out behavior improves (C2 remains unresolved).

## References

1. H. Kang, Y. Zhang, N. L. Kuang, N. Majamaki, N. Jaitly, Y.-A. Ma, and L.
   Qin. “LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning.” 2025,
   revised 2026. <https://arxiv.org/abs/2510.04573>.
2. J. Jiang et al. “DiffuReason: Bridging Latent Reasoning and Generative
   Refinement for Sequential Recommendation.” 2026.
   <https://arxiv.org/abs/2602.09744>.
3. N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao.
   “Reflexion: Language Agents with Verbal Reinforcement Learning.” 2023.
   <https://arxiv.org/abs/2303.11366>.
4. R. Rafailov et al. “Direct Preference Optimization: Your Language Model Is
   Secretly a Reward Model.” 2023. <https://arxiv.org/abs/2305.18290>.

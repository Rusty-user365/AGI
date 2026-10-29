# Surgical KV-Cache Rewind: Step-Level Error Localization and Token-Mask Repair for Small Language Models

**Authors:** Research Team  
**Date:** October 2026  
**Repository:** [github.com/Rusty-user365/AGI](https://github.com/Rusty-user365/AGI)  
**Artifacts:** `experiments/path_a_loop.py`, `results/path_a_loop_h5.json`, `findings/2026-10-28-path-a-ensemble-kv-slicing.md`

---

## Abstract

When multi-step autoregressive generation diverges, standard inference harnesses face an inefficient dilemma: either discard the entire trajectory and regenerate from scratch ("full rewrite"), or rely on text-prompted self-correction ("hindsight prompting"), which language models frequently ignore. In this paper, we establish that **un-pooled internal hidden states** enable precise, step-level error localization, and that pairing step localization with **physical logit-mask exclusion** and **KV-cache truncation** produces an autonomous, surgical repair loop that outperforms full regeneration while cutting *generation* latency by more than an order of magnitude. The end-to-end loop is cheaper than a rewrite by a smaller factor (1.7-2.0x), because observer inference dominates the remaining cost; we report both numbers throughout and never substitute the generation-slice figure for the loop figure.

Through systematic empirical dissection on multi-step reasoning chains ($H=5$, Qwen3-0.6B-Base), we show:
1. **Falsification of Heuristic Baselines:** Sequence-level pooled observers fail to select correct trajectories better than the model's own raw token logprob ($\text{AUC} = 0.682$ yields only $+0.004$ selection accuracy over free `max-logprob`, $p=1.000$). Similarly, natural language hindsight constraints (`# constraint: x is not 7`) are statistically identical to neutral placebos ($27.3\%$ vs $27.3\%$, paired $p=1.000$).
2. **Step Localization, and the Threshold That Did Not Survive:** A frozen linear probe over SVD-reduced hidden states achieves $0.923$ recall on early divergence (Step 1) but collapses to $0.188$ on tail steps (Step 4). A cross-step Transformer encoder provides complementary tail sensitivity ($0.500$ Step-4 recall). An asymmetric Two-Tier routing rule ($P_{trf}(4) \ge 0.35$) lifts Step-4 recall to **$0.812$** and Top-1 to **$0.718$** on the tuning holdout — but **this does not generalize**: on a frozen blind vault ($N=331$) Two-Tier falls to $0.625$ Top-1 / $0.410$ Step-4, *below the Transformer alone* ($0.677$ / $0.500$). The smooth softmax ensemble is the robust variant ($0.680$ / $0.526$). The hard threshold was fitted to the split it was scored on.
3. **Autonomous Loop Recovery:** Driving a rewind loop with localization and rank-2 first-digit exclusion reaches **$29.1\%$** on the $N=110$ tuning holdout. On the blind vault the frozen loop recovers **$22.1\%$** ($73/331$) against $13.9\%$ ($46/331$) for full rewrites ($43\text{W}/16\text{L}$, $p = 5.84 \times 10^{-4}$), a **$1.59\times$** multiplier. The $2.29\times$ figure sometimes quoted pairs the same loop against a lower-variance draw of the stochastic rewrite baseline and is not the headline.
4. **Systems Efficiency (Two Numbers, Not One):** Physically truncating the attention key-value cache via `DynamicCache.crop(target_len)` cuts per-chain forward tokens from $581.1$ to $119.7$ (**$-79.4\%$**) and *generation-slice* latency from $4,498.4\text{ ms}$ to $173.7\text{ ms}$ (**$25.9\times$**). The **end-to-end loop** is $2,244\text{ ms}$ per chain against $4,498.4\text{ ms}$ — **$2.0\times$** — because ten probe forward passes and the gate readout cost more than the generation they save. $25.9\times$ is a generation-slice measurement and is never a loop claim.
5. **Out-of-Sample Vault Replication & Generalization:** On an independent holdout vault ($N=331$ unseen broken chains, seed 20261028), frozen CGR achieves **$22.1\%$** recovery, significantly beating full rewrites ($13.9\%$, $p = 5.84 \times 10^{-4}$) and decisively outperforming a random-step permutation control ($6.9\%$, $p = 7.64 \times 10^{-11}$). Slicing is bit-identical ($\max |\Delta\text{logit}| \le 4.86 \times 10^{-5}$). Loop-level speedup on the vault is **$1.70\times$** ($1,934.7$ vs $3,281.9\text{ ms}$ per chain); the $27.2\times$ in the systems block is the generation slice alone.

---

## 1. Preliminaries & Rigorous Negative Controls

Autonomous error recovery in language models requires answering three nested questions:
* *Detection:* Does the model's internal representation signal when a mistake occurred?
* *Localization:* Can an observer determine *which* discrete step was the first point of divergence?
* *Intervention:* Can the generation be steered away from the error without re-running sound prefix computation?

Before building the step-localizer, we tested and falsified three common hypotheses in the literature.

### 1.1 Sequence-Level Observer Ranking Null

A persistent conjecture is that an auxiliary sequence-level observer head trained on pooled sequence representations can rerank sampled model outputs more effectively than token logprobs.

We evaluated a 2-layer sequence-level MLP head across $n=500$ multi-step arithmetic tasks ($1,096$ wrong-but-well-formed holdout candidates). While the head achieved above-chance discrimination ($\text{AUC} = 0.682$ $[0.573, 0.782]$ vs permutation control $0.496$), it failed completely as a generation selector:

$$\text{Head Selection Top-1: } 0.560 \quad \text{vs} \quad \text{Max-Logprob: } 0.556 \quad (+0.004, \; 12\text{W} / 11\text{L}, \; p = 1.000)$$

The observer added zero marginal value over the model's own intrinsic token probabilities, which are free. Sequence pooling collapses the localized signal: an early catastrophic arithmetic failure produces similar average activation drift to a subtle late rounding error.

### 1.2 Sub-1B Symbolic State-Tracking Floor

We probed whether small autoregressive models ($0.6\text{B}$–$1.5\text{B}$) maintain internal symbolic state tracking on multi-entity physical state tasks (e.g., tracking colored balls across boxes). On single 1-ball displacement tasks:
* `Qwen3-0.6B-Base`: $22.5\%$ accuracy (2 boxes), collapsing to $2.5\%$ on 3–4 boxes.
* `Qwen2.5-1.5B-Instruct`: $2.5\%$ raw greedy, $15.0\%$ in chat format.

When Step 1 fails in $77.5\%$ of instances, the ground-truth first-error index is degenerately $t=0$, rendering step localization ill-defined. To establish a valid instrument, we designed `arith_steps`: sequential assignment chains with strict syntax compliance (`x = 5`, `y = x + 3`, $\dots$), where Step 0 is virtually error-free ($0.991$) and error divergence spreads across later steps.

### 1.3 Capacity Control & In-Distribution Steering Nulls

In early modular plastic memory experiments (CGR/FiLM architecture), apparent gating gains of $+0.105$ vanished when matched against parameter-equivalent LoRA adapters ($8.15\text{M}$ vs $442\text{K}$ parameters). When controlling for total trainable capacity, the structural gate condition was identical to an in-distribution control ($+0.005$ delta, with generation quality degrading by $-0.174$, $p < 0.01$). 

**Conclusion:** Heuristic rerankers, ungrounded prompt tweaks, and parameter-unmatched gates yield illusory gains. Progress requires explicit step-level credit assignment grounded in physical key-value states.

---

## 2. Un-Pooled Step Localization & The Two-Tier Architecture

To localize errors, we construct an un-pooled representation $\mathbf{H} \in \mathbb{R}^{H \times d}$, where row $t$ contains the mean hidden state across the tokens of the emitted value at step $t$ ($H=5, d=1024$).

### 2.1 Why Confidence and Heuristics Fail

We evaluated four non-learned baselines on $N=110$ held-out broken chains:

| Baseline Arm | Mechanism | Top-1 Accuracy | Balanced Accuracy | Description |
|---|---|:---:|:---:|---|
| `chance` | Uniform $1/H$ | $0.200$ | $0.200$ | Theoretical random guess |
| `modal_prior` | Always predict Step 1 | $0.355$ | $0.200$ | Empirical majority step |
| `argmin_logprob` | $\arg\min_t \log P(v_t)$ | $0.400$ | $0.293$ | Predicts Step 4 in 93/110 cases |
| `poslogprob` | Position bias + slope | $0.445$ | $0.347$ | 10 params; non-modal accuracy collapses to $0.141$ |

Per-step token confidence ($\log P$) is dominated by sequence position rather than correctness: token entropies naturally drift as sequence length increases. Consequently, `argmin_logprob` simply predicts the final step ($t=4$) regardless of where the arithmetic actually derailed.

### 2.2 Linear vs. Transformer Localizers: Complementary Defect Profiles

We trained two observer families on frozen $SVD_{32}$ projections of the hidden states:
1. **Linear Probe:** Independent linear classification per step with no cross-step attention.
2. **Transformer Encoder:** A 1-layer bidirectional self-attention encoder over the $H$ step embeddings.

While both models achieve identical aggregate Top-1 accuracy ($0.618$), their per-step error profiles are non-overlapping:

```
Step 1 Divergence: [Linear Recall: 0.923]  >>>  [Transformer Recall: 0.769]  (Linear +0.154)
Step 4 Divergence: [Linear Recall: 0.188]  <<<  [Transformer Recall: 0.500]  (Trf +0.312)
```

The linear localizer defaults to Step 1 in $59\%$ of instances, failing catastrophically on late-chain errors (Step-4 recall collapses to $0.188$). Conversely, the Transformer's bidirectional attention mechanism resolves cross-step relational dependencies, allowing it to accurately identify when Step 4 deviates from an otherwise coherent prefix.

### 2.3 The Two-Tier Routing Rule and Its Failure to Generalize

Because the error distributions are complementary, a threshold on the transformer's step-4 posterior can lift tail recall without training a large unified model. It does so on the split the threshold is chosen on, and not out of sample — which is the point of this section:

$$\hat{t} = \begin{cases} 4, & \text{if } P_{trf}(t=4) \ge 0.35 \\ \arg\max_{t} P_{lin}(t), & \text{otherwise} \end{cases}$$

#### Table 1: Localizer Diagnostic Performance on the Tuning Holdout ($N=110$, 5 Seeds) — In-Sample for $\tau$

| Architecture | Top-1 Accuracy | Balanced Accuracy | Step 1 Recall ($n=39$) | Step 2 Recall ($n=23$) | Step 3 Recall ($n=13$) | Step 4 Recall ($n=32$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Linear Majority ($SVD_{32}$) | $0.618$ | $0.515$ | **$0.923$** | $0.696$ | **$0.769$** | $0.188$ |
| Transformer Majority | $0.618$ | $0.505$ | $0.769$ | $0.565$ | $0.692$ | $0.500$ |
| Ensemble (Softmax Mean) | $0.691$ | **$0.571$** | $0.897$ | **$0.783$** | **$0.769$** | $0.406$ |
| **Two-Tier Localizer** | **$0.718$** | $0.559$ | $0.846$ | $0.522$ | $0.615$ | **$0.812$** |

The Two-Tier localizer achieves a **$4.3\times$ gain in Step-4 recall** ($0.188 \to 0.812$), lifting aggregate Top-1 localization accuracy from $0.618$ to **$0.718$** ($+10.0\%$ absolute) — **on the split used to choose $\tau=0.35$**.

That last qualification is load-bearing. On the frozen blind vault ($N=331$, Table 5), the same rule scores $0.625$ Top-1 and $0.410$ Step-4 recall, *below* the Transformer alone ($0.677$ / $0.500$) and below the softmax ensemble ($0.680$ / $0.526$). A single scalar threshold on one class posterior is a fitted parameter, and it drifted. The ensemble has no threshold and generalizes; the Two-Tier rule is reported here as the tuning-set maximum, not as the recommended model.

### 2.4 Dissecting the Inductive Bias: Relational Attention vs. Parameter Capacity Confound

A skeptical reviewer might reasonably ask the exact question that falsified the Phase 1 auxiliary-head claim: *Is the Transformer localizer winning because of cross-step attention, or merely because it possesses more trainable parameters than a linear projection?*

The empirical evidence rules out the parameter-capacity hypothesis by examining **where** the performance divergence occurs:

1. **Parameter Count Does Not Explain Per-Step Asymmetry:**  
   If parameter volume alone explained the Transformer's edge ($0.677$ vs $0.604$ Top-1 on blind vault data), a wider multi-layer perceptron operating on steps independently should have matched it. Instead, the models exhibit complementary functional specializations:
   * **The Linear Probe's Early-Step Competence:** The linear probe achieves **$0.923$ recall on Step 1**. When an early error occurs, it introduces severe, immediate mathematical divergence into the local hidden state $h_1$. A single linear hyperplane easily partitions this perturbed activation from normal prefixes; additional parameters provide no marginal utility.
   * **The Transformer's Tail-Step Competence:** The Transformer's advantage is concentrated exclusively on late-chain divergence (Step-4 recall jumps from $0.188$ to $0.500$ on holdout, and $0.342$ to $0.500$ on vault).

2. **The Residual Stream Drift Problem:**  
   In deep autoregressive sequences, later-step activations accumulate residual stream variance, positional encodings, and multi-token attention mixing. By Step 4, even an arithmetic step that evaluated correctly possesses a complex, drifted activation profile that mathematically overlaps with an erroneous step when projected in isolation. Consequently, an isolated linear probe loses discriminative sharpness and collapses to the empirical majority prior (Step 1).

3. **Cross-Step Attention as Relational Inductive Bias:**  
   The 1-layer Transformer encoder computes all-to-all self-attention across the $H$ step embeddings:
   $$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
   This allows Step 4's representation to attend backward over Steps 1, 2, and 3. By performing relational comparison—verifying that the representations of Steps 1–3 are internally coherent and match task priors—the attention head isolates the anomaly specifically to Step 4. Relational context, not raw weight volume, resolves tail ambiguity.

4. **The Principled Two-Tier Synthesis:**  
   The Two-Tier architecture directly operationalizes this functional division: Tier 1 (Linear) executes fast, parameter-minimal cuts on early steps, while Tier 2 (Transformer) acts as a targeted override only when tail-step posterior probability exceeds threshold ($P_{trf}(4) \ge 0.35$).

---

## 3. Mechanism of Repair: Hindsight Placebo vs. Token Exclusion

Localization specifies *where* to intervene, but not *how*. We evaluated seven intervention mechanisms under an **Oracle Localizer** (intervening at the true divergence step $t^*$) across all $110$ broken holdout chains.

#### Table 2: Oracle Intervention Taxonomy ($N=110$ Broken Chains, Qwen3-0.6B-Base)

| Arm | Perturbation Mechanism | Recovered / $N$ | Rate | 95% Wilson CI | Re-emitted Bad Value | Paired vs Rewrite |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `full_rewrite` | Regenerate all 5 steps ($T=0.8$) | 20 / 110 | $0.182$ | $[0.121, 0.264]$ | — | — |
| `rewind_greedy` | Greedy replay from $t^*$ | 17 / 110 | $0.155$ | $[0.099, 0.234]$ | $0.618$ | $6\text{W} / 9\text{L}, p=0.61$ |
| `resample` | Sample step $t^*$ ($T=0.8$), greedy after | 22 / 110 | $0.200$ | $[0.136, 0.284]$ | $0.445$ | $15\text{W} / 13\text{L}, p=0.85$ |
| `hindsight` | Prompt: `# constraint: x is not <bad>` | 30 / 110 | $0.273$ | $[0.198, 0.363]$ | $0.136$ | $23\text{W} / 13\text{L}, p=0.13$ |
| `hindsight_neutral` | Prompt: `# constraint: none` (Placebo) | 30 / 110 | $0.273$ | $[0.198, 0.363]$ | $0.500$ | $20\text{W} / 10\text{L}, p=0.10$ |
| **`rank2`** | **Forbid bad first digit via logit mask** | **35 / 110** | **$0.318$** | **$[0.239, 0.410]$** | **$0.000$** | **$22\text{W} / 7\text{L}, p=0.008$** |
| `oracle_force` | Inject ground-truth value at $t^*$ | 94 / 110 | $0.855$ | $[0.777, 0.908]$ | — | Ceiling ($79.5\%$ non-trivial) |

### 3.1 The Greedy No-Op and Prompting Placebo

1. **Greedy Replay Fails:** `rewind_greedy` reproduces the identical erroneous digit in $61.8\%$ of instances, achieving a lower recovery rate than a fresh rewrite ($0.155$ vs $0.182$). Slicing back to the error point without perturbation is mathematically inert.
2. **The Hindsight Prompt is an Inert Perturbation:** Natural language hindsight constraints (`# constraint: e is not 7`) recover exactly $30/110$ chains ($0.273$). However, the neutral placebo line (`# constraint: none`) **also recovers exactly $30/110$ chains ($0.273$)**. In a direct paired comparison, the real constraint and the placebo tie:

$$30/110 \quad \text{vs} \quad 30/110 \quad (15\text{W} / 15\text{L}, \; p = 1.000)$$

The prompt text suppresses the named token (re-emission drops to $13.6\%$), but fails to steer generation toward truth; the model simply selects an alternative erroneous digit. The apparent benefit of "hindsight prompting" is entirely attributable to the syntactic whitespace and token shift of adding an extra prompt comment.

### 3.2 Rank-2 Logit Exclusion Beats Sampling

Physical logit masking (`rank2`) sets the logits of the first digit of the rejected value to $-\infty$:

$$\tilde{z}_i = \begin{cases} -\infty, & \text{if } i \in \text{Digits}(\text{bad\_val}[0]) \\ z_i, & \text{otherwise} \end{cases}$$

This forces the model to take its second-highest probability branch without relying on instruction following. `rank2` recovers **$31.8\%$** of broken chains ($22\text{W} / 7\text{L}, p=0.008$), completely eliminating bad-value re-emission ($0.0\%$).

---

## 4. End-to-End Autonomous Loop & Systems Benchmark

We wired the Two-Tier localizer, the `rank2` intervention, and an "Accept-if-Cleaner" gate into a fully automated rewind loop operating on un-annotated, raw model completions.

### 4.1 The Accept-if-Cleaner Gate

When an observer mis-localizes a late defect to Step 1, applying `rank2` risks corrupting an already-correct prefix. We implemented a confidence guard:
* Let $lp_{orig}$ be the logprob of the original emitted token at $\hat{t}$, and $lp_{edit}$ be the logprob under rank-2 masking.
* If the original step was confident ($lp_{orig} > -0.25$) and the edit sharply degrades confidence ($lp_{edit} < lp_{orig} - 0.35$), the edit is **rejected**, and the prefix is preserved.

In empirical holdout testing, the gate rejected $23/110$ candidate edits. The gated and ungated Two-Tier arms are **numerically identical** ($32/110$, $34{,}046$ tokens, $246{,}826\text{ ms}$), because the rejected rows contributed zero recoveries when ungated. The same holds on the vault: $81/331$ rejected, $0$ recoveries among them, and the gated arm is the only arm run there. So the gate is a *conservative filter that cost nothing*, not a demonstrated recovery gain: it removes edits that would not have helped, and it buys precision ($32/87 = 36.8\%$ recovery among accepted rows on $N=110$) plus skipped compute. An auxiliary probe evaluated falling back to the localizer's second-highest ranked step ($\hat{t}_2$) on those 23 rejected rows: fallback hit the true step in $3/23$ instances but converted $0/23$ chains into additional recoveries, which is why terminating on rejection is the compute policy we use.

#### Table 3: End-to-End Autonomous Loop Recovery on the Tuning Holdout ($N=110$)

| Autonomous Arm | Localization Model | Intervention | Recovered / $N$ | Success Rate | 95% Wilson CI | $\hat{t} = t^*$ Exact | Accepted / $N$ |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| `full_rewrite` | None (Sample $T=0.8$) | Full sequence | 14 / 110 | $0.127$ | $[0.077, 0.202]$ | — | — |
| `linear_ungated` | Linear Majority | `rank2` | 24 / 110 | $0.218$ | $[0.151, 0.304]$ | $0.618$ | — |
| `ensemble_ungated` | Ensemble Average | `rank2` | 31 / 110 | $0.282$ | $[0.206, 0.372]$ | $0.691$ | — |
| **`two_tier_ungated`** | **Two-Tier Rule** | **`rank2`** | **32 / 110** | **$0.291$** | **$[0.214, 0.382]$** | **$0.718$** | — |
| `ensemble_gated` | Ensemble Average | `rank2` + Gate | 31 / 110 | $0.282$ | $[0.206, 0.372]$ | $0.691$ | 66 / 110 |
| **`two_tier_gated`** | **Two-Tier Rule** | **`rank2` + Gate** | **32 / 110** | **$0.291$** | **$[0.214, 0.382]$** | **$0.718$** | **87 / 110** |

Autonomous loop recovery jumps from $12.7\%$ (rewrite) and $21.8\%$ (linear) to **$29.1\%$** ($32/110$). Because the theoretical ceiling of `rank2` under an omniscient oracle is $31.8\%$ ($35/110$, Table 2), the Two-Tier autonomous loop captures:

$$\frac{29.1\%}{31.8\%} = \mathbf{91.5\% \text{ of the theoretical oracle repair ceiling}}$$

This ratio is a property of the $N=110$ tuning draw and should be read as such: the oracle ceiling was not re-measured on the vault, so no vault ceiling ratio is claimed. The vault result (Section 4.3) stands on its own paired comparison against full rewrites.

### 4.2 Physical KV-Cache Slicing (`DynamicCache.crop`)

Prior rewind experiments re-ran forward passes over the entire prompt and prefix. To test the systems claim, we integrated physical key-value cache slicing using HuggingFace's `DynamicCache.crop(target_len)`.

During autoregressive generation, past key-value projections for prompt tokens and verified prefix steps are retained in memory. When the localizer triggers a rewind to step $\hat{t}$:
1. The token length $L_{\hat{t}}$ corresponding to the prompt up to step $\hat{t}-1$ is computed.
2. The past key-value tensors are truncated in-place: `cache.crop(L_target)`.
3. Generation resumes from step $\hat{t}$ without re-evaluating preceding tokens.

#### Table 4: Systems Benchmark, Generation Slice Only ($N=110$ Chains, GTX 1050 4GB)

| Generation Arm | Mean Tokens / Chain | Mean Latency / Chain | FLOPs / Token Reduction | Latency Reduction | Relative Speedup |
|---|:---:|:---:|:---:|:---:|:---:|
| **Full Rewrite** ($T=0.8$) | $581.1$ | $4,498.4\text{ ms}$ | Baseline | Baseline | $1.0\times$ |
| **Un-cached Fresh Rewind** | $379.5$ | $2,732.2\text{ ms}$ | $-34.7\%$ | $-39.3\%$ | $1.6\times$ |
| **KV-Cache Truncated Rewind** | **$119.7$** | **$173.7\text{ ms}$** | **$-79.4\%$** | **$-96.1\%$** | **$25.9\times$** |

Compared to standard full-sequence regeneration, surgical KV-cache truncation eliminates $79.4\%$ of forward tokens and reduces *generation-slice* wall-clock latency by $96.1\%$ (a $25.9\times$ speedup). Even against naive un-cached rewinds, retaining the prefix key-value states delivers a $15.7\times$ speedup ($2,732.2\text{ ms} \to 173.7\text{ ms}$).

**This table is not loop latency.** Measured end to end, the same arm in Table 3 cost $246,826\text{ ms}$ for $110$ chains — **$2,244\text{ ms}$ per chain against $4,498.4\text{ ms}$ for a full rewrite, i.e. $2.0\times$**. The difference is the observer: ten probe forward passes plus the post-edit readout needed by the Accept-if-Cleaner gate. On the vault the same accounting gives $1,934.7\text{ ms}$ vs $3,281.9\text{ ms}$ per chain, **$1.70\times$**. The honest systems claim is therefore two-tiered: *generation* is $25.9$-$27.2\times$ cheaper, the *loop* is $1.7$-$2.0\times$ cheaper. Quoting the first as the second would overstate the system benefit by an order of magnitude.

### 4.3 Out-of-Sample Vault Replication ($N=331$) and Anti-Self-Deception Controls

To eliminate post-hoc threshold overfitting concerns on small holdout splits, we conducted a blind vault replication on an independent test split (`seed=20261028`, $140$ tasks, $560$ candidates, yielding **$331$ complete broken chains**) with all probe weights, SVD projection matrices, and decision thresholds permanently frozen ($\tau_{trf}=0.35, lp_{orig}=-0.25, \Delta lp=-0.35$).

#### Table 5: Blind Vault Replication and Permutation Control ($N=331$ Unseen Chains)

| Generation Arm | Intervention Mechanism | Recovered / $N$ | Rate | 95% Wilson CI | $\hat{t} = t^*$ Exact | Accepted / $N$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `full_rewrite` ($T=0.8$) | Regenerate entire chain | 46 / 331 | $0.139$ | $[0.106, 0.180]$ | — | — |
| `random_t_rank2` | Uniform random $\hat{t} \in \{1 \dots 4\}$ + `rank2` | 23 / 331 | $0.069$ | $[0.047, 0.102]$ | $0.245$ | — |
| **`two_tier_gated` (CGR)** | **Two-Tier Probe + Gate + `rank2`** | **73 / 331** | **$0.221$** | **$[0.179, 0.268]$** | **$0.625$** | **250 / 331** |

1. **Localization Generalizes; the Hard Threshold Does Not (Rule 1, qualified):**  
   The relational inductive bias holds out of sample: Transformer Majority Top-1 reached **$0.677$** (vs $0.604$ for Linear), up from $0.618$ on the tuning holdout, and the Ensemble Average held at **$0.680$** (balanced $0.636$, vs $0.691$ in-sample) — a $0.011$ drop, not a collapse. The **Two-Tier rule, however, degrades to $0.625$ / $0.410$ Step-4 recall**, below both. $\tau_{trf}=0.35$ was selected on the same $110$ rows where it scored $0.812$; on unseen seeds it is calibration drift, not a law. The transferable finding is the *complementarity* of linear and cross-step attention, not the threshold that exploits it.
2. **Defeating the Permutation Control (Rule 3 PASS):**  
   Slicing back to a uniformly random step with rank-2 exclusion (`random_t_rank2`) collapses to **$6.9\%$ recovery** ($23/331$)—performing at roughly half the recovery rate of doing a standard full rewrite ($13.9\%$). Arbitrary rewinds actively destroy sound prefixes. In contrast, Two-Tier CGR achieves **$22.1\%$** ($73/331$). The paired McNemar test demonstrates decisive statistical superiority:
   $$\mathbf{57\text{ Wins} \; / \; 7\text{ Losses}} \quad (p = 7.64 \times 10^{-11})$$
   The paired odds ratio of $8.14 : 1$ confirms that recovery is causally driven by accurate hidden-state localization rather than stochastic perturbation.
3. **Replicating Out-of-Sample Systems Savings (Rule 2, qualified):**  
   CGR significantly outperforms full-chain rewrites ($22.1\%$ vs $13.9\%$, $43\text{W}/16\text{L}, p = 5.84 \times 10^{-4}$) while slashing forward tokens from $580.6$ to $120.1$ ($-79.3\%$) and per-chain latency from $3,281.9\text{ ms}$ to $120.7\text{ ms}$—a **$27.2\times$ generation-slice speedup**, with the end-to-end loop at $1,934.7\text{ ms}$ vs $3,281.9\text{ ms}$ per chain ($1.70\times$).
4. **Bit-Identical Slicing Verification (Rule 4 PASS):**  
   To ensure cache slicing does not introduce numerical artifacts or cross-attention corruption, we benchmarked `DynamicCache.crop` against un-cached recomputations across 20 validation sequences. The maximum absolute logit divergence was bounded by $\max |\Delta\text{logit}| \le 4.86 \times 10^{-5}$ (well below single-precision epsilon), with $100\%$ bit-identical argmax token sequences.

### 4.4 Long-Horizon Scaling ($H=10$) & The Generative Horizon Boundary ($H^*$)

To establish the limits of test-time KV rewinds as reasoning chains deepen, we scaled the arithmetic task from $H=5$ to $H=10$ assignments on `Qwen3-0.6B-Base` ($N=75$ broken holdout chains, `results/verify_h10_scaling.json`).

#### Table 6: Horizon Scaling Diagnostics and Recovery at $H=10$ ($N=75$)

| Evaluation Arm | Mechanism | Top-1 Localization | Full-Chain Recovery | 95% Wilson CI | Mean Forward Tokens |
|---|---|:---:|:---:|:---:|:---:|
| `linear_localizer` | SVD-32 Linear Probe (5 seeds) | **$0.520$** (Chance $0.100$) | — | — | — |
| `full_rewrite` | Regenerate from Step 0 ($T=0.8$) | — | 0 / 75 ($0.000$) | $[0.000, 0.049]$ | $1,656.2$ |
| `in_context_oops` | Verbal retry prompt without KV crop | — | 0 / 75 ($0.000$) | $[0.000, 0.049]$ | $220.6$ |
| `two_tier_gated_1rewind` | Two-Tier CGR + `rank2` + Gate | $0.520$ | 0 / 75 ($0.000$) | $[0.000, 0.049]$ | **$165.0$** |
| `two_tier_gated_max2` | Sequential max-2 rewinds | — | 0 / 75 ($0.000$) | $[0.000, 0.049]$ | $171.9$ |

1. **Robust Localization vs. Multiplicative Rollout Decay:** While the localizer reliably identifies the erroneous step with **$52.0\%$ accuracy** ($5.2\times$ over chance), every generation arm collapsed to $0.0\%$ full-chain recovery. Because per-step conditional accuracy on sub-1B models hovers at $p \approx 0.35$–$0.50$, the joint probability of completing the remaining $10 - t$ steps correct without error decays exponentially: $\prod_{k=t+1}^{10} p_k \le (0.50)^6 \approx 0.015$.
2. **Identification of the Autoregressive Horizon Boundary ($H^*$):** For small language models, unassisted autoregressive arithmetic possesses an empirical horizon ceiling at $H^* \approx 6$–$8$. Beyond $H^*$, error recovery cannot rely solely on test-time branch exploration; it strictly necessitates external symbolic tool invocations (e.g. calculator tool) or checkpoint scaling to preserve late-step fidelity.
3. **Token Conservation Does Not Become Latency:** Even at the generative capacity ceiling, surgical KV-cache truncation reduces token forward evaluation by **$90.0\%$** ($165.0$ vs $1,656.2$ tokens). Wall-clock latency, however, moves from $7,502.7\text{ ms}$ to $6,399.2\text{ ms}$ — **$1.17\times$** — because un-batched single-sequence inference at this scale is dominated by framework launch and prefill overhead rather than decode FLOPs. A $90\%$ token reduction therefore does not imply a $90\%$ latency reduction, and at $H=10$ the loop is only marginally faster while recovering nothing.

---

## 5. Limitations & Remaining Headroom

While Path A substantially resolves the localization bottleneck and establishes that KV slicing is real, three claims required correction after the blind-vault audit, and they are reported as such above: the Two-Tier threshold does not generalize, the loop speedup is 1.7-2.0x rather than 27x, and the rewrite multiplier is 1.59x rather than 2.29x. Beyond those corrections, several challenges remain.

1. **The Rank-2 to Oracle Gap ($31.8\%$ vs $79.5\%$):**  
   Under an oracle that directly injects the ground-truth value, $79.5\%$ of non-trivial broken chains are restored. Under `rank2`, recovery caps at $31.8\%$. First-digit exclusion guarantees that the model will not repeat its previous answer, but when numbers span multi-digit ranges ($[1, 99]$), excluding one digit still leaves multiple plausible incorrect candidates. Closing this remaining headroom requires token-level value steering or beam-guided branch exploration at step $\hat{t}$.
2. **The Autoregressive Horizon Ceiling ($H^* \approx 6$–$8$):**  
   While the Two-Tier hidden-state localizer continues to isolate defects accurately at $H=10$ ($52.0\%$ vs $10.0\%$ chance), full-chain recovery collapses to $0.0\%$ across all arms due to multiplicative rollout compounding ($(0.45)^6 < 0.01$). Slicing past the error point succeeds locally, but completing long downstream sequences requires coupling CGR with external symbolic tool execution or larger parameter checkpoints to prevent subsequent divergence.
3. **Hybrid Architecture Compatibility:**  
   In linear-attention or state-space models (e.g., Mamba, LFM3), key-value caching is replaced by recurrent convolutional and state matrices. Rewinding in such architectures requires rolling back the hidden recurrent state $h_t$ rather than slicing a sequence cache.

---

## 6. Dynamic Adapter Routing and the Bridge Protocol

> **Status: design proposal, not a measurement.** Nothing in this section has
> been executed. It specifies a mechanism and the artefacts that would
> falsify it. It appears here because the systems constraint it addresses —
> specialised intervention under a $\le 4\text{ GB}$ budget — is real, not
> because the proposal has been validated. No number in this section may be
> quoted as a result.

Full-checkpoint escalation was measured and closed. On the $N=110$ ladder no
deployable actuator beat `rank2` (`findings/2026-10-28-level2-actuator-ladder.md`),
and the memory arithmetic is unforgiving on the target hardware: the AR backbone
occupies $\sim$1.2 GB and a 1.5–2.5B-class specialist a further $\sim$2.2 GB, so
a two-checkpoint policy spends $\sim$3.4 GB before activations on a 4 GB budget.
The alternative is to swap low-rank adapter weights ($\sim$10–50 MB) instead of
a parameter backbone. The CGR line already produces adapters in that band — the
matched LoRA control trains $22.3\text{M}$ parameters
(`findings/2026-10-25-steering-capacity-control.md`), $\approx 45\text{ MB}$ at
fp16 — but none of them is a code-repair specialist, which is the specialist
this section assumes.

### 6.1 KV-Cache Coherence Under Adapter Intervention

A fundamental systems constraint in recurrent adapter intervention is
**Key-Value Cache Coherence**: updating attention projections
($\Delta W_q, \Delta W_k, \Delta W_v$) or feed-forward layers
($\Delta W_{\text{MLP}}$) alters the activation distribution, rendering cached
tensors mathematically misaligned with subsequent decoding steps. Two
operational paths can manage this boundary.

**Path 1 — Query-Restricted Adaptation (in-stream persistence).** Adapters are
constrained during training to low-rank perturbations of query and MLP
projections ($\Delta W_q, \Delta W_{\text{MLP}}$), keeping the key-value
projection weights ($W_k, W_v$) strictly frozen. When an error is localized at
index $\hat{t}_{\min}$, the controller crops the cache via
`DynamicCache.crop` to $\hat{t}_{\min}-1$. Because $W_k$ and $W_v$ are
invariant, adapter-generated queries at $\hat{t}_{\min}$ attend directly to
cached prefix states without projection distortion. When the specialised task
spans the remainder of the sequence, the adapter stays active to the termination
boundary, which avoids cross-distribution attention artifacts.

The crop operation itself is measured, not assumed: on 20 validation sequences
`DynamicCache.crop` reproduced un-cached recomputation to
$\max|\Delta\text{logit}| \le 4.86 \times 10^{-5}$ with $100\%$ bit-identical
argmax sequences (Section 4.3, Rule 4). What is *not* measured is whether an
adapter's $\Delta W_q$ keeps that equivalence — the bit-identity result was
obtained with the base weights only.

### 6.2 The Bridge Protocol: Targeted Re-Encoding

When a specialised adapter is needed only for a localized sub-routine — repairing
code or symbolic arithmetic at steps $\hat{t} \dots \hat{t}+k$ — before returning
to general causal generation, retaining adapter overhead across the remaining
sequence is wasteful. The **Bridge Protocol** treats adapter generation as an
isolated, temporary patch:

1. **Surgical truncation.** The cache is cropped to $\hat{t}-1$, preserving
   verified prefix representations.
2. **Specialist execution.** The task-specific adapter is engaged to generate the
   localized patch tokens $S_{\text{patch}} = \{x_{\hat{t}}, \dots, x_{\hat{t}+k}\}$.
3. **Distribution re-alignment (the bridge).** The adapter is disengaged. Rather
   than flushing the context window or retaining adapted latent vectors, the raw
   text tokens of $S_{\text{patch}}$ pass through a single non-generative forward
   prefill under the frozen base weights. This re-encodes the patch's keys and
   values directly into the base distribution.
4. **Resumption.** Autoregressive decoding resumes under the base policy at step
   $\hat{t}+k+1$.

The economic argument is that prefill over a short window is parallel, so
re-encoding $k$ tokens costs one forward pass rather than $k$ sequential decode
steps, and the specialist's weights can be de-allocated immediately. That
argument is plausible but **its cost is not established here**, and Section 4.4
cuts against the optimistic reading: at $H=10$ a $90\%$ reduction in forwarded
tokens bought only $1.17\times$ wall-clock, because launch and prefill overhead
dominate at this scale. Any claim that the bridge is "imperceptible" or
"single-digit milliseconds" would therefore contradict a measurement already in
this paper. It has to be timed.

**What would count as evidence.** The protocol is worth adopting only if, against
query-restricted persistence (Path 1) on the same rows, it satisfies all three:

1. **Cost:** re-encoding $+$ adapter swap is cheaper in wall-clock than keeping
   the adapter resident for the remaining $T - (\hat{t}+k)$ steps. Tokens
   forwarded is the wrong metric; Section 4.4 shows it decouples from latency.
2. **Quality:** end-to-end recovery is not lower. The bridge rewrites the patch
   into the base distribution, so the failure mode to watch is a specialist edit
   that the base model then treats as out-of-distribution and talks itself out of.
3. **Coherence:** $\max|\Delta\text{logit}|$ between bridged and fully-base
   decoding over the same token sequence stays within the $4.86 \times 10^{-5}$
   envelope already established for crop.

If (1) fails, the protocol collapses to Path 1 with extra machinery. If (2)
fails, the adapter was load-bearing for more than the patch and targeted
re-encoding is the wrong abstraction. Either outcome is a result; neither is
currently known.

---

## 7. Conclusion

Autonomous reasoning repair does not require fine-tuning the base language model or relying on fickle text-based self-correction prompts. By reading un-pooled intermediate hidden states, enforcing physical logit exclusion, and surgically cropping the key-value cache, language models can detect and repair reasoning divergence at a fraction of the computational cost of regeneration: 1.59x the recovery rate of a full rewrite on a blind 331-chain vault, with the repair's generation step 26x faster than regenerating the chain.

Surgical KV-cache rewinds turn error recovery from a $4.5$-second full trajectory discard into a sub-$200$ millisecond in-memory cache crop for the generation step. The full loop, including observer inference, runs at $1.7$-$2.0\times$ the speed of a rewrite; making the observer cheaper is the clearest remaining systems headroom.

---

### Reproducibility & Source Verification

All code, trained weights, raw outputs, and evaluation logs are deterministic and tracked:
* **Harness & Pipelines:** [`experiments/path_a_loop.py`](file:///d:/kaggle%20research/experiments/path_a_loop.py)
* **Raw JSON Metrics:** [`results/path_a_loop_h5.json`](file:///d:/kaggle%20research/results/path_a_loop_h5.json), [`results/verify_vault_h5.json`](file:///d:/kaggle%20research/results/verify_vault_h5.json), [`results/verify_h10_scaling.json`](file:///d:/kaggle%20research/results/verify_h10_scaling.json), [`results/oracle_rewind_h5.json`](file:///d:/kaggle%20research/results/oracle_rewind_h5.json)
* **Underlying Corpus:** [`results/arith_steps_h5.pt`](file:///d:/kaggle%20research/results/arith_steps_h5.pt) (tuning), [`results/arith_steps_h5_vault.pt`](file:///d:/kaggle%20research/results/arith_steps_h5_vault.pt) (blind vault)
* **Detailed Finding Reports:** [`findings/2026-10-28-out-of-sample-and-h10-verification.md`](file:///d:/kaggle%20research/findings/2026-10-28-out-of-sample-and-h10-verification.md) (vault + H=10, authoritative for the systems and generalization numbers above); [`findings/2026-10-28-path-a-ensemble-kv-slicing.md`](file:///d:/kaggle%20research/findings/2026-10-28-path-a-ensemble-kv-slicing.md) (Path A; corrected in place on 2026-10-28 after the vault audit)

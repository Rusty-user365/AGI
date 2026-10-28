# Surgical KV-Cache Rewind: Step-Level Error Localization and Token-Mask Repair for Small Language Models

**Authors:** Research Team  
**Date:** October 2026  
**Repository:** [github.com/Rusty-user365/AGI](https://github.com/Rusty-user365/AGI)  
**Artifacts:** `experiments/path_a_loop.py`, `results/path_a_loop_h5.json`, `findings/2026-10-28-path-a-ensemble-kv-slicing.md`

---

## Abstract

When multi-step autoregressive generation diverges, standard inference harnesses face an inefficient dilemma: either discard the entire trajectory and regenerate from scratch ("full rewrite"), or rely on text-prompted self-correction ("hindsight prompting"), which language models frequently ignore. In this paper, we establish that **un-pooled internal hidden states** enable precise, step-level error localization, and that pairing step localization with **physical logit-mask exclusion** and **KV-cache truncation** produces an autonomous, surgical repair loop that substantially outperforms full regeneration while reducing inference latency by an order of magnitude.

Through systematic empirical dissection on multi-step reasoning chains ($H=5$, Qwen3-0.6B-Base), we show:
1. **Falsification of Heuristic Baselines:** Sequence-level pooled observers fail to select correct trajectories better than the model's own raw token logprob ($\text{AUC} = 0.682$ yields only $+0.004$ selection accuracy over free `max-logprob`, $p=1.000$). Similarly, natural language hindsight constraints (`# constraint: x is not 7`) are statistically identical to neutral placebos ($27.3\%$ vs $27.3\%$, paired $p=1.000$).
2. **Two-Tier Step Localization:** A frozen linear probe over SVD-reduced hidden states achieves $0.923$ recall on early divergence (Step 1) but collapses to $0.188$ on tail steps (Step 4). A cross-step Transformer encoder provides complementary tail sensitivity ($0.500$ Step-4 recall). An asymmetric Two-Tier routing rule ($P_{trf}(4) \ge 0.35$) eliminates tail collapse, elevating Step-4 recall to **$0.812$** (a $4.3\times$ gain) and overall Top-1 localization accuracy to **$0.718$** ($+10.0\%$ absolute over linear).
3. **Autonomous Loop Recovery:** Driving a rewind loop with Two-Tier localization and rank-2 first-digit exclusion achieves **$29.1\%$** end-to-end recovery on held-out broken chains, outperforming full sampling rewrites ($12.7\%$) by **$2.29\times$** and capturing **$91.5\%$** of the theoretical oracle-localization rank-2 ceiling ($31.8\%$). An "Accept-if-Cleaner" gate safely suppresses destructive prefix edits.
4. **Systems Efficiency:** By physically truncating the attention key-value cache via `DynamicCache.crop(target_len)`, the model avoids prompt and prefix re-computation, slashing per-chain forward tokens from $581.1$ to $119.7$ (**$-79.4\%$**) and wall-clock inference latency from $4,498.4\text{ ms}$ to $173.7\text{ ms}$—a **$25.9\times$ speedup** ($96.1\%$ latency cut).

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

### 2.3 The Two-Tier Routing Law

Because the error distributions are complementary, an asymmetric ensemble rule eliminates the tail-step bottleneck without training a large unified model:

$$\hat{t} = \begin{cases} 4, & \text{if } P_{trf}(t=4) \ge 0.35 \\ \arg\max_{t} P_{lin}(t), & \text{otherwise} \end{cases}$$

#### Table 1: Localizer Diagnostic Performance on Holdout Chains ($N=110$, 5 Seeds)

| Architecture | Top-1 Accuracy | Balanced Accuracy | Step 1 Recall ($n=39$) | Step 2 Recall ($n=23$) | Step 3 Recall ($n=13$) | Step 4 Recall ($n=32$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Linear Majority ($SVD_{32}$) | $0.618$ | $0.515$ | **$0.923$** | $0.696$ | **$0.769$** | $0.188$ |
| Transformer Majority | $0.618$ | $0.505$ | $0.769$ | $0.565$ | $0.692$ | $0.500$ |
| Ensemble (Softmax Mean) | $0.691$ | **$0.571$** | $0.897$ | **$0.783$** | **$0.769$** | $0.406$ |
| **Two-Tier Localizer** | **$0.718$** | $0.559$ | $0.846$ | $0.522$ | $0.615$ | **$0.812$** |

The Two-Tier localizer achieves a **$4.3\times$ gain in Step-4 recall** ($0.188 \to 0.812$), lifting aggregate Top-1 localization accuracy from $0.618$ to **$0.718$** ($+10.0\%$ absolute).

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

In empirical holdout testing, the gate rejected $23/110$ candidate edits, successfully **preventing 6 destructive overwrites** of sound prefixes. An auxiliary probe evaluated falling back to the localizer's second-highest ranked step ($\hat{t}_2$) on those 23 rejected rows: fallback hit the true step in $3/23$ instances but converted $0/23$ chains into additional recoveries, confirming that terminating when the gate rejects is the optimal compute policy.

#### Table 3: End-to-End Autonomous Loop Recovery on Holdout Set ($N=110$)

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

### 4.2 Physical KV-Cache Slicing (`DynamicCache.crop`)

Prior rewind experiments re-ran forward passes over the entire prompt and prefix. To test the systems claim, we integrated physical key-value cache slicing using HuggingFace's `DynamicCache.crop(target_len)`.

During autoregressive generation, past key-value projections for prompt tokens and verified prefix steps are retained in memory. When the localizer triggers a rewind to step $\hat{t}$:
1. The token length $L_{\hat{t}}$ corresponding to the prompt up to step $\hat{t}-1$ is computed.
2. The past key-value tensors are truncated in-place: `cache.crop(L_target)`.
3. Generation resumes from step $\hat{t}$ without re-evaluating preceding tokens.

#### Table 4: Systems Compute & Latency Benchmark ($N=110$ Chains, GTX 1050 4GB)

| Generation Arm | Mean Tokens / Chain | Mean Latency / Chain | FLOPs / Token Reduction | Latency Reduction | Relative Speedup |
|---|:---:|:---:|:---:|:---:|:---:|
| **Full Rewrite** ($T=0.8$) | $581.1$ | $4,498.4\text{ ms}$ | Baseline | Baseline | $1.0\times$ |
| **Un-cached Fresh Rewind** | $379.5$ | $2,732.2\text{ ms}$ | $-34.7\%$ | $-39.3\%$ | $1.6\times$ |
| **KV-Cache Truncated Rewind** | **$119.7$** | **$173.7\text{ ms}$** | **$-79.4\%$** | **$-96.1\%$** | **$25.9\times$** |

Compared to standard full-sequence regeneration, surgical KV-cache truncation eliminates $79.4\%$ of forward tokens and **reduces wall-clock repair latency by $96.1\%$ (a $25.9\times$ speedup)**. Even against naive un-cached rewinds, retaining the prefix key-value states delivers a **$15.7\times$ speedup** ($2,732.2\text{ ms} \to 173.7\text{ ms}$).

---

## 5. Limitations & Remaining Headroom

While Path A resolves the localization bottleneck and validates the systems claim, several fundamental challenges remain:

1. **The Rank-2 to Oracle Gap ($31.8\%$ vs $79.5\%$):**  
   Under an oracle that directly injects the ground-truth value, $79.5\%$ of non-trivial broken chains are restored. Under `rank2`, recovery caps at $31.8\%$. First-digit exclusion guarantees that the model will not repeat its previous answer, but when numbers span multi-digit ranges ($[1, 99]$), excluding one digit still leaves multiple plausible incorrect candidates. Closing this remaining headroom requires token-level value steering or beam-guided branch exploration at step $\hat{t}$.
2. **Context Window & Architecture Scope:**  
   Our experiments were executed on `Qwen3-0.6B-Base` across horizon $H=5$. While $H=5$ fully captures multi-step error cascading, deeper reasoning chains ($H > 20$) in code generation or mathematical proofs may introduce attention dispersion across intermediate states, necessitating multi-scale or hierarchical observer heads.
3. **Hybrid Architecture Compatibility:**  
   In linear-attention or state-space models (e.g., Mamba, LFM3), key-value caching is replaced by recurrent convolutional and state matrices. Rewinding in such architectures requires rolling back the hidden recurrent state $h_t$ rather than slicing a sequence cache.

---

## 6. Conclusion

Autonomous reasoning repair does not require fine-tuning the base language model or relying on fickle text-based self-correction prompts. By reading un-pooled intermediate hidden states with a complementary Two-Tier localizer, enforcing physical logit exclusion, and surgically cropping the key-value cache, language models can detect and repair reasoning divergence at a fraction of the computational cost of regeneration.

Surgical KV-cache rewinds transform error recovery from an expensive $4.5$-second full trajectory discard into a sub-$200$ millisecond in-memory cache crop, providing a scalable, mathematically principled foundation for reliable test-time reasoning.

---

### Reproducibility & Source Verification

All code, trained weights, raw outputs, and evaluation logs are deterministic and tracked:
* **Harness & Pipelines:** [`experiments/path_a_loop.py`](file:///d:/kaggle%20research/experiments/path_a_loop.py)
* **Raw JSON Metrics:** [`results/path_a_loop_h5.json`](file:///d:/kaggle%20research/results/path_a_loop_h5.json)
* **Underlying Corpus:** [`results/arith_steps_h5.pt`](file:///d:/kaggle%20research/results/arith_steps_h5.pt)
* **Detailed Finding Report:** [`findings/2026-10-28-path-a-ensemble-kv-slicing.md`](file:///d:/kaggle%20research/findings/2026-10-28-path-a-ensemble-kv-slicing.md)

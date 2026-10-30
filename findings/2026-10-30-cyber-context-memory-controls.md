# Cyber context and experience-memory controls

**Date:** 2026-10-30. Exploratory local development experiment on the pinned
`Qwen/Qwen3-0.6B` chat backbone and cyber adapter checkpoint 2500. Base and
adapter weights were frozen. This was not a live multi-step DAG or benchmark.

## Question and method

The prior factorial-to-cyber run produced a repeated Python function instead of
JSON. We reconstructed prior conversation tokens from three previously verified
coder outputs: factorial, clamp after verifier feedback, and even-sum. We paired
each with two related SQLite questions: selecting a user by name or by integer
ID. All conditions used the same cyber adapter, greedy decoding, seed 42,
160-token cap and fresh full-prompt cyber prefill. The six rows are combinations
of three contexts and two tasks, not six independently sampled task families.

Four conditions: cyber task alone; same task following the stored code; same
code context with a directly inserted format rule; and same code context with
that rule retrieved from the scoped MiniLM/Chroma experience memory. The rule:

`[SYSTEM MEMORY: For this task, output only a JSON object with "sql" and "params".]`

The tag is text in the user message, **not a privileged system instruction**.
The rule was written after the earlier factorial failure. The memory contained
additional distractor rules; the JSON rule was selected for both tasks. A poem
scope check excluded the cyber rule. This is an exploratory test and does not
show that the rule was automatically learned or generalizes to unseen tasks.

## Results

| Condition | SQL contract passed | Strict bare JSON and contract | Repeated Python |
| --- | ---: | ---: | ---: |
| Cyber alone | 6/6 | 0/6 | 0/6 |
| Prior code, no rule | 2/6 | 2/6 | 4/6 |
| Prior code + direct rule | 6/6 | 6/6 | 0/6 |
| Prior code + retrieved rule | 6/6 | 6/6 | 0/6 |

Cyber alone produced a fenced JSON block for each task: the SQL was correct
when the outer fence was stripped, but bare JSON failed. The standalone prompt
is identical across the three context rows for each task, so the 6/6 tally is
three repeats of **two distinct standalone prompts**, not six independent
baseline tasks. With prior code, the cyber adapter repeated the factorial and
even-sum functions for both SQL tasks; the clamp-feedback context produced a
valid result without the rule. The controlled contrast shows the added rule
helped on these development cases, but does not isolate why earlier code
changed the result: content, length, chat structure and other context features
were not separately ablated.

The direct and retrieved conditions had identical prompt token IDs and generated
identical text in all 6/6 rows. The retriever selected the intended rule, but
this comparison supplies no evidence that retrieval improves on directly
providing the same instruction. A follow-up must use multiple unrelated rules,
independent sessions and held-out examples, and test incorrect retrieval and
expiry as well as cases where no rule is applicable. A scope is supplied by the
caller; the current memory does not autonomously infer trustworthy scope.

Raw prompts (including special tokens), token IDs, response text, per-case
checks, actual retrieved metadata and timing are saved at
`results/cyber_context_memory_controls_v1.json`. Runner:
`experiments/cyber_context_memory_controls.py`. The SQL checker uses a restricted
in-memory SQLite authorizer, but is not a general-purpose security audit.
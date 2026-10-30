# Five-rule retrieval ablation after coder output

**Date:** 2026-10-30. Exploratory local development test using the chat-capable
Qwen/Qwen3-0.6B at revision c1899de289a04d12100db370d81485cdf75e47ca and cyber
adapter checkpoint 2500, CPU fp32. Weights were unchanged. Results and exact
prompts: `results/multirule_memory_ablation_v1.json`; runnable source:
`experiments/multirule_memory_ablation.py`.

Five handcrafted rules (SQL JSON, Python formatting, algebra, web UI and
phishing) were stored in the **same** supplied session scope. MiniLM embeddings
and Chroma ranked all five candidates; top-1 was injected. This tests retrieval
among competing candidates rather than separating the five rules by scope. The
retrieval query was the current user task, while the generative context contained
the saved actual coder-produced factorial function followed by that task. The
rule and scope were manually selected for this development test; the system did
not detect errors, infer scope, or learn a rule autonomously.

| Check | Result |
| --- | --- |
| SQL rule selected as the only top-1 result | 3/3 SQL prompts |
| Other domain control queries selected their intended rules | 4/4 |
| Poem outside any scope retrieved a rule | No |
| Same poem with the experiment scope supplied retrieved a rule | Yes: math rule |
| Cyber SQL behavior after coder output, no memory | 0/3 |
| Cyber SQL behavior after coder output, retrieved rule | 2/3 |

The out-of-scope poem returns nothing because the caller supplies no active
scope and no global rules exist. The same poem with the experiment scope gets a
math rule: the current top-1 retrieval has no similarity threshold and does not
autonomously abstain for an irrelevant query. This is a negative-control failure
for general no-match detection, despite successful scope isolation.

The no-memory responses repeated Python. With the retrieved rule, all three
outputs were bare JSON and the original name and ID SQL tasks passed their
behavior checks. On a reworded name query, the SQL selected `user_id` although
the test database has an `id` column, and therefore failed. The reworded prompt
said "user id" rather than explicitly spelling out the schema; this ambiguity
limits interpretation of the failure. Selecting the right format rule does not
ensure a functional query.

The two original tasks and rule were already examined in the prior controls;
the third is a small rewording, not a blind held-out task family. The four
non-SQL retrieval controls were hand-authored. The sample is too small to make
a statistical or generalization claim, and no direct-instruction arm was rerun
here. In the previous four-condition test, direct and retrieved arms had the
same input tokens and same outputs in all six cases; retrieval did not improve
on supplying the same rule directly. A meaningful next study should hold out
rules and task families, calibrate and evaluate abstention on negative examples,
and use paired baseline/direct/retrieved conditions with the same prompt.
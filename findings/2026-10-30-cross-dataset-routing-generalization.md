# Cross-dataset routing generalization: exploratory cyber experiments

Date: 2026-10-30

## Result

A frozen Qwen/Qwen3-0.6B base model and its step-2500 cyber LoRA were evaluated with a binary correctness router trained on hidden states from a separate SecEval sample. On CyberMetric-80 the router scored **57/80 (71.25%)**, versus **54/80 (67.5%)** for both the base and cyber adapter. A paired oracle choosing a correct answer when either arm was correct scored 61/80.

The router switched on 9 questions, including 3 useful switches (base wrong, cyber right) and **0 harmful switches** (base right, cyber wrong). This is a positive observation for this sample, not proof of safety or zero risk. It flagged 7 of the 26 base errors; 19 were not flagged. Accuracy gains, error detection and absence of harmful switches require independent replication.

| Router training -> test | Base | Cyber | Router | Oracle | Switches | Useful / harmful |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 SecEval -> 80 CyberMetric | 54 | 54 | 57 | 61 | 9 | 3 / 0 |
| 200 SecEval -> 40 disjoint SecEval | 26 | 23 | 26 | 29 | 12 | 1 / 1 |
| 80 CyberMetric -> 100 SecQA | 97 | 90 | 97 | 97 | 9 | 0 / 0 |

The CyberMetric-trained router retained the base's 97/100 result on SecQA despite nine switches; the cyber arm alone scored 90/100. This observation does not demonstrate that the router detects a near-ceiling benchmark or has a general safety mechanism.

## Method and limitations

The SecEval set used 200 training and 40 disjoint test single-answer questions selected with a fixed shuffle. The base answered 124/200 training questions correctly, leaving 76 labeled errors. SecEval source revision `205dab7b0888a06f4b53ca7d9c7093e1326683e1`; source SHA-256 `194c3a511104be422a678675a71d44315d0c0392b040745155aeebc321e692e0`. Exact and high character-similarity question overlaps with CyberMetric and SecQA were filtered; semantic overlap, paraphrases, and prior model-training exposure are not ruled out.

The heads use the final four generated answer-token hidden states, train-only standardization and an eight-dimensional SVD, followed by the existing LinearStep and TransformerStep components adapted to binary correctness. Their sigmoid scores are averaged and a fixed score below 0.50 triggers a switch. The scores have not been calibrated as correctness probabilities. The former arithmetic localizer was a step classifier; this is a binary adaptation, not the identical arithmetic task or its trained weights. These findings do not prove the heads learned a universal uncertainty signal, nor do they establish superiority over confidence baselines.

Generative models and LoRA weights were frozen. The pre-existing CyberMetric and SecQA results had been inspected in prior experiments; this is cross-dataset evaluation, not a pristine blind benchmark. Baseline and cyber answers use the same prompts, greedy decoding and strict answer parser; 2 cyber outputs on CyberMetric, 5 on held-out SecEval, and 7 on SecQA were invalid under the required format. A more permissive parser would change measured scores and must be reported separately rather than silently substituted.

The original routing evaluation selected between separately generated model outputs. It does not test multi-step DAG localization, safe commit verification, rollback and base-cache bridge replay, or end-to-end runtime savings. The fixed 40-question SecEval test is small, the +3/80 CyberMetric difference is exploratory, and no causal or general reliability claim follows from the results.

Raw data: `results/cyber_router_crossdataset_v1.json`; code: `experiments/cyber_router_crossdataset.py`; SecEval heads: `results/cyber_router_crossdataset_v1_heads.pt`; CyberMetric-trained heads: `results/cyber_router_cybermetric_to_secqa_heads.pt`. Sources: [SecEval](https://github.com/XuanwuAI/SecEval), [CyberMetric](https://github.com/cybermetric/CyberMetric), [SecQA](https://huggingface.co/datasets/zefang-liu/secqa).

## Live generation verification

The completed live run regenerated all 80 base answers, applied the frozen SecEval-trained heads, and started cyber generation for each of nine switches with the complete prompt and an empty cache. All 80 base outputs and routing decisions matched the offline records; all nine switched adapter outputs matched the saved standalone outputs. A forward hook on the underlying model verified the full prompt and cache length zero on each switch. Live routing reproduced 57/80, with 3 useful and 0 harmful switches. See `results/cyber_router_live_cybermetric80_v3.json` and `experiments/cyber_router_live.py`. Two earlier incomplete attempts stopped when a hook on the PEFT wrapper observed no calls; the successful run hooked the underlying model actually used for generation. This verifies a single-question router implementation, not the full CGR DAG/bridge.
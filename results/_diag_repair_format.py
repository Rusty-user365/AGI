"""Does the revised repair prompt produce a body, or another restart?

One problem, one greedy attempt, two hinted repairs. Prints the continuation
head so the format can be judged before spending GPU hours.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from experiments.arith_pilot import load as load_model
from experiments.code_bench import load as load_corpus, run_tests
from experiments.code_loop import (build_repair_prompt, feedback_of,
                                   localize_cut)

rows = load_corpus(Path("data/humaneval.jsonl.gz"))
model, tok = load_model(Path("checkpoints/hf/qwen3-0.6b-base"), torch.float32,
                        "cuda", "auto")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


@torch.no_grad()
def gen(prompt, sample, n=1):
    enc = tok([prompt], return_tensors="pt").to("cuda")
    enc.pop("token_type_ids", None)
    kw = (dict(do_sample=True, temperature=0.8, top_p=0.95,
               num_return_sequences=n) if sample else dict(do_sample=False))
    out = model.generate(**enc, max_new_tokens=192,
                         pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id, **kw)
    from experiments.code_bench import truncate
    return [truncate(t) for t in
            tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)]


for tid in ("HumanEval/0", "HumanEval/1"):
    r = next(x for x in rows if x["task_id"] == tid)
    plines = len(r["prompt"].splitlines())
    print("=" * 78)
    print(tid)
    body = gen(r["prompt"], False)[0]
    res = run_tests(r["prompt"] + body, r["test"], r["entry_point"])
    print(f"greedy pass={res['ok']} kind={res['failed_line']!r}")
    print("--- greedy body ---")
    print(body[:400])
    cut, kind = localize_cut(body.splitlines(), res, plines)
    rp = build_repair_prompt(r["prompt"], body.splitlines(), cut,
                             feedback_of(res))
    print(f"--- repair prompt (cut={cut}, kind={kind}) ---")
    print(rp[:200].replace("\n", " | "))
    print("... tail:", repr(rp[-80:]))
    for i, s in enumerate(gen(rp, True, 2)):
        print(f"--- repair {i} head ---")
        print(s[:300])
        rr = run_tests(r["prompt"] + s, r["test"], r["entry_point"])
        print(f"    pass={rr['ok']}")

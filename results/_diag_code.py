"""Diagnostic: what does the model actually emit after the FAILED comment?"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import torch

from experiments.arith_pilot import load as load_model
from experiments.code_bench import load as load_corpus, run_tests
from experiments.code_loop import (build_repair_prompt, feedback_of,
                                   localize_cut, top_level_start)

rows = load_corpus(Path("data/humaneval.jsonl.gz"))
r = rows[0]
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
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                            skip_special_tokens=True)


print("=== prompt ===")
print(r["prompt"])

body = gen(r["prompt"], False)[0]
print("=== greedy body ===")
print(body)
res = run_tests(r["prompt"] + body, r["test"], r["entry_point"])
print("pass:", res["ok"], "| failed_line:", res["failed_line"])

lines = body.splitlines()
cut, kind = localize_cut(lines, res, len(r["prompt"].splitlines()))
print(f"cut={cut} kind={kind} line={lines[cut]!r}")

for attempt in range(2):
    rp = build_repair_prompt(r["prompt"], lines, cut, feedback_of(res))
    print(f"=== repair prompt tail (attempt {attempt}) ===")
    print(rp[-420:])
    suffix = gen(rp, True)[0]
    print("=== suffix ===")
    print(suffix)
    body = "\n".join(lines[:cut]) + ("\n" if cut else "") + suffix
    res = run_tests(r["prompt"] + body, r["test"], r["entry_point"])
    print("pass:", res["ok"], "| failed_line:", res["failed_line"])
    lines = body.splitlines()
    cut, kind = localize_cut(lines, res, len(r["prompt"].splitlines()))
    print(f"next cut={cut} kind={kind}")

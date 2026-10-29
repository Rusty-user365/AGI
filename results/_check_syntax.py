"""Did the infill produce syntactically valid Python, and how far did it drift?"""
import ast
import json
import sys
from pathlib import Path

from experiments.code_bench import load

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/diffusion/mdlm_infill.jsonl")
corpus = {r["task_id"]: r for r in load(Path("data/humaneval.jsonl.gz"))}
recs = {json.loads(l)["task_id"]: json.loads(l) for l in
        path.read_text(encoding="utf-8").splitlines()
        if l.strip()}
ar = {json.loads(l)["task_id"]: json.loads(l) for l in
      Path("results/ar05b/rows.jsonl").read_text(encoding="utf-8").splitlines()
      if l.strip()}

for tid in sorted(recs):
    r = recs[tid]
    prompt = corpus[tid]["prompt"]
    body0 = ar[tid]["arms"]["greedy"]["body"]
    body1 = r["arm"]["body_final"]
    try:
        ast.parse(prompt + body1)
        ok = "valid"
    except SyntaxError as e:
        ok = "SYNTAX ERROR: %s (line %s)" % (e.msg, e.lineno)
    changed = sum(1 for a, b in zip(body0, body1) if a != b) + abs(len(body0) - len(body1))
    print("%-16s %-42s changed %d/%d chars" % (tid, ok, changed, len(body0)))

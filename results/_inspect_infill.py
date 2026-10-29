import json
from pathlib import Path

recs = [json.loads(l) for l in
        Path("results/diffusion/mdlm_infill.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()]
ar = {json.loads(l)["task_id"]: json.loads(l) for l in
      Path("results/ar05b/rows.jsonl").read_text(encoding="utf-8").splitlines()
      if l.strip()}

for r in sorted(recs, key=lambda x: x["task_id"]):
    tid = r["task_id"]
    g = ar[tid]["arms"]["greedy"]
    print("=" * 70)
    print(tid, "| kind", g["kind"], "| nfe", r["arm"]["nfe"],
          "| attempts", r["arm"]["attempts"], "| cuts", r["arm"]["cuts"])
    print("--- original failing body (first 6 lines) ---")
    print("\n".join(g["body"].splitlines()[:6]))
    print("--- diffusion final body (first 10 lines) ---")
    print("\n".join(r["arm"]["body_final"].splitlines()[:10]))

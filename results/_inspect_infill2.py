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
    a = r["arm"]
    print("=" * 72)
    print(tid, "| kind", g["kind"], "| nfe", a["nfe"], "| budget", a["budget"],
          "| attempts", a["attempts"])
    print("spans:", a["cuts"])
    body = g["body"]
    for k, (lo, hi) in enumerate(a["cuts"]):
        print("  span %d chars: %r" % (k, body[lo:hi]))
    print("--- final body ---")
    print("\n".join(a["body_final"].splitlines()[:12]))

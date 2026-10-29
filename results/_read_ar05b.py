import json
from pathlib import Path

d = json.loads(Path("results/ar05b/summary.json").read_text(encoding="utf-8"))
print("n", d["n"], "vram", round(d["vram_peak_mb"]))
for a, v in d["arms"].items():
    print("%-9s %3d/%d rate=%.3f ci=[%.3f,%.3f] tok=%d tok/solve=%.0f tests/solve=%.2f"
          % (a, v["ok"], v["n"], v["rate"], v["ci"][0], v["ci"][1],
             v["tokens"], v["tokens_per_solve"], v["tests_per_solve"]))
print("paired:")
for k, v in d["paired"].items():
    print("  %-14s %dW/%dL p=%.4g" % (k, v["wins"], v["losses"], v["p"]))
print("by kind:", d["by_greedy_failure_kind"])
print("loc:", d["localization"])

rows = [json.loads(l) for l in
        Path("results/ar05b/rows.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()]
print("\nfailing greedy problems:")
for r in sorted(rows, key=lambda x: x["task_id"]):
    g = r["arms"]["greedy"]
    if not g["ok"]:
        c = r["arms"]["cgr"]
        print("  %-16s kind=%-9s cgr=%d reps=%d gen_lens=%s cuts=%s"
              % (r["task_id"], g["kind"], c["ok"], c["repairs"],
                 c.get("gen_lens"), c.get("cuts")))

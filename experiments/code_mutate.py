"""How many code failures are one statement away from passing?

For each failing candidate, mutate one operator or integer literal at a time and
re-run the official test. If a single mutation passes, the failure is *local*:
a symbolic search can find the faulty step, and the mutant is a free fix. If no
single mutation passes, the failure is *structural* (the algorithm is wrong),
and backtracking to one statement cannot repair it.

This is the prevalence measurement that decides whether an executable localizer
is worth building for the code domain.

CPU only; no model is loaded. Uses the stored candidate bodies.

Usage:
    python experiments/code_mutate.py
    python experiments/code_mutate.py --arm greedy --max-mutants 60
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.code_bench import load as load_corpus, run_tests

OPS = re.compile(r"<=|>=|==|!=|//|\band\b|\bor\b|[<>+\-*%]")
FLIP = {
    "<": "<=", "<=": "<", ">": ">=", ">=": ">",
    "==": "!=", "!=": "==", "+": "-", "-": "+",
    "*": "//", "//": "*", "%": "//", "and": "or", "or": "and",
}
INT = re.compile(r"\b\d+\b")


def mutations(line: str):
    """Yield (column, old, new) for one-line single-token mutations."""
    for m in OPS.finditer(line):
        old = m.group(0)
        new = FLIP.get(old)
        if new is not None:
            yield m.start(), old, new
    for m in INT.finditer(line):
        n = int(m.group(0))
        for d in (1, -1):
            if n + d >= 0:
                yield m.start(), m.group(0), str(n + d)


def try_mutants(body: str, r: dict, max_mutants: int):
    """Return (fix, tried) where fix is (line_no, old, new) or None."""
    lines = body.splitlines()
    tried = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith('"""'):
            continue
        for col, old, new in mutations(line):
            if tried >= max_mutants:
                return None, tried
            mutant = "\n".join(lines[:i] + [line[:col] + new + line[col + len(old):]]
                               + lines[i + 1:])
            tried += 1
            res = run_tests(r["prompt"] + mutant, r["test"], r["entry_point"])
            if res["ok"]:
                return (i + 1, old, new), tried
    return None, tried


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="results/code_pilot_rows.jsonl")
    ap.add_argument("--data", default="data/humaneval.jsonl.gz")
    ap.add_argument("--arm", default="greedy", choices=["greedy", "cgr"])
    ap.add_argument("--max-mutants", type=int, default=60)
    ap.add_argument("--out", default="results/code_mutate.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    recs = [json.loads(l) for l in
            (root / args.rows).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    corpus = {r["task_id"]: r for r in load_corpus(root / args.data)}

    local, structural, detail = 0, 0, []
    for rec in sorted(recs, key=lambda x: x["task_id"]):
        arm = rec["arms"][args.arm]
        if arm["ok"]:
            continue
        body = arm.get("body0") if args.arm == "cgr" else arm.get("body")
        if not body:
            continue
        fix, tried = try_mutants(body, corpus[rec["task_id"]], args.max_mutants)
        if fix:
            local += 1
            detail.append({"task_id": rec["task_id"], "local": True,
                           "line": fix[0], "old": fix[1], "new": fix[2],
                           "tried": tried})
            print(f"  LOCAL  {rec['task_id']:<16} line {fix[0]}: "
                  f"{fix[1]!r} -> {fix[2]!r}  ({tried} mutants)")
        else:
            structural += 1
            detail.append({"task_id": rec["task_id"], "local": False,
                           "tried": tried})
            print(f"  STRUCT {rec['task_id']:<16} ({tried} mutants)")

    n = local + structural
    print(f"\narm={args.arm}  failures={n}  local={local}  structural={structural}")
    if n:
        print(f"single-mutation fixable: {local}/{n} = {local / n:.2f}")
    print("If local is high, a symbolic search is a free actuator on those "
          "failures; if near zero, the failures are structural and backtracking "
          "to one statement cannot repair them.")

    out = root / args.out
    out.write_text(json.dumps({"arm": args.arm, "failures": n, "local": local,
                               "structural": structural, "detail": detail},
                              indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

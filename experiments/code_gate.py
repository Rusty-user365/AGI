"""Futility gate for the staged code pilot, as pre-registered.

Reads the resumable rows file and answers one question: after Stage 1, should
the run continue to 164 or stop?

Pre-registered rule (findings/2026-10-29-code-pilot-preregistration.md):

    stop if cgr has zero paired wins against best_of, or
    stop if cgr trails best_of by 3 or more problems outright.

The gate is a single look, declared before the run. It is a decision aid, not a
result: the primary analysis is the paired McNemar on all rows collected.

Usage:
    python experiments/code_gate.py
    python experiments/code_gate.py --rows results/code_pilot_rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.level2_ladder import mcnemar
from experiments.localizer import wilson


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="results/code_pilot_rows.jsonl")
    ap.add_argument("--stage1", type=int, default=24)
    ap.add_argument("--out", default="results/code_pilot_gate.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = Path(args.rows)
    if not path.is_absolute():
        path = root / path
    recs = load(path)
    recs.sort(key=lambda x: x["task_id"])
    n = len(recs)

    g = [bool(x["arms"]["greedy"]["ok"]) for x in recs]
    b = [bool(x["arms"]["best_of"]["ok"]) for x in recs]
    c = [bool(x["arms"]["cgr"]["ok"]) for x in recs]
    mc = mcnemar(c, b)
    wins, losses = mc["wins"], mc["losses"]

    ck = sum(c)
    bk = sum(b)
    gk = sum(g)
    stop = (wins == 0) or (ck <= bk - 3)

    print(f"rows: {n}  (stage-1 target {args.stage1})")
    print(f"  greedy  {gk:>3}/{n}  [{wilson(gk, n)[0]:.3f},{wilson(gk, n)[1]:.3f}]")
    print(f"  best_of {bk:>3}/{n}  [{wilson(bk, n)[0]:.3f},{wilson(bk, n)[1]:.3f}]")
    print(f"  cgr     {ck:>3}/{n}  [{wilson(ck, n)[0]:.3f},{wilson(ck, n)[1]:.3f}]")
    print(f"  cgr vs best_of: {wins}W/{losses}L  p={mc['p']:.3g}")
    print(f"  cgr - best_of = {ck - bk:+d}")

    tok_b = sum(x["arms"]["best_of"]["tokens"] for x in recs)
    tok_c = sum(x["arms"]["cgr"]["tokens"] for x in recs)
    tst_b = sum(x["arms"]["best_of"]["tests"] for x in recs)
    tst_c = sum(x["arms"]["cgr"]["tests"] for x in recs)
    print(f"  tokens best={tok_b} cgr={tok_c} ({tok_c / max(tok_b, 1):.3f}x)")
    print(f"  tests  best={tst_b} cgr={tst_c} ({tst_c / max(tst_b, 1):.2f}x)")

    if n < args.stage1:
        verdict = "incomplete"
        print(f"\nVERDICT: incomplete ({n}/{args.stage1}) -- keep running")
    elif stop:
        verdict = "stop_for_futility"
        print("\nVERDICT: STOP. Futility gate fired "
              f"(wins={wins}, cgr-best={ck - bk:+d}).")
        print("Report as: no detectable signal at N=%d; stopped for futility."
              % n)
    else:
        verdict = "continue_to_164"
        print(f"\nVERDICT: CONTINUE to 164 (wins={wins}, cgr-best={ck - bk:+d})")

    payload = {
        "rows": n, "stage1": args.stage1, "verdict": verdict,
        "arms": {"greedy": gk, "best_of": bk, "cgr": ck},
        "paired_cgr_vs_best": mc,
        "tokens": {"best_of": tok_b, "cgr": tok_c},
        "tests": {"best_of": tst_b, "cgr": tst_c},
        "rule": "stop if wins==0 or cgr <= best_of - 3",
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0 if verdict != "stop_for_futility" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Decompose the code-pilot token advantage: early stopping or repair?

`best_of_4` always pays for four samples, even when the first one passes. `cgr`
stops as soon as a test passes, so on problems greedy already solves it spends
one sample. That makes the raw token totals incomparable: the loop can look
twice as efficient without the repair doing anything.

This script builds the fair baseline offline from the stored samples:

    sequential  test the four stored samples in order, stop at the first pass

Same solved set as `best_of_4`, adaptive cost. If `cgr` does not beat
`sequential` on tokens at equal accuracy, the token advantage was early
stopping, not repair.

No model is loaded and no GPU is used. Tests are re-run on stored bodies, so
token counts for the sequential arm are re-tokenized approximations (a few
tokens per body); `cgr` and `best_of` keep their exact recorded counts.

Usage:
    python experiments/code_decompose.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from experiments.code_bench import load as load_corpus, run_tests
from experiments.level2_ladder import mcnemar
from experiments.localizer import wilson


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="results/code_pilot_rows.jsonl")
    ap.add_argument("--data", default="data/humaneval.jsonl.gz")
    ap.add_argument("--model", default="checkpoints/hf/qwen3-0.6b-base")
    ap.add_argument("--out", default="results/code_pilot_decompose.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    rows_path = root / args.rows
    recs = [json.loads(l) for l in
            rows_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs.sort(key=lambda x: x["task_id"])
    corpus = {r["task_id"]: r for r in load_corpus(root / args.data)}
    tok = AutoTokenizer.from_pretrained(str(root / args.model))

    seq_ok, seq_tokens, seq_tests = [], [], []
    detail = []
    for rec in recs:
        r = corpus[rec["task_id"]]
        bodies = rec["arms"]["best_of"]["bodies"]
        L = len(tok(r["prompt"])["input_ids"])
        spent, tests, ok = 0, 0, False
        for body in bodies:
            spent += L + len(tok(body)["input_ids"])
            tests += 1
            res = run_tests(r["prompt"] + body, r["test"], r["entry_point"])
            if res["ok"]:
                ok = True
                break
        seq_ok.append(ok)
        seq_tokens.append(spent)
        seq_tests.append(tests)
        detail.append({"task_id": rec["task_id"], "ok": ok, "tokens": spent,
                       "tests": tests,
                       "greedy_ok": rec["arms"]["greedy"]["ok"],
                       "cgr_ok": rec["arms"]["cgr"]["ok"],
                       "cgr_tokens": rec["arms"]["cgr"]["tokens"],
                       "cgr_tests": rec["arms"]["cgr"]["tests"]})

    n = len(recs)
    c_ok = [bool(x["arms"]["cgr"]["ok"]) for x in recs]
    b_ok = [bool(x["arms"]["best_of"]["ok"]) for x in recs]
    c_tok = sum(x["arms"]["cgr"]["tokens"] for x in recs)
    b_tok = sum(x["arms"]["best_of"]["tokens"] for x in recs)
    s_tok = sum(seq_tokens)
    c_tst = sum(x["arms"]["cgr"]["tests"] for x in recs)
    s_tst = sum(seq_tests)

    # Greedy-then-sequential: the harness ablation. Same greedy first attempt as
    # cgr, same early stop, but unhinted whole-body samples instead of hinted
    # localized repairs. Whatever cgr gains over this is the treatment itself.
    gs_ok, gs_tok, gs_tst = [], [], []
    for rec, s_ok, s_t, s_tests in zip(recs, seq_ok, seq_tokens, seq_tests):
        g = rec["arms"]["greedy"]
        if g["ok"]:
            gs_ok.append(True)
            gs_tok.append(g["tokens"])
            gs_tst.append(g["tests"])
        else:
            gs_ok.append(s_ok)
            gs_tok.append(g["tokens"] + s_t)
            gs_tst.append(g["tests"] + s_tests)

    print(f"rows: {n}")
    print(f"{'arm':<12} {'ok':>7} {'tokens':>9} {'tok/solve':>10} {'tests':>7}")
    for name, ok, tokn, tst in (
        ("best_of_4", sum(b_ok), b_tok, sum(x["arms"]["best_of"]["tests"] for x in recs)),
        ("sequential", sum(seq_ok), s_tok, s_tst),
        ("greedy_seq", sum(gs_ok), sum(gs_tok), sum(gs_tst)),
        ("cgr", sum(c_ok), c_tok, c_tst),
    ):
        print(f"{name:<12} {ok:>3}/{n:<3} {tokn:>9} {tokn / max(ok, 1):>10.0f} {tst:>7}")

    mc = mcnemar(c_ok, seq_ok)
    mg = mcnemar(c_ok, gs_ok)
    print(f"\ncgr vs sequential: {mc['wins']}W/{mc['losses']}L p={mc['p']:.3g}")
    print(f"cgr vs greedy_seq: {mg['wins']}W/{mg['losses']}L p={mg['p']:.3g}")
    print(f"cgr tokens / sequential tokens = {c_tok / max(s_tok, 1):.3f}x")
    print(f"cgr tokens / greedy_seq tokens = {c_tok / max(sum(gs_tok), 1):.3f}x")
    print(f"cgr tokens / best_of tokens    = {c_tok / max(b_tok, 1):.3f}x")
    print(f"sequential tokens / best_of    = {s_tok / max(b_tok, 1):.3f}x")

    # Split by whether greedy already solved it: that is where early stopping
    # pays and where the repair loop never runs.
    easy = [d for d in detail if d["greedy_ok"]]
    hard = [d for d in detail if not d["greedy_ok"]]
    for name, grp in (("greedy-solved", easy), ("greedy-failed", hard)):
        if not grp:
            continue
        ct = sum(d["cgr_tokens"] for d in grp)
        st = sum(d["tokens"] for d in grp)
        print(f"\n{name} (n={len(grp)}): cgr {sum(d['cgr_ok'] for d in grp)} solved, "
              f"{ct} tok | seq {sum(d['ok'] for d in grp)} solved, {st} tok "
              f"| ratio {ct / max(st, 1):.2f}x")

    print("\nper-problem (greedy_failed):")
    for d in hard:
        print(f"  {d['task_id']:<16} cgr={int(d['cgr_ok'])}/{d['cgr_tokens']:>5}t "
              f"seq={int(d['ok'])}/{d['tokens']:>5}t "
              f"best_tok={d['cgr_tokens']:>5}")

    payload = {
        "rows": n,
        "arms": {
            "best_of_4": {"ok": sum(b_ok), "tokens": b_tok,
                          "tests": sum(x["arms"]["best_of"]["tests"] for x in recs)},
            "sequential": {"ok": sum(seq_ok), "tokens": s_tok, "tests": s_tst},
            "greedy_seq": {"ok": sum(gs_ok), "tokens": sum(gs_tok),
                           "tests": sum(gs_tst)},
            "cgr": {"ok": sum(c_ok), "tokens": c_tok, "tests": c_tst},
        },
        "cgr_vs_sequential": mc,
        "cgr_vs_greedy_seq": mg,
        "ratio_cgr_seq": c_tok / max(s_tok, 1),
        "ratio_cgr_greedy_seq": c_tok / max(sum(gs_tok), 1),
        "ratio_cgr_best": c_tok / max(b_tok, 1),
        "note": "sequential tokens are re-tokenized approximations; cgr and "
                "best_of use exact recorded counts",
        "detail": detail,
    }
    out = root / args.out
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

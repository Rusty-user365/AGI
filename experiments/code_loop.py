"""Code loop: greedy vs best-of-k vs localized repair, at matched tokens.

Pre-registration: `findings/2026-10-29-code-pilot-preregistration.md`.

The claim under test:

    Given a failing test, re-proposing only the localized region with the
    failing assertion in context solves more problems per generated token than
    redrawing whole functions.

Arms, per problem:

    greedy      one body, tests decide
    best_of_4   four full bodies, tests select the first that passes
    cgr_4       one body, test, localize, re-prompt from the faulty statement
                with the failing assertion, re-test; repairs continue until the
                problem's best_of_4 token budget is spent (max 8 repairs)

Tokens are counted, not estimated: prompt tokens are paid on every call,
including re-prefills, plus every generated token. That makes the comparison
conservative for the repair arm, which pays the prompt again on each attempt.

Localization is pre-registered and deliberately simple:

    exception   last traceback frame naming prog.py -> enclosing top-level line
    assert      cut = 0: rewrite the whole body with the failing assertion
    other       cut = 0: the body is malformed or untraceable, do not carry it

An exception localizes itself; an assert does not. The two are reported
separately, because pooling them would hide the only case where a learned
localizer has a job. The revision from "last `return`" to cut = 0 is recorded in
the pre-registration with the diagnostic that forced it.

Usage:
    python experiments/code_loop.py --limit 2
    python experiments/code_loop.py
    python experiments/code_loop.py --resume
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments._device import (manual_seed_all, peak_backend, peak_mb,
                                 reset_peak, resolve as _resolve_device)
from experiments.arith_pilot import load as load_model
from experiments.code_bench import (failure_kind, load as load_corpus,
                                    run_tests, truncate)
from experiments.level2_ladder import mcnemar
from experiments.localizer import wilson

FRAME = re.compile(r'File "[^"]*prog\.py", line (\d+)')


@torch.no_grad()
def generate(model, tok, prompt, args, sample: bool, n: int = 1,
             max_new: int | None = None):
    """Return (texts, tokens). Tokens = n * prompt + every generated token."""
    enc = tok([prompt], return_tensors="pt").to(args.device)
    enc.pop("token_type_ids", None)
    L = int(enc["input_ids"].shape[1])
    kw = (dict(do_sample=True, temperature=args.temperature, top_p=args.top_p,
               num_return_sequences=n) if sample else dict(do_sample=False))
    out = model.generate(**enc, max_new_tokens=args.max_new if max_new is None
                         else max_new,
                         pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id, **kw)
    texts = [truncate(t) for t in
             tok.batch_decode(out[:, L:], skip_special_tokens=True)]
    # Count what was actually generated, before truncation. Counting the
    # re-tokenized, truncated text would undercount a completion that ran into a
    # stop sequence, which would let the repair arm spend more than its budget
    # while appearing matched. Every row of a batch is generated to the same
    # length, so that length is the compute actually paid.
    gen_len = int(out.shape[1]) - L
    tokens = n * L + n * gen_len
    # gen_len is returned because a forward pass is the unit a diffusion arm can
    # be matched against: L KV-cached decoding steps for L tokens, S denoising
    # steps for S passes. Without the per-attempt length the match would have to
    # be approximated from the arm total, which is not per-problem.
    return texts, tokens, gen_len


def top_level_start(lines: list[str], idx: int) -> int:
    """Walk back to the enclosing 4-space statement, so a repair keeps syntax."""
    for i in range(idx, -1, -1):
        ln = lines[i]
        if ln.startswith("    ") and not ln.startswith("        "):
            return i
    return 0


def localize_cut(body_lines: list[str], res: dict, prompt_lines: int):
    """Return (cut_index, kind). See the module docstring for the rule."""
    stderr = res.get("stderr") or ""
    kind = failure_kind(stderr)
    if kind != "exception":
        # Cutting at the last `return` can only rewrite the final statement, so
        # a bug in a loop above it is unreachable. Rewrite the whole body and
        # let the failing assertion (or the syntax error) do the localizing.
        # A malformed body must not be carried across a cut either.
        return 0, kind
    idx = None
    frames = FRAME.findall(stderr)
    if frames:
        line = int(frames[-1])
        # The test body lives in the same file, so a frame can name a line
        # outside the candidate. If the deepest frame is in the test, the
        # exception was raised by the harness, not by the body: rewrite the body
        # rather than cutting at a clamped, meaningless index.
        if prompt_lines < line <= prompt_lines + len(body_lines):
            idx = line - prompt_lines - 1
        else:
            return 0, kind
    if idx is None:
        for i, ln in enumerate(body_lines):
            if ln.strip().startswith("return"):
                idx = i
        if idx is None:
            for i in range(len(body_lines) - 1, -1, -1):
                if body_lines[i].strip():
                    idx = i
                    break
    if idx is None:
        return 0, kind
    idx = max(0, min(idx, len(body_lines) - 1))
    return top_level_start(body_lines, idx), kind


def build_repair_prompt(prompt: str, body_lines: list[str], cut: int,
                        feedback: str | None) -> str:
    """Hint above the prompt, then the shared prefix; the body continues clean.

    The hint must not sit *inside* the body. With an in-body `# FAILED:` comment
    the base model treated the comment as the end of the function and restarted
    a new `def`, which truncation then cut -- the repair arm was losing to a
    formatting artifact rather than to the architecture. A comment block above
    the prompt keeps the continuation point exactly where a normal completion
    starts, so the model writes a body.
    """
    fb = (feedback or "test failed").strip().splitlines()
    fb = fb[-1] if fb else "test failed"
    head = ("# The function below failed a test. Rewrite the body so it passes.\n"
            f"# Failing test: {fb[:300]}\n")
    text = head + prompt.rstrip("\n")
    if cut > 0:
        text += "\n" + "\n".join(body_lines[:cut])
    return text + "\n"


def feedback_of(res: dict) -> str:
    if res.get("failed_line"):
        return res["failed_line"]
    err = (res.get("stderr") or "").strip().splitlines()
    return err[-1] if err else "test failed"


def arm_greedy(model, tok, r, args):
    texts, tokens, gen_len = generate(model, tok, r["prompt"], args, False)
    res = run_tests(r["prompt"] + texts[0], r["test"], r["entry_point"],
                    args.timeout)
    return {"ok": res["ok"], "tokens": tokens, "tests": 1, "gen_len": gen_len,
            "kind": None if res["ok"] else failure_kind(res["stderr"]),
            "body": texts[0]}


def arm_best_of(model, tok, r, args, k: int):
    texts, tokens, gen_len = generate(model, tok, r["prompt"], args, True, k)
    tests = 0
    kind = None
    for t in texts:
        tests += 1
        res = run_tests(r["prompt"] + t, r["test"], r["entry_point"],
                        args.timeout)
        if res["ok"]:
            return {"ok": True, "tokens": tokens, "tests": tests,
                    "gen_len": gen_len, "kind": None, "bodies": texts}
        kind = failure_kind(res["stderr"])
    return {"ok": False, "tokens": tokens, "tests": tests,
            "gen_len": gen_len, "kind": kind, "bodies": texts}


def arm_cgr(model, tok, r, args, budget: int):
    prompt_lines = len(r["prompt"].splitlines())
    texts, tokens, gen_len = generate(model, tok, r["prompt"], args, False)
    gen_lens = [gen_len]
    body = texts[0]
    res = run_tests(r["prompt"] + body, r["test"], r["entry_point"],
                    args.timeout)
    tests = 1
    initial_kind = None if res["ok"] else failure_kind(res["stderr"])
    first_repair_ok = None
    repairs = 0
    cuts = []
    while (not res["ok"] and repairs < args.max_repairs):
        body_lines = body.splitlines()
        cut, _kind = localize_cut(body_lines, res, prompt_lines)
        rp = build_repair_prompt(r["prompt"], body_lines, cut,
                                 feedback_of(res))
        rp_len = len(tok(rp)["input_ids"])
        remaining = budget - tokens - rp_len
        if remaining < 8:
            break                       # no budget left for a real repair
        texts2, tok2, gen2 = generate(model, tok, rp, args, True, 1,
                                       max_new=min(args.max_new, remaining))
        tokens += tok2
        gen_lens.append(gen2)
        body = "\n".join(body_lines[:cut]) + ("\n" if cut else "") + texts2[0]
        res = run_tests(r["prompt"] + body, r["test"], r["entry_point"],
                        args.timeout)
        tests += 1
        repairs += 1
        cuts.append(cut)
        if repairs == 1:
            first_repair_ok = res["ok"]
    return {"ok": res["ok"], "tokens": tokens, "tests": tests,
            "kind": initial_kind, "repairs": repairs,
            "first_repair_ok": first_repair_ok, "cuts": cuts,
            "gen_lens": gen_lens, "budget": budget, "body0": texts[0],
            "body_final": body}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/humaneval.jsonl.gz")
    ap.add_argument("--model", default="checkpoints/hf/qwen3-0.6b-base")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                    help="fp16 is required for a 1.5B model on a 4 GB card")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--best-k", type=int, default=4)
    ap.add_argument("--max-repairs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rows-out", default="results/code_pilot_rows.jsonl")
    ap.add_argument("--out", default="results/code_pilot_h5.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.device = str(_resolve_device(args.device))

    root = Path(__file__).resolve().parents[1]
    data = Path(args.data)
    if not data.is_absolute():
        data = root / data
    rows_out = Path(args.rows_out)
    if not rows_out.is_absolute():
        rows_out = root / rows_out

    rows = load_corpus(data)
    if args.limit:
        rows = rows[:args.limit]

    done = set()
    if args.resume and rows_out.exists():
        for line in rows_out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["task_id"])
        print(f"resume: {len(done)} problems already done")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    model, tok = load_model(model_path, {"fp32": torch.float32,
                                         "fp16": torch.float16,
                                         "bf16": torch.bfloat16}[args.dtype],
                            args.device, "auto")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    reset_peak(args.device)

    t0 = time.time()
    with rows_out.open("a", encoding="utf-8") as fh:
        for j, r in enumerate(rows):
            if r["task_id"] in done:
                continue
            # Per-problem seed: the run is reproducible, and a resumed run
            # regenerates a problem exactly as an uninterrupted run would.
            seed = args.seed + j
            torch.manual_seed(seed)
            manual_seed_all(seed)
            g = arm_greedy(model, tok, r, args)
            b = arm_best_of(model, tok, r, args, args.best_k)
            c = arm_cgr(model, tok, r, args, b["tokens"])
            rec = {"task_id": r["task_id"], "entry_point": r["entry_point"],
                   "arms": {"greedy": g, "best_of": b, "cgr": c}}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  {j + 1:>3}/{len(rows)} {r['task_id']:<16} "
                  f"g={int(g['ok'])} b={int(b['ok'])} c={int(c['ok'])} "
                  f"tok g{b['tokens']} c{c['tokens']} "
                  f"rep={c['repairs']} kind={g['kind']} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    recs = [json.loads(l) for l in
            rows_out.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs.sort(key=lambda x: x["task_id"])
    n = len(recs)
    names = ["greedy", "best_of", "cgr"]

    summary = {}
    for a in names:
        ok = sum(int(x["arms"][a]["ok"]) for x in recs)
        tokens = sum(x["arms"][a]["tokens"] for x in recs)
        tests = sum(x["arms"][a]["tests"] for x in recs)
        lo, hi = wilson(ok, n)
        summary[a] = {"ok": ok, "n": n, "rate": ok / max(n, 1), "ci": [lo, hi],
                      "tokens": tokens, "tokens_per_solve": tokens / max(ok, 1),
                      "tests": tests, "tests_per_solve": tests / max(ok, 1)}

    paired = {
        "cgr_vs_best": mcnemar([x["arms"]["cgr"]["ok"] for x in recs],
                               [x["arms"]["best_of"]["ok"] for x in recs]),
        "best_vs_greedy": mcnemar([x["arms"]["best_of"]["ok"] for x in recs],
                                  [x["arms"]["greedy"]["ok"] for x in recs]),
        "cgr_vs_greedy": mcnemar([x["arms"]["cgr"]["ok"] for x in recs],
                                 [x["arms"]["greedy"]["ok"] for x in recs]),
    }

    by_kind = {}
    for kind in ("assert", "exception", "other", None):
        grp = [x for x in recs if x["arms"]["greedy"]["kind"] == kind]
        if not grp:
            continue
        by_kind[str(kind)] = {
            "n": len(grp),
            **{a: sum(int(x["arms"][a]["ok"]) for x in grp) for a in names},
        }

    repaired = [x for x in recs if x["arms"]["cgr"]["repairs"] > 0]
    loc = {
        "repaired": len(repaired),
        "first_repair_ok": sum(int(bool(x["arms"]["cgr"]["first_repair_ok"]))
                               for x in repaired),
        "mean_repairs": (sum(x["arms"]["cgr"]["repairs"] for x in recs) / max(n, 1)),
    }

    payload = {
        "config": vars(args),
        "n": n,
        "arms": summary,
        "paired": paired,
        "by_greedy_failure_kind": by_kind,
        "localization": loc,
        "vram_peak_mb": peak_mb(args.device),
        "peak_backend": peak_backend(args.device),
    }

    print("\n" + "=" * 78)
    print(f"CODE LOOP (N={n}, best_k={args.best_k})")
    print("=" * 78)
    print(f"{'arm':<10} {'ok':>7} {'rate':>6} {'95% CI':>16} "
          f"{'tokens':>9} {'tok/solve':>10} {'tests/solve':>11}")
    for a in names:
        s = summary[a]
        print(f"{a:<10} {s['ok']:>3}/{n:<3} {s['rate']:>6.3f} "
              f"[{s['ci'][0]:.3f},{s['ci'][1]:.3f}] {s['tokens']:>9} "
              f"{s['tokens_per_solve']:>10.0f} {s['tests_per_solve']:>11.2f}")
    for k_, v in paired.items():
        print(f"  {k_}: {v['wins']}W/{v['losses']}L p={v['p']:.2e}")
    for k_, v in by_kind.items():
        print(f"  kind={k_}: n={v['n']} greedy={v['greedy']} "
              f"best={v['best_of']} cgr={v['cgr']}")
    print(f"  localization: {loc['first_repair_ok']}/{loc['repaired']} first "
          f"repairs passed, mean repairs {loc['mean_repairs']:.2f}")
    print(f"peak VRAM {payload['vram_peak_mb']:.0f} MB")

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

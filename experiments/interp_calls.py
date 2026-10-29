"""Call-allocation curve: what is "we know what to put where" worth?

The actuator ladder closed the subtractive branch: no ban beats rank-2 at
32/110, while injecting the correct value at the localized step reaches 70/110.
That makes the interpreter the weapon and the localizer the aiming system. This
script prices the aiming system.

The interpreter here is **deployable, not an oracle**: values are computed from
the definition block in the prompt by `eval_lines`, i.e. exactly what a Python
subprocess would return. The generator's stored `truth` is used only as an
assertion that the evaluator is correct; a mismatch aborts the run.

Arms, on the same 110 holdout rows:

    k0_rank2         current CGR: first-digit ban at t_hat, greedy finish (32/110)
    k1_loc           interpreter at t_hat
    k2_loc           interpreter at t_hat + the runner-up step
    k3_loc           interpreter at the top three localizer steps
    k1_rand          interpreter at 1 uniformly random step
    k2_rand          interpreter at 2 uniformly random steps
    k3_rand          interpreter at 3 uniformly random steps
    k1_oracleloc     interpreter at the true error step (perfect binding)
    k5_all           interpreter at every step (must be 110/110; sanity check)

Localized ranking: two-tier argmax is rank 1 (so k1_loc is anchored to the
committed loop's 70/110), then the remaining steps in descending ensemble
probability (mean of linear and transformer softmax).

The claim this can support: localized calls reach a given accuracy with fewer
interpreter calls than random calls. It cannot support "more puzzles solved",
because k5_all solves everything by construction.

The localizer is trained on `--train-data` and evaluated on `--data`. They are
the same file for the h5 holdout run; for the frozen vault, `--train-data` stays
at the h5 file so the localizer weights are frozen and the vault is genuinely
out of sample. The reproduction anchors (32/110, 70/110, 94/110, 0.718) are
claims about the h5 holdout and are enforced only when `--data` is that file.

Usage:
    python experiments/interp_calls.py --limit 6
    python experiments/interp_calls.py
    python experiments/interp_calls.py --data results/arith_steps_h5_vault.pt \
        --train-data results/arith_steps_h5.pt \
        --out results/interp_calls_vault.json \
        --rows-out results/interp_calls_vault_rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.arith_pilot import load
from experiments.arith_steps import build_prompt, parse_value
from experiments.level2_ladder import (MaskDigitSet, first_digit_mask, gen,
                                       mcnemar, digit_ids_of)
from experiments.localizer import (Standardizer, balanced_accuracy, load_candidates,
                                   run_seed, svd_basis, wilson)
from experiments.localizer_pair import majority, per_step_recall
from experiments.oracle_rewind import MaskBadFirstDigit, finish_chain

LINE = re.compile(r"^([a-z]) = (\d+)$")
STEP = re.compile(r"^([a-z]) = ([a-z]) ([+\-*]) (\d+)$")


def eval_lines(lines: list[str]) -> dict[str, int]:
    """Evaluate the definition block exactly as a subprocess would.

    `a = 4`, then `b = a + 3`, `c = b * 2`, ... No model, no ground truth.
    """
    val: dict[str, int] = {}
    for ln in lines:
        m = LINE.match(ln)
        if m:
            val[m.group(1)] = int(m.group(2))
            continue
        m = STEP.match(ln)
        if not m:
            raise ValueError(f"unparseable definition line: {ln!r}")
        tgt, src, op, k = m.group(1), m.group(2), m.group(3), int(m.group(4))
        if src not in val:
            raise ValueError(f"undefined source {src!r} in {ln!r}")
        v = val[src]
        val[tgt] = v - k if op == "-" else (v * k if op == "*" else v + k)
    return val


def interp_values(lines: list[str], order: list[str]) -> list[int]:
    val = eval_lines(lines)
    return [val[o] for o in order]


def fill_with(emitted, truth, injected: set[int], lines, order, start, steps,
              model, tok, args):
    """Replay the chain: keep the original prefix, write truth at `injected`.

    Everything before the first intervention is left exactly as emitted. That
    is what makes `k1_loc` identical to the ladder's `oracle_value` arm and
    `k1_oracleloc` identical to `oracle_force`: those arms reuse
    `emitted[:t]` and only regenerate downstream. Regenerating the prefix
    instead is a different, unanchored experiment (it scored 72/110 rather
    than 70/110 and failed the anchor).
    """
    inj = set(injected)
    first = min(inj)
    seq = list(emitted[:first])
    for i in range(first, steps):
        if i in inj:
            seq.append(int(truth[i]))
            continue
        prompt = build_prompt(lines, order, start, seq, i)
        v = parse_value(gen(model, tok, prompt, args, False))
        seq.append(v)
        if v is None:
            break
    return seq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="results/arith_steps_h5.pt")
    ap.add_argument("--train-data", default="results/arith_steps_h5.pt",
                    help="localizer training corpus; leave at the h5 file when "
                         "--data is the frozen vault so the weights are frozen")
    ap.add_argument("--model", default="checkpoints/hf/qwen3-0.6b-base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rows-out", default="results/interp_calls_rows.jsonl")
    ap.add_argument("--out", default="results/interp_calls_h5.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path
    train_path = Path(args.train_data)
    if not train_path.is_absolute():
        train_path = root / train_path
    # The anchors are claims about the h5 holdout run. They cannot hold on a
    # different corpus, and a flag someone can forget to set would silently hide
    # a broken arm, so the gate is tied to the eval file itself.
    anchor = data_path.resolve() == (root / "results/arith_steps_h5.pt").resolve()
    print(f"eval data: {data_path.name} | train data: {train_path.name}")
    print(f"anchors: {'enforced (h5 holdout)' if anchor else 'skipped (eval data is not the h5 holdout)'}")
    rows_out = Path(args.rows_out)
    if not rows_out.is_absolute():
        rows_out = root / rows_out

    raw = torch.load(data_path, weights_only=False)
    steps = int(raw["config"]["steps"])
    tasks = {(t["split"], t["idx"]): t for t in raw["tasks"]}

    # ---- evaluator correctness gate ---------------------------------------
    bad = 0
    for t in raw["tasks"]:
        if t["split"] != "holdout":
            continue
        got = interp_values(t["lines"], t["order"])
        if got != list(t["truth"]):
            bad += 1
    assert bad == 0, f"{bad} holdout tasks: subprocess evaluator != stored truth"
    print(f"evaluator: 0 mismatches on holdout definition blocks")

    hold = [c for c in load_candidates(data_path, steps) if c["split"] == "holdout"]
    train = [c for c in load_candidates(train_path, steps) if c["split"] == "train"]
    assert train, f"no train candidates in {train_path}"
    assert hold, f"no holdout candidates in {data_path}"
    print(f"localizer: {len(train)} train candidates, {len(hold)} eval candidates")
    lab = torch.tensor([c["label"] for c in hold])

    feats = torch.stack([c["feat"] for c in train])
    sd = Standardizer(feats, False)
    proj = svd_basis(feats, 32)
    run_args = argparse.Namespace(width=64, heads=4, dropout=0.1, epochs=150,
                                  batch=32, lr=1e-3, weight_decay=1e-4,
                                  device=args.device)
    xh = sd(torch.stack([c["feat"] for c in hold])).to(args.device)
    xh_proj = xh @ proj.to(args.device)
    lph = torch.stack([c["logprob"] for c in hold]).to(args.device)

    lin_preds, lin_p, trf_p = [], [], []
    for s in range(args.seeds):
        rl, _, ml = run_seed(train, {"h": hold}, run_args, steps, s, sd, proj, "linear")
        rt, _, mt = run_seed(train, {"h": hold}, run_args, steps, s, sd, None, "transformer")
        lin_preds.append(rl["h"]["pred"])
        with torch.no_grad():
            lin_p.append(F.softmax(ml(xh_proj, lph), dim=-1))
            trf_p.append(F.softmax(mt(xh, lph), dim=-1))
    pl_maj = majority(lin_preds, steps)
    lin_mean = torch.stack(lin_p).mean(0)
    trf_mean = torch.stack(trf_p).mean(0)
    two_tier = pl_maj.clone()
    for i in range(len(hold)):
        if trf_mean[i, 4] >= 0.35:
            two_tier[i] = 4
    if anchor:
        assert abs(float((two_tier == lab).float().mean()) - 0.7181818) < 1e-4, \
            "two-tier localizer did not reproduce 0.718"
    ens = (lin_mean + trf_mean) / 2.0

    def ranked_steps(i: int) -> list[int]:
        first = int(two_tier[i])
        rest = [s for s in range(steps) if s != first]
        rest.sort(key=lambda s: -float(ens[i, s]))
        return [first] + rest

    print(f"localizer: two-tier {float((two_tier == lab).float().mean()):.3f} | "
          f"step4 {per_step_recall(two_tier, lab, steps)[4]['recall']:.3f}")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    model, tok = load(model_path, torch.float32, args.device, "auto")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.cuda.reset_peak_memory_stats()

    rows = [r for r in raw["rows"]
            if r["split"] == "holdout" and r["complete"] and r["first_error"] is not None]
    if args.limit:
        rows = rows[:args.limit]
    key = {(c["split"], c["idx"], c["cand"]): i for i, c in enumerate(hold)}
    missing = [r for r in rows if (r["split"], r["idx"], r["cand"]) not in key]
    assert not missing, (f"{len(missing)}/{len(rows)} holdout rows have no localizer "
                         f"candidate; the curve would silently drop them")

    done = set()
    if args.resume and rows_out.exists():
        for line in rows_out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["idx"], rec["cand"]))
        print(f"resume: {len(done)} rows already done")

    digits = digit_ids_of(tok)
    t0 = time.time()
    with rows_out.open("a", encoding="utf-8") as fh:
        for j, r in enumerate(rows):
            if (r["idx"], r["cand"]) in done:
                continue
            i = key.get((r["split"], r["idx"], r["cand"]))
            if i is None:
                continue
            t_true = int(r["first_error"])
            t_hat = int(two_tier[i])
            emitted = list(r["emitted"])
            task = tasks[(r["split"], r["idx"])]
            lines, order, start = task["lines"], task["order"], task["start"]
            truth = interp_values(lines, order)
            rank = ranked_steps(i)
            rng = random.Random(90000 + j)

            rec = {"idx": r["idx"], "cand": r["cand"], "t_true": t_true,
                   "t_hat": t_hat, "rank": rank, "ok": {}, "steps": {}}

            def record(name, injected, ok):
                rec["ok"][name] = bool(ok)
                rec["steps"][name] = sorted(injected)

            # k0: current subtractive actuator
            prompt = build_prompt(lines, order, start, emitted[:t_hat], t_hat)
            bad_val = emitted[t_hat] if t_hat < len(emitted) else None
            proc = (MaskBadFirstDigit(*first_digit_mask(tok, int(bad_val), digits))
                    if bad_val is not None else None)
            v = parse_value(gen(model, tok, prompt, args, False, proc))
            seq = list(emitted[:t_hat]) + [v] if v is not None else []
            if v is not None:
                seq = finish_chain(model, tok, lines, order, start, seq, t_hat + 1,
                                   steps, args.device, args.max_new)
            record("k0_rank2", {t_hat},
                   len(seq) == steps and all(a == b for a, b in zip(seq, truth)))

            # localized calls
            for k in (1, 2, 3):
                inj = set(rank[:k])
                seq = fill_with(emitted, truth, inj, lines, order, start, steps,
                                model, tok, args)
                record(f"k{k}_loc", inj,
                       len(seq) == steps and all(a == b for a, b in zip(seq, truth)))

            # random calls, same k
            for k in (1, 2, 3):
                inj = set(rng.sample(range(steps), k))
                seq = fill_with(emitted, truth, inj, lines, order, start, steps,
                                model, tok, args)
                record(f"k{k}_rand", inj,
                       len(seq) == steps and all(a == b for a, b in zip(seq, truth)))

            # perfect binding
            seq = fill_with(emitted, truth, {t_true}, lines, order, start, steps,
                            model, tok, args)
            record("k1_oracleloc", {t_true},
                   len(seq) == steps and all(a == b for a, b in zip(seq, truth)))

            # all steps
            seq = fill_with(emitted, truth, set(range(steps)), lines, order, start,
                            steps, model, tok, args)
            record("k5_all", set(range(steps)),
                   len(seq) == steps and all(a == b for a, b in zip(seq, truth)))

            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (j + 1) % 10 == 0 or (j + 1) == len(rows):
                print(f"  {j + 1:>3}/{len(rows)} | t_hat={t_hat} t*={t_true} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    recs = [json.loads(l) for l in rows_out.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    recs.sort(key=lambda x: (x["idx"], x["cand"]))
    n = len(recs)
    names = ["k0_rank2", "k1_loc", "k2_loc", "k3_loc", "k1_rand", "k2_rand",
             "k3_rand", "k1_oracleloc", "k5_all"]
    summary = {}
    for a in names:
        ok = sum(int(x["ok"][a]) for x in recs)
        lo, hi = wilson(ok, n)
        calls = sum(len(x["steps"][a]) for x in recs) / max(n, 1)
        summary[a] = {"ok": ok, "n": n, "rate": ok / max(n, 1), "ci": [lo, hi],
                      "calls": calls}

    paired = {}
    for k in (1, 2, 3):
        paired[f"k{k}_loc_vs_rand"] = mcnemar(
            [x["ok"][f"k{k}_loc"] for x in recs],
            [x["ok"][f"k{k}_rand"] for x in recs])
    paired["k1_loc_vs_k0"] = mcnemar([x["ok"]["k1_loc"] for x in recs],
                                     [x["ok"]["k0_rank2"] for x in recs])

    # calls needed by random search to match k1_loc's accuracy
    curve = {k: {"loc": summary[f"k{k}_loc"]["ok"], "rand": summary[f"k{k}_rand"]["ok"]}
             for k in (1, 2, 3)}

    # ---- anchors: all three must reproduce the committed runs exactly -------
    # k0_rank2  -> results/level2_ladder_h5.json  (rank2)
    # k1_loc    -> results/level2_ladder_h5.json  (oracle_value)
    # k1_oracleloc -> results/oracle_rewind_h5.json (oracle_force)
    anchors = {}
    if anchor:
        anchors["k0_rank2"] = {"got": summary["k0_rank2"]["ok"], "expected": 32,
                               "source": "level2_ladder_h5.json rank2"}
        anchors["k1_loc"] = {"got": summary["k1_loc"]["ok"], "expected": 70,
                             "source": "level2_ladder_h5.json oracle_value"}
        anchors["k1_oracleloc"] = {"got": summary["k1_oracleloc"]["ok"],
                                   "expected": 94,
                                   "source": "oracle_rewind_h5.json oracle_force"}
        for name, a in anchors.items():
            assert a["got"] == a["expected"], (
                f"anchor failed: {name} = {a['got']}/{n}, {a['source']} has "
                f"{a['expected']}/110; the arms are not comparable")
        print("anchors reproduced: k0_rank2 32/110, k1_loc 70/110, "
              "k1_oracleloc 94/110")

    payload = {
        "config": vars(args),
        "n": n,
        "anchor_enforced": anchor,
        "train_candidates": len(train),
        "anchors": anchors,
        "localizer": {
            "two_tier_top1": float((two_tier == lab).float().mean()),
            "two_tier_bal": balanced_accuracy(two_tier, lab, steps),
            "t_exact": sum(int(x["t_hat"] == x["t_true"]) for x in recs) / max(n, 1),
        },
        "arms": summary,
        "paired": paired,
        "curve": curve,
        "vram_peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
    }

    print("\n" + "=" * 80)
    print(f"CALL-ALLOCATION CURVE (N={n})")
    print("=" * 80)
    print(f"{'arm':<15} {'ok':>7} {'rate':>6} {'95% CI':>16} {'calls':>6} {'ok/call':>8}")
    for a in names:
        s = summary[a]
        print(f"{a:<15} {s['ok']:>3}/{n:<3} {s['rate']:>6.3f} "
              f"[{s['ci'][0]:.3f},{s['ci'][1]:.3f}] {s['calls']:>6.2f} "
              f"{s['ok'] / max(s['calls'] * n, 1e-9):>8.3f}")
    for key_, pv in paired.items():
        print(f"  {key_}: {pv['wins']}W/{pv['losses']}L p={pv['p']:.2e}")
    print(f"peak VRAM {payload['vram_peak_mb']:.0f} MB")

    out_file = Path(args.out)
    if not out_file.is_absolute():
        out_file = root / out_file
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

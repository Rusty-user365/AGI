"""Level-2 actuator ladder: is single-token rank-2 the bottleneck, or the value?

The Level-2 proposal (swap in a diffusion/infilling specialist) changes four
things at once: model identity, attention directionality, actuator granularity,
and escalation policy. Any result would be uninterpretable. This script changes
one thing at a time, on the model we already have, before any second model is
loaded.

Arms, all at the frozen two-tier localizer's `t_hat` unless stated otherwise,
all ungated (the gate is a no-op on recovery: `path_a_loop_h5.json` gated and
ungated two-tier arms are identical at 32/110):

    rank2               forbid the first digit of `emitted[t_hat]`. Baseline;
                        must reproduce 32/110.
    span_digitset       forbid every digit appearing in `emitted[t_hat]`.
                        The strict "mask the whole value" actuator. Caveat: for
                        a multi-digit bad value this also forbids the truth
                        whenever the truth shares a digit, so it can be worse
                        than rank2 for mechanical reasons.
    resample_neq        sample at T=0.8, reject only the exact bad value,
                        up to 4 draws. The loosest possible actuator upgrade.
    resample_neq_best4  sample 4, drop any draw equal to the bad value, keep the
                        highest step logprob. Selection uses no ground truth.
    oracle_value        inject truth[t_hat] at t_hat. The ceiling any actuator
                        could reach *through this localizer*: if this does not
                        clear rank2 by much, no infiller can help, because
                        infilling is strictly weaker than knowing the value.
    resample_at_tstar   at the true error step, 4 unmasked samples, roll each
                        forward, success if any recovers. Upper bound on any
                        actuator that only changes the value at t*.

Pre-registered decision rule (written into the output payload before the run):

    build Level 2  iff  (best actuator beats rank2, paired exact McNemar
                         p < 0.05)  AND  (oracle_value - rank2 >= 15 chains).

Otherwise Level 2 is closed as measured, not as failed-to-load.

Rows are appended to a JSONL file as they finish, so a crash or power-off
loses at most one row; `--resume` skips completed (idx, cand) pairs.

Usage:
    python experiments/level2_ladder.py --limit 6
    python experiments/level2_ladder.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.arith_pilot import load
from experiments.arith_steps import build_prompt, parse_value, readout
from experiments.localizer import (Standardizer, balanced_accuracy, load_candidates,
                                   run_seed, sign_test, svd_basis, wilson)
from experiments.localizer_pair import majority, per_step_recall
from experiments.oracle_rewind import MaskBadFirstDigit, finish_chain

try:
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover
    LogitsProcessor = object


class MaskDigitSet(LogitsProcessor):
    """Forbid a set of digit tokens until the value has started.

    Qwen tokenises ` 15` as [space, '1', '5'], so the mask stays on until a
    digit token has actually been emitted.
    """

    def __init__(self, banned: set[int], digit_ids: set[int]):
        self.banned = [int(b) for b in banned]
        self.digits = set(int(d) for d in digit_ids)
        self.done = False

    def __call__(self, input_ids, scores):
        if not self.done:
            if int(input_ids[0, -1]) in self.digits:
                self.done = True
            else:
                for b in self.banned:
                    scores[:, b] = -float("inf")
        return scores


def digit_ids_of(tok) -> list[int]:
    return [tok(" " + str(d))["input_ids"][-1] for d in range(10)]


@torch.no_grad()
def gen(model, tok, prompt, args, sample, processor=None):
    """gen_step with a pluggable logits processor (gen_step only takes the
    (int, list) rank-2 mask tuple)."""
    enc = tok([prompt], return_tensors="pt").to(args.device)
    enc.pop("token_type_ids", None)
    kw = (dict(do_sample=True, temperature=args.temperature, top_p=args.top_p)
          if sample else dict(do_sample=False))
    if processor is not None:
        kw["logits_processor"] = [processor]
    out = model.generate(**enc, max_new_tokens=args.max_new,
                         pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id, **kw)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                            skip_special_tokens=True)[0]


def first_digit_mask(tok, value: int, digits: list[int]):
    ids = tok(" " + str(value))["input_ids"]
    return (next(i for i in ids if i in digits), digits)


def span_mask(tok, value: int, digits: list[int]):
    ids = tok(" " + str(value))["input_ids"]
    banned = {i for i in ids if i in digits}
    return (banned, set(digits))


def chain_ok(seq, truth) -> bool:
    return len(seq) == len(truth) and all(a == b for a, b in zip(seq, truth))


def run_arm(model, tok, arm, lines, order, truth, start, emitted, t_hat,
            t_true, steps, args):
    """One actuator arm. Returns (ok, value_used, same_value, samples)."""
    bad = emitted[t_hat] if t_hat < len(emitted) else None
    digits = digit_ids_of(tok)

    if arm == "oracle_value":
        v = int(truth[t_hat])
        seq = list(emitted[:t_hat]) + [v]
        seq = finish_chain(model, tok, lines, order, start, seq, t_hat + 1,
                           steps, args.device, args.max_new)
        return chain_ok(seq, truth), v, v == bad, 1

    prompt = build_prompt(lines, order, start, emitted[:t_hat], t_hat)

    if arm == "rank2":
        proc = (MaskBadFirstDigit(*first_digit_mask(tok, int(bad), digits))
                if bad is not None else None)
        text = gen(model, tok, prompt, args, False, proc)
        v = parse_value(text)
        n_samples = 1

    elif arm == "span_digitset":
        proc = (MaskDigitSet(*span_mask(tok, int(bad), digits))
                if bad is not None else None)
        text = gen(model, tok, prompt, args, False, proc)
        v = parse_value(text)
        n_samples = 1

    elif arm == "resample_neq":
        v, n_samples = None, 0
        for _ in range(args.neq_draws):
            text = gen(model, tok, prompt, args, True)
            n_samples += 1
            v = parse_value(text)
            if v is not None and v != bad:
                break

    elif arm == "resample_neq_best4":
        cands = []
        for _ in range(args.best_draws):
            text = gen(model, tok, prompt, args, True)
            vv = parse_value(text)
            if vv is None:
                continue
            lp, _ = readout(model, tok, prompt, f" {vv}\n", args.device)
            cands.append((float(lp) if lp is not None else -1e9, vv))
        n_samples = args.best_draws
        keep = [c for c in cands if c[1] != bad] or cands
        v = max(keep)[1] if keep else None

    else:
        raise ValueError(arm)

    if v is None:
        return False, None, False, n_samples
    seq = list(emitted[:t_hat]) + [v]
    seq = finish_chain(model, tok, lines, order, start, seq, t_hat + 1, steps,
                       args.device, args.max_new)
    return chain_ok(seq, truth), v, v == bad, n_samples


def ceiling_at_tstar(model, tok, lines, order, truth, start, emitted, t_true,
                     steps, args):
    """k unmasked samples at the true step; success if any chain recovers."""
    for k in range(1, args.ceiling_k + 1):
        prompt = build_prompt(lines, order, start, emitted[:t_true], t_true)
        text = gen(model, tok, prompt, args, True)
        v = parse_value(text)
        if v is None:
            continue
        seq = list(emitted[:t_true]) + [v]
        seq = finish_chain(model, tok, lines, order, start, seq, t_true + 1,
                           steps, args.device, args.max_new)
        if chain_ok(seq, truth):
            return True, k
    return False, None


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Paired exact McNemar for arm a vs arm b."""
    w = sum(1 for x, y in zip(a, b) if x and not y)
    l = sum(1 for x, y in zip(a, b) if y and not x)
    return {"wins": w, "losses": l, "p": sign_test(w, l)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="results/arith_steps_h5.pt")
    ap.add_argument("--model", default="checkpoints/hf/qwen3-0.6b-base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--neq-draws", type=int, default=4)
    ap.add_argument("--best-draws", type=int, default=4)
    ap.add_argument("--ceiling-k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rows-out", default="results/level2_ladder_rows.jsonl")
    ap.add_argument("--out", default="results/level2_ladder_h5.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path
    rows_out = Path(args.rows_out)
    if not rows_out.is_absolute():
        rows_out = root / rows_out

    raw = torch.load(data_path, weights_only=False)
    steps = int(raw["config"]["steps"])
    cands = load_candidates(data_path, steps)
    train = [c for c in cands if c["split"] == "train"]
    hold = [c for c in cands if c["split"] == "holdout"]
    lab = torch.tensor([c["label"] for c in hold])

    # ---- frozen two-tier localizer, identical to path_a_loop.py -------------
    feats = torch.stack([c["feat"] for c in train])
    sd = Standardizer(feats, False)
    proj = svd_basis(feats, 32)
    run_args = argparse.Namespace(width=64, heads=4, dropout=0.1, epochs=150,
                                  batch=32, lr=1e-3, weight_decay=1e-4,
                                  device=args.device)
    xh = sd(torch.stack([c["feat"] for c in hold])).to(args.device)
    xh_proj = xh @ proj.to(args.device)
    lph = torch.stack([c["logprob"] for c in hold]).to(args.device)

    lin_preds, trf_probs = [], []
    for s in range(args.seeds):
        rl, _, ml = run_seed(train, {"h": hold}, run_args, steps, s, sd, proj, "linear")
        rt, _, mt = run_seed(train, {"h": hold}, run_args, steps, s, sd, None, "transformer")
        lin_preds.append(rl["h"]["pred"])
        with torch.no_grad():
            trf_probs.append(F.softmax(mt(xh, lph), dim=-1))
    pl_maj = majority(lin_preds, steps)
    trf_p = torch.stack(trf_probs).mean(0)
    two_tier = pl_maj.clone()
    for i in range(len(hold)):
        if trf_p[i, 4] >= 0.35:
            two_tier[i] = 4

    print(f"localizer: linear {float((pl_maj == lab).float().mean()):.3f} | "
          f"two-tier {float((two_tier == lab).float().mean()):.3f} | "
          f"step4 {per_step_recall(two_tier, lab, steps)[4]['recall']:.3f}")
    assert abs(float((two_tier == lab).float().mean()) - 0.7181818) < 1e-4, \
        "two-tier localizer did not reproduce path_a_loop_h5.json (0.718)"

    # ---- generation model ---------------------------------------------------
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    model, tok = load(model_path, torch.float32, args.device, "auto")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.cuda.reset_peak_memory_stats()

    tasks = {(t["split"], t["idx"]): t for t in raw["tasks"]}
    rows = [r for r in raw["rows"]
            if r["split"] == "holdout" and r["complete"] and r["first_error"] is not None]
    if args.limit:
        rows = rows[:args.limit]
    key = {(c["split"], c["idx"], c["cand"]): i for i, c in enumerate(hold)}

    done = set()
    if args.resume and rows_out.exists():
        for line in rows_out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["idx"], rec["cand"]))
        print(f"resume: {len(done)} rows already done")

    arms = ["rank2", "span_digitset", "resample_neq", "resample_neq_best4",
            "oracle_value"]
    t0_all = time.time()
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
            lines, order, truth, start = (task["lines"], task["order"],
                                          task["truth"], task["start"])
            rec = {"idx": r["idx"], "cand": r["cand"], "t_true": t_true,
                   "t_hat": t_hat, "bad": emitted[t_hat] if t_hat < len(emitted) else None,
                   "ok": {}, "val": {}, "same_value": {}, "samples": {}}

            for ai, arm in enumerate(arms):
                torch.manual_seed(10000 * j + ai)
                ok, v, same, n_s = run_arm(model, tok, arm, lines, order, truth,
                                           start, emitted, t_hat, t_true, steps, args)
                rec["ok"][arm] = bool(ok)
                rec["val"][arm] = v
                rec["same_value"][arm] = bool(same)
                rec["samples"][arm] = n_s

            if args.ceiling_k > 0:
                torch.manual_seed(10000 * j + 99)
                ok_c, k_used = ceiling_at_tstar(model, tok, lines, order, truth,
                                                start, emitted, t_true, steps, args)
                rec["ceiling_at_tstar"] = {"ok": bool(ok_c), "k": k_used}

            fh.write(json.dumps(rec) + "\n")
            fh.flush()

            if (j + 1) % 10 == 0 or (j + 1) == len(rows):
                print(f"  {j + 1:>3}/{len(rows)} | "
                      f"rank2={rec['ok']['rank2']} "
                      f"span={rec['ok']['span_digitset']} "
                      f"neq={rec['ok']['resample_neq']} "
                      f"best4={rec['ok']['resample_neq_best4']} "
                      f"oracle={rec['ok']['oracle_value']} "
                      f"| t_hat={t_hat} t*={t_true} "
                      f"({time.time() - t0_all:.0f}s)", flush=True)

    # ---- aggregate ----------------------------------------------------------
    recs = [json.loads(l) for l in rows_out.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    recs.sort(key=lambda x: (x["idx"], x["cand"]))
    n = len(recs)
    summary = {}
    for arm in arms:
        ok = sum(int(x["ok"][arm]) for x in recs)
        lo, hi = wilson(ok, n)
        summary[arm] = {
            "ok": ok, "n": n, "rate": ok / max(n, 1), "ci": [lo, hi],
            "same_value": sum(int(x["same_value"][arm]) for x in recs) / max(n, 1),
            "samples": sum(int(x["samples"][arm]) for x in recs) / max(n, 1),
        }
    if args.ceiling_k > 0:
        cok = sum(int(x.get("ceiling_at_tstar", {}).get("ok", False)) for x in recs)
        lo, hi = wilson(cok, n)
        summary["resample_at_tstar"] = {"ok": cok, "n": n, "rate": cok / max(n, 1),
                                        "ci": [lo, hi], "k": args.ceiling_k}

    base = [x["ok"]["rank2"] for x in recs]
    paired = {}
    for arm in arms[1:]:
        paired[arm] = mcnemar([x["ok"][arm] for x in recs], base)
    if args.ceiling_k > 0:
        paired["resample_at_tstar"] = mcnemar(
            [x.get("ceiling_at_tstar", {}).get("ok", False) for x in recs], base)

    headroom = summary["oracle_value"]["ok"] - summary["rank2"]["ok"]
    deployable = ["span_digitset", "resample_neq", "resample_neq_best4"]
    best = max(deployable, key=lambda a: summary[a]["ok"])
    build = (summary[best]["ok"] > summary["rank2"]["ok"]
             and paired[best]["p"] < 0.05 and headroom >= 15)
    verdict = {
        "deployable_actuators": deployable,
        "best_deployable": best,
        "best_deployable_vs_rank2": paired[best],
        "headroom_at_that": headroom,
        "ceiling_arms": {"oracle_value": summary["oracle_value"]["ok"],
                         "resample_at_tstar": summary.get("resample_at_tstar", {}).get("ok")},
        "decision": "build_level2" if build else "close_level2_actuator_branch",
        "rule": ("build iff the best DEPLOYABLE actuator beats rank2 at McNemar "
                 "p<0.05 and oracle_value - rank2 >= 15 chains. oracle_value and "
                 "resample_at_tstar are ceilings, not actuators: they are excluded "
                 "from the best-actuator search."),
    }

    payload = {
        "config": vars(args),
        "n": n,
        "localizer": {
            "linear_top1": float((pl_maj == lab).float().mean()),
            "two_tier_top1": float((two_tier == lab).float().mean()),
            "two_tier_bal": balanced_accuracy(two_tier, lab, steps),
            "t_exact": sum(int(x["t_hat"] == x["t_true"]) for x in recs) / max(n, 1),
        },
        "arms": summary,
        "paired_vs_rank2": paired,
        "verdict": verdict,
        "split_by_localization": {
            "t_hat_eq_tstar": {
                "n": sum(1 for x in recs if x["t_hat"] == x["t_true"]),
                **{a: sum(int(x["ok"][a]) for x in recs if x["t_hat"] == x["t_true"])
                   for a in arms},
            },
            "t_hat_ne_tstar": {
                "n": sum(1 for x in recs if x["t_hat"] != x["t_true"]),
                **{a: sum(int(x["ok"][a]) for x in recs if x["t_hat"] != x["t_true"])
                   for a in arms},
            },
        },
        "vram_peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
    }

    print("\n" + "=" * 78)
    print(f"LEVEL-2 ACTUATOR LADDER (N={n})")
    print("=" * 78)
    print(f"{'arm':<22} {'ok':>6} {'rate':>6} {'95% CI':>16} {'same':>6} {'p vs rank2':>11}")
    for arm in arms + (["resample_at_tstar"] if args.ceiling_k > 0 else []):
        s = summary[arm]
        p = paired.get(arm, {}).get("p")
        ps = f"{p:.2e}" if p is not None else "-"
        print(f"{arm:<22} {s['ok']:>3}/{n:<3} {s['rate']:>6.3f} "
              f"[{s['ci'][0]:.3f},{s['ci'][1]:.3f}] "
              f"{s.get('same_value', 0.0):>6.2f} {ps:>11}")
    print(f"\nt_exact = {payload['localizer']['t_exact']:.3f}")
    print(f"headroom (oracle_value - rank2) = {headroom}")
    print(f"DECISION: {verdict['decision']}  (best={best}, "
          f"p={paired[best]['p']:.2e})")
    print(f"peak VRAM {payload['vram_peak_mb']:.0f} MB")

    out_file = Path(args.out)
    if not out_file.is_absolute():
        out_file = root / out_file
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

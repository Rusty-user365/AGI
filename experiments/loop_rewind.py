"""Does a *predicted* step, rather than the true one, still repair the chain?

`oracle_rewind.py` measures the repair mechanism with the true `first_error`.
That answers "is the fix worth localizing for" but not "is this localizer good
enough to drive it". This closes the loop: the linear localizer is trained on
the train split, predicts a step for each holdout candidate, and the rank-2
exclusion runs at the *predicted* step.

Two variants, because they fail differently:

    pred_real     rewind to the model's own emitted prefix `emitted[:t']`.
                  This is what a real loop does: the DAG keeps the tokens that
                  were generated, so if t' > first_error the error is still in
                  the prefix and masking a later step cannot remove it.
    pred_truth    rewind to `truth[:t']`, i.e. assume a verifier certified every
                  step before t'. Differs from `pred_real` only when t' is late,
                  and isolates localization error from prefix contamination.

Reference points from the oracle run on the same 110 rows: full rewrite 0.182,
rank-2 at the true step 0.318, force-truth at the true step 0.855.

No KV cache: "steps generated" counts fresh forwards over the prefix.

Usage:
    python experiments/loop_rewind.py --data results/arith_steps_h5.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.arith_pilot import load
from experiments.arith_steps import build_prompt, parse_value
from experiments.localizer import Standardizer, load_candidates, run_seed, svd_basis
from experiments.oracle_rewind import finish_chain, gen_step, wilson
from experiments._device import resolve as _resolve_device


def mask_for(tok, value: int):
    """Forbid the first digit of `value`, leaving the leading space alone."""
    digit_ids = [tok(" " + str(d))["input_ids"][-1] for d in range(10)]
    ids = tok(" " + str(value))["input_ids"]
    return next(i for i in ids if i in digit_ids), digit_ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="results/arith_steps_h5.pt")
    ap.add_argument("--model", default="checkpoints/hf/qwen3-0.6b-base")
    ap.add_argument("--arch", default="auto")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-new", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/loop_rewind_h5.json")
    args = ap.parse_args()
    args.device = str(_resolve_device(args.device))

    path = Path(args.data)
    raw = torch.load(path, weights_only=False)
    steps = int(raw["config"]["steps"])
    cands = load_candidates(path, steps)
    train = [c for c in cands if c["split"] == "train"]
    hold = [c for c in cands if c["split"] == "holdout"]

    # ---- localizer: train on train, predict the step for each holdout row ----
    feats = torch.stack([c["feat"] for c in train])
    sd = Standardizer(feats, False)
    proj = svd_basis(feats, 32)
    run_args = argparse.Namespace(width=64, heads=4, dropout=0.1, epochs=150,
                                  batch=32, lr=1e-3, weight_decay=1e-4,
                                  device="cpu")
    preds = []
    for s in range(args.seeds):
        res, _acc, _m = run_seed(train, {"h": hold}, run_args, steps, s, sd,
                                 proj, "linear")
        preds.append(res["h"]["pred"])
    stack = torch.stack(preds)
    pred_t = []
    for i in range(stack.shape[1]):
        pred_t.append(int(torch.bincount(stack[:, i], minlength=steps).argmax()))
    lab = torch.tensor([c["label"] for c in hold])
    hit = float((torch.tensor(pred_t) == lab).float().mean())
    print(f"localizer majority over {args.seeds} seeds: top1 {hit:.3f} "
          f"on {len(hold)} candidates")

    # ---- align corpus rows with the localizer's candidate order ----
    key = {(c["split"], c["idx"], c["cand"]): i for i, c in enumerate(hold)}
    tasks = {(t["split"], t["idx"]): t for t in raw["tasks"]}
    rows = [r for r in raw["rows"]
            if r["split"] == "holdout" and r["complete"]
            and r["first_error"] is not None]
    if args.limit:
        rows = rows[:args.limit]

    model, tok = load(Path(args.model).resolve(), torch.float32, args.device,
                      args.arch)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    out = {v: {"ok": 0, "n": 0, "steps": 0, "t_exact": 0, "t_late": 0}
           for v in ("pred_real", "pred_truth")}
    per_row = []
    for j, r in enumerate(rows):
        i = key.get((r["split"], r["idx"], r["cand"]))
        if i is None:
            continue
        t_hat = pred_t[i]
        t_true = int(r["first_error"])
        emitted = list(r["emitted"])
        task = tasks[(r["split"], r["idx"])]
        lines, order, truth, start = (task["lines"], task["order"],
                                      task["truth"], task["start"])

        for variant in ("pred_real", "pred_truth"):
            prefix = list(emitted[:t_hat]) if variant == "pred_real" \
                else list(truth[:t_hat])
            # The value being replaced is the one the model produced at t_hat,
            # whether or not it is in the prefix we rewind to.
            if t_hat >= len(emitted) or emitted[t_hat] is None:
                continue
            bad = int(emitted[t_hat])
            prompt = build_prompt(lines, order, start, prefix, t_hat)
            text = gen_step(model, tok, prompt, args.device, args.max_new, False,
                            mask=mask_for(tok, bad))
            v = parse_value(text)
            generated = 1
            seq = prefix + [v]
            if v is not None:
                seq = finish_chain(model, tok, lines, order, start, seq,
                                   t_hat + 1, steps, args.device, args.max_new)
                generated += steps - t_hat - 1
            ok = (len(seq) == steps
                  and all(seq[k] == truth[k] for k in range(steps)))
            a = out[variant]
            a["n"] += 1
            a["ok"] += int(ok)
            a["steps"] += generated
            a["t_exact"] += int(t_hat == t_true)
            a["t_late"] += int(t_hat > t_true)
            per_row.append({"idx": r["idx"], "cand": r["cand"], "t_true": t_true,
                            "t_hat": t_hat, "variant": variant, "ok": bool(ok),
                            "generated": generated})
        if (j + 1) % 10 == 0:
            print(f"  {j + 1}/{len(rows)}", flush=True)

    print(f"\n{'variant':<12} {'recovered':>10} {'rate':>6} {'95% CI':>16} "
          f"{'steps/gen':>9} {'t exact':>8} {'t late':>7}")
    for v, a in out.items():
        n = max(a["n"], 1)
        lo, hi = wilson(a["ok"], a["n"])
        print(f"{v:<12} {a['ok']:>4}/{a['n']:<5} {a['ok']/n:>6.3f} "
              f"[{lo:.3f}, {hi:.3f}] {a['steps']/n:>9.2f} "
              f"{a['t_exact']/n:>8.3f} {a['t_late']/n:>7.3f}")

    payload = {"config": vars(args), "steps": steps, "n": len(rows),
               "localizer_top1": hit, "localizer_seeds": args.seeds,
               "variants": {v: dict(a, rate=a["ok"] / max(a["n"], 1))
                            for v, a in out.items()},
               "per_row": per_row}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

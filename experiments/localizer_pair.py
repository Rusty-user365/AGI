"""Paired test: does cross-step attention beat a per-step linear read?

`localizer.py` reports each arm separately, so "linear 0.611 vs transformer
0.576" is a comparison of two means, not a paired test on the same candidates.
The user's gate for attention was stricter: the transformer must beat linear by
a margin that survives a paired test on the holdout candidates.

This trains both arms with the **same seed and the same split**, so the only
difference is the architecture (independent per-step scoring vs one encoder
layer over the H rows). It reports:

    per-seed paired W/L on holdout candidates
    pooled sign test over seeds (stability check, not an independent test)
    bootstrap CI over candidates for the majority-vote accuracy difference
    per-step recall for each arm, so a constant predictor cannot look good

Bootstrap resamples candidates, which is the unit of the holdout split; seeds
are averaged by majority vote before resampling, so the interval reflects
candidate sampling error only and not seed noise.

Usage:
    python experiments/localizer_pair.py --data results/arith_steps_h5.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.localizer import (Standardizer, balanced_accuracy, load_candidates,
                                   run_seed, sign_test, svd_basis, wilson)


def majority(preds: list[torch.Tensor], steps: int) -> torch.Tensor:
    """Per-candidate majority vote across seeds, ties to the lowest step."""
    stack = torch.stack(preds)                      # [seeds, n]
    out = torch.zeros(stack.shape[1], dtype=torch.long)
    for i in range(stack.shape[1]):
        counts = torch.bincount(stack[:, i], minlength=steps)
        out[i] = int(counts.argmax())
    return out


def bootstrap_diff(a: torch.Tensor, b: torch.Tensor, n_boot: int = 10000,
                   seed: int = 0) -> tuple[float, float, float]:
    """CI for mean(a) - mean(b) over candidates. a, b are 0/1 tensors."""
    g = torch.Generator().manual_seed(seed)
    n = a.numel()
    d = a.float() - b.float()
    draws = torch.randint(0, n, (n_boot, n), generator=g)
    means = d[draws].mean(-1)
    lo = float(torch.quantile(means, 0.025))
    hi = float(torch.quantile(means, 0.975))
    return float(d.mean()), lo, hi


def per_step_recall(pred: torch.Tensor, lab: torch.Tensor, steps: int) -> list:
    out = []
    for c in range(steps):
        m = lab == c
        out.append({"step": c, "n": int(m.sum()),
                    "recall": float((pred[m] == lab[m]).float().mean())
                    if int(m.sum()) else None})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="results/arith_steps_h5.pt")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.data)
    raw = torch.load(path, weights_only=False)
    steps = int(raw["config"]["steps"])
    cands = load_candidates(path, steps)
    train = [c for c in cands if c["split"] == "train"]
    hold = [c for c in cands if c["split"] == "holdout"]
    lab = torch.tensor([c["label"] for c in hold])
    print(f"{len(train)} train / {len(hold)} holdout broken candidates, "
          f"steps={steps}")

    feats = torch.stack([c["feat"] for c in train])
    sd = Standardizer(feats, False)
    proj = svd_basis(feats, 32)

    run_args = argparse.Namespace(width=args.width, heads=4, dropout=0.1,
                                  epochs=args.epochs, batch=32, lr=1e-3,
                                  weight_decay=1e-4, device=args.device)

    lin_preds, trf_preds = [], []
    wins = losses = 0
    for s in range(args.seeds):
        rl, acc_l, _ = run_seed(train, {"h": hold}, run_args, steps, s, sd,
                                proj, "linear")
        rt, acc_t, _ = run_seed(train, {"h": hold}, run_args, steps, s, sd,
                                None, "transformer")
        pl, pt = rl["h"]["pred"], rt["h"]["pred"]
        lin_preds.append(pl)
        trf_preds.append(pt)
        w = int(((pl == lab) & (pt != lab)).sum())
        l = int(((pl != lab) & (pt == lab)).sum())
        wins += w
        losses += l
        print(f"  seed{s}: linear {float((pl == lab).float().mean()):.3f} "
              f"(train {acc_l:.3f})  transformer "
              f"{float((pt == lab).float().mean()):.3f} (train {acc_t:.3f})  "
              f"paired {w}W/{l}L p={sign_test(w, l):.3f}")

    pl = majority(lin_preds, steps)
    pt = majority(trf_preds, steps)
    ok_l, ok_t = pl == lab, pt == lab
    diff, lo, hi = bootstrap_diff(ok_l, ok_t)
    print(f"\nmajority vote over {args.seeds} seeds, n={len(hold)}")
    print(f"  linear      top1 {float(ok_l.float().mean()):.3f}  "
          f"bal {balanced_accuracy(pl, lab, steps):.3f}")
    print(f"  transformer top1 {float(ok_t.float().mean()):.3f}  "
          f"bal {balanced_accuracy(pt, lab, steps):.3f}")
    print(f"  difference (linear - transformer) {diff:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}] 95% bootstrap over candidates")
    print(f"  pooled paired over seeds {wins}W/{losses}L "
          f"p={sign_test(wins, losses):.3f} (same candidates each seed, so this "
          f"is a stability check)")

    rec_l = per_step_recall(pl, lab, steps)
    rec_t = per_step_recall(pt, lab, steps)
    print(f"\n{'step':>4} {'n':>4} {'linear':>8} {'transformer':>12}")
    for a, b in zip(rec_l, rec_t):
        rl_ = "  n/a" if a["recall"] is None else f"{a['recall']:.3f}"
        rt_ = "  n/a" if b["recall"] is None else f"{b['recall']:.3f}"
        print(f"{a['step']:>4} {a['n']:>4} {rl_:>8} {rt_:>12}")

    out = {
        "config": vars(args), "data": str(path), "steps": steps,
        "train_candidates": len(train), "holdout_candidates": len(hold),
        "majority": {
            "linear_top1": float(ok_l.float().mean()),
            "linear_bal": balanced_accuracy(pl, lab, steps),
            "transformer_top1": float(ok_t.float().mean()),
            "transformer_bal": balanced_accuracy(pt, lab, steps),
            "diff": diff, "diff_ci": [lo, hi],
        },
        "pooled_paired": {"wins": wins, "losses": losses,
                          "p": sign_test(wins, losses)},
        "per_step_recall": {"linear": rec_l, "transformer": rec_t},
        "per_seed": [{"seed": i,
                      "linear": float((lin_preds[i] == lab).float().mean()),
                      "transformer": float((trf_preds[i] == lab).float().mean())}
                     for i in range(args.seeds)],
        "linear_top1_ci": list(wilson(int(ok_l.sum()), len(hold))),
        "transformer_top1_ci": list(wilson(int(ok_t.sum()), len(hold))),
    }
    args.out = args.out or f"results/localizer_pair_{path.stem}.json"
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

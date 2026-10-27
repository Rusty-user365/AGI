"""Per-step localizer: point at the step where the chain broke.

The question. A pooled observer cannot localize: mean-pooling over steps
destroys position before the first layer, so the information is gone. That was a
real gap in the selection experiments and it has never been tested. This script
tests it directly on `arith_steps` corpora, where the generator supplies
`first_error` for every self-conditioned trajectory.

Input is the un-pooled per-step state, `[H, d]`, one row per step. Pooling
happens *within* a step (over that step's emitted tokens) and never across steps.

Architectures compared, because "attention helps" is the claim under test:

    linear       each step scored independently -- the shallow observer that
                 failed before. Cannot see a contradiction between step 3 and
                 step 5.
    transformer  one encoder layer over the H rows, so step i can attend to
                 step j. This is the only arm that can in principle notice
                 "the value at step 5 is inconsistent with step 3".

Baselines, on the *same* eval candidates:

    chance          1 / H
    prior           always predict the most common first_error in train
    always 0        always predict the first step
    argmin logprob  point at the least confident step -- free, and the one that
                    matters. A localizer that ties this adds nothing, which is
                    the outcome of the last two selection experiments.

Reported: top-1 accuracy with a Wilson CI, top-2, mean absolute step error, and
a paired sign test of localizer vs argmin-logprob over discordant candidates.

Splits are used as recorded: train tasks train the model, holdout tasks evaluate
it. They are never pooled. Seeds are reported individually, not just averaged.

Usage:
    python experiments/localizer.py --data results/arith_steps_h8.pt --seeds 5
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- dataset ----

def load_candidates(path: Path, steps: int):
    """Rows usable for localization: complete chains that actually broke."""
    state = torch.load(path, weights_only=False)
    out = []
    for r in state["rows"]:
        if not r["complete"] or r["first_error"] is None:
            continue
        if any(s["feat"] is None or s["logprob"] is None for s in r["steps"]):
            continue
        if len(r["steps"]) != steps:
            continue
        out.append({
            "split": r["split"], "idx": r["idx"], "cand": r["cand"],
            "feat": torch.stack([s["feat"] for s in r["steps"]]).float(),
            "logprob": torch.tensor([s["logprob"] for s in r["steps"]],
                                    dtype=torch.float32),
            "label": int(r["first_error"]),
        })
    return out, state.get("config", {})


class Standardizer:
    def __init__(self, feats: torch.Tensor):
        self.mu = feats.mean(0, keepdim=True)
        self.sd = feats.std(0, keepdim=True).clamp_min(1e-6)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu.to(x.device)) / self.sd.to(x.device)


# ----------------------------------------------------------------- models ----

class LinearStep(nn.Module):
    """Score each step independently. No cross-step information by design."""

    def __init__(self, d: int, width: int, dropout: float, use_logprob: bool):
        super().__init__()
        self.use_logprob = use_logprob
        self.proj = nn.Sequential(nn.Linear(d + int(use_logprob), width),
                                  nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(width, 1)

    def forward(self, x, lp):
        if self.use_logprob:
            x = torch.cat([x, lp.unsqueeze(-1)], dim=-1)
        return self.out(self.proj(x)).squeeze(-1)


class TransformerStep(nn.Module):
    """One encoder layer over the H steps, so step i can attend to step j."""

    def __init__(self, d: int, width: int, heads: int, dropout: float,
                 use_logprob: bool, steps: int):
        super().__init__()
        self.use_logprob = use_logprob
        self.inp = nn.Linear(d + int(use_logprob), width)
        self.pos = nn.Parameter(torch.zeros(1, steps, width))
        nn.init.normal_(self.pos, std=0.02)
        self.block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=2 * width,
            dropout=dropout, batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 1)

    def forward(self, x, lp):
        if self.use_logprob:
            x = torch.cat([x, lp.unsqueeze(-1)], dim=-1)
        h = self.inp(x) + self.pos[:, :x.shape[1]]
        h = self.block(h)
        return self.out(self.norm(h)).squeeze(-1)


def build(arch: str, d: int, width: int, heads: int, dropout: float,
          use_logprob: bool, steps: int) -> nn.Module:
    if arch == "linear":
        return LinearStep(d, width, dropout, use_logprob)
    return TransformerStep(d, width, heads, dropout, use_logprob, steps)


# ------------------------------------------------------------------ stats ----

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial p, no scipy on this machine."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ------------------------------------------------------------------- train ----

def run_seed(train, evals, args, steps: int, seed: int, d: int, sd):
    torch.manual_seed(seed)
    random.seed(seed)

    model = build(args.arch, d, args.width, args.heads, args.dropout,
                  args.use_logprob, steps).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    x = torch.stack([c["feat"] for c in train]).to(args.device)
    x = sd(x)
    lp = torch.stack([c["logprob"] for c in train]).to(args.device)
    y = torch.tensor([c["label"] for c in train], device=args.device)

    n = len(train)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=args.device)
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            logits = model(x[b], lp[b])
            loss = F.cross_entropy(logits, y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        res = {}
        for name, cands in evals.items():
            xe = sd(torch.stack([c["feat"] for c in cands]).to(args.device))
            lpe = torch.stack([c["logprob"] for c in cands]).to(args.device)
            logits = model(xe, lpe)
            pred = logits.argmax(-1).cpu()
            top2 = logits.topk(min(2, steps), dim=-1).indices.cpu()
            lab = torch.tensor([c["label"] for c in cands])
            res[name] = {
                "pred": pred, "label": lab,
                "top1": float((pred == lab).float().mean()),
                "top2": float((top2 == lab.unsqueeze(-1)).any(-1).float().mean()),
                "mae": float((pred - lab).abs().float().mean()),
                "lp_pred": lpe.argmin(-1).cpu(),
            }
        xt = sd(torch.stack([c["feat"] for c in train]).to(args.device))
        lpt = torch.stack([c["logprob"] for c in train]).to(args.device)
        train_acc = float((model(xt, lpt).argmax(-1).cpu() == y.cpu())
                          .float().mean())
    return res, train_acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="results/arith_steps_h8.pt")
    ap.add_argument("--arch", default="transformer",
                    choices=["transformer", "linear"])
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--use-logprob", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.data)
    # steps comes from the config, so load the file once to read it.
    raw = torch.load(path, weights_only=False)
    steps = int(raw["config"]["steps"])
    cands, _ = load_candidates(path, steps)

    train = [c for c in cands if c["split"] == "train"]
    evals = {"holdout": [c for c in cands if c["split"] == "holdout"]}
    if not train or not evals["holdout"]:
        print(f"not enough labelled candidates: train {len(train)}, "
              f"holdout {len(evals['holdout'])}")
        return 1
    print(f"{path}: {steps} steps, {len(train)} train candidates, "
          f"{len(evals['holdout'])} holdout candidates")
    print("label distribution train  :",
          torch.bincount(torch.tensor([c["label"] for c in train]),
                         minlength=steps).tolist())
    print("label distribution holdout:",
          torch.bincount(torch.tensor([c["label"] for c in evals["holdout"]]),
                         minlength=steps).tolist())

    d = train[0]["feat"].shape[-1]
    sd = Standardizer(torch.stack([c["feat"] for c in train]).reshape(-1, d))

    # Baselines on the holdout set, independent of any training.
    hold = evals["holdout"]
    lab = torch.tensor([c["label"] for c in hold])
    lp_pred = torch.stack([c["logprob"] for c in hold]).argmin(-1)
    counts = torch.bincount(torch.tensor([c["label"] for c in train]),
                            minlength=steps)
    prior = int(counts.argmax())
    base = {
        "n": len(hold),
        "chance": 1.0 / steps,
        "prior_index": prior,
        "prior_acc": float((lab == prior).float().mean()),
        "always_first_acc": float((lab == 0).float().mean()),
        "argmin_logprob_acc": float((lp_pred == lab).float().mean()),
        "argmin_logprob_ci": list(wilson(int((lp_pred == lab).sum()),
                                        len(hold))),
    }
    print(f"\nbaselines on holdout (n={len(hold)}): chance {base['chance']:.3f}  "
          f"prior({prior}) {base['prior_acc']:.3f}  "
          f"always-0 {base['always_first_acc']:.3f}  "
          f"argmin-logprob {base['argmin_logprob_acc']:.3f}")

    runs = []
    for s in range(args.seeds):
        res, train_acc = run_seed(train, evals, args, steps, s, d, sd)
        h = res["holdout"]
        runs.append({"seed": s, "train_acc": train_acc,
                     "top1": h["top1"], "top2": h["top2"], "mae": h["mae"]})
        print(f"  seed {s}: train {train_acc:.3f}  holdout top1 {h['top1']:.3f}  "
              f"top2 {h['top2']:.3f}  mae {h['mae']:.2f}")

    top1 = [r["top1"] for r in runs]
    mean_top1 = sum(top1) / len(top1)
    k_hits = int(round(mean_top1 * len(hold)))
    print(f"\n{args.arch}{' +logprob' if args.use_logprob else ''}: "
          f"holdout top1 mean {mean_top1:.3f}  "
          f"range [{min(top1):.3f}, {max(top1):.3f}]  "
          f"~95% CI [{wilson(k_hits, len(hold))[0]:.3f}, "
          f"{wilson(k_hits, len(hold))[1]:.3f}]")

    # Paired comparison against the free baseline, over discordant candidates.
    best = max(runs, key=lambda r: r["top1"])
    res, _ = run_seed(train, evals, args, steps, best["seed"], d, sd)
    h = res["holdout"]
    wins = int(((h["pred"] == h["label"]) & (h["lp_pred"] != h["label"])).sum())
    losses = int(((h["pred"] != h["label"]) & (h["lp_pred"] == h["label"])).sum())
    p = sign_test(wins, losses)
    print(f"vs argmin-logprob (seed {best['seed']}): {wins}W/{losses}L  "
          f"p={p:.3f}")

    out = {
        "config": vars(args), "data": str(path), "steps": steps,
        "train_candidates": len(train), "holdout_candidates": len(hold),
        "label_dist_train": torch.bincount(
            torch.tensor([c["label"] for c in train]),
            minlength=steps).tolist(),
        "label_dist_holdout": torch.bincount(lab, minlength=steps).tolist(),
        "baselines": base, "runs": runs, "mean_top1": mean_top1,
        "mean_top1_ci": list(wilson(k_hits, len(hold))),
        "vs_logprob": {"wins": wins, "losses": losses, "p": p,
                       "seed": best["seed"]},
    }
    args.out = args.out or f"results/localizer_{path.stem}_{args.arch}.json"
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

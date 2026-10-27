"""Per-step localizer: point at the step where the chain broke.

The question. A pooled observer cannot localize: mean-pooling over steps
destroys position before the first layer. That gap has never been tested. This
script tests it on `arith_steps` corpora, where the generator supplies
`first_error` for every self-conditioned trajectory.

Input is the un-pooled per-step state, `[H, d]`, one row per step. Pooling
happens *within* a step (over that step's emitted tokens) and never across steps.

The confound this script exists to control. Hidden states carry position: RoPE
and residual-norm growth both make the step index linearly readable. Error
labels are not uniform over steps either (the modal first-error step holds ~56%
of broken candidates in the h5 pilot). So a reader over `[H, d]` can beat
`argmin logprob` and the label prior *separately* by learning
"prior + confidence" and never extracting an error signal at all.

The baseline that closes that hole is `poslogprob`: a per-position bias and a
per-position logprob slope, i.e. the strongest model that uses **only** position
and confidence. It is trained on the same rows with the same seed.

    arch=linear        each step scored independently, no cross-step information
    arch=transformer   one encoder layer over the H rows, so step i sees step j
    arch=poslogprob    logits[t] = bias[t] + w[t] * logprob[t]   (the confound)

A hidden-state reader earns its place only by beating `poslogprob` on paired
candidates, not by beating `argmin logprob` alone.

Capacity. 126-240 trajectories do not support a 1024->64 trainable projection
(65k parameters). Two controls, both optional:

    --frozen-proj 32   SVD basis fitted on train features, then frozen. The
                       attention block only learns interactions in 32 dims.
    --demean-position  subtract the per-step mean feature (train) so the step
                       index cannot be read off the mean shift.

Reported: top-1 with a Wilson CI, balanced accuracy (mean per-class recall, so
predicting the modal step scores 1/H rather than 0.56), accuracy on non-modal
labels, top-2, mean absolute step error, and paired sign tests against both
`argmin logprob` and `poslogprob`.

Splits are used as recorded: train tasks train, holdout tasks evaluate. Never
pooled.

Usage:
    python experiments/localizer.py --data results/arith_steps_h5.pt \
        --arch transformer --frozen-proj 32 --seeds 5
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
    return out


class Standardizer:
    def __init__(self, feats: torch.Tensor, demean_position: bool = False):
        # feats: [N, H, d]
        self.pos_mu = feats.mean(0, keepdim=True) if demean_position else None
        f = feats - self.pos_mu if demean_position else feats
        self.mu = f.mean((0, 1), keepdim=True)
        self.sd = f.std((0, 1), keepdim=True).clamp_min(1e-6)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_mu is not None:
            x = x - self.pos_mu.to(x.device)
        return (x - self.mu.to(x.device)) / self.sd.to(x.device)


def svd_basis(feats: torch.Tensor, q: int) -> torch.Tensor:
    """Frozen [d, q] basis from centred train features."""
    x = feats.reshape(-1, feats.shape[-1])
    x = x - x.mean(0, keepdim=True)
    _u, _s, v = torch.pca_lowrank(x, q=q)
    return v


# ----------------------------------------------------------------- models ----

class LinearStep(nn.Module):
    """Score each step independently. No cross-step information by design.

    `use_lp` appends the step's own log-probability as one extra input column.
    That is the redundancy control: if the hidden state already carries what
    confidence carries, adding confidence changes nothing (the same test that
    killed the selection head). If it helps, the two are complementary.
    """

    def __init__(self, d: int, width: int, dropout: float,
                 use_lp: bool = False):
        super().__init__()
        self.use_lp = use_lp
        self.proj = nn.Sequential(nn.Linear(d + int(use_lp), width), nn.GELU(),
                                  nn.Dropout(dropout))
        self.out = nn.Linear(width, 1)

    def forward(self, x, lp):
        if self.use_lp:
            x = torch.cat([x, lp.unsqueeze(-1)], dim=-1)
        return self.out(self.proj(x)).squeeze(-1)


class TransformerStep(nn.Module):
    """One encoder layer over the H steps, so step i can attend to step j."""

    def __init__(self, d: int, width: int, heads: int, dropout: float,
                 steps: int):
        super().__init__()
        self.inp = nn.Linear(d, width)
        self.pos = nn.Parameter(torch.zeros(1, steps, width))
        nn.init.normal_(self.pos, std=0.02)
        self.block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=2 * width,
            dropout=dropout, batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 1)

    def forward(self, x, lp):
        h = self.inp(x) + self.pos[:, :x.shape[1]]
        return self.out(self.norm(self.block(h))).squeeze(-1)


class PosLogprob(nn.Module):
    """The position confound, as a model: bias[t] + w[t] * logprob[t].

    Ten parameters. Uses no hidden state. Any hidden-state reader must beat this
    or it has learned nothing beyond "which step is usually wrong" plus
    "how confident was the model".
    """

    def __init__(self, steps: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(steps))
        self.w = nn.Parameter(torch.zeros(steps))

    def forward(self, x, lp):
        return self.bias.unsqueeze(0) + self.w.unsqueeze(0) * lp


def build(arch: str, d: int, width: int, heads: int, dropout: float,
          steps: int, use_lp: bool = False) -> nn.Module:
    if arch == "linear":
        return LinearStep(d, width, dropout, use_lp)
    if arch == "poslogprob":
        return PosLogprob(steps)
    return TransformerStep(d, width, heads, dropout, steps)


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


def balanced_accuracy(pred: torch.Tensor, lab: torch.Tensor, steps: int) -> float:
    """Mean per-class recall: predicting one class scores 1 / H, not its prior."""
    recalls = []
    for c in range(steps):
        m = lab == c
        if int(m.sum()) > 0:
            recalls.append(float((pred[m] == lab[m]).float().mean()))
    return sum(recalls) / len(recalls) if recalls else 0.0


def nonmodal_accuracy(pred: torch.Tensor, lab: torch.Tensor,
                      modal: int) -> tuple[float, int]:
    m = lab != modal
    n = int(m.sum())
    return (float((pred[m] == lab[m]).float().mean()) if n else 0.0), n


# ------------------------------------------------------------------- train ----

def run_seed(train, evals, args, steps: int, seed: int, sd, proj,
             arch: str) -> tuple[dict, float, nn.Module]:
    torch.manual_seed(seed)
    random.seed(seed)

    d_in = proj.shape[1] if proj is not None else train[0]["feat"].shape[-1]
    model = build(arch, d_in, args.width, args.heads, args.dropout, steps,
                  getattr(args, "linear_logprob", False)).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    def prep(cands):
        x = sd(torch.stack([c["feat"] for c in cands]).to(args.device))
        if proj is not None:
            x = x @ proj.to(args.device)
        lp = torch.stack([c["logprob"] for c in cands]).to(args.device)
        return x, lp

    x, lp = prep(train)
    y = torch.tensor([c["label"] for c in train], device=args.device)
    n = len(train)
    for _epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=args.device)
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            loss = F.cross_entropy(model(x[b], lp[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        res = {}
        for name, cands in evals.items():
            xe, lpe = prep(cands)
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
        xt, lpt = prep(train)
        train_acc = float((model(xt, lpt).argmax(-1).cpu() == y.cpu())
                          .float().mean())
    return res, train_acc, model


def describe(name: str, h: dict, steps: int, prior: int) -> dict:
    nonmodal = nonmodal_accuracy(h["pred"], h["label"], prior)
    out = {
        "name": name,
        "top1": h["top1"], "top2": h["top2"], "mae": h["mae"],
        "bal_acc": balanced_accuracy(h["pred"], h["label"], steps),
        "nonmodal_acc": nonmodal[0], "nonmodal_n": nonmodal[1],
    }
    print(f"  {name:<14} top1 {out['top1']:.3f}  bal {out['bal_acc']:.3f}  "
          f"nonmodal {out['nonmodal_acc']:.3f} (n={out['nonmodal_n']})  "
          f"top2 {out['top2']:.3f}  mae {out['mae']:.2f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="results/arith_steps_h5.pt")
    ap.add_argument("--arch", default="transformer",
                    choices=["transformer", "linear"])
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--frozen-proj", type=int, default=0,
                    help="0 = keep the full hidden size")
    ap.add_argument("--demean-position", action="store_true")
    ap.add_argument("--linear-logprob", action="store_true",
                    help="linear arm only: append the step logprob as a feature")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.data)
    raw = torch.load(path, weights_only=False)
    steps = int(raw["config"]["steps"])
    cands = load_candidates(path, steps)

    train = [c for c in cands if c["split"] == "train"]
    hold = [c for c in cands if c["split"] == "holdout"]
    if not train or not hold:
        print(f"not enough labelled candidates: train {len(train)}, "
              f"holdout {len(hold)}")
        return 1
    print(f"{path}: {steps} steps, {len(train)} train / {len(hold)} holdout "
          f"broken candidates")

    lab = torch.tensor([c["label"] for c in hold])
    lab_tr = torch.tensor([c["label"] for c in train])
    print("first_error train  :", torch.bincount(lab_tr, minlength=steps).tolist())
    print("first_error holdout:", torch.bincount(lab, minlength=steps).tolist())
    prior = int(torch.bincount(lab_tr, minlength=steps).argmax())

    sd = Standardizer(torch.stack([c["feat"] for c in train]),
                      args.demean_position)
    proj = None
    if args.frozen_proj:
        proj = svd_basis(torch.stack([c["feat"] for c in train]),
                         min(args.frozen_proj, train[0]["feat"].shape[-1]))
        print(f"frozen SVD projection: {train[0]['feat'].shape[-1]} -> "
              f"{proj.shape[1]} dims (fitted on train)")

    # Free baselines, independent of any training.
    lp_pred = torch.stack([c["logprob"] for c in hold]).argmin(-1)
    base = {
        "n": len(hold), "chance": 1.0 / steps, "prior_index": prior,
        "prior_acc": float((lab == prior).float().mean()),
        "always_first_acc": float((lab == 0).float().mean()),
        "argmin_logprob_acc": float((lp_pred == lab).float().mean()),
        "argmin_logprob_ci": list(wilson(int((lp_pred == lab).sum()), len(hold))),
        "argmin_logprob_bal": balanced_accuracy(lp_pred, lab, steps),
        "argmin_logprob_nonmodal": list(nonmodal_accuracy(lp_pred, lab, prior)),
    }
    print(f"\nbaselines (n={len(hold)}): chance {base['chance']:.3f}  "
          f"prior({prior}) {base['prior_acc']:.3f}  "
          f"always-0 {base['always_first_acc']:.3f}")
    print(f"  argmin-logprob  top1 {base['argmin_logprob_acc']:.3f}  "
          f"bal {base['argmin_logprob_bal']:.3f}  "
          f"nonmodal {base['argmin_logprob_nonmodal'][0]:.3f} "
          f"(n={base['argmin_logprob_nonmodal'][1]})")

    runs, vs_lp, vs_pp = [], {"wins": 0, "losses": 0}, {"wins": 0, "losses": 0}
    for s in range(args.seeds):
        res, train_acc, _m = run_seed(train, {"holdout": hold}, args, steps, s,
                                      sd, proj, args.arch)
        rb, _tr, _mb = run_seed(train, {"holdout": hold}, args, steps, s,
                                sd, proj, "poslogprob")
        h, hb = res["holdout"], rb["holdout"]
        d_main = describe(f"seed{s} {args.arch}", h, steps, prior)
        d_pp = describe(f"seed{s} poslogprob", hb, steps, prior)
        w_lp = int(((h["pred"] == h["label"]) & (lp_pred != h["label"])).sum())
        l_lp = int(((h["pred"] != h["label"]) & (lp_pred == h["label"])).sum())
        w_pp = int(((h["pred"] == h["label"]) & (hb["pred"] != h["label"])).sum())
        l_pp = int(((h["pred"] != h["label"]) & (hb["pred"] == h["label"])).sum())
        vs_lp["wins"] += w_lp
        vs_lp["losses"] += l_lp
        vs_pp["wins"] += w_pp
        vs_pp["losses"] += l_pp
        runs.append({"seed": s, "train_acc": train_acc,
                     "main": d_main, "poslogprob": d_pp,
                     "vs_argmin_logprob": {"wins": w_lp, "losses": l_lp,
                                           "p": sign_test(w_lp, l_lp)},
                     "vs_poslogprob": {"wins": w_pp, "losses": l_pp,
                                       "p": sign_test(w_pp, l_pp)}})
        print(f"    train acc {train_acc:.3f} | vs argmin-lp {w_lp}W/{l_lp}L "
              f"p={sign_test(w_lp, l_lp):.3f} | vs poslogprob {w_pp}W/{l_pp}L "
              f"p={sign_test(w_pp, l_pp):.3f}")

    top1 = [r["main"]["top1"] for r in runs]
    bal = [r["main"]["bal_acc"] for r in runs]
    pp_top1 = [r["poslogprob"]["top1"] for r in runs]
    k_hits = int(round(sum(top1) / len(top1) * len(hold)))
    print(f"\n{args.arch}"
          f"{' +demean-pos' if args.demean_position else ''}"
          f"{' +svd' + str(args.frozen_proj) if args.frozen_proj else ''}: "
          f"top1 mean {sum(top1)/len(top1):.3f} "
          f"[{min(top1):.3f}, {max(top1):.3f}]  "
          f"bal mean {sum(bal)/len(bal):.3f}  "
          f"| poslogprob top1 mean {sum(pp_top1)/len(pp_top1):.3f}")
    print(f"pooled paired over {args.seeds} seeds (same candidates, so this is "
          f"a stability check, not an independent test):")
    print(f"  vs argmin-logprob {vs_lp['wins']}W/{vs_lp['losses']}L "
          f"p={sign_test(vs_lp['wins'], vs_lp['losses']):.3f}")
    print(f"  vs poslogprob     {vs_pp['wins']}W/{vs_pp['losses']}L "
          f"p={sign_test(vs_pp['wins'], vs_pp['losses']):.3f}")

    # Confusion matrix and prediction histogram, best seed. One hot column means
    # the model learned the label prior and nothing else.
    best = max(range(len(runs)), key=lambda i: runs[i]["main"]["top1"])
    res, _tr, _m = run_seed(train, {"holdout": hold}, args, steps, best, sd,
                            proj, args.arch)
    h = res["holdout"]
    conf = [[int(((h["pred"] == i) & (h["label"] == j)).sum())
             for j in range(steps)] for i in range(steps)]
    print(f"\nconfusion, best seed {best} (rows predicted, cols true):")
    for i, row in enumerate(conf):
        print(f"  {i}: " + " ".join(f"{v:>3}" for v in row))

    is_greedy = torch.tensor([c["cand"] == 0 for c in hold])
    by_arm = {}
    for name, mask in (("greedy", is_greedy), ("sampled", ~is_greedy)):
        n = int(mask.sum())
        if n == 0:
            continue
        by_arm[name] = {
            "n": n,
            "localizer": float((h["pred"][mask] == h["label"][mask])
                               .float().mean()),
            "argmin_logprob": float((lp_pred[mask] == h["label"][mask])
                                    .float().mean()),
            "prior": float((h["label"][mask] == prior).float().mean()),
        }
        a = by_arm[name]
        print(f"  {name:<8} n={n:>3}  localizer {a['localizer']:.3f}  "
              f"argmin-lp {a['argmin_logprob']:.3f}  prior {a['prior']:.3f}")

    out = {
        "config": vars(args), "data": str(path), "steps": steps,
        "train_candidates": len(train), "holdout_candidates": len(hold),
        "label_dist_train": torch.bincount(lab_tr, minlength=steps).tolist(),
        "label_dist_holdout": torch.bincount(lab, minlength=steps).tolist(),
        "baselines": base, "runs": runs,
        "mean_top1": sum(top1) / len(top1),
        "mean_top1_ci": list(wilson(k_hits, len(hold))),
        "mean_bal_acc": sum(bal) / len(bal),
        "mean_poslogprob_top1": sum(pp_top1) / len(pp_top1),
        "pooled_vs_argmin_logprob": dict(vs_lp,
                                         p=sign_test(vs_lp["wins"],
                                                     vs_lp["losses"])),
        "pooled_vs_poslogprob": dict(vs_pp, p=sign_test(vs_pp["wins"],
                                                        vs_pp["losses"])),
        "confusion_pred_rows": conf,
        "pred_hist": torch.bincount(h["pred"], minlength=steps).tolist(),
        "logprob_hist": torch.bincount(lp_pred, minlength=steps).tolist(),
        "by_arm": by_arm,
    }
    tag = args.arch
    if args.frozen_proj:
        tag += f"_svd{args.frozen_proj}"
    if args.demean_position:
        tag += "_demeanpos"
    if args.linear_logprob:
        tag += "_lp"
    args.out = args.out or f"results/localizer_{path.stem}_{tag}.json"
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stateless judge: can a prefix predict the verifier's verdict?

The proposal: a judge that reviews the artifact every now and then, blind to the
writer's reasoning, and commits only above a threshold. This script tests the
predictive half of that idea on data we already have, with no model load and no
GPU work beyond tensor math.

Task. Given the first p steps of a chain, predict whether the finished chain is
correct. Label = `first_error is None`, which is the verifier's verdict, not a
human preference. p runs over 1..steps-1, so the judge is asked before the
answer exists.

Why this is the right label. The exam analogy is only safe if the external
examiner is exact and stationary. Ours is: the verifier. A judge trained on
"what a human might like" can be gamed; a judge trained on "what the verifier
will say" cannot change the target. The risk that remains is distribution
shift, not reward hacking, and that is measurable.

Blind by construction. The judge reads the hidden state at the prefix and the
step logprobs. It never reads generated commentary, so it cannot be persuaded by
the writer's own narrative. In arithmetic there is no commentary; in code this
choice will matter.

Three warnings from our own runs, encoded here:

  1. The gate was a no-op on recovery (gated == ungated exactly). A threshold on
     the existing score bought precision and skipped compute, not accuracy.
  2. tau = 0.35 for the two-tier rule was tuned in-sample and degraded on the
     frozen vault. So tau is chosen on TRAIN only and its transfer to holdout
     is reported as a first-class number.
  3. Prose feedback was inert (30/110 vs 30/110 placebo). The judge therefore
     gets no text, only state.

The operating point is chosen for precision, not balanced accuracy: a false
commit accepts a wrong chain, while a false reject costs one more verifier
call. The asymmetry is the whole point.

Usage:
    python experiments/global_judge.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from experiments.localizer import Standardizer, svd_basis


def auc(scores: torch.Tensor, y: torch.Tensor) -> float:
    """Rank AUC with ties counted as half. Positive class = chain is correct."""
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).float().mean()
    eq = (pos[:, None] == neg[None, :]).float().mean()
    return float(gt + 0.5 * eq)


def build(raw, prefix: int, split: str):
    """One example per chain: state at `prefix` steps, label = chain correct."""
    feats, lps, labels, idxs = [], [], [], []
    for r in raw["rows"]:
        if r["split"] != split or not r["complete"]:
            continue
        steps = r["steps"]
        if len(steps) != raw["config"]["steps"]:
            continue
        if any(s["feat"] is None or s["logprob"] is None for s in steps[:prefix]):
            continue
        h = torch.stack([s["feat"] for s in steps[:prefix]]).float()
        lp = torch.tensor([s["logprob"] for s in steps[:prefix]], dtype=torch.float32)
        feats.append(h)
        lps.append(lp)
        labels.append(1 if r["first_error"] is None else 0)
        idxs.append((r["idx"], r["cand"]))
    return feats, lps, torch.tensor(labels), idxs


def stack_feats(feats, lps):
    """Last-step state, mean state, and three logprob summaries."""
    h_last = torch.stack([f[-1] for f in feats])
    h_mean = torch.stack([f.mean(0) for f in feats])
    lp = torch.stack(lps)
    return h_last, h_mean, lp


def pick_tau(scores: torch.Tensor, y: torch.Tensor, target_prec: float) -> dict:
    """Lowest threshold whose commits have precision >= target on this split."""
    order = torch.argsort(scores, descending=True)
    best = {"tau": None, "precision": 0.0, "recall": 0.0, "commits": 0}
    tp = 0
    for k, i in enumerate(order.tolist(), start=1):
        tp += int(y[i] == 1)
        prec = tp / k
        if prec >= target_prec:
            best = {"tau": float(scores[i]), "precision": prec,
                    "recall": tp / max(int((y == 1).sum()), 1), "commits": k}
    return best


def apply_tau(scores: torch.Tensor, y: torch.Tensor, tau: float) -> dict:
    commit = scores >= tau
    k = int(commit.sum())
    tp = int((commit & (y == 1)).sum())
    return {"commits": k, "precision": tp / k if k else float("nan"),
            "recall": tp / max(int((y == 1).sum()), 1),
            "false_commits": int((commit & (y == 0)).sum())}


def center_within_task(last, mean, lp, ids):
    """Subtract each task's mean, leaving only within-task variation.

    If the judge is reading task difficulty rather than chain quality, this
    removes exactly that component and its holdout AUC should rise.
    """
    groups: dict = {}
    for i, (idx, _cand) in enumerate(ids):
        groups.setdefault(idx, []).append(i)
    last, mean, lp = last.clone(), mean.clone(), lp.clone()
    for g in groups.values():
        if len(g) < 2:
            continue
        last[g] -= last[g].mean(0, keepdim=True)
        mean[g] -= mean[g].mean(0, keepdim=True)
        lp[g] -= lp[g].mean(0, keepdim=True)
    return last, mean, lp


def task_mix(ids, y) -> dict:
    groups: dict = {}
    for i, (idx, _cand) in enumerate(ids):
        groups.setdefault(idx, []).append(i)
    out = {"all_correct": 0, "all_wrong": 0, "mixed": 0}
    for g in groups.values():
        k = int(y[g].sum())
        if k == len(g):
            out["all_correct"] += 1
        elif k == 0:
            out["all_wrong"] += 1
        else:
            out["mixed"] += 1
    return out


class LogReg(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


def train_logreg(x, y, epochs=400, lr=0.05, wd=1e-3):
    model = LogReg(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(x), y.float())
        loss.backward()
        opt.step()
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="results/arith_steps_h5.pt")
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--center", action="store_true",
                    help="remove each task's mean feature (relative judge only)")
    ap.add_argument("--out", default="results/global_judge_h5.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = Path(args.data)
    if not path.is_absolute():
        path = root / path
    raw = torch.load(path, weights_only=False)
    steps = int(raw["config"]["steps"])

    print(f"{'p':>2} {'n_tr':>5} {'n_ho':>5} {'base':>5} "
          f"{'AUC_lp_last':>11} {'AUC_lp_mean':>11} {'AUC_lp_min':>10} "
          f"{'AUC_judge':>9} {'tau':>7} {'ho_prec':>7} {'ho_rec':>6} {'fp':>3}")

    results = {}
    for p in range(1, steps):
        tr_f, tr_lp, tr_y, tr_id = build(raw, p, "train")
        ho_f, ho_lp, ho_y, ho_id = build(raw, p, "holdout")
        tr_last, tr_mean, tr_lp3 = stack_feats(tr_f, tr_lp)
        ho_last, ho_mean, ho_lp3 = stack_feats(ho_f, ho_lp)
        if args.center:
            tr_last, tr_mean, tr_lp3 = center_within_task(
                tr_last, tr_mean, tr_lp3, tr_id)
            ho_last, ho_mean, ho_lp3 = center_within_task(
                ho_last, ho_mean, ho_lp3, ho_id)
        if p == 1:
            print("task mix train:", task_mix(tr_id, tr_y),
                  "| holdout:", task_mix(ho_id, ho_y))

        sd = Standardizer(torch.stack(tr_f))
        proj = svd_basis(torch.stack(tr_f), 32)

        def make_x(last, mean, lp):
            # Standardizer expects [N, H, d]; add and drop a position axis.
            zl = sd(last.unsqueeze(1)).squeeze(1) @ proj
            zm = sd(mean.unsqueeze(1)).squeeze(1) @ proj
            return torch.cat([zl, zm, lp], dim=1)

        x_tr = make_x(tr_last, tr_mean, tr_lp3)
        x_ho = make_x(ho_last, ho_mean, ho_lp3)
        d_state = x_tr.shape[1] - 3
        variants = {
            "state": (x_tr[:, :d_state], x_ho[:, :d_state]),
            "logprob": (x_tr[:, d_state:], x_ho[:, d_state:]),
            "all": (x_tr, x_ho),
        }
        aucs = {}
        models = {}
        for name, (a, b) in variants.items():
            m = train_logreg(a, tr_y)
            with torch.no_grad():
                aucs[name] = {"holdout": auc(m(b), ho_y), "train": auc(m(a), tr_y)}
            models[name] = m

        model = models["all"]
        with torch.no_grad():
            s_tr = model(x_tr)
            s_ho = model(x_ho)

        # baselines: higher logprob = more likely correct
        b_last = auc(-ho_lp3[:, -1], ho_y)
        b_mean = auc(-ho_lp3.mean(1), ho_y)
        b_min = auc(-ho_lp3.min(1).values, ho_y)
        a_judge = aucs["all"]["holdout"]

        tau = pick_tau(s_tr, tr_y, args.target_precision)
        applied = apply_tau(s_ho, ho_y, tau["tau"]) if tau["tau"] is not None else \
            {"commits": 0, "precision": float("nan"), "recall": 0.0,
             "false_commits": 0}

        base = float(ho_y.float().mean())
        tau_s = f"{tau['tau']:>7.2f}" if tau["tau"] is not None else "   none"
        print(f"{p:>2} {len(tr_y):>5} {len(ho_y):>5} {base:>5.2f} "
              f"{b_last:>11.3f} {b_mean:>11.3f} {b_min:>10.3f} "
              f"{a_judge:>9.3f} {tau_s} "
              f"{applied['precision']:>7.3f} {applied['recall']:>6.3f} "
              f"{applied['false_commits']:>3} "
              f"| state {aucs['state']['holdout']:.3f} "
              f"(tr {aucs['state']['train']:.3f}) "
              f"lp {aucs['logprob']['holdout']:.3f} "
              f"(tr {aucs['logprob']['train']:.3f})")

        results[p] = {
            "n_train": int(len(tr_y)), "n_holdout": int(len(ho_y)),
            "holdout_base_rate": base,
            "auc": {"lp_last": b_last, "lp_mean": b_mean, "lp_min": b_min,
                    "judge": a_judge},
            "auc_variants": aucs,
            "tau_train": tau,
            "holdout_at_tau": applied,
            "transfer_gap": (tau["precision"] - applied["precision"]
                             if tau["tau"] is not None else None),
        }

    payload = {"target_precision": args.target_precision,
               "center_within_task": bool(args.center),
               "prefix": results}
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    best = max(results.items(), key=lambda kv: kv[1]["auc"]["judge"])
    print(f"best prefix: p={best[0]}, judge AUC {best[1]['auc']['judge']:.3f} "
          f"vs best logprob baseline "
          f"{max(best[1]['auc']['lp_last'], best[1]['auc']['lp_mean'], best[1]['auc']['lp_min']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

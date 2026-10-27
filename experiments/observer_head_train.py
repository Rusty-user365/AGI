"""Can a head trained directly on the selection objective rank repairs?

The seven shipped heads were trained on scaffold episode features and score
within-task AUC 0.425-0.550, all intervals containing chance. That is a fact
about those heads, not about whether repair correctness is learnable from this
backbone's features: the diagnostic also found an oracle gap of 0.409
(0.308 base rate to 0.717 best-of-4), so there is a large amount of signal in
the task for a selector to find.

This trains a predictor on cached features against the correctness label and
tests it the same way: within-task AUC on tasks the model has never seen.

Two feature sets are compared, because the dilution argument is testable:

  full      cat([pooled, cond, clock]) over the whole document -- what the
            seven heads actually consume. The document is 99% unchanged text,
            so a signal about the repaired block is diluted by construction.
  block     pooled hidden state of the repaired block alone. Out of
            distribution for the backbone, but the label is not diluted.

Both are evaluated on the same task split, so the comparison is paired.

A null here is decisive for the plan: if a predictor trained directly on the
selection objective, with the labels in hand, still cannot beat chance on
held-out tasks, then no routing logic built on this backbone can, and the
scratchpad should be abandoned rather than extended.

Usage:
    python experiments/observer_head_train.py --features results/observer_features_test.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from experiments.observer_diagnostic import auc, bootstrap, within_task_auc


def split_tasks(task: torch.Tensor, frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    uniq = task.unique()
    g = torch.Generator().manual_seed(seed)
    perm = uniq[torch.randperm(len(uniq), generator=g)]
    n = max(1, int(round(frac * len(perm))))
    return perm[:n], perm[n:]


def fit_logistic(Xtr: torch.Tensor, ytr: torch.Tensor, k: int,
                 weight_decay: float, steps: int, lr: float) -> nn.Module:
    model = nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos = float(ytr.sum())
    neg = float(len(ytr) - pos)
    pw = torch.tensor([neg / max(pos, 1.0)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(Xtr).squeeze(-1), ytr)
        loss.backward()
        opt.step()
    model.eval()
    return model


def evaluate(probs: torch.Tensor, y: torch.Tensor, task: torch.Tensor,
             reps: int, seed: int) -> dict:
    """Within-task AUC on the held-out tasks, with a cluster bootstrap CI."""
    per_task = []
    for t in task.unique().tolist():
        idx = (task == t).nonzero().flatten().tolist()
        rows = [{"correct": int(y[i]), "scores": {"pred": float(probs[i])},
                 "tokens": 0} for i in idx]
        per_task.append(rows)
    w = within_task_auc(per_task, "pred")
    bs = bootstrap(per_task, "pred", reps=reps, seed=seed)
    discordant = sum(1 for rows in per_task
                     if any(r["correct"] for r in rows)
                     and not all(r["correct"] for r in rows))
    hits = sum(1 for rows in per_task
               if max(rows, key=lambda r: r["scores"]["pred"])["correct"])
    base = float(y.float().mean())
    return {
        "within_task_auc": w,
        "within_task_ci": bs["within_ci"],
        "discordant_tasks": discordant,
        "top1": hits / max(len(per_task), 1),
        "base_rate": base,
        "gain_vs_base": hits / max(len(per_task), 1) - base,
        "oracle_top1": sum(1 for rows in per_task
                           if any(r["correct"] for r in rows)) / max(len(per_task), 1),
    }


def run_feature_set(name: str, X: torch.Tensor, y: torch.Tensor,
                    task: torch.Tensor, args,
                    parseable: torch.Tensor | None = None) -> dict:
    tr_tasks, ho_tasks = split_tasks(task, args.train_frac, args.seed)
    tr = torch.isin(task, tr_tasks)
    ho = torch.isin(task, ho_tasks)

    mu = X[tr].mean(0, keepdim=True)
    sd = X[tr].std(0, keepdim=True).clamp(min=1e-6)
    Z = (X - mu) / sd

    # PCA fitted on train only. 1544 dimensions against a few hundred rows
    # overfits without this, and the point is to give the signal its best shot.
    k = min(args.components, int(tr.sum()), Z.shape[1])
    _, _, Vh = torch.linalg.svd(Z[tr], full_matrices=False)
    P = Vh[:k].T
    Ztr, Zho = Z[tr] @ P, Z[ho] @ P
    pmu = Ztr.mean(0, keepdim=True)
    psd = Ztr.std(0, keepdim=True).clamp(min=1e-6)
    Ztr, Zho = (Ztr - pmu) / psd, (Zho - pmu) / psd

    model = fit_logistic(Ztr, y[tr].float(), k, args.weight_decay,
                         args.steps, args.lr)
    with torch.no_grad():
        ptr = model(Ztr).squeeze(-1)
        pho = model(Zho).squeeze(-1)

    out = {
        "features": name,
        "dim_in": X.shape[1],
        "components": k,
        "train": evaluate(ptr, y[tr], task[tr], args.bootstrap, args.seed),
        "heldout": evaluate(pho, y[ho], task[ho], args.bootstrap, args.seed),
    }
    # Fluency control. A block-only feature cannot know which thesis is
    # authoritative, so a high AUC may only mean "this text is not garbled".
    # Restricting evaluation to candidates that parse at all removes the
    # garbled ones and leaves the question the DAG actually needs answered:
    # among well-formed repairs, can the observer pick the correct one?
    if args.parseable_only and parseable is not None:
        m = parseable[ho] == 1
        if int(m.sum()) >= 8:
            out["heldout_parseable"] = evaluate(
                pho[m], y[ho][m], task[ho][m], args.bootstrap, args.seed)
            pc = out["heldout_parseable"]["within_task_ci"]
            out["passes_parseable"] = bool(
                pc and (pc[0] > 0.5 or pc[1] < 0.5)
                and out["heldout_parseable"]["discordant_tasks"] >= args.min_discordant)
    h = out["heldout"]
    ci = h["within_task_ci"]
    out["passes"] = bool(ci and (ci[0] > 0.5 or ci[1] < 0.5)
                         and h["discordant_tasks"] >= args.min_discordant)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="results/observer_features_test.pt")
    ap.add_argument("--components", type=int, default=32)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--min-discordant", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds; overrides --seed and reports "
                         "stability across splits")
    ap.add_argument("--out", default="results/observer_head_train.json")
    ap.add_argument("--permute", action="store_true",
                    help="shuffle labels across the whole dataset and rerun; "
                         "held-out AUC must collapse to 0.5")
    ap.add_argument("--parseable-only", action="store_true",
                    help="also report held-out AUC restricted to candidates "
                         "that parse, which removes the garbled-text confound")
    args = ap.parse_args()

    d = torch.load(args.features, weights_only=False)
    y, task = d["labels"].long(), d["task"].long()
    parseable = d.get("parseable")
    if parseable is not None:
        parseable = parseable.long()
    if args.permute:
        g = torch.Generator().manual_seed(1234)
        y = y[torch.randperm(len(y), generator=g)]
        print("labels permuted: this run should score 0.5")
    print(f"{len(y)} rows, {task.unique().numel()} tasks, base rate {float(y.float().mean()):.3f}, "
          f"oracle {d.get('oracle_top1'):.3f}")

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed])
    sets = (("full", "feats"), ("block", "block_feats"))
    runs: dict[str, list[dict]] = {name: [] for name, _ in sets}

    for seed in seeds:
        args.seed = seed
        print(f"\nseed {seed}")
        print(f"{'features':<8} {'dim':>5} {'k':>3} {'split':<8} {'withinAUC':>10} "
              f"{'95% CI':>16} {'disc':>5} {'top1':>6} {'gain':>7}")
        for name, key in sets:
            r = run_feature_set(name, d[key].float(), y, task, args,
                                parseable=parseable)
            r["seed"] = seed
            runs[name].append(r)
            for split in ("train", "heldout", "heldout_parseable"):
                s = r.get(split)
                if s is None:
                    continue
                ci = s["within_task_ci"]
                ci_s = "nan" if not ci else f"[{ci[0]:.3f},{ci[1]:.3f}]"
                wa = s["within_task_auc"]
                print(f"{name:<8} {r['dim_in']:>5} {r['components']:>3} {split:<8} "
                      f"{'nan' if wa is None else f'{wa:.3f}':>10} {ci_s:>16} "
                      f"{s['discordant_tasks']:>5} {s['top1']:>6.3f} "
                      f"{s['gain_vs_base']:>+7.3f}")
            print(f"{'':<8} passes={r['passes']}"
                  + (f" parseable_passes={r.get('passes_parseable')}"
                     if "passes_parseable" in r else ""))

    if len(seeds) > 1:
        print(f"\n{'features':<8} {'mean heldout AUC':>17} {'min':>7} {'max':>7} "
              f"{'passes':>7}/{len(seeds)}")
        for name, _ in sets:
            vals = [r["heldout"]["within_task_auc"] for r in runs[name]
                    if r["heldout"]["within_task_auc"] is not None]
            if not vals:
                print(f"{name:<8} {'nan':>17}")
                continue
            print(f"{name:<8} {sum(vals) / len(vals):>17.3f} {min(vals):>7.3f} "
                  f"{max(vals):>7.3f} "
                  f"{sum(1 for r in runs[name] if r['passes']):>7}/{len(seeds)}")

    Path(args.out).write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    for name, _ in sets:
        vals = [r["heldout"]["within_task_auc"] for r in runs[name]
                if r["heldout"]["within_task_auc"] is not None]
        if vals:
            print(f"best held-out: {name} mean AUC={sum(vals) / len(vals):.3f} "
                  f"passes={sum(1 for r in runs[name] if r['passes'])}/{len(seeds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

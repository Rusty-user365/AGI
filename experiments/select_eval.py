"""Does a learned observer beat cheap selectors at picking the correct candidate?

Consumes any corpus written by `arith_bench.py` or `wordbench.py`. The corpus
supplies, per candidate, a correctness label, a mean token log-probability, and
a pooled hidden state. This trains a head on the train tasks and scores it on
tasks it has never seen, against every baseline that costs nothing:

    greedy        one deterministic pass
    random        pick uniformly among admitted candidates (expected = p1)
    majority      self-consistency: the modal value, logprob tie-break
    logprob       the candidate the model itself is most confident in
    head          the learned observer
    oracle        any admitted candidate correct -- the ceiling

The head has to beat *all* of these to justify a control loop, not just random.
Self-consistency and logprob are the honest competitors: if a free statistic
matches the head, the loop is a reranker for something the model already knows.

Admission is the same for every arm: a candidate counts only if its completion
is a bare integer. A task with no admitted candidate is scored incorrect for
every arm, including the oracle.

Splits are by task and disjoint by construction, so this measures transfer to
unseen problems rather than memorisation.

Usage:
    python experiments/select_eval.py --corpus results/wordbench.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pairwise_auc(scores: torch.Tensor, y: torch.Tensor,
                 task: torch.Tensor) -> tuple[float | None, int]:
    """Concordance over (correct, incorrect) pairs sharing a task.

    Pooling pairs rather than averaging per-task AUCs keeps tasks with more
    discordant pairs from being outvoted by tasks with a single one.
    """
    num = 0.0
    den = 0
    discordant = 0
    for t in task.unique().tolist():
        m = task == t
        s, yy = scores[m], y[m]
        pos, neg = s[yy == 1], s[yy == 0]
        if len(pos) and len(neg):
            discordant += 1
        if not len(pos) or not len(neg):
            continue
        d = pos[:, None] - neg[None, :]
        num += float((d > 0).sum()) + 0.5 * float((d == 0).sum())
        den += d.numel()
    return (num / den if den else None), discordant


def cluster_bootstrap(scores: torch.Tensor, y: torch.Tensor, task: torch.Tensor,
                      reps: int, seed: int) -> tuple[float, float] | None:
    """Resample tasks, not candidates: candidates within a task are correlated."""
    uniq = task.unique()
    if len(uniq) < 2:
        return None
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(reps):
        pick = uniq[torch.randint(0, len(uniq), (len(uniq),), generator=g)]
        idx = torch.cat([(task == t).nonzero().flatten() for t in pick])
        a, _ = pairwise_auc(scores[idx], y[idx], task[idx])
        if a is not None:
            vals.append(a)
    if not vals:
        return None
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])


def fit_head_rank(Xtr, ytr, task_tr, k: int, wd: float, steps: int, lr: float,
                  seed: int, bs: int = 32):
    """ListNet: softmax over the candidates of one task, against normalised labels.

    Pointwise BCE treats each candidate as an independent example, but the
    decision is a comparison inside a task. This optimises the comparison
    directly, which is the objective a router actually needs.
    """
    torch.manual_seed(seed)
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp(min=1e-6)
    Z = (Xtr - mu) / sd
    k = min(k, int(Z.shape[0]), Z.shape[1])
    _, _, Vh = torch.linalg.svd(Z, full_matrices=False)
    P = Vh[:k].T
    Ztr = Z @ P
    pmu = Ztr.mean(0, keepdim=True)
    psd = Ztr.std(0, keepdim=True).clamp(min=1e-6)
    Ztr = (Ztr - pmu) / psd

    model = nn.Linear(k, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    uniq = task_tr.unique()
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        pick = uniq[torch.randint(0, len(uniq), (bs,), generator=g)]
        loss = torch.zeros(())
        used = 0
        for t in pick.tolist():
            m = task_tr == t
            if int(m.sum()) < 2:
                continue
            s = model(Ztr[m]).squeeze(-1)
            yt = ytr[m]
            tot = yt.sum()
            if tot <= 0:
                continue
            tgt = yt / tot
            loss = loss - (tgt * torch.log_softmax(s, 0)).sum()
            used += 1
        if used == 0:
            continue
        opt.zero_grad()
        (loss / used).backward()
        opt.step()
    model.eval()
    return {"model": model, "mu": mu, "sd": sd, "P": P, "pmu": pmu, "psd": psd}


def fit_head(Xtr, ytr, k: int, wd: float, steps: int, lr: float, seed: int):
    """Standardise, project with a train-only SVD, then logistic regression."""
    torch.manual_seed(seed)
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp(min=1e-6)
    Z = (Xtr - mu) / sd
    k = min(k, int(Z.shape[0]), Z.shape[1])
    _, _, Vh = torch.linalg.svd(Z, full_matrices=False)
    P = Vh[:k].T
    Ztr = Z @ P
    pmu = Ztr.mean(0, keepdim=True)
    psd = Ztr.std(0, keepdim=True).clamp(min=1e-6)
    Ztr = (Ztr - pmu) / psd

    model = nn.Linear(k, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pos = float(ytr.sum())
    neg = float(len(ytr) - pos)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / max(pos, 1.0)]))
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(model(Ztr).squeeze(-1), ytr).backward()
        opt.step()
    model.eval()
    return {"model": model, "mu": mu, "sd": sd, "P": P, "pmu": pmu, "psd": psd}


def apply_head(h, X):
    with torch.no_grad():
        Z = (X - h["mu"]) / h["sd"]
        return h["model"](((Z @ h["P"]) - h["pmu"]) / h["psd"]).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="results/wordbench.pt")
    ap.add_argument("--components", type=int, default=32)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permute", action="store_true",
                    help="shuffle labels before training; head AUC must fall to 0.5")
    ap.add_argument("--min-discordant", type=int, default=30)
    ap.add_argument("--use-logprob", action="store_true",
                    help="append the model's own logprob to the feature vector; "
                         "if this does not beat logprob alone, the hidden state "
                         "adds nothing the model does not already encode")
    ap.add_argument("--loss", default="bce", choices=["bce", "rank"],
                    help="bce = pointwise; rank = ListNet within each task")
    ap.add_argument("--out", default="results/select_eval.json")
    args = ap.parse_args()

    d = torch.load(args.corpus, weights_only=False)
    rows = [r for r in d["rows"] if r["feat"] is not None]
    dropped = len(d["rows"]) - len(rows)
    if dropped:
        print(f"dropped {dropped} rows with no hidden state")
    if not rows:
        print("empty corpus")
        return 1

    key = {(r["split"], r["idx"]) for r in rows}
    kid = {k: i for i, k in enumerate(sorted(key))}
    task = torch.tensor([kid[(r["split"], r["idx"])] for r in rows])
    is_tr = torch.tensor([r["split"] == "train" for r in rows])
    is_ho = torch.tensor([r["split"] == "holdout" for r in rows])
    if not bool(is_tr.any()) or not bool(is_ho.any()):
        print("need both train and holdout rows")
        return 1

    y = torch.tensor([r["correct"] for r in rows], dtype=torch.float)
    if args.permute:
        g = torch.Generator().manual_seed(1234)
        y = y[torch.randperm(len(y), generator=g)]
        print("labels permuted: head AUC should collapse to 0.5")
    lp = torch.tensor([r["logprob"] if r["logprob"] is not None else -1e9
                       for r in rows])
    admitted = torch.tensor([bool(r.get("clean", r.get("parseable", 1)))
                             for r in rows])
    X = torch.stack([r["feat"] for r in rows]).float()
    if args.use_logprob:
        X = torch.cat([X, lp.unsqueeze(-1) / 10.0], dim=-1)
    values = [r["value"] for r in rows]
    cand = torch.tensor([r["cand"] for r in rows])

    tkey = {kid[(r["split"], r["idx"])]: r["answer"] for r in rows}

    print(f"{len(rows)} rows, {len(key)} tasks, admitted "
          f"{float(admitted.float().mean()):.3f}")

    if args.loss == "rank":
        h = fit_head_rank(X[is_tr], y[is_tr], task[is_tr], args.components,
                          args.weight_decay, args.steps, args.lr, args.seed)
    else:
        h = fit_head(X[is_tr], y[is_tr], args.components, args.weight_decay,
                     args.steps, args.lr, args.seed)
    head_all = apply_head(h, X)

    res = {"config": vars(args), "rows": len(rows), "tasks": len(key),
           "admitted_rate": float(admitted.float().mean()), "arms": {}}

    for name, m in (("head_train", is_tr), ("head_holdout", is_ho)):
        sc, yy, tt = head_all[m], y[m], task[m]
        a, disc = pairwise_auc(sc, yy, tt)
        ci = cluster_bootstrap(sc, yy, tt, args.bootstrap, args.seed)
        res["arms"][name] = {"within_task_auc": a, "ci": ci,
                             "discordant_tasks": disc,
                             "tasks": int(tt.unique().numel())}
        ci_s = "nan" if not ci else f"[{ci[0]:.3f},{ci[1]:.3f}]"
        print(f"{name:<14} within-task AUC "
              f"{'nan' if a is None else f'{a:.3f}'} {ci_s}  discordant {disc}")

    # ---- accuracy arms, holdout only -------------------------------------
    ho_idx = is_ho.nonzero().flatten()
    t_ho = task[ho_idx]
    y_ho = y[ho_idx]
    v_ho = [values[i] for i in ho_idx.tolist()]
    lp_ho = lp[ho_idx]
    ad_ho = admitted[ho_idx]
    c_ho = cand[ho_idx]
    sc_ho = head_all[ho_idx]
    tasks_ho = t_ho.unique().tolist()
    n_tasks = len(tasks_ho)

    def acc(hits: int) -> dict:
        lo, hi = wilson(hits, n_tasks)
        return {"hits": hits, "n": n_tasks, "acc": hits / n_tasks,
                "wilson": [lo, hi]}

    def per_task_correct(scores):
        """Boolean correctness per holdout task, in tasks_ho order."""
        out = []
        for t in tasks_ho:
            m = (t_ho == t) & ad_ho
            if not m.any():
                out.append(False)
                continue
            idx = m.nonzero().flatten()
            best = idx[int(torch.argmax(scores[idx]))]
            out.append(bool(v_ho[best] is not None and v_ho[best] == tkey[t]))
        return torch.tensor(out, dtype=torch.bool)

    # random: expected accuracy of a uniform pick among admitted candidates
    exp_hits = 0.0
    for t in tasks_ho:
        m = (t_ho == t) & ad_ho
        if m.any():
            exp_hits += float(y_ho[m].mean())
    res["arms"]["random"] = {"expected_acc": exp_hits / n_tasks, "n": n_tasks}

    # self-consistency: modal value among admitted, logprob tie-break
    maj_ok = []
    for t in tasks_ho:
        m = (t_ho == t) & ad_ho
        if not m.any():
            maj_ok.append(False)
            continue
        idx = m.nonzero().flatten().tolist()
        counts: dict[int, int] = {}
        for j in idx:
            if v_ho[j] is not None:
                counts[v_ho[j]] = counts.get(v_ho[j], 0) + 1
        if not counts:
            maj_ok.append(False)
            continue
        best_v = max(counts, key=lambda v: (counts[v],
                                            max(lp_ho[j] for j in idx
                                                if v_ho[j] == v)))
        maj_ok.append(best_v == tkey[t])
    maj_ok = torch.tensor(maj_ok, dtype=torch.bool)

    ok = {
        "majority": maj_ok,
        "logprob": per_task_correct(lp_ho),
        "head": per_task_correct(sc_ho),
    }

    orc = 0
    for t in tasks_ho:
        m = (t_ho == t) & ad_ho
        if m.any() and any(v_ho[j] == tkey[t]
                           for j in m.nonzero().flatten().tolist()):
            orc += 1
    res["arms"]["oracle"] = acc(orc)

    g_ok = []
    for t in tasks_ho:
        m = (t_ho == t) & (c_ho == 0)
        if not m.any():
            g_ok.append(False)
            continue
        j = m.nonzero().flatten()[0]
        g_ok.append(bool(v_ho[j] is not None and v_ho[j] == tkey[t]))
    ok["greedy"] = torch.tensor(g_ok, dtype=torch.bool)

    for name in ("greedy", "majority", "logprob", "head"):
        res["arms"][name] = acc(int(ok[name].sum()))

    print(f"\n{'arm':<10} {'acc':>6} {'95% CI':>18}")
    for name in ("greedy", "random", "majority", "logprob", "head", "oracle"):
        a = res["arms"][name]
        if "expected_acc" in a:
            print(f"{name:<10} {a['expected_acc']:>6.3f} {'(expected)':>18}")
        else:
            lo, hi = a["wilson"]
            print(f"{name:<10} {a['acc']:>6.3f} {f'[{lo:.3f},{hi:.3f}]':>18}")

    # Paired comparison. A one-task gap in point accuracy means nothing; what
    # matters is how often the head wins and loses on the same tasks, which is
    # an exact sign test over discordant pairs.
    from math import comb

    def sign_test(a: torch.Tensor, b: torch.Tensor) -> dict:
        w = int(((a) & (~b)).sum())
        l = int(((~a) & (b)).sum())
        n = w + l
        if n == 0:
            return {"wins": 0, "losses": 0, "p": 1.0}
        k = min(w, l)
        p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
        return {"wins": w, "losses": l, "p": p}

    res["paired"] = {}
    print(f"\n{'head vs':<10} {'win':>5} {'loss':>5} {'p':>7}")
    for other in ("greedy", "majority", "logprob"):
        s = sign_test(ok["head"], ok[other])
        res["paired"][other] = s
        print(f"{other:<10} {s['wins']:>5} {s['losses']:>5} {s['p']:>7.3f}")

    head_ci = res["arms"]["head_holdout"]["ci"]
    disc = res["arms"]["head_holdout"]["discordant_tasks"]
    best_free = max(res["arms"]["logprob"]["acc"],
                    res["arms"]["majority"]["acc"],
                    res["arms"]["greedy"]["acc"])
    res["gate"] = {
        "auc_ci_excludes_chance": bool(
            head_ci and (head_ci[0] > 0.5 or head_ci[1] < 0.5)),
        "sufficient_discordant": bool(disc >= args.min_discordant),
        "beats_best_free_arm": bool(res["arms"]["head"]["acc"] > best_free),
        "beats_best_free_p_lt_0.05": bool(
            min(res["paired"][o]["p"] for o in res["paired"]) < 0.05
            and all(res["paired"][o]["wins"] >= res["paired"][o]["losses"]
                    for o in res["paired"])),
    }
    res["gate"]["passes"] = bool(all(res["gate"].values()))
    print(f"\ngate {res['gate']}")

    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

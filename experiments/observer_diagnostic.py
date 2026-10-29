"""Does any of the seven heads predict whether a candidate repair is correct?

This is the gate on the whole scratchpad idea. A rejection loop can only help if
the observer's score is correlated with being right. If the best head sits at
AUC 0.5, generate-and-reject cannot beat picking a candidate at random, and the
loop is decoration.

Measured per (task, candidate) pair:

  correct   whether the repaired block names the authoritative thesis
            (the same rule the pre-registered gate uses)
  score     each head's output on the repaired document, via
            forward_episodes -> build_features -> heads
  tokens    document length, kept because pooled hidden states track length

Three estimators, because they answer different questions:

  pooled AUC        ranks candidates across all tasks. Confounded by task
                    difficulty: a hard task contributes several wrong
                    candidates and they share a score level.
  within-task AUC   probability that a correct candidate outranks an incorrect
                    one *from the same prompt*. This is what best-of-N
                    selection actually uses, and it removes task difficulty by
                    construction. Primary statistic.
  ICC               share of score variance that is between-task. A fixed
                    threshold Gate is dominated by prompt difficulty when this
                    is high; a tournament is not.

Confound control: AUC of token count alone, and within-task AUC recomputed on
candidate pairs whose lengths differ by at most --len-tol tokens. If a head's
signal survives that, it is not a token counter.

Calibration: Brier and ECE, plus precision/coverage at score quantiles. A Gate
needs a threshold that holds across prompts; a reranker only needs ordering.

Decision rule (fixed before the run):

  rerank_usable   within-task AUC 95% CI excludes 0.5, in either direction.
                  A head at 0.35 is as usable as 0.65 once the sign is flipped.
  gate_usable     some threshold holds precision >= base_rate + 0.15 at
                  coverage >= 0.30, on the honest half of the split.

Reading a null result: these heads were trained on synthetic episode features,
not on (draft, correct) pairs, so an AUC near 0.5 means *these heads as trained*
cannot rank repairs. It does not mean no observer can: the next step would be
retraining the heads on cached features from correct and incorrect drafts, which
is cheap because the backbone stays frozen.

Usage:
    python experiments/observer_diagnostic.py --n 120 --candidates 4
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
from experiments._device import resolve as _resolve_device

from cgr.config import Config
from cgr.data.corpus import build_corpus
from cgr.data.documents import make_task_set
from cgr.data.grammar import blocked_spans, parse_block
from cgr.train import load_system, pick_device

CLOCK = [0.0, 0.0, 6.0, 10.0, 0.0, 1.0, 0.0, 0.6]
HEADS = ("value", "outcome_max", "tau", "disagreement",
         "epistemic_conf", "scope_conf", "source_conf")


# ---------------------------------------------------------------- statistics
def auc(scores: list[float], labels: list[int]) -> float | None:
    """Rank-based AUC with average ranks for ties. None if one class is empty."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(ranks[i] for i in range(len(scores)) if labels[i])
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def within_task_auc(per_task: list[list[dict]], key: str,
                    max_len_diff: int | None = None) -> float | None:
    """Concordance over (correct, incorrect) pairs drawn from the same task.

    max_len_diff restricts to pairs whose token counts are close, which is the
    direct control for a head that is secretly counting tokens.
    """
    num = 0.0
    den = 0
    for rows in per_task:
        pos = [r for r in rows if r["correct"]]
        neg = [r for r in rows if not r["correct"]]
        for p in pos:
            for n in neg:
                if max_len_diff is not None and abs(p["tokens"] - n["tokens"]) > max_len_diff:
                    continue
                den += 1
                sp, sn = p["scores"][key], n["scores"][key]
                num += 1.0 if sp > sn else (0.5 if sp == sn else 0.0)
    return num / den if den else None


def bootstrap(per_task: list[list[dict]], key: str, reps: int = 1000,
              seed: int = 0) -> dict:
    """Cluster bootstrap over tasks. Pairs within a task are not independent."""
    rng = random.Random(seed)
    pooled, within = [], []
    flat_scores = [r["scores"][key] for rows in per_task for r in rows]
    flat_labels = [int(r["correct"]) for rows in per_task for r in rows]
    n = len(per_task)
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        sample = [per_task[i] for i in idx]
        s = [r["scores"][key] for rows in sample for r in rows]
        y = [int(r["correct"]) for rows in sample for r in rows]
        a = auc(s, y)
        w = within_task_auc(sample, key)
        if a is not None:
            pooled.append(a)
        if w is not None:
            within.append(w)
    def pct(xs):
        if not xs:
            return None
        xs = sorted(xs)
        lo = xs[int(0.025 * (len(xs) - 1))]
        hi = xs[int(0.975 * (len(xs) - 1))]
        return [lo, hi]
    return {"pooled_ci": pct(pooled), "within_ci": pct(within),
            "n_obs": len(flat_labels), "base_rate": sum(flat_labels) / max(len(flat_labels), 1),
            "_flat": (flat_scores, flat_labels)}


def icc(per_task: list[list[dict]], key: str) -> float | None:
    """Between-task share of score variance. High means prompt difficulty dominates."""
    groups = [[r["scores"][key] for r in rows] for rows in per_task]
    groups = [g for g in groups if g]
    if len(groups) < 2:
        return None
    allv = [v for g in groups for v in g]
    gm = sum(allv) / len(allv)
    between = sum(len(g) * (sum(g) / len(g) - gm) ** 2 for g in groups)
    total = sum((v - gm) ** 2 for v in allv)
    return between / total if total > 0 else None


def platt(scores: list[float], labels: list[int], iters: int = 400) -> tuple[float, float]:
    """Fit p = sigmoid(a*logit_rank + b) by gradient descent. Rank-preserving."""
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    z = [(s - lo) / span for s in scores]
    a, b = 1.0, 0.0
    lr = 0.5
    n = max(len(z), 1)
    for _ in range(iters):
        ga = gb = 0.0
        for zi, yi in zip(z, labels):
            p = 1.0 / (1.0 + math.exp(-(a * zi + b)))
            ga += (p - yi) * zi
            gb += (p - yi)
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


def apply_platt(scores: list[float], ab: tuple[float, float],
                ref: tuple[float, float] | None = None) -> list[float]:
    a, b = ab
    lo, hi = ref if ref else (min(scores), max(scores))
    span = (hi - lo) or 1.0
    return [1.0 / (1.0 + math.exp(-(a * ((s - lo) / span) + b))) for s in scores]


def brier_ece(probs: list[float], labels: list[int], bins: int = 10) -> dict:
    n = max(len(probs), 1)
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / n
    ece = 0.0
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (k == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        ece += abs(conf - acc) * len(idx) / n
    return {"brier": brier, "ece": ece}


def threshold_table(scores: list[float], labels: list[int],
                    quantiles: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9)) -> list[dict]:
    """Precision / coverage for a 'accept if score >= t' rule at score quantiles."""
    order = sorted(scores)
    out = []
    for q in quantiles:
        t = order[min(int(q * (len(order) - 1)), len(order) - 1)]
        acc = [(s, y) for s, y in zip(scores, labels) if s >= t]
        if not acc:
            continue
        hits = sum(y for _, y in acc)
        out.append({"quantile": q, "threshold": t, "coverage": len(acc) / len(scores),
                    "precision": hits / len(acc)})
    return out


# ---------------------------------------------------------------- model side
def role_of(index: int, n_blocks: int) -> str:
    if index == 0:
        return "intro"
    if index == n_blocks - 1:
        return "conclusion"
    return "body"


@torch.no_grad()
def head_scores(system, tok, blocks: list[str]) -> tuple[dict[str, float], int, torch.Tensor]:
    ids, _ = blocked_spans(blocks, tok)
    t = torch.tensor(ids, device=system.device).unsqueeze(0)
    clock = torch.tensor([CLOCK], device=system.device)
    f = system.forward_episodes(t, clock)
    preds = f["preds"]
    one = lambda x: x.reshape(-1)[0]
    conf = lambda k: float(torch.softmax(preds[k], -1).max())
    scores = {
        "value": float(torch.sigmoid(one(preds["value"]))),
        "tau": float(torch.nn.functional.softplus(one(preds["tau"]))),
        "disagreement": float(torch.sigmoid(one(preds["disagreement"]))),
        "epistemic_conf": conf("epistemic"),
        "scope_conf": conf("scope"),
        "source_conf": conf("source"),
        "outcome_max": float(preds["outcome"].reshape(-1).max()),
    }
    return scores, len(ids), f["feats"][0].detach()


@torch.no_grad()
def block_feature(system, tok, text: str, max_len: int) -> torch.Tensor:
    """Pooled hidden state of the repaired block alone.

    The full-document feature is 99% unchanged text, so any signal about the
    repaired span is diluted by construction. This is the same backbone applied
    only to the block, which is out of distribution but gives the label a chance
    to be visible.
    """
    ids = tok.encode(text, add_bos=True, add_eos=True)[:max_len]
    t = torch.tensor(ids, device=system.device).unsqueeze(0)
    return system.backbone.pooled(t)[0].detach()


def load_partial(path: Path, args) -> list[list[dict]]:
    if not path.exists():
        return []
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if prev.get("status") != "partial" or prev.get("signature") != signature(args):
        return []
    return prev.get("per_task", [])


def signature(args) -> dict:
    return {"checkpoint": args.checkpoint, "split": args.split, "n": args.n,
            "candidates": args.candidates, "steps": args.steps,
            "seed": args.seed, "cond_mode": args.cond_mode}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",
                    default="checkpoints/diffugpt_5k_e10_cgr_fixup_reasoner/stage_b.pt")
    ap.add_argument("--split", default="test", choices=["test", "unseen"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cond-mode", default="none",
                    choices=["none", "reasoner", "learned"])
    ap.add_argument("--len-tol", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="candidate diversity; 0.0 makes every candidate "
                         "identical and the diagnostic meaningless")
    ap.add_argument("--order", default=None,
                    choices=[None, "left_to_right", "random"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--dump-features", default=None,
                    help="write cached backbone features and labels here, for "
                         "training a head directly on the selection objective")
    ap.add_argument("--out", default="results/observer_diagnostic.json")
    args = ap.parse_args()
    args.device = str(_resolve_device(args.device))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    system, cfg = load_system(args.checkpoint)
    system.to(device).eval()
    tok, splits = build_corpus(cfg, tok=system.tok)

    tasks = make_task_set(
        splits[args.split], args.n, seed=args.seed,
        fits=lambda texts: len(blocked_spans(texts, tok)[0]) <= cfg.model.max_len
    )["local"]

    dump = ({"feats": [], "block_feats": [], "labels": [], "parseable": [],
             "task": []}
            if args.dump_features else None)
    # Feature dumps are positional, so a partial resume would misalign them.
    per_task: list[list[dict]] = [] if dump else load_partial(out_path, args)
    if per_task:
        print(f"resuming with {len(per_task)}/{len(tasks)} tasks from {out_path}", flush=True)

    for ti in range(len(per_task), len(tasks)):
        task = tasks[ti]
        rows = []
        for ci in range(args.candidates):
            cand = system.repair(
                list(task.corrupt_blocks), task.target_block, mode="local",
                steps=args.steps, seed=args.seed + ci,
                reasoning_mode=args.cond_mode,
                temperature=args.temperature, order=args.order)
            role = role_of(task.target_block, len(cand))
            tid, _, _ = parse_block(cand[task.target_block], role)
            sc, ntok, feats = head_scores(system, tok, cand)
            rows.append({"correct": int(tid == task.authoritative),
                         "scores": sc, "tokens": ntok, "candidate": ci,
                         "text": cand[task.target_block],
                         "parseable": int(tid is not None)})
            if dump is not None:
                dump["feats"].append(feats.cpu())
                dump["block_feats"].append(
                    block_feature(system, tok, cand[task.target_block],
                                  cfg.model.max_len).cpu())
                dump["labels"].append(int(tid == task.authoritative))
                dump["parseable"].append(int(tid is not None))
                dump["task"].append(ti)
        per_task.append(rows)
        if (ti + 1) % 10 == 0 or ti + 1 == len(tasks):
            out_path.write_text(json.dumps(
                {"status": "partial", "signature": signature(args),
                 "per_task": per_task}, indent=1), encoding="utf-8")
            print(f"  {ti + 1}/{len(tasks)} tasks", flush=True)

    # ------------------------------------------------------------- readout
    # length as its own "head" so the same estimators measure the confound
    for rows in per_task:
        for r in rows:
            r["scores"]["tokens"] = float(r["tokens"])

    flat = [r for rows in per_task for r in rows]
    labels = [r["correct"] for r in flat]
    mean_correct = sum(labels) / max(len(labels), 1)
    oracle = sum(1 for rows in per_task if any(r["correct"] for r in rows)) / max(len(per_task), 1)

    len_auc = auc([r["scores"]["tokens"] for r in flat], labels)
    len_within = within_task_auc(per_task, "tokens")

    # Within-task AUC is only defined on discordant tasks: at least one correct
    # and at least one incorrect candidate. A run that is all-0 (temperature
    # too high) or all-1 (too easy) has plenty of tasks and no signal, so the
    # adequacy check counts discordant tasks, not evaluated tasks.
    discordant = sum(1 for rows in per_task
                     if any(r["correct"] for r in rows)
                     and not all(r["correct"] for r in rows))
    within_pairs = sum(len([r for r in rows if r["correct"]])
                       * len([r for r in rows if not r["correct"]])
                       for rows in per_task)

    # A selection experiment needs distinct candidates. Greedy decoding with a
    # deterministic commit order returns the same text for every seed, so this
    # is reported first: if it is 1.0, every AUC below is measuring noise.
    distinct = [len({r["text"] for r in rows}) for rows in per_task]
    diversity = {
        "mean_distinct_per_task": sum(distinct) / max(len(distinct), 1),
        "frac_tasks_single_candidate": sum(1 for d in distinct if d <= 1) / max(len(distinct), 1),
        "candidates": args.candidates,
        "temperature": args.temperature,
        "order": args.order or getattr(cfg.model, "infill_order", "left_to_right"),
        "source": ("token sampling" if args.temperature > 0 else "commit order only"),
    }
    if args.temperature == 0.0 and diversity["order"] == "left_to_right":
        print("WARNING: temperature 0 and left_to_right order give identical "
              "candidates for every seed; the AUCs below are undefined.")

    report = {
        "status": "complete",
        "signature": signature(args),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "cond_mode": args.cond_mode,
        "n_tasks": len(per_task),
        "candidates": args.candidates,
        "steps": args.steps,
        "pairs": len(flat),
        "mean_candidate_correct": mean_correct,
        "oracle_top1": oracle,
        "discordant_tasks": discordant,
        "discordant_frac": discordant / max(len(per_task), 1),
        "within_task_pairs": within_pairs,
        "diversity": diversity,
        "length_confound": {
            "auc": len_auc,
            "within_task_auc": len_within,
            "note": "token count used as a head; if this is far from 0.5 the "
                    "pooled AUCs are length-confounded and within-task is the "
                    "only honest number",
        },
        "rule": {
            "rerank_usable": "within-task AUC 95% CI excludes 0.5, either "
                             "direction, with at least 30 tasks",
            "gate_usable": "a threshold holds precision >= base_rate + 0.15 at "
                           "coverage >= 0.30 on the held-out half, with at "
                           "least 30 tasks",
        },
        "heads": {},
    }

    half = len(per_task) // 2
    fit_tasks, hold_tasks = per_task[:half], per_task[half:]

    print(f"\npairs={len(flat)}  base rate={mean_correct:.3f}  "
          f"oracle top1={oracle:.3f}")
    print(f"discordant tasks: {discordant}/{len(per_task)} "
          f"({discordant / max(len(per_task), 1):.2f})  "
          f"within-task pairs={within_pairs}  "
          f"{'adequate' if discordant >= 30 else 'INADEQUATE for a CI'}")
    print(f"diversity: {diversity['mean_distinct_per_task']:.2f} distinct of "
          f"{args.candidates} candidates ({diversity['source']}); "
          f"{diversity['frac_tasks_single_candidate']:.2f} of tasks had only one")
    print(f"length alone: pooled AUC={len_auc:.3f} "
          f"within-task AUC={len_within if len_within is None else round(len_within, 3)}")
    print(f"\n{'head':<16} {'pooled':>7} {'95% CI':>16} {'within':>7} {'95% CI':>16} "
          f"{'len-ctl':>8} {'top1':>6} {'gain':>7} {'ICC':>6}")

    for k in HEADS:
        s = [r["scores"][k] for r in flat]
        a = auc(s, labels)
        w = within_task_auc(per_task, k)
        wctl = within_task_auc(per_task, k, max_len_diff=args.len_tol)
        bs = bootstrap(per_task, k, reps=args.bootstrap, seed=args.seed)
        hits = sum(1 for rows in per_task
                   if max(rows, key=lambda r: r["scores"][k])["correct"])
        top1 = hits / max(len(per_task), 1)
        ab = platt([r["scores"][k] for rows in fit_tasks for r in rows],
                   [r["correct"] for rows in fit_tasks for r in rows])
        ref = (min([r["scores"][k] for rows in fit_tasks for r in rows]),
               max([r["scores"][k] for rows in fit_tasks for r in rows]))
        hold_scores = [r["scores"][k] for rows in hold_tasks for r in rows]
        hold_labels = [r["correct"] for rows in hold_tasks for r in rows]
        probs = apply_platt(hold_scores, ab, ref=ref)
        cal = brier_ece(probs, hold_labels)
        tab = threshold_table(hold_scores, hold_labels)
        base = sum(hold_labels) / max(len(hold_labels), 1)
        gate_ok = any(t["precision"] >= base + 0.15 and t["coverage"] >= 0.30 for t in tab)

        ci_p = bs["pooled_ci"]
        ci_w = bs["within_ci"]
        # A degenerate CI on a handful of tasks must not read as a positive,
        # and the count that matters is discordant tasks, not evaluated ones.
        enough = discordant >= 30
        rerank_ok = bool(enough and ci_w and (ci_w[0] > 0.5 or ci_w[1] < 0.5))
        report["heads"][k] = {
            "auc": a, "auc_ci": ci_p,
            "within_task_auc": w, "within_task_ci": ci_w,
            "within_task_auc_len_controlled": wctl,
            "top1": top1, "gain_vs_mean": top1 - mean_correct,
            "icc": icc(per_task, k),
            "brier": cal["brier"], "ece": cal["ece"],
            "calibration_note": "Platt fit on first half, scored on second half",
            "thresholds": tab, "heldout_base_rate": base,
            "rerank_usable": rerank_ok,
            "gate_usable": bool(enough and gate_ok),
            "sufficient_n": enough,
        }
        fmt = lambda x: "nan" if x is None else f"{x:.3f}"
        fmt_ci = lambda c: "nan" if not c else f"[{c[0]:.3f},{c[1]:.3f}]"
        print(f"{k:<16} {fmt(a):>7} {fmt_ci(ci_p):>16} {fmt(w):>7} {fmt_ci(ci_w):>16} "
              f"{fmt(wctl):>8} {top1:>6.3f} {top1 - mean_correct:>+7.3f} "
              f"{fmt(icc(per_task, k)):>6}")

    print(f"\n{'head':<16} {'Brier':>7} {'ECE':>6} {'prec@cov.5':>11} "
          f"{'prec@cov.3':>11} {'gate':>6} {'rerank':>7}")
    for k, h in report["heads"].items():
        t50 = next((t for t in h["thresholds"] if t["quantile"] == 0.5), None)
        t90 = next((t for t in h["thresholds"] if t["quantile"] == 0.9), None)
        p50 = t50["precision"] if t50 else float("nan")
        p90 = t90["precision"] if t90 else float("nan")
        print(f"{k:<16} {h['brier']:>7.3f} {h['ece']:>6.3f} {p50:>11.3f} "
              f"{p90:>11.3f} {str(h['gate_usable']):>6} {str(h['rerank_usable']):>7}")

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if dump is not None:
        dpath = Path(args.dump_features)
        dpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "feats": torch.stack(dump["feats"]),
            "block_feats": torch.stack(dump["block_feats"]),
            "labels": torch.tensor(dump["labels"], dtype=torch.long),
            "parseable": torch.tensor(dump["parseable"], dtype=torch.long),
            "task": torch.tensor(dump["task"], dtype=torch.long),
            "signature": signature(args),
            "base_rate": mean_correct,
            "oracle_top1": oracle,
        }, dpath)
        print(f"wrote features {dpath} "
              f"({len(dump['labels'])} rows, {dump['feats'][0].numel()}-dim)")
    print(f"\nwrote {args.out}")
    best = max((h for h in report["heads"].items()),
               key=lambda kv: abs((kv[1]["within_task_auc"] or 0.5) - 0.5))
    bw = best[1]["within_task_auc"]
    print(f"strongest within-task signal: {best[0]} "
          f"AUC={'nan' if bw is None else f'{bw:.3f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

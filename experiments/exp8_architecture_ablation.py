"""Paired unseen-thesis comparison: joint CGR training vs LoRA-only denoising.

This isolates the effect of the trained reasoner side channel while holding
the DiffuGPT backbone, tokenizer, synthetic corpus, optimizer budget and repair
decoder fixed. It does not test a DAG or claim that the gate participates in
decoding; the current architecture has no DAG repair path.

Example:
  python experiments/exp8_architecture_ablation.py \
    --cgr-checkpoint checkpoints/diffugpt/stage_b.pt \
    --baseline-checkpoint checkpoints/diffugpt_backbone/stage_b.pt \
    --split unseen
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cgr.config import Config
from cgr.data.corpus import build_corpus
from cgr.data.documents import make_task_set, matched_budgets
from cgr.data.grammar import blocked_spans
from cgr.metrics import paired_bootstrap
from cgr.model.system import CGRSystem
from cgr.train import load_system, pick_device
from experiments.exp4_repair_taxonomy import (edit_quality, n_tokens_of,
                                               run_model, score_document)


def load(path: str, device: torch.device):
    system, cfg = load_system(path)
    system.to(device).eval()
    return system, cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cgr-checkpoint", required=True)
    ap.add_argument("--baseline-checkpoint", required=True)
    ap.add_argument("--split", choices=["test", "heldout", "unseen"],
                    default="unseen")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    cgr, cfg = load(args.cgr_checkpoint, device)
    baseline, base_cfg = load(args.baseline_checkpoint, device)
    if cfg.model.backbone != "diffugpt" or base_cfg.model.backbone != "diffugpt":
        raise SystemExit("both checkpoints must use the DiffuGPT backbone")
    if cgr.tok.spec() != baseline.tok.spec():
        raise SystemExit("checkpoints use different tokenizers")
    if cfg.model.infill_order != base_cfg.model.infill_order:
        raise SystemExit("checkpoints use different infill orders")

    tok, splits = build_corpus(cfg, tok=cgr.tok)
    records = splits[args.split]
    if not records:
        raise SystemExit(f"split {args.split!r} is empty for this config")
    selection: dict = {}
    n_tok = n_tokens_of(tok)
    tasks = make_task_set(
        records, args.n, seed=args.seed,
        fits=lambda texts: len(blocked_spans(texts, tok)[0]) <= cfg.model.max_len,
        stats=selection)
    defects = ("local", "cross_block", "cross_block_partial")
    rows = {}
    for defect in defects:
        for method in ("one_block", "global"):
            key = f"{defect}/{method}"
            per_model = {"cgr": [], "backbone": [], "noop": []}
            budgets = {"cgr": [], "backbone": []}
            # On the unseen split a local defect replaces the only reason the
            # thesis has, and that thesis's reason pool never appeared in
            # training, so "specific" is undefined rather than zero. Mirror
            # exp4's rule instead of silently scoring 0.
            spec_defined = not (args.split == "unseen" and defect == "local")
            for task in tasks[defect]:
                per_model["noop"].append(score_document(
                    task, task.texts(), specificity_defined=spec_defined))
                for name, system in (("cgr", cgr), ("backbone", baseline)):
                    spent = matched_budgets(task, n_tok)[method]
                    out, changed = run_model(
                        system, task, method, spent, n_tok, tok,
                        steps=args.steps, seed=args.seed,
                        use_reasoner=(name == "cgr"))
                    score = score_document(task, out,
                                           specificity_defined=spec_defined)
                    score["blocks_changed"] = len(changed)
                    score["budget"] = spent.as_dict()
                    # Preservation is measured on the blocks the editor was
                    # allowed to leave alone. Without it, "rewrote everything"
                    # and "repaired the defect" can look identical.
                    eq = edit_quality(system, task, task.texts(), out, tok,
                                      changed)
                    score.update(eq)
                    budgets[name].append(spent.as_dict())
                    per_model[name].append(score)
            rows[key] = {
                "n": len(per_model["cgr"]),
                "cgr_mean_c_target": _mean(per_model["cgr"], "c_target"),
                "backbone_mean_c_target": _mean(per_model["backbone"], "c_target"),
                "noop_mean_c_target": _mean(per_model["noop"], "c_target"),
                "cgr_mean_blocks_changed": _mean(budgets["cgr"], "blocks_used"),
                "backbone_mean_blocks_changed": _mean(budgets["backbone"], "blocks_used"),
                "cgr_mean_tokens_spent": _mean(budgets["cgr"], "tokens_used"),
                "backbone_mean_tokens_spent": _mean(budgets["backbone"], "tokens_used"),
                "c_target_cgr_minus_backbone": _paired(
                    per_model, "c_target", args.seed),
                "q_target_cgr_minus_backbone": _paired(
                    per_model, "q_target", args.seed),
                "cgr_mean_preservation": _mean(per_model["cgr"], "preservation"),
                "backbone_mean_preservation": _mean(per_model["backbone"], "preservation"),
                "preservation_cgr_minus_backbone": _paired(
                    per_model, "preservation", args.seed),
                "cgr_mean_improvement": _mean(per_model["cgr"], "improvement"),
                "backbone_mean_improvement": _mean(per_model["backbone"], "improvement"),
                "improvement_cgr_minus_backbone": _paired(
                    per_model, "improvement", args.seed),
                "cgr_doc_consistent_rate": _doc_consistent_rate(per_model["cgr"]),
                "backbone_doc_consistent_rate": _doc_consistent_rate(per_model["backbone"]),
                "noop_doc_consistent_rate": _doc_consistent_rate(per_model["noop"]),
                "cgr_anchor_rate": _mean(per_model["cgr"], "anchor"),
                "backbone_anchor_rate": _mean(per_model["backbone"], "anchor"),
            }

    payload = {
        "experiment": "exp8_architecture_ablation",
        "preset": "diffugpt", "seed": args.seed, "split": args.split,
        "selection": selection, "requested_n": args.n,
        "cgr_checkpoint": args.cgr_checkpoint,
        "baseline_checkpoint": args.baseline_checkpoint,
        "comparison": "trained CGR reasoner conditioning vs LoRA-only masked-denoising baseline",
        "limits": [
            "The decoder and edit scopes are identical between conditions.",
            "The gate does not participate in this repair comparison; no DAG is implemented.",
            "Unseen-split specificity for local defects is not evaluated here.",
        ],
        "conditions": rows,
    }
    out = Path(args.out or f"results/exp8_architecture_ablation_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _mean(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _paired(per_model, key, seed):
    """Paired bootstrap over examples where both conditions have a value.

    ``q_target`` is deliberately undefined on the unseen local condition, so
    the pair is dropped rather than coerced to zero.
    """
    a, b = [], []
    for base_row, cgr_row in zip(per_model["backbone"], per_model["cgr"]):
        if base_row.get(key) is None or cgr_row.get(key) is None:
            continue
        a.append(base_row[key])
        b.append(cgr_row[key])
    if not a:
        return None
    return paired_bootstrap(a, b, seed=seed)


def _doc_consistent_rate(rows):
    """Fraction of edited documents that parse and agree as a whole.

    This is stricter than a per-block parse rate: counts, indices, reason
    fields and the thesis must all be consistent across the document.
    """
    return sum(bool(r["consistent"]) for r in rows) / len(rows) if rows else None


if __name__ == "__main__":
    raise SystemExit(main())

"""Why does a repair attempt fail? A per-block failure taxonomy.

`check_backbone.py` reports one number (on-thesis rate). That number cannot
distinguish four very different failures:

    empty / mask leak   the decoder left the mask token in the span, or the
                        span decoded to nothing
    unparseable         text was produced, but it does not match any template
    wrong thesis        it parses, but supports a thesis other than T*
    not specific        it parses and supports T*, but the reason is not drawn
                        from T*'s pool (the local-quality target)
    success             parses, supports T*, and is specific

The distinction matters because the fixes are different: a mask leak is a
decoder bug, unparseable output is a format/objective problem, and
"not specific" is a content problem that the on-thesis gate cannot see.

The script also checks the plumbing the gate assumes: that the target span is
non-empty, that the decoded target is the block the parser is asked about, and
that unmasked blocks are byte-identical.

    python scripts/diagnose_repair.py --checkpoint checkpoints/diffugpt_5k_e10_cgr/stage_b.pt \
        --split test --n 100 --steps 8 --dump 10
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cgr.data.corpus import build_corpus
from cgr.data.documents import (anchor_ok, block_quality, coherence,
                                make_task_set, task_from_blocks)
from cgr.data.grammar import (THESES, THESIS_POOL, blocked_spans,
                              is_consistent, parse_block)
from cgr.train import load_system, pick_device


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Honest at small n and near 0, unlike normal."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def classify(gen: str, target_ids: list[int], tok, task, idx: int) -> dict:
    """Assign exactly one failure category to one generated block."""
    tid, reason, ridx = parse_block(gen, "body")
    clean = task.blocks[idx]
    thesis = THESES[task.authoritative][0]
    pool = THESIS_POOL[task.authoritative]
    norm = " ".join(gen.split())
    mask_leaks = sum(1 for t in target_ids if t == tok.mask_id)
    rec = {
        "gen": gen,
        "clean": task.clean_blocks[idx],
        "corrupt": clean.text,
        "thesis_id": tid,
        "reason": reason,
        "reason_index": ridx,
        "expected_thesis": task.authoritative,
        "expected_reason_index": clean.reason_index,
        "mask_leaks": mask_leaks,
        "gen_tokens": len(tok.encode(gen, add_bos=False, add_eos=False)),
        "clean_tokens": len(tok.encode(task.clean_blocks[idx],
                                      add_bos=False, add_eos=False)),
        "exact_copy": gen.strip() == task.clean_blocks[idx].strip(),
        "has_reason_word": "reason" in gen.lower(),
        # Format vs content, measured independently of the exact parser. If the
        # thesis phrase and a pool reason both appear, the model knew *what* to
        # say and failed on *how* to say it. That distinction decides whether
        # the fix is more training, a different objective, or constrained
        # decoding -- and it is not visible in the parse rate.
        "content_ok": bool(thesis in norm and any(r in norm for r in pool)),
        "prefix_ok": bool(re.match(r"^(?:reason|point)\s+\d+\s*:", norm)),
        "ends_period": norm.endswith("."),
    }
    if not gen.strip():
        cat = "empty"
    elif mask_leaks:
        cat = "mask_leak"
    elif tid is None:
        cat = "unparseable"
    elif tid != task.authoritative:
        cat = "wrong_thesis"
    elif ridx != clean.reason_index:
        cat = "wrong_slot"
    elif reason not in THESIS_POOL[task.authoritative]:
        cat = "not_specific"
    else:
        cat = "success"
    rec["category"] = cat
    return rec


@torch.no_grad()
def teacher_forced_span(system, tok, ids: list[int], s: int, e: int,
                        cond) -> dict:
    """Top-1 on the target span with the *clean* block visible to the model.

    The free-running decoder commits one token at a time, so a format error
    early in the span can cascade. This probe masks the whole span in one pass
    and asks the model to predict every token from bidirectional context. It
    separates "the model does not know the block" from "the sampler drifts".
    """
    t = torch.tensor(ids, device=system.device).unsqueeze(0)
    masked = t.clone()
    masked[:, s:e] = tok.mask_id
    logits = system.backbone(masked, cond)
    # logits[:, j] predicts token j + 1. Targets are ids[s:e], so the
    # predicting positions are s-1 .. e-2.
    lo = max(s - 1, 0)
    pred = logits[0, lo:e - 1].argmax(-1)
    tgt = t[0, s:e]
    correct = pred == tgt
    nll = torch.nn.functional.cross_entropy(
        logits[0, lo:e - 1].float(), tgt, reduction="mean")
    return {
        "tf_top1": float(correct.float().mean().item()),
        "tf_exact": bool(correct.all().item()),
        "tf_nll": float(nll.item()),
        "tf_n": int(tgt.numel()),
    }


@torch.no_grad()
def run(system, cfg, tok, tasks, steps: int, seed: int,
        use_reasoner: bool) -> list[dict]:
    rows = []
    for task in tasks:
        idx = task.edited[0]
        blocks = task.texts()
        ids, spans = blocked_spans(blocks, tok)
        if not (0 <= idx < len(spans)):
            rows.append({"category": "plumbing_bad_index", "idx": idx,
                         "n_spans": len(spans)})
            continue
        s, e = spans[idx]
        if e <= s:
            rows.append({"category": "plumbing_empty_span", "idx": idx,
                         "span": [s, e]})
            continue
        t = torch.tensor(ids, device=system.device).unsqueeze(0)
        cond = system.encode(t)[1] if use_reasoner else None
        order = getattr(cfg.model, "infill_order", "confidence")
        gen_ids = system.backbone.infill(
            t, [(s, e)], cond=cond, steps=steps, seed=seed, order=order)
        flat = gen_ids[0].tolist()
        target_ids = flat[s:e]
        gen = tok.decode(target_ids)

        # Re-parse the *whole* produced document, exactly as the scorer does,
        # so a decode bug that shifts later spans is visible rather than hidden
        # behind a block-level string comparison.
        new_blocks = []
        _, new_spans = blocked_spans(blocks, tok)
        for (a, b) in new_spans:
            new_blocks.append(tok.decode(flat[a:b]))
        parsed = task_from_blocks(new_blocks, task.authoritative).blocks
        rec = classify(gen, target_ids, tok, task, idx)
        rec.update(teacher_forced_span(system, tok, ids, s, e, cond))
        rec.update({
            "span": [s, e],
            "span_tokens": e - s,
            "n_blocks_out": len(new_blocks),
            "n_blocks_in": len(blocks),
            "anchor": bool(anchor_ok(blocks, new_blocks, task.locked_indices)),
            "doc_consistent": bool(is_consistent(new_blocks)),
            "c_edit": coherence(parsed, task.authoritative)["c_edit"],
            "block_quality": block_quality(parsed[idx], task.authoritative),
        })
        rows.append(rec)
    return rows


def summarise(rows: list[dict], n_requested: int) -> dict:
    cats = Counter(r.get("category", "unknown") for r in rows)
    n = len(rows)
    on_thesis = sum(1 for r in rows if r.get("thesis_id") == r.get("expected_thesis")
                    and r.get("thesis_id") is not None)
    specific = sum(1 for r in rows if r.get("category") == "success")
    parseable = sum(1 for r in rows if r.get("thesis_id") is not None)
    lo_t, hi_t = wilson(on_thesis, n)
    lo_s, hi_s = wilson(specific, n)
    lo_p, hi_p = wilson(parseable, n)

    def rate(key: str) -> float | None:
        return sum(1 for r in rows if r.get(key)) / n if n else None

    return {
        "n": n, "n_requested": n_requested,
        "categories": dict(cats),
        "parseable": parseable / n if n else None,
        "parseable_ci95": [lo_p, hi_p],
        "on_thesis": on_thesis / n if n else None,
        "on_thesis_ci95": [lo_t, hi_t],
        "specific": specific / n if n else None,
        "specific_ci95": [lo_s, hi_s],
        "content_ok": rate("content_ok"),
        "prefix_ok": rate("prefix_ok"),
        "ends_period": rate("ends_period"),
        "anchor": rate("anchor"),
        "doc_consistent": rate("doc_consistent"),
        "mask_leaks": sum(1 for r in rows if r.get("mask_leaks")),
        "empty": sum(1 for r in rows if r.get("category") == "empty"),
        "mean_gen_tokens": (sum(r.get("gen_tokens", 0) for r in rows) / n
                            if n else None),
        "mean_clean_tokens": (sum(r.get("clean_tokens", 0) for r in rows) / n
                              if n else None),
        "mean_span_tokens": (sum(r.get("span_tokens", 0) for r in rows) / n
                             if n else None),
        "tf_top1": (sum(r.get("tf_top1", 0.0) for r in rows) / n
                    if n else None),
        "tf_exact": rate("tf_exact"),
        "tf_nll": (sum(r.get("tf_nll", 0.0) for r in rows) / n
                   if n else None),
        "mean_c_edit": (sum(r.get("c_edit", 0.0) for r in rows) / n
                        if n else None),
        "mean_block_quality": (sum(r.get("block_quality", 0) for r in rows) / n
                               if n else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["test", "heldout", "unseen"],
                    default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--use-reasoner", action="store_true")
    ap.add_argument("--dump", type=int, default=5,
                    help="how many failure examples to print")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = pick_device(args.device)
    system, cfg = load_system(args.checkpoint)
    system.to(dev).eval()
    tok, splits = build_corpus(cfg, tok=system.tok)
    records = splits[args.split]
    if not records:
        raise SystemExit(f"split {args.split!r} is empty")

    stats: dict = {}
    tasks = make_task_set(
        records, args.n, seed=args.seed,
        fits=lambda texts: len(blocked_spans(texts, tok)[0])
        <= cfg.model.max_len, stats=stats)
    local = tasks["local"]
    rows = run(system, cfg, tok, local, args.steps, args.seed,
               args.use_reasoner)
    s = summarise(rows, args.n)

    print(f"checkpoint   {args.checkpoint}")
    print(f"split        {args.split}   (selection {stats})")
    print(f"n            {s['n']} (requested {args.n})")
    print(f"steps        {args.steps}   reasoner={args.use_reasoner}")
    print(f"span/clean   {s['mean_span_tokens']:.1f} / "
          f"{s['mean_clean_tokens']:.1f} tokens")
    print(f"parseable    {s['parseable']:.3f}  "
          f"[{s['parseable_ci95'][0]:.3f}, {s['parseable_ci95'][1]:.3f}]")
    print(f"on-thesis    {s['on_thesis']:.3f}  "
          f"[{s['on_thesis_ci95'][0]:.3f}, {s['on_thesis_ci95'][1]:.3f}]")
    print(f"specific     {s['specific']:.3f}  "
          f"[{s['specific_ci95'][0]:.3f}, {s['specific_ci95'][1]:.3f}]")
    print(f"format       prefix {s['prefix_ok']:.3f}  "
          f"period {s['ends_period']:.3f}  content {s['content_ok']:.3f}")
    print(f"teacher-forced span top1 {s['tf_top1']:.3f}  "
          f"exact {s['tf_exact']:.3f}  nll {s['tf_nll']:.3f}")
    print(f"anchor       {s['anchor']:.3f}   doc-consistent "
          f"{s['doc_consistent']:.3f}   c_edit {s['mean_c_edit']:.3f}")
    print("categories   " + json.dumps(s["categories"]))

    fails = [r for r in rows if r.get("category") != "success"]
    for r in fails[:args.dump]:
        print("\n--- " + r.get("category", "?") + " ---")
        print(f"span {r.get('span')} ({r.get('span_tokens')} tokens), "
              f"mask_leaks={r.get('mask_leaks')}, "
              f"gen_tokens={r.get('gen_tokens')}")
        print("clean  :", r.get("clean"))
        print("corrupt:", r.get("corrupt"))
        print("gen    :", repr(r.get("gen")))

    payload = {
        "experiment": "diagnose_repair",
        "checkpoint": args.checkpoint, "split": args.split,
        "n": args.n, "steps": args.steps, "seed": args.seed,
        "use_reasoner": bool(args.use_reasoner),
        "selection": stats, "summary": s,
        "failures": fails[:max(args.dump, 20)],
    }
    out = Path(args.out or
               f"results/diagnose_{Path(args.checkpoint).parent.name}_"
               f"{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

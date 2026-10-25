"""Diagnostic: does the pre-trained backbone work on *this* corpus?

Run before any repair experiment, and again after fine-tuning. It answers three
questions in order, because a failure at step 1 makes the later numbers
uninterpretable:

  1. Does the checkpoint load, and is the side channel an exact identity?
  2. Can the backbone denoise a *synthetic* document at the training mask
     ratio? (Natural-text accuracy is reported for context, not as the gate.)
  3. Can it infill one block of a synthetic document well enough for the
     parser to accept the result? This is the editor floor.

The gate the paper uses is `informative = infill_on_thesis >= 0.2` in
`exp4`. This script is the cheaper leading indicator: if the parse rate on a
single masked body block is ~0, no repair comparison is meaningful yet.

    python scripts/check_backbone.py --preset diffugpt
    python scripts/check_backbone.py --preset diffugpt --checkpoint checkpoints/diffugpt/stage_b.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cgr.config import Config
from cgr.data.corpus import build_corpus
from cgr.data.documents import make_task_set
from cgr.data.grammar import blocked_spans, parse_block
from cgr.model.system import CGRSystem
from cgr.tokenizer import make_tokenizer
from cgr.train import pick_device

NATURAL = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Remote work raises output because commuting hours become working hours.",
    "Street trees cool cities by blocking direct sunlight and lowering surface heat.",
    "Peer review slows discovery: queues add months per round and reviewers are scarce.",
    "Open data improves policy because outside checks find errors and models can be reproduced.",
    "Slow feedback harms learning because errors are corrected only after the exam.",
    "Vertical farms waste energy because lighting replaces free sunlight all year.",
    "A city that plants trees along its streets lowers surface temperature in summer.",
]


def load(args) -> tuple[CGRSystem, Config]:
    if args.checkpoint:
        from cgr.train import load_system
        return load_system(args.checkpoint)
    cfg = Config.preset(args.preset)
    cfg.model.dropout = 0.0
    tok = make_tokenizer(cfg)
    if tok is None:  # the canvas builds its vocabulary from the corpus
        tok, _ = build_corpus(cfg)
    return CGRSystem(cfg, tok), cfg


@torch.no_grad()
def denoise_acc(model, tok, texts, ratio=0.15, seed=0, max_len=128):
    """Top-1 on masked positions, shift-aware, restricted to real tokens."""
    g = torch.Generator().manual_seed(seed)
    correct = total = 0
    for t in texts:
        ids = torch.tensor(
            tok.encode(t, add_bos=False, add_eos=False)[:max_len],
            device=next(model.parameters()).device)
        if ids.numel() < 4:
            continue
        x = ids.unsqueeze(0)
        mask = (torch.rand(x.shape, generator=g) < ratio).to(x.device)
        mask[:, 0] = False
        noisy = x.masked_fill(mask, tok.mask_id)
        logits = model(noisy)
        pred = logits[:, :-1].argmax(-1)
        sel = mask[:, 1:]
        correct += (pred[sel] == x[:, 1:][sel]).sum().item()
        total += int(sel.sum())
    return correct / max(total, 1), total


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Honest at small n and near 0, unlike normal."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@torch.no_grad()
def block_infill_rate(system, tok, tasks, steps=8, seed=0,
                      use_reasoner=False):
    """Counts of one-block repairs that parse, and that parse on-thesis."""
    parseable = on_thesis = 0
    for t in tasks:
        blocks = list(t.corrupt_blocks)
        out = system.repair(blocks, t.target_block, mode="local",
                            steps=steps, seed=seed,
                            use_reasoner=use_reasoner)
        if t.target_block == 0:
            role = "intro"
        elif t.target_block == len(blocks) - 1:
            role = "conclusion"
        else:
            role = "body"
        tid, _, _ = parse_block(out[t.target_block], role)
        if tid is not None:
            parseable += 1
            on_thesis += int(tid == t.authoritative)
    return parseable, on_thesis, len(tasks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="diffugpt")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test",
                    choices=["test", "unseen"],
                    help="evaluate the ordinary test split or the held-out "
                         "unseen-thesis split; never mix them")
    ap.add_argument("--use-reasoner", action="store_true",
                    help="condition the infill on the CGR reasoner, as the "
                         "deployed CGR editor does; the default is the "
                         "reasoner-free path used for the original gate")
    ap.add_argument("--device", default="auto",
                    help="cpu, cuda, or auto")
    ap.add_argument("--json-out", default=None,
                    help="write the gate numbers to a JSON file so the paper "
                         "table can cite them instead of a console log")
    args = ap.parse_args()

    system, cfg = load(args)
    system.to(pick_device(args.device))
    system.eval()
    tok = system.tok
    print(f"preset      {cfg.preset_name}")
    print(f"backbone    {cfg.model.backbone}  d_model={cfg.model.d_model} "
          f"layers={cfg.model.n_layers} max_len={cfg.model.max_len}")
    print(f"tokenizer   {cfg.model.tokenizer}  vocab={tok.vocab_size}")
    print(f"params      {system.n_params()}")

    # 1. identity checks on whatever weights are loaded
    ids = torch.tensor(tok.encode("this essay argues that street trees cool "
                                  "cities with 3 reasons ."),
                       device=system.device)[:cfg.model.max_len]
    ids = ids.unsqueeze(0)
    with torch.no_grad():
        plain = system.backbone(ids).clone()
        cond = torch.zeros(1, cfg.model.d_model, device=system.device)
        steered = system.backbone(ids, cond)
        snap = system.plastic_snapshot()
        system.backbone.plastic_a.add_(
            torch.randn_like(system.backbone.plastic_a))
        after = system.backbone(ids)
        system.restore_plastic(snap)
    print(f"\ncond identity      {float((plain - steered).abs().max()):.3e} "
          f"(0.0 = side channel is inert at load)")
    print(f"plastic perturbation delta {float((plain - after).abs().max()):.3e} "
          f"(output change after adding noise to plastic adapter)")

    # 2. denoising
    _, splits = build_corpus(cfg, tok=tok)
    corpus_texts = [" ".join(r.blocks) for r in splits["test"][:200]]
    denoise = {}
    for name, texts in (("synthetic", corpus_texts), ("natural", NATURAL)):
        acc, n = denoise_acc(system.backbone, tok, texts, seed=args.seed,
                             max_len=cfg.model.max_len)
        denoise[name] = {"top1": acc, "masked_tokens": n}
        print(f"denoise top1 {name:<9} {acc:.3f}  (n={n} masked tokens)")

    # 3. the editor floor
    tasks = make_task_set(
        splits[args.split], args.n, seed=args.seed,
        fits=lambda texts: len(blocked_spans(texts, tok)[0])
        <= cfg.model.max_len)
    local = tasks["local"]
    parseable, on_thesis, n = block_infill_rate(
        system, tok, local, steps=args.steps, seed=args.seed,
        use_reasoner=args.use_reasoner)
    p_lo, p_hi = wilson(parseable, n)
    t_lo, t_hi = wilson(on_thesis, n)
    print(f"\nsplit {args.split}  reasoner={args.use_reasoner}")
    print(f"local block infill: parseable {parseable / max(n, 1):.3f} "
          f"[{p_lo:.3f}, {p_hi:.3f}]  "
          f"on-thesis {on_thesis / max(n, 1):.3f} "
          f"[{t_lo:.3f}, {t_hi:.3f}]  (n={n})")
    print("gate: on-thesis >= 0.2 before any repair comparison is meaningful")

    if args.json_out:
        payload = {
            "experiment": "editor_gate",
            "checkpoint": args.checkpoint,
            "preset": cfg.preset_name,
            "split": args.split,
            "use_reasoner": bool(args.use_reasoner),
            "n": n,
            "steps": args.steps,
            "seed": args.seed,
            "parseable": parseable,
            "on_thesis": on_thesis,
            "parseable_rate": parseable / max(n, 1),
            "on_thesis_rate": on_thesis / max(n, 1),
            "parseable_ci95": [p_lo, p_hi],
            "on_thesis_ci95": [t_lo, t_hi],
            "floor": 0.20,
            "clears_floor": (on_thesis / max(n, 1)) >= 0.20,
            "denoise": denoise,
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

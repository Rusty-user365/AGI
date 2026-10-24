"""Construct every preset, forward one real batch, report the true parameter count.

This exists because the draft claimed ~5M params and a compute budget that was
never measured. The point of this script is to make the number in the paper a
number produced by code rather than by memory.

The data config is shrunk for the check (we are validating shapes, not
throughput), but model/train config are used exactly as the preset defines them.
Run: python scripts/check_presets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgr.config import Config                      # noqa: E402
from cgr.data.corpus import build_corpus           # noqa: E402
from cgr.model.system import CGRSystem             # noqa: E402
from cgr.train import build_episodes, collate      # noqa: E402


def check(name: str) -> dict:
    cfg = Config.preset(name)
    # shrink only the data volume; keep the model/train config untouched
    cfg.data.n_docs = 64
    cfg.data.n_val = 16
    cfg.data.n_test = 16

    tok, splits = build_corpus(cfg)
    cfg.model.vocab_size = tok.vocab_size
    episodes = build_episodes(cfg, tok, splits, n_claims=8, n_repairs=8,
                              seed=cfg.train.seed)
    batch = collate(episodes[:4], tok, cfg.model.max_len)

    system = CGRSystem(cfg, tok)
    system.train()  # grads must flow; we are checking reachability, not accuracy

    # Seed the memory so the read path (and memory.alpha) is actually exercised.
    with torch.no_grad():
        probe = system.backbone.pooled(batch["ids"][:1])[0]
    system.memory.add("probe entry", probe, scope="none", verified=True)
    mem_texts = ["probe entry"] * batch["ids"].shape[0]

    out = system.forward_episodes(batch["ids"], batch["clock"], hard=True,
                                  memory_texts=mem_texts)
    denoise = system.backbone.masked_denoising_loss(batch["ids"], out["cond"])
    r = system.reasoner.loss(out["pooled"], batch["labels"]["value"])
    h, _ = system.heads.loss(out["preds"], batch["labels"],
                             cfg.train.head_weights)
    g = torch.nn.functional.cross_entropy(out["action_logits"],
                                          batch["actions"])
    total = sum(p.numel() for p in system.parameters())

    # every parameter must be reachable from the loss, or it is dead weight the
    # paper would be silently counting
    loss = denoise + r + h + g
    loss.backward()
    gradless = [n for n, p in system.named_parameters()
                if p.requires_grad and p.grad is None]
    dead = list(gradless)

    print(f"[{name}] params={total:,} vocab={tok.vocab_size} "
          f"ids={tuple(batch['ids'].shape)} loss={float(loss.detach()):.4f}")
    if dead:
        print(f"  WARNING no grad: {dead[:8]}"
              f"{' ...' if len(dead) > 8 else ''}")
    else:
        print("  all trainable params received gradient")
    return {"preset": name, "params": total, "vocab": tok.vocab_size,
            "gradless": dead}


if __name__ == "__main__":
    rows = [check(n) for n in ("smoke", "dev", "paper")]
    print()
    for r in rows:
        print(f"{r['preset']:6s} {r['params']:>10,}")
    bad = [r["preset"] for r in rows if r["gradless"]]
    sys.exit(1 if bad else 0)

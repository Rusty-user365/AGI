"""Training: Stage A (supervised warmup) and Stage B (live gate).

Stage A trains backbone + reasoner + heads + gate jointly against the
scaffolding's labels. Stage B replaces argmax with Gumbel-softmax and adds a
KL leash to the scaffolding policy, which is annealed.

Stage C (RL on verified outcomes) is specified in the paper and not run. It is
also not implemented here, deliberately: shipping an untested RL loop and
calling it "specified" is how the paper got into trouble in the first place.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Config
from .data.corpus import build_corpus, make_repair_set
from .data.facts import make_claim_set
from .model.gate import A, ACTIONS
from .model.system import CGRSystem
from .scaffold import (episodes_from_claims, episodes_from_records,
                       episodes_from_repairs)
from .tokenizer import Tokenizer


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------

def collate(episodes, tok: Tokenizer, max_len: int) -> dict:
    L = min(max(e.ids.numel() for e in episodes), max_len)
    ids = torch.full((len(episodes), L), tok.pad_id, dtype=torch.long)
    for i, e in enumerate(episodes):
        n = min(e.ids.numel(), L)
        ids[i, :n] = e.ids[:n]
    clock = torch.stack([e.clock for e in episodes])
    labels = {k: torch.stack([e.labels[k] for e in episodes])
              for k in episodes[0].labels}
    actions = torch.tensor([e.action for e in episodes], dtype=torch.long)
    return {"ids": ids, "clock": clock, "labels": labels, "actions": actions}


def build_episodes(cfg: Config, tok: Tokenizer, splits: dict,
                   n_claims: int | None = None, n_repairs: int | None = None,
                   seed: int = 0):
    n_claims = n_claims or len(splits["train"])
    n_repairs = n_repairs or max(200, len(splits["train"]) // 2)
    noise = 0.1

    docs = episodes_from_records(splits["train"], noise=noise, seed=seed)
    claims_pos = make_claim_set(n_claims // 2, seed=seed + 1, marker_present=True)
    claims_neg = make_claim_set(n_claims // 2, seed=seed + 2, marker_present=False)
    claims = episodes_from_claims(claims_pos + claims_neg, tok, noise=noise,
                                  seed=seed + 3)
    repairs = make_repair_set(splits["train"], n_repairs // 2, seed=seed + 4)
    reps = []
    for kind in ("local", "cross_block"):
        reps += episodes_from_repairs(repairs[kind], tok, noise=noise, seed=seed + 5)

    episodes = docs + claims + reps
    rng = random.Random(seed)
    rng.shuffle(episodes)
    return episodes


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------

def train(cfg: Config, out_dir: str | Path = "checkpoints",
          verbose: bool = True) -> dict:
    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok, splits = build_corpus(cfg)
    # Size the vocabulary to the corpus, not to the draft's 32k. A 512-row
    # embedding table for a 183-token vocabulary would be 90% padding, and the
    # model could emit ids the tokenizer cannot decode.
    cfg.model.vocab_size = tok.vocab_size
    episodes = build_episodes(cfg, tok, splits, seed=cfg.train.seed)

    system = CGRSystem(cfg, tok)
    opt = torch.optim.AdamW(system.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    hist = {"stage_a": [], "stage_b": []}
    t0 = time.time()

    def run_epoch(epoch: int, stage: str) -> dict:
        system.train()
        rng = random.Random(cfg.train.seed + epoch)
        order = list(range(len(episodes)))
        rng.shuffle(order)
        agg = {"loss": 0.0, "n": 0, "gate_acc": 0.0, "gate_n": 0}
        for i in range(0, len(order), cfg.train.batch_size):
            batch = collate([episodes[j] for j in order[i:i + cfg.train.batch_size]],
                            tok, cfg.model.max_len)
            out_f = system.forward_episodes(
                batch["ids"], batch["clock"],
                memory_texts=None, hard=(stage == "A"))
            # 1. backbone denoising
            denoise_loss = system.backbone.masked_denoising_loss(
                batch["ids"], out_f["cond"])
            # 2. reasoner denoising
            r_loss = system.reasoner.loss(out_f["pooled"],
                                          batch["labels"]["value"])
            # 3. heads
            h_loss, h_stats = system.heads.loss(out_f["preds"], batch["labels"],
                                                cfg.train.head_weights)
            # 4. gate
            if stage == "A":
                g_loss = F.cross_entropy(out_f["action_logits"], batch["actions"])
            else:
                tau = max(0.5, cfg.model.gumbel_tau * (1.0 - epoch / max(cfg.train.stage_b_epochs, 1)))
                soft, _ = system.gate(out_f["feats"], hard=False, tau=tau)
                oracle = F.one_hot(batch["actions"], cfg.model.n_actions).float()
                g_loss = F.kl_div((soft + 1e-8).log(), oracle,
                                  reduction="batchmean") * cfg.train.kl_leash
            loss = denoise_loss + r_loss + h_loss + g_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(system.parameters(), cfg.train.grad_clip)
            opt.step()

            with torch.no_grad():
                pred_a = out_f["action_logits"].argmax(-1)
                agg["gate_acc"] += (pred_a == batch["actions"]).float().mean().item()
                agg["gate_n"] += 1
            agg["loss"] += float(loss.detach())
            agg["n"] += 1
        agg["loss"] /= max(agg["n"], 1)
        agg["gate_acc"] /= max(agg["gate_n"], 1)
        agg["epoch"] = epoch
        agg["stage"] = stage
        return agg

    for ep in range(cfg.train.stage_a_epochs):
        h = run_epoch(ep, "A")
        hist["stage_a"].append(h)
        if verbose:
            print(f"[A {ep}] loss={h['loss']:.4f} gate_acc={h['gate_acc']:.3f}")
    for ep in range(cfg.train.stage_b_epochs):
        h = run_epoch(ep, "B")
        hist["stage_b"].append(h)
        if verbose:
            print(f"[B {ep}] loss={h['loss']:.4f} gate_acc={h['gate_acc']:.3f}")

    torch.save({"model": system.state_dict(),
                "cfg": cfg.__dict__,
                "tokenizer": tok.itos,
                "n_params": system.n_params()}, out / "stage_b.pt")
    tok.save(out / "tokenizer.json")
    cfg.to_json(out / "config.json")

    summary = {"elapsed_s": time.time() - t0, "params": system.n_params(),
               "history": hist, "vocab": tok.vocab_size,
               "n_episodes": len(episodes)}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2),
                                            encoding="utf-8")
    if verbose:
        print(json.dumps(summary["params"], indent=2))
    return summary


def load_system(path: str | Path) -> tuple[CGRSystem, Config]:
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg_path = Path(path).parent / "config.json"
    cfg = Config.from_json(cfg_path) if cfg_path.exists() else Config.preset("smoke")
    tok = Tokenizer(ck["tokenizer"][5:])
    system = CGRSystem(cfg, tok)
    system.load_state_dict(ck["model"])
    system.eval()
    return system, cfg


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()
    train(Config.preset(args.preset), args.out)

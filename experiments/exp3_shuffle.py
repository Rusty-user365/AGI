"""Experiment 3 — is the side channel causal? (shuffle ablation + control)

Four conditions, plus the control the draft was missing:

  full       steering vector from this input
  shuffled   steering vector from a different input in the batch
  constant   steering vector from zeroed thought-blocks
  none       no steering at all (backbone pass 1 only)
  oracle     steering vector computed from the CLEAN document while the input
             is corrupted  <-- positive control

The control is the point. If `oracle` is much better than `full`, then the
steering vector *can* carry information the input lacks, and a null
shuffled-vs-full result means the model failed to learn to use the channel —
an optimisation/power result, not evidence that reasoning is decorative. If
`oracle` is no better than `full`, the channel is structurally incapable of
carrying anything, and the paper must say so.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from _common import arg_parser, banner, get_system, table, write_result
from cgr.data.corpus import (build_corpus, corrupt_local, corrupt_nonlocal)
from cgr.data.grammar import blocked_spans, is_consistent
from cgr.metrics import bootstrap_ci, minimum_detectable_effect
from cgr.train import collate, build_episodes

PREDICTION = {
    "full_vs_shuffled": "full better than shuffled if reasoning is causal",
    "shuffled_vs_constant": "shuffled ~ constant if content is irrelevant",
    "none": "worst, if the channel contributes at all",
    "oracle": "oracle better than full => channel can carry information",
    "interpretation": "a null shuffled result is only interpretable if oracle "
                      "shows the channel can carry information",
}


def fixed_mask_loss(system, ids, cond, seed=1234, ratio=0.5) -> float:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        return float(system.backbone.masked_denoising_loss(
            ids, cond, mask_ratio=ratio, generator=g))


def main() -> int:
    ap = arg_parser("Experiment 3: shuffle ablation with a positive control")
    args = ap.parse_args()
    n = args.n or 100

    banner("Experiment 3 — side-channel causality")
    system, cfg = get_system(args.preset, seed=args.seed,
                             checkpoint=args.checkpoint)
    system.eval()
    tok, splits = build_corpus(cfg)

    import random
    rng = random.Random(args.seed)
    recs = splits["test"][:n]
    from cgr.scaffold import episodes_from_records
    clean_eps = episodes_from_records(recs, noise=0.0, seed=0)
    clean_ids = collate(clean_eps, tok, cfg.model.max_len)

    corrupt_blocks = []
    for r in recs:
        if rng.random() < 0.5:
            ex = corrupt_local(r.blocks, rng)
        else:
            ex = corrupt_nonlocal(r.blocks, rng, r.thesis_id)
        corrupt_blocks.append(ex.corrupt_blocks)

    cor_ids, spans = [], []
    for b in corrupt_blocks:
        ids, sp = blocked_spans(b, tok)
        cor_ids.append(ids)
        spans.append(sp)
    L = min(max(len(i) for i in cor_ids), cfg.model.max_len)
    ids = torch.full((len(cor_ids), L), tok.pad_id, dtype=torch.long)
    for i, x in enumerate(cor_ids):
        ids[i, :min(len(x), L)] = torch.tensor(x[:L])

    clock = torch.stack([torch.tensor([0., 0., 6., 10., 0., 1., 0., 0.6])
                         for _ in range(len(recs))])

    # steering vectors for each condition
    with torch.no_grad():
        pooled0 = system.backbone.pooled(ids)
        blocks = system.reasoner.encode(pooled0)
        cond_full = system.reasoner.combine(blocks)
        cond_shuf = system.reasoner.combine(blocks.roll(1, dims=0))
        cond_const = system.reasoner.combine(torch.zeros_like(blocks))
        # positive control: steer from the CLEAN document
        pooled_clean = system.backbone.pooled(clean_ids["ids"])
        cond_oracle = system.reasoner.combine(system.reasoner.encode(pooled_clean))

    conditions = {
        "full": cond_full,
        "shuffled": cond_shuf,
        "constant": cond_const,
        "none": None,
        "oracle": cond_oracle,
    }

    rows, per_cond = [], {}
    losses = {}
    for name, cond in conditions.items():
        with torch.no_grad():
            out = system.forward_episodes(ids, clock, reasoning_mode="none",
                                          cond_override=cond)
            preds = system.heads.predict(out["feats"])
        loss = fixed_mask_loss(system, ids, cond if cond is not None else None)
        # value head should say "repairable" (label 1.0)
        val_mae = float((preds["value"] - 1.0).abs().mean())
        losses[name] = loss
        per_cond[name] = {"denoise_loss": loss, "value_mae": val_mae}

    # generation consistency per condition (small n; generation is expensive)
    gen_n = min(20, len(recs))
    gen_cons = {}
    for name, cond in conditions.items():
        ok = 0
        for i in range(gen_n):
            blocks_i = corrupt_blocks[i]
            ids_i, sp_i = blocked_spans(blocks_i, tok)
            t = torch.tensor(ids_i).unsqueeze(0)
            c = None if cond is None else cond[i:i + 1]
            with torch.no_grad():
                g = system.backbone.infill(t, [(0, t.shape[1])], cond=c, steps=4,
                                           seed=args.seed)
            flat = g[0].tolist()
            new_blocks = [tok.decode(flat[s:e]) for s, e in sp_i]
            ok += int(is_consistent(new_blocks))
        gen_cons[name] = ok / max(gen_n, 1)

    base = losses["full"]
    for name in conditions:
        rows.append({
            "condition": name,
            "loss": losses[name],
            "vs_full": losses[name] - base,
            "value_mae": per_cond[name]["value_mae"],
            "repair_consistency": gen_cons[name],
        })
    table(rows, ["condition", "loss", "vs_full", "value_mae", "repair_consistency"])

    # paired bootstrap: full vs shuffled, per-document losses
    def doc_losses(cond):
        out = []
        for i in range(len(recs)):
            li = ids[i:i + 1]
            c = None if cond is None else cond[i:i + 1]
            out.append(fixed_mask_loss(system, li, c, seed=args.seed + i))
        return out

    full_d = doc_losses(cond_full)
    shuf_d = doc_losses(cond_shuf)
    diff = [a - b for a, b in zip(shuf_d, full_d)]  # positive => full better
    mean, lo, hi = bootstrap_ci(diff)
    mde = minimum_detectable_effect(len(diff), float(torch.tensor(diff).std()))

    channel_carries_info = (losses["oracle"] < losses["full"] - 1e-3)
    reasoning_causal = (mean > 0) and (lo > 0)

    notes = []
    if not channel_carries_info:
        notes.append(
            "The positive control did not beat `full`: the steering vector "
            "could not carry information the input lacked. A null shuffle "
            "result is therefore uninterpretable and must not be reported as "
            "'reasoning is decorative'.")
    if not reasoning_causal:
        notes.append(
            f"full vs shuffled: {mean:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}]), "
            f"MDE {mde:.4f}. The side channel is not measurably causal here. "
            f"Report as a null result with the power calculation attached.")
    if losses["constant"] <= losses["shuffled"] + 1e-3:
        notes.append("shuffled ~ constant: content is irrelevant, only the "
                     "presence of a steering vector matters.")

    write_result("exp3_shuffle", {
        "experiment": "exp3_shuffle",
        "preset": args.preset, "seed": args.seed, "n": len(recs),
        "preregistered_prediction": PREDICTION,
        "conditions": per_cond,
        "summary": {
            "losses": losses,
            "repair_consistency": gen_cons,
            "full_vs_shuffled_mean_diff": mean,
            "full_vs_shuffled_ci": [lo, hi],
            "mde": mde,
            "channel_carries_information": channel_carries_info,
            "reasoning_causal": reasoning_causal,
        },
        "notes": notes,
    })
    print("\n" + "\n".join(f"  ! {x}" for x in notes) if notes else "\n  no anomalies")
    return 0


if __name__ == "__main__":
    sys.exit(main())

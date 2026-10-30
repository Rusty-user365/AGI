"""Historical fresh-prompt conditional-repair experiment, with audited reporting.

This implementation does not generate repairs from a persistent cropped cache.
Token counters are formula estimates. Timings exclude parts of the controller.
See results/measurement_audit_20261030.json. Historical results are not replaced.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.arith_pilot import load
from experiments.arith_steps import build_prompt, parse_value, readout
from experiments.localizer import (Standardizer, balanced_accuracy, load_candidates,
                                   run_seed, sign_test, svd_basis, wilson)
from experiments.localizer_pair import bootstrap_diff, majority, per_step_recall
from experiments.oracle_rewind import MaskBadFirstDigit, finish_chain, gen_step
from experiments._device import resolve as _resolve_device


def mask_for(tok, value: int):
    """Forbid the first digit of value, leaving leading space alone."""
    digit_ids = [tok(" " + str(d))["input_ids"][-1] for d in range(10)]
    ids = tok(" " + str(value))["input_ids"]
    return next(i for i in ids if i in digit_ids), digit_ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="results/arith_steps_h5.pt")
    ap.add_argument("--model", default="checkpoints/hf/qwen3-0.6b-base")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/path_a_loop_h5_audited_v2.json")
    args = ap.parse_args()
    if Path(args.out).exists() and not getattr(args, "diagnostics_only", False):
        raise FileExistsError("Preserve recorded evidence: choose a new --out path")
    print("AUDIT: conditional repair; token counts are estimates; timings are not full-controller or cache speedups.")
    args.device = str(_resolve_device(args.device))

    root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path

    raw = torch.load(data_path, weights_only=False)
    steps = int(raw["config"]["steps"])
    cands = load_candidates(data_path, steps)
    train = [c for c in cands if c["split"] == "train"]
    hold = [c for c in cands if c["split"] == "holdout"]
    lab = torch.tensor([c["label"] for c in hold])

    print(f"Loaded: {len(train)} train, {len(hold)} holdout candidates, steps={steps}")

    # =========================================================================
    # 1. Train Localizer Ensemble (Linear SVD32 + Transformer Full-Width)
    # =========================================================================
    feats = torch.stack([c["feat"] for c in train])
    sd = Standardizer(feats, False)
    proj = svd_basis(feats, 32)
    run_args = argparse.Namespace(width=64, heads=4, dropout=0.1, epochs=150,
                                  batch=32, lr=1e-3, weight_decay=1e-4,
                                  device=args.device)

    print(f"Training {args.seeds} seeds of Linear and Transformer Localizers on {args.device}...")
    lin_models, trf_models = [], []
    lin_preds, trf_preds = [], []
    lin_logits_seeds, trf_logits_seeds = [], []

    xh = sd(torch.stack([c["feat"] for c in hold])).to(args.device)
    xh_proj = xh @ proj.to(args.device)
    lph = torch.stack([c["logprob"] for c in hold]).to(args.device)

    for s in range(args.seeds):
        rl, acc_l, ml = run_seed(train, {"h": hold}, run_args, steps, s, sd, proj, "linear")
        rt, acc_t, mt = run_seed(train, {"h": hold}, run_args, steps, s, sd, None, "transformer")
        lin_models.append(ml)
        trf_models.append(mt)
        lin_preds.append(rl["h"]["pred"])
        trf_preds.append(rt["h"]["pred"])

        with torch.no_grad():
            lin_logits_seeds.append(ml(xh_proj, lph))
            trf_logits_seeds.append(mt(xh, lph))

    # Evaluate Localizer profiles on Holdout
    pl_maj = majority(lin_preds, steps)
    pt_maj = majority(trf_preds, steps)

    lin_probs = torch.stack([F.softmax(l, dim=-1) for l in lin_logits_seeds]).mean(0)
    trf_probs = torch.stack([F.softmax(l, dim=-1) for l in trf_logits_seeds]).mean(0)
    ens_probs = (lin_probs + trf_probs) / 2.0
    p_ens = ens_probs.argmax(-1).cpu()

    # Two-Tier Rule: If Transformer predicts step 4 (or Trf step4 prob >= 0.35), predict 4; else Linear
    two_tier_preds = pl_maj.clone()
    for i in range(len(hold)):
        if trf_probs[i, 4] >= 0.35:
            two_tier_preds[i] = 4

    rec_lin = per_step_recall(pl_maj, lab, steps)
    rec_trf = per_step_recall(pt_maj, lab, steps)
    rec_ens = per_step_recall(p_ens, lab, steps)
    rec_tt = per_step_recall(two_tier_preds, lab, steps)

    print("\n--- Localizer Diagnostic Summary on Holdout (n=110) ---")
    print(f"Linear Majority:     Top-1={float((pl_maj == lab).float().mean()):.3f}, Bal={balanced_accuracy(pl_maj, lab, steps):.3f}, Step4 Recall={rec_lin[4]['recall']:.3f}")
    print(f"Transformer Majority:Top-1={float((pt_maj == lab).float().mean()):.3f}, Bal={balanced_accuracy(pt_maj, lab, steps):.3f}, Step4 Recall={rec_trf[4]['recall']:.3f}")
    print(f"Ensemble Average:    Top-1={float((p_ens == lab).float().mean()):.3f}, Bal={balanced_accuracy(p_ens, lab, steps):.3f}, Step4 Recall={rec_ens[4]['recall']:.3f}")
    print(f"Two-Tier (Trf step4):Top-1={float((two_tier_preds == lab).float().mean()):.3f}, Bal={balanced_accuracy(two_tier_preds, lab, steps):.3f}, Step4 Recall={rec_tt[4]['recall']:.3f}")

    # =========================================================================
    # 2. Load Generation Model
    # =========================================================================
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    model, tok = load(model_path, torch.float32, args.device, "auto")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tasks = {(t["split"], t["idx"]): t for t in raw["tasks"]}
    rows = [r for r in raw["rows"]
            if r["split"] == "holdout" and r["complete"] and r["first_error"] is not None]
    if args.limit:
        rows = rows[:args.limit]
    key = {(c["split"], c["idx"], c["cand"]): i for i, c in enumerate(hold)}

    # Helper function to compute candidate error scores under the ensemble
    def score_candidate(feat_t, lp_t):
        with torch.no_grad():
            f_norm = sd(feat_t.unsqueeze(0).to(args.device))
            lp_dev = lp_t.unsqueeze(0).to(args.device)
            p_l = torch.stack([F.softmax(m(f_norm @ proj.to(args.device), lp_dev).squeeze(0), dim=-1) for m in lin_models]).mean(0)
            p_t = torch.stack([F.softmax(m(f_norm, lp_dev).squeeze(0), dim=-1) for m in trf_models]).mean(0)
            return ((p_l + p_t) / 2.0).cpu()

    # =========================================================================
    # 3. Execute End-to-End Rewind Loop with Arms & Gate
    # =========================================================================
    arms = {
        "full_rewrite": {"ok": 0, "n": 0, "tokens": 0, "time_ms": 0.0, "accepted": 0, "rejected": 0},
        "linear_ungated": {"ok": 0, "n": 0, "tokens": 0, "time_ms": 0.0, "t_exact": 0, "accepted": 0, "rejected": 0},
        "ensemble_ungated": {"ok": 0, "n": 0, "tokens": 0, "time_ms": 0.0, "t_exact": 0, "accepted": 0, "rejected": 0},
        "two_tier_ungated": {"ok": 0, "n": 0, "tokens": 0, "time_ms": 0.0, "t_exact": 0, "accepted": 0, "rejected": 0},
        "ensemble_gated": {"ok": 0, "n": 0, "tokens": 0, "time_ms": 0.0, "t_exact": 0, "accepted": 0, "rejected": 0},
        "two_tier_gated": {"ok": 0, "n": 0, "tokens": 0, "time_ms": 0.0, "t_exact": 0, "accepted": 0, "rejected": 0},
    }

    systems = {
        "uncached_rewind_tokens": 0,
        "prefix_prefill_token_estimate": 0,
        "uncached_rewind_time_ms": 0.0,
        "prefix_prefill_time_ms": 0.0,
    }

    print(f"\nRunning end-to-end evaluation on {len(rows)} holdout rows...")

    for j, r in enumerate(rows):
        i = key.get((r["split"], r["idx"], r["cand"]))
        if i is None:
            continue

        t_true = int(r["first_error"])
        emitted = list(r["emitted"])
        task = tasks[(r["split"], r["idx"])]
        lines, order, truth, start = task["lines"], task["order"], task["truth"], task["start"]

        t_lin = int(pl_maj[i])
        t_ens = int(p_ens[i])
        t_tt = int(two_tier_preds[i])

        # ---------------------------------------------------------------------
        # Arm 1: Full Rewrite (T=0.8, sample baseline)
        # ---------------------------------------------------------------------
        t0 = time.time()
        rw_seq = []
        rw_tokens = 0
        for s_idx in range(steps):
            p = build_prompt(lines, order, start, rw_seq, s_idx)
            enc_p = tok(p, return_tensors="pt")["input_ids"]
            from experiments.arith_steps import step_value
            txt = step_value(model, tok, p, args.device, args.max_new, sample=True, temperature=0.8, top_p=0.95)
            val = parse_value(txt)
            rw_seq.append(val)
            rw_tokens += enc_p.shape[1] + 2
            if val is None:
                break
        t_rw = (time.time() - t0) * 1000
        ok_rw = (rw_seq == truth)
        arms["full_rewrite"]["n"] += 1
        arms["full_rewrite"]["ok"] += int(ok_rw)
        arms["full_rewrite"]["tokens"] += rw_tokens
        arms["full_rewrite"]["time_ms"] += t_rw

        # ---------------------------------------------------------------------
        # Pre-edit localizer scores for the chain
        # ---------------------------------------------------------------------
        orig_feat = hold[i]["feat"]
        orig_lp = hold[i]["logprob"]
        p_err_before = score_candidate(orig_feat, orig_lp) # [steps]
        max_err_before = float(p_err_before.max())

        # Cache tokens for systems timing benchmark
        prompt_tokens_by_step = [tok(build_prompt(lines, order, start, emitted[:s_k], s_k), return_tensors="pt")["input_ids"][0] for s_k in range(steps)]

        # ---------------------------------------------------------------------
        # Arms 2-6: Rewinds with rank2
        # ---------------------------------------------------------------------
        loop_configs = [
            ("linear_ungated", t_lin, False),
            ("ensemble_ungated", t_ens, False),
            ("two_tier_ungated", t_tt, False),
            ("ensemble_gated", t_ens, True),
            ("two_tier_gated", t_tt, True),
        ]

        # Generate repaired sequence for distinct t_hat targets to avoid redundant decodes
        repaired_by_t = {}
        tokens_by_t = {}
        time_by_t = {}

        for arm_name, t_hat, is_gated in loop_configs:
            if t_hat not in repaired_by_t:
                bad_val = int(emitted[t_hat]) if (t_hat < len(emitted) and emitted[t_hat] is not None) else 0
                mask = mask_for(tok, bad_val)
                p_hat = build_prompt(lines, order, start, emitted[:t_hat], t_hat)
                
                t0_start = time.time()
                txt = gen_step(model, tok, p_hat, args.device, args.max_new, False, mask=mask)
                v_hat = parse_value(txt)
                seq_rep = emitted[:t_hat] + [v_hat]
                if v_hat is not None:
                    seq_rep = finish_chain(model, tok, lines, order, start, seq_rep, t_hat + 1, steps, args.device, args.max_new)
                elapsed_ms = (time.time() - t0_start) * 1000
                
                tok_count = sum(tok(build_prompt(lines, order, start, seq_rep[:k], k))["input_ids"].__len__() + 2 for k in range(t_hat, steps))
                repaired_by_t[t_hat] = seq_rep
                tokens_by_t[t_hat] = tok_count
                time_by_t[t_hat] = elapsed_ms

            seq_repaired = repaired_by_t[t_hat]
            tok_used = tokens_by_t[t_hat]
            time_used = time_by_t[t_hat]

            # Measure systems KV-cache slicing benchmark on the ensemble target
            if arm_name == "ensemble_ungated":
                # Fresh forward tokens / latency
                systems["uncached_rewind_tokens"] += tok_used
                systems["uncached_rewind_time_ms"] += time_used
                
                # KV cache truncated forward
                t0_cac = time.time()
                target_prompt_tokens = prompt_tokens_by_step[t_hat].to(args.device)
                out_prefix = model(input_ids=target_prompt_tokens.unsqueeze(0), use_cache=True)
                cac = out_prefix.past_key_values
                # Crop cache back to target_prompt_tokens length (verifying crop functionality)
                cac.crop(target_prompt_tokens.shape[0])
                cac_tokens = target_prompt_tokens.shape[0] + (steps - t_hat) * 2
                t_cac_ms = (time.time() - t0_cac) * 1000
                systems["prefix_prefill_token_estimate"] += cac_tokens
                systems["prefix_prefill_time_ms"] += t_cac_ms

            final_seq = seq_repaired
            if is_gated:
                # Read out original step logprob vs edit logprob at t_hat
                p_eval = build_prompt(lines, order, start, emitted[:t_hat], t_hat)
                val_orig_str = f" {emitted[t_hat]}\n" if (t_hat < len(emitted) and emitted[t_hat] is not None) else " None\n"
                val_rep_str = f" {seq_repaired[t_hat]}\n" if (t_hat < len(seq_repaired) and seq_repaired[t_hat] is not None) else " None\n"
                lp_orig, _ = readout(model, tok, p_eval, val_orig_str, args.device)
                lp_edit, _ = readout(model, tok, p_eval, val_rep_str, args.device)

                # Overwrite gate check:
                # Reject edit if the original step was clean (lp_orig > -0.25) and rank-2 degraded confidence (lp_edit < lp_orig - 0.35)
                is_clean_orig = (lp_orig is not None and lp_orig > -0.25)
                is_degraded_edit = (lp_edit is not None and lp_orig is not None and (lp_edit < lp_orig - 0.35))

                if is_clean_orig and is_degraded_edit:
                    arms[arm_name]["rejected"] += 1
                    final_seq = emitted
                else:
                    arms[arm_name]["accepted"] += 1
                    final_seq = seq_repaired

            ok_arm = (final_seq == truth)
            arms[arm_name]["n"] += 1
            arms[arm_name]["ok"] += int(ok_arm)
            arms[arm_name]["tokens"] += tok_used
            arms[arm_name]["time_ms"] += time_used
            arms[arm_name]["t_exact"] += int(t_hat == t_true)

        if (j + 1) % 10 == 0 or (j + 1) == len(rows):
            print(f"  {j + 1:>3}/{len(rows)} | "
                  f"Rewrite: {arms['full_rewrite']['ok']}/{arms['full_rewrite']['n']} | "
                  f"Linear: {arms['linear_ungated']['ok']}/{arms['linear_ungated']['n']} | "
                  f"Ens: {arms['ensemble_ungated']['ok']}/{arms['ensemble_ungated']['n']} | "
                  f"TwoTier: {arms['two_tier_ungated']['ok']}/{arms['two_tier_ungated']['n']} | "
                  f"Gated: {arms['ensemble_gated']['ok']}/{arms['ensemble_gated']['n']} | "
                  f"TwoTierGated: {arms['two_tier_gated']['ok']}/{arms['two_tier_gated']['n']}",
                  flush=True)

    # =========================================================================
    # 4. Final Analysis & Statistical Comparisons
    # =========================================================================
    n_total = len(rows)
    print("\n" + "=" * 80)
    print(f"PATH A FINAL EMPIRICAL RESULTS (N={n_total})")
    print("=" * 80)
    print(f"{'Arm':<20} {'Recovered':>10} {'Rate':>7} {'95% Wilson CI':>18} {'t_exact':>8} {'Accepted':>9}")
    print("-" * 80)
    for name, a in arms.items():
        rate = a["ok"] / max(a["n"], 1)
        lo, hi = wilson(a["ok"], a["n"])
        t_ex = f"{a['t_exact']/a['n']:.3f}" if "t_exact" in a else "—"
        acc = f"{a.get('accepted', 0)}/{a['n']}" if name.endswith("_gated") else "—"
        print(f"{name:<20} {a['ok']:>4}/{a['n']:<5} {rate:>7.3f} [{lo:.3f}, {hi:.3f}] {t_ex:>8} {acc:>9}")

    print("\n" + "=" * 80)
    print("INCOMPLETE TIMING DIAGNOSTICS: NO SPEEDUP CLAIM")
    print("=" * 80)
    avg_unc_tok = systems["uncached_rewind_tokens"] / max(n_total, 1)
    avg_cac_tok = systems["prefix_prefill_token_estimate"] / max(n_total, 1)
    avg_unc_time = systems["uncached_rewind_time_ms"] / max(n_total, 1)
    avg_cac_time = systems["prefix_prefill_time_ms"] / max(n_total, 1)
    avg_rw_tok = arms["full_rewrite"]["tokens"] / max(n_total, 1)
    avg_rw_time = arms["full_rewrite"]["time_ms"] / max(n_total, 1)

    print(f"Full Rewrite:             {avg_rw_tok:>6.1f} tokens/chain | {avg_rw_time:>6.1f} ms/chain")
    print(f"Un-cached Fresh Rewind:   {avg_unc_tok:>6.1f} tokens/chain | {avg_unc_time:>6.1f} ms/chain")
    print(f"Prefix prefill only (no decode):{avg_cac_tok:>6.1f} tokens/chain | {avg_cac_time:>6.1f} ms/chain")

    payload = {
        "measurement_scope": {
            "repair_uses_persistent_cropped_cache": False,
            "full_controller_latency_measured": False,
            "token_counts_are_estimates": True,
            "task_clustering_accounted_for": False,
            "historical_interpretation_audit": "results/measurement_audit_20261030.json",
        },
        "config": vars(args),
        "n": n_total,
        "localizer_eval": {
            "linear_top1": float((pl_maj == lab).float().mean()),
            "linear_bal": balanced_accuracy(pl_maj, lab, steps),
            "linear_step4_rec": rec_lin[4]["recall"],
            "transformer_top1": float((pt_maj == lab).float().mean()),
            "transformer_bal": balanced_accuracy(pt_maj, lab, steps),
            "transformer_step4_rec": rec_trf[4]["recall"],
            "ensemble_top1": float((p_ens == lab).float().mean()),
            "ensemble_bal": balanced_accuracy(p_ens, lab, steps),
            "ensemble_step4_rec": rec_ens[4]["recall"],
            "two_tier_top1": float((two_tier_preds == lab).float().mean()),
            "two_tier_bal": balanced_accuracy(two_tier_preds, lab, steps),
            "two_tier_step4_rec": rec_tt[4]["recall"],
            "per_step_recall": {
                "linear": rec_lin, "transformer": rec_trf,
                "ensemble": rec_ens, "two_tier": rec_tt
            }
        },
        "arms": {
            k: dict(v, rate=v["ok"] / max(v["n"], 1), ci=list(wilson(v["ok"], v["n"])))
            for k, v in arms.items()
        },
        "systems": {
            "full_rewrite_tokens": avg_rw_tok,
            "full_rewrite_time_ms": avg_rw_time,
            "uncached_rewind_tokens": avg_unc_tok,
            "uncached_rewind_time_ms": avg_unc_time,
            "prefix_prefill_token_estimate": avg_cac_tok,
            "prefix_prefill_time_ms": avg_cac_time,
        }
    }

    out_file = Path(args.out)
    if not out_file.is_absolute():
        out_file = root / out_file
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote full empirical results to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

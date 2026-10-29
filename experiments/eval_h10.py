"""Evaluation harness for Scenario 2 & 3: The Long-Horizon 'Context Rot' Scaling Test (H=10).

Evaluates:
  1. full_rewrite (from Step 0)
  2. in_context_oops (verbal retry prompt without KV truncation)
  3. two_tier_gated_kv_crop (1 rewind)
  4. two_tier_gated_kv_crop_max2 (up to 2 sequential rewinds)
"""
from pathlib import Path
import argparse
import json
import math
import random
import sys
import time
import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from experiments.arith_pilot import load
from experiments.arith_steps import build_prompt, parse_value, readout, wilson, FEWSHOT, step_value
from experiments.localizer import load_candidates, Standardizer, svd_basis, run_seed, balanced_accuracy
from experiments.localizer_pair import majority, per_step_recall
from experiments.oracle_rewind import MaskBadFirstDigit, gen_step, finish_chain
from experiments._device import resolve as _resolve_device

def mask_for(tok, value: int):
    digit_ids = [tok(" " + str(d))["input_ids"][-1] for d in range(10)]
    ids = tok(" " + str(value))["input_ids"]
    return next(i for i in ids if i in digit_ids), digit_ids

def paired_stats(y_a: list[int], y_b: list[int]):
    b = sum(1 for a, c in zip(y_a, y_b) if a == 1 and c == 0)
    c = sum(1 for a, c in zip(y_a, y_b) if a == 0 and c == 1)
    n_disc = b + c
    if n_disc == 0:
        return {"wins_a": 0, "wins_b": 0, "p_value": 1.0}
    k = min(b, c)
    total = sum(math.comb(n_disc, i) * (0.5 ** n_disc) for i in range(k + 1))
    p = min(1.0, 2.0 * total)
    return {"wins_a": b, "wins_b": c, "n_disc": n_disc, "p_value": float(p)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/arith_steps_h10.pt")
    parser.add_argument("--model", default="checkpoints/hf/qwen3-0.6b-base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-new", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="results/verify_h10_scaling.json")
    args = parser.parse_args()
    args.device = str(_resolve_device(args.device))

    steps = 10
    print(f"Loading H=10 corpus from {args.data}...", flush=True)
    raw = torch.load(root / args.data, weights_only=False)
    tasks = {(t["split"], t["idx"]): t for t in raw["tasks"]}
    cands = load_candidates(root / args.data, steps)
    train = [c for c in cands if c["split"] == "train"]
    hold = [c for c in cands if c["split"] == "holdout"]
    rows = [r for r in raw["rows"] if r["split"] == "holdout" and r["complete"] and r["first_error"] is not None]

    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"Loaded {len(train)} train candidates, {len(hold)} holdout candidates ({len(rows)} broken holdout rows).", flush=True)

    # Train Localizers on H=10 Train Split
    feats_train = torch.stack([c["feat"] for c in train])
    sd = Standardizer(feats_train, False)
    proj = svd_basis(feats_train, 32)
    run_args = argparse.Namespace(width=64, heads=4, dropout=0.1, epochs=150, batch=32, lr=1e-3, weight_decay=1e-4, device=args.device)

    lin_models, trf_models = [], []
    print("Training 5 seeds of Linear and Transformer Localizers on H=10 train set...", flush=True)
    for s in range(args.seeds):
        _, _, ml = run_seed(train, {}, run_args, steps, s, sd, proj, "linear")
        _, _, mt = run_seed(train, {}, run_args, steps, s, sd, None, "transformer")
        ml.eval()
        mt.eval()
        lin_models.append(ml)
        trf_models.append(mt)

    # Compute Localizer Predictions on Holdout Broken Candidates
    key = {(c["split"], c["idx"], c["cand"]): i for i, c in enumerate(hold)}
    xh = sd(torch.stack([c["feat"] for c in hold])).to(args.device)
    xh_proj = (xh @ proj.to(args.device))
    lph = torch.stack([c["logprob"] for c in hold]).to(args.device)

    with torch.no_grad():
        lin_logits_seeds = [m(xh_proj, lph) for m in lin_models]
        trf_logits_seeds = [m(xh, lph) for m in trf_models]

    lin_preds = [l.argmax(-1).cpu() for l in lin_logits_seeds]
    pl_maj = majority(lin_preds, steps)
    trf_probs = torch.stack([F.softmax(l, dim=-1) for l in trf_logits_seeds]).mean(0).cpu()

    # Two-Tier Rule for H=10: if Transformer assigns probability >= 0.25 to any tail step (steps 7, 8, 9), route to tail
    two_tier_preds = []
    for i in range(len(hold)):
        tail_prob = trf_probs[i, 7:].sum().item()
        if tail_prob >= 0.30:
            best_tail = 7 + int(trf_probs[i, 7:].argmax().item())
            two_tier_preds.append(best_tail)
        else:
            two_tier_preds.append(int(pl_maj[i]))

    # Diagnostic accuracies on H=10 holdout. (pred, lab) is the signature;
    # the 2026-10-28 vault run had this backwards, so check before editing.
    h_lab = torch.tensor([c["label"] for c in hold])
    tt_p = torch.tensor(two_tier_preds)
    tt_top1 = float((tt_p == h_lab).float().mean())
    tt_bal = balanced_accuracy(tt_p, h_lab, steps)
    lin_top1 = float((pl_maj == h_lab).float().mean())
    lin_bal = balanced_accuracy(pl_maj, h_lab, steps)

    print("-" * 75)
    print(f"H=10 LOCALIZER DIAGNOSTIC (N={len(hold)} candidates)")
    print("-" * 75)
    print(f"Linear Majority:   Top-1 = {lin_top1:.3f}, Bal = {lin_bal:.3f}")
    print(f"Two-Tier (Tail):   Top-1 = {tt_top1:.3f}, Bal = {tt_bal:.3f}")
    print("-" * 75)

    # Load Model
    print(f"\nLoading model {args.model} for H=10 evaluation...", flush=True)
    model, tok = load(root / args.model, torch.float16, args.device, "auto")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    arms = {
        "full_rewrite": {"ok": 0, "n": 0, "tokens": 0, "time_ms": 0.0, "recovered_list": []},
        "in_context_oops": {"ok": 0, "n": 0, "tokens": 0, "time_ms": 0.0, "recovered_list": []},
        "two_tier_gated_1rewind": {"ok": 0, "n": 0, "tokens": 0, "time_ms": 0.0, "recovered_list": [], "accepted": 0, "rejected": 0},
        "two_tier_gated_max2": {"ok": 0, "n": 0, "tokens": 0, "time_ms": 0.0, "recovered_list": [], "accepted": 0, "rejected": 0, "second_rewind_count": 0},
    }

    print(f"\nRunning end-to-end evaluation on {len(rows)} broken H=10 rows...", flush=True)

    for j, r in enumerate(rows):
        i_cand = key.get((r["split"], r["idx"], r["cand"]))
        if i_cand is None:
            continue
        t_true = int(r["first_error"])
        emitted = list(r["emitted"])
        task = tasks[(r["split"], r["idx"])]
        lines, order, truth, start = task["lines"], task["order"], task["truth"], task["start"]
        t_pred = int(two_tier_preds[i_cand])

        # -------------------------------------------------------------
        # Arm 1: Full Rewrite (T=0.8, from Step 0)
        # -------------------------------------------------------------
        t0_rw = time.time()
        rw_seq = []
        rw_tokens = 0
        for s_idx in range(steps):
            p = build_prompt(lines, order, start, rw_seq, s_idx)
            enc_p = tok(p, return_tensors="pt")["input_ids"]
            txt = step_value(model, tok, p, args.device, args.max_new, sample=True, temperature=0.8, top_p=0.95)
            val = parse_value(txt)
            rw_seq.append(val)
            rw_tokens += enc_p.shape[1] + 2
            if val is None:
                break
        t_rw_ms = (time.time() - t0_rw) * 1000
        ok_rw = (rw_seq == truth)
        arms["full_rewrite"]["n"] += 1
        arms["full_rewrite"]["ok"] += int(ok_rw)
        arms["full_rewrite"]["tokens"] += rw_tokens
        arms["full_rewrite"]["time_ms"] += t_rw_ms
        arms["full_rewrite"]["recovered_list"].append(int(ok_rw))

        # -------------------------------------------------------------
        # Arm 2: In-Context Verbal Retry ("Oops, recalculate from step t")
        # -------------------------------------------------------------
        t0_oops = time.time()
        first_name = lines[0].split(" = ")[0]
        vals_full = [f"{first_name} -> {start}"] + [f"{order[k]} -> {emitted[k]}" for k in range(steps)]
        p_oops = (FEWSHOT + "\n".join(lines) + "\n" + "\n".join(vals_full) +
                  f"\n# Oops, recompute from step {order[t_pred]}:\n{order[t_pred]} ->")
        bad_val_oops = int(emitted[t_pred]) if (t_pred < len(emitted) and emitted[t_pred] is not None) else 0
        mask_oops = mask_for(tok, bad_val_oops)
        txt_oops = gen_step(model, tok, p_oops, args.device, args.max_new, False, mask=mask_oops)
        v_oops = parse_value(txt_oops)
        seq_oops = emitted[:t_pred] + [v_oops]
        if v_oops is not None:
            seq_oops = finish_chain(model, tok, lines, order, start, seq_oops, t_pred + 1, steps, args.device, args.max_new)
        t_oops_ms = (time.time() - t0_oops) * 1000
        tok_oops = tok(p_oops)["input_ids"].__len__() + (steps - t_pred) * 2

        ok_oops = (seq_oops == truth)
        arms["in_context_oops"]["n"] += 1
        arms["in_context_oops"]["ok"] += int(ok_oops)
        arms["in_context_oops"]["tokens"] += tok_oops
        arms["in_context_oops"]["time_ms"] += t_oops_ms
        arms["in_context_oops"]["recovered_list"].append(int(ok_oops))

        # -------------------------------------------------------------
        # Arm 3: Two-Tier Gated 1-Rewind with KV-Cache Truncation
        # -------------------------------------------------------------
        t0_cgr = time.time()
        bad_val_cgr = int(emitted[t_pred]) if (t_pred < len(emitted) and emitted[t_pred] is not None) else 0
        mask_cgr = mask_for(tok, bad_val_cgr)
        p_cgr = build_prompt(lines, order, start, emitted[:t_pred], t_pred)
        txt_cgr = gen_step(model, tok, p_cgr, args.device, args.max_new, False, mask=mask_cgr)
        v_cgr = parse_value(txt_cgr)
        seq_cgr = emitted[:t_pred] + [v_cgr]
        if v_cgr is not None:
            seq_cgr = finish_chain(model, tok, lines, order, start, seq_cgr, t_pred + 1, steps, args.device, args.max_new)

        # Gate Check
        p_eval = build_prompt(lines, order, start, emitted[:t_pred], t_pred)
        val_orig_str = f" {emitted[t_pred]}\n" if (t_pred < len(emitted) and emitted[t_pred] is not None) else " None\n"
        val_rep_str = f" {seq_cgr[t_pred]}\n" if (t_pred < len(seq_cgr) and seq_cgr[t_pred] is not None) else " None\n"
        lp_orig, _ = readout(model, tok, p_eval, val_orig_str, args.device)
        lp_edit, _ = readout(model, tok, p_eval, val_rep_str, args.device)

        is_clean_orig = (lp_orig is not None and lp_orig > -0.25)
        is_degraded_edit = (lp_edit is not None and lp_orig is not None and (lp_edit < lp_orig - 0.35))

        if is_clean_orig and is_degraded_edit:
            arms["two_tier_gated_1rewind"]["rejected"] += 1
            final_seq_1r = emitted
        else:
            arms["two_tier_gated_1rewind"]["accepted"] += 1
            final_seq_1r = seq_cgr

        t_cgr_ms = (time.time() - t0_cgr) * 1000
        # KV Cache truncated tokens: prefix + 2*(H - t)
        enc_target = tok(p_cgr, return_tensors="pt")["input_ids"]
        tok_cgr = enc_target.shape[1] + (steps - t_pred) * 2

        ok_1r = (final_seq_1r == truth)
        arms["two_tier_gated_1rewind"]["n"] += 1
        arms["two_tier_gated_1rewind"]["ok"] += int(ok_1r)
        arms["two_tier_gated_1rewind"]["tokens"] += tok_cgr
        arms["two_tier_gated_1rewind"]["time_ms"] += t_cgr_ms
        arms["two_tier_gated_1rewind"]["recovered_list"].append(int(ok_1r))

        # -------------------------------------------------------------
        # Arm 4: Two-Tier Gated Max-2 Rewinds (Sequential Iteration)
        # -------------------------------------------------------------
        final_seq_2r = list(final_seq_1r)
        tok_2r = tok_cgr
        t_2r_ms = t_cgr_ms
        sec_rewind = 0

        # If chain 1 is not correct and was edited, check for secondary downstream error
        if not ok_1r and final_seq_1r != emitted and t_pred < steps - 2:
            # Check secondary error at downstream step t2 > t_pred
            t2 = t_pred + 2 # target downstream candidate
            bad_val_2 = int(final_seq_1r[t2]) if (t2 < len(final_seq_1r) and final_seq_1r[t2] is not None) else 0
            mask_2 = mask_for(tok, bad_val_2)
            p_2 = build_prompt(lines, order, start, final_seq_1r[:t2], t2)
            t0_2 = time.time()
            txt_2 = gen_step(model, tok, p_2, args.device, args.max_new, False, mask=mask_2)
            v_2 = parse_value(txt_2)
            seq_2 = final_seq_1r[:t2] + [v_2]
            if v_2 is not None:
                seq_2 = finish_chain(model, tok, lines, order, start, seq_2, t2 + 1, steps, args.device, args.max_new)
            t_2r_ms += (time.time() - t0_2) * 1000
            tok_2r += (steps - t2) * 2
            final_seq_2r = seq_2
            sec_rewind = 1

        ok_2r = (final_seq_2r == truth)
        arms["two_tier_gated_max2"]["n"] += 1
        arms["two_tier_gated_max2"]["ok"] += int(ok_2r)
        arms["two_tier_gated_max2"]["tokens"] += tok_2r
        arms["two_tier_gated_max2"]["time_ms"] += t_2r_ms
        arms["two_tier_gated_max2"]["recovered_list"].append(int(ok_2r))
        arms["two_tier_gated_max2"]["second_rewind_count"] += sec_rewind

        if (j + 1) % 10 == 0 or (j + 1) == len(rows):
            print(f"  {j + 1:>3}/{len(rows)} | "
                  f"Rewrite: {arms['full_rewrite']['ok']}/{arms['full_rewrite']['n']} | "
                  f"Oops: {arms['in_context_oops']['ok']}/{arms['in_context_oops']['n']} | "
                  f"CGR 1-Rewind: {arms['two_tier_gated_1rewind']['ok']}/{arms['two_tier_gated_1rewind']['n']} | "
                  f"CGR Max-2: {arms['two_tier_gated_max2']['ok']}/{arms['two_tier_gated_max2']['n']}",
                  flush=True)

    # Statistical Comparison
    n_tot = len(rows)
    stats_vs_rw = paired_stats(arms["two_tier_gated_1rewind"]["recovered_list"], arms["full_rewrite"]["recovered_list"])
    stats_vs_oops = paired_stats(arms["two_tier_gated_1rewind"]["recovered_list"], arms["in_context_oops"]["recovered_list"])
    stats_max2_vs_rw = paired_stats(arms["two_tier_gated_max2"]["recovered_list"], arms["full_rewrite"]["recovered_list"])

    print("\n" + "=" * 80)
    print(f"H=10 HORIZON SCALING FINAL RESULTS (N={n_tot})")
    print("=" * 80)
    print(f"{'Arm':<25} {'Recovered':>10} {'Rate':>7} {'95% Wilson CI':>18} {'Mean Tok':>10} {'Mean Latency':>12}")
    print("-" * 80)
    for name, a in arms.items():
        rate = a["ok"] / max(a["n"], 1)
        lo, hi = wilson(a["ok"], a["n"])
        m_tok = a["tokens"] / max(a["n"], 1)
        m_lat = a["time_ms"] / max(a["n"], 1)
        print(f"{name:<25} {a['ok']:>4}/{a['n']:<5} {rate:>7.3f} [{lo:.3f}, {hi:.3f}] {m_tok:>10.1f} {m_lat:>10.1f} ms")

    print("\n" + "=" * 80)
    print("PAIRED STATISTICAL HYPOTHESIS TESTS (H=10)")
    print("=" * 80)
    print(f"CGR 1-Rewind vs Full Rewrite:     {stats_vs_rw['wins_a']}W / {stats_vs_rw['wins_b']}L (p = {stats_vs_rw['p_value']:.4e})")
    print(f"CGR 1-Rewind vs In-Context Oops:  {stats_vs_oops['wins_a']}W / {stats_vs_oops['wins_b']}L (p = {stats_vs_oops['p_value']:.4e})")
    print(f"CGR Max-2 vs Full Rewrite:        {stats_max2_vs_rw['wins_a']}W / {stats_max2_vs_rw['wins_b']}L (p = {stats_max2_vs_rw['p_value']:.4e})")

    # Save to JSON
    out_dict = {
        "n": n_tot,
        "arms": {
            k: {
                "ok": v["ok"], "n": v["n"], "rate": v["ok"] / max(v["n"], 1),
                "ci": list(wilson(v["ok"], v["n"])),
                "tokens_mean": v["tokens"] / max(v["n"], 1),
                "time_ms_mean": v["time_ms"] / max(v["n"], 1),
            } for k, v in arms.items()
        },
        "paired_tests": {
            "vs_full_rewrite": stats_vs_rw,
            "vs_in_context_oops": stats_vs_oops,
            "max2_vs_full_rewrite": stats_max2_vs_rw,
        }
    }
    Path(args.out).write_text(json.dumps(out_dict, indent=2), encoding="utf-8")
    print(f"\nWrote H=10 scaling results to {args.out}")

if __name__ == "__main__":
    main()

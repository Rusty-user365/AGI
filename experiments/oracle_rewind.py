"""Does rewinding to the broken step actually fix anything?

This isolates the repair mechanism from the localizer. Localization is replaced
by the oracle (`first_error` from the generator), so the result is an **upper
bound** on what any localizer could achieve. If oracle rewind does not beat a
full rewrite, no observer can rescue the architecture, and the localizer
question stops mattering.

Arms, all starting from the verified prefix `truth[:t]`:

    A full rewrite      regenerate the whole chain, same decoding as the run
                        that broke (T=0.8). The "Oops, let me rewrite" baseline.
    B rewind greedy     re-decode step t greedily from the true prefix. For a
                        greedy-origin trajectory this must reproduce the same
                        wrong value -- the deterministic no-op.
    C rewind + hindsight  inject "# constraint: <name> is not <bad>" and re-decode
                        step t greedily. The Groundhog Day fix.
    C0 rewind + neutral line  the same extra prompt line with no information in
                        it ("# constraint: none"). The control for C: if C0
                        matches C, the constraint text is not what moved the
                        distribution, any perturbation is.
    D rewind + rank-2   forbid the bad value's first token at step t, take the
                        argmax of the rest, then greedy.
    E rewind + resample re-decode step t at T=0.8, greedy afterwards.
    Oracle force truth  set step t to the true value, greedy afterwards. Upper
                        bound on downstream recoverability.

Every arm records whether the whole chain ends correct and how many steps it
generated. Compute saving is `H - t` steps for B-E and 0 for A, so the arms are
comparable only on accuracy; the saving is arithmetic.

Note this corpus has no KV cache: each step is a fresh forward over the prefix,
so "steps generated" is the compute unit, not cache slicing.

Usage:
    python experiments/oracle_rewind.py --data results/arith_steps_h5.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.arith_pilot import load
from experiments.arith_steps import (build_prompt, parse_value, step_value)
from experiments._device import resolve as _resolve_device

try:
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover
    LogitsProcessor = object


class MaskBadFirstDigit(LogitsProcessor):
    """Forbid the first digit of a rejected value, leaving the space alone.

    Qwen tokenises ` 15` as [space, '1', '5'], so masking the first id would
    forbid the space rather than the value. The mask therefore stays on until a
    digit token has actually been emitted.
    """

    def __init__(self, bad_digit_id: int, digit_ids: list[int]):
        self.bad = bad_digit_id
        self.digits = set(int(d) for d in digit_ids)
        self.done = False

    def __call__(self, input_ids, scores):
        if not self.done:
            if int(input_ids[0, -1]) in self.digits:
                self.done = True
            else:
                scores[:, self.bad] = -float("inf")
        return scores


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


@torch.no_grad()
def gen_step(model, tok, prompt, device, max_new, sample, temperature=0.8,
             top_p=0.95, mask=None):
    enc = tok([prompt], return_tensors="pt").to(device)
    enc.pop("token_type_ids", None)
    kw = dict(do_sample=True, temperature=temperature, top_p=top_p) if sample \
        else dict(do_sample=False)
    if mask is not None:
        kw["logits_processor"] = [MaskBadFirstDigit(*mask)]
    gen = model.generate(**enc, max_new_tokens=max_new,
                         pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id, **kw)
    return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                            skip_special_tokens=True)[0]


def finish_chain(model, tok, lines, order, start, emitted, from_step, steps,
                 device, max_new):
    """Greedy completion of steps from_step..steps-1 using the emitted prefix."""
    for i in range(from_step, steps):
        prompt = build_prompt(lines, order, start, emitted, i)
        v = parse_value(gen_step(model, tok, prompt, device, max_new, False))
        emitted.append(v)
        if v is None:
            break
    return emitted


def run_arm(model, tok, task, arm, t, bad, args, device):
    lines, order, truth, start = task
    steps = len(truth)
    emitted = list(truth[:t])          # verified prefix
    generated = 0

    if arm == "full_rewrite":
        emitted = []
        for i in range(steps):
            prompt = build_prompt(lines, order, start, emitted, i)
            v = parse_value(gen_step(model, tok, prompt, device, args.max_new,
                                     True, args.temperature, args.top_p))
            emitted.append(v)
            generated += 1
            if v is None:
                break
        ok = len(emitted) == steps and all(emitted[i] == truth[i]
                                           for i in range(steps))
        return ok, generated, emitted

    if arm == "oracle_force":
        v = truth[t]
    else:
        constraint = None
        if arm == "hindsight":
            constraint = f"# constraint: {order[t]} is not {bad}"
        elif arm == "hindsight_neutral":
            constraint = "# constraint: none"
        mask = None
        if arm == "rank2":
            digit_ids = [tok(" " + str(d))["input_ids"][-1] for d in range(10)]
            ids = tok(" " + str(bad))["input_ids"]
            bad_digit = next(i for i in ids if i in digit_ids)
            mask = (bad_digit, digit_ids)
        prompt = build_prompt(lines, order, start, emitted, t,
                              constraint=constraint)
        text = gen_step(model, tok, prompt, device, args.max_new,
                        arm == "resample", args.temperature, args.top_p,
                        mask=mask)
        v = parse_value(text)
        generated += 1
        if v is None:
            return False, generated, emitted + [None]

    emitted.append(v)
    emitted = finish_chain(model, tok, lines, order, start, emitted, t + 1,
                           steps, device, args.max_new)
    generated += steps - t - 1
    ok = len(emitted) == steps and all(emitted[i] == truth[i]
                                       for i in range(steps))
    return ok, generated, emitted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="results/arith_steps_h5.pt")
    ap.add_argument("--model", default="checkpoints/hf/qwen3-0.6b-base")
    ap.add_argument("--arch", default="auto")
    ap.add_argument("--dtype", default="float32", choices=["float16", "float32"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-new", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=0, help="0 = all holdout rows")
    ap.add_argument("--out", default="results/oracle_rewind.json")
    args = ap.parse_args()
    args.device = str(_resolve_device(args.device))

    state = torch.load(Path(args.data), weights_only=False)
    steps = int(state["config"]["steps"])
    tasks = {(t["split"], t["idx"]): t for t in state["tasks"]}

    rows = [r for r in state["rows"]
            if r["split"] == "holdout" and r["complete"]
            and r["first_error"] is not None]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} broken holdout trajectories, steps={steps}")

    model, tok = load(Path(args.model).resolve(), torch.float32, args.device,
                      args.arch)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    arms = ["full_rewrite", "rewind_greedy", "hindsight", "hindsight_neutral",
            "rank2", "resample", "oracle_force"]
    res = {a: {"ok": 0, "n": 0, "steps": 0, "wins_vs_rewrite": 0,
               "losses_vs_rewrite": 0, "same_value": 0} for a in arms}
    per_row = []
    random.seed(0)

    for j, r in enumerate(rows):
        t = int(r["first_error"])
        bad = r["emitted"][t] if t < len(r["emitted"]) else None
        if bad is None:
            continue
        task = tasks[(r["split"], r["idx"])]
        task = (task["lines"], task["order"], task["truth"], task["start"])
        outcomes = {}
        for arm in arms:
            ok, generated, emitted = run_arm(model, tok, task, arm, t, bad,
                                             args, args.device)
            a = res[arm]
            a["n"] += 1
            a["ok"] += int(ok)
            a["steps"] += generated
            outcomes[arm] = ok
            if arm != "full_rewrite" and arm != "oracle_force":
                a["same_value"] += int(len(emitted) > t and emitted[t] == bad)
        base = outcomes["full_rewrite"]
        for arm in arms:
            if arm in ("full_rewrite", "oracle_force"):
                continue
            res[arm]["wins_vs_rewrite"] += int(outcomes[arm] and not base)
            res[arm]["losses_vs_rewrite"] += int(base and not outcomes[arm])
        per_row.append({"split": r["split"], "idx": r["idx"],
                        "cand": r["cand"], "t": t, "bad": bad,
                        "ok": outcomes})
        if (j + 1) % 10 == 0:
            print(f"  {j + 1}/{len(rows)}", flush=True)

    print(f"\n{'arm':<15} {'recovered':>9} {'rate':>6} {'95% CI':>15} "
          f"{'steps/gen':>9} {'same value':>10} {'W/L vs rewrite':>14}")
    for arm in arms:
        a = res[arm]
        n = max(a["n"], 1)
        lo, hi = wilson(a["ok"], a["n"])
        print(f"{arm:<15} {a['ok']:>4}/{a['n']:<4} {a['ok']/n:>6.3f} "
              f"[{lo:.3f}, {hi:.3f}] {a['steps']/n:>9.2f} "
              f"{a['same_value']/n:>10.3f} "
              f"{a['wins_vs_rewrite']:>5}W/{a['losses_vs_rewrite']:<4}L")

    out = {"config": vars(args), "n": len(rows), "steps": steps,
           "arms": {a: dict(res[a], rate=res[a]["ok"] / max(res[a]["n"], 1))
                    for a in arms},
           "per_row": per_row}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Step-by-step state tracking with per-step ground truth.

Why this corpus exists. `statetrack.py` records one answer per task (the final
counts), so there is no per-step label and `Localization Accuracy` is undefined,
not low. A Transformer localizer over un-pooled per-step states cannot be
trained or scored without knowing which step broke. This script emits the state
after every move, one line per step, and therefore gets `first_error` for free
from the generator.

The format is *demonstrated* in the few-shot block. The earlier probe showed the
model answering with the state after one move and never decrementing the source
box; but that prompt only ever showed a final answer, so the model was never
told to emit intermediate states. That is a prompt confound, not a capability
measurement, and it is fixed here.

Per-step labels:

    truth_states   states[0] = initial, states[i] = after move i
    pred_states    parsed from "Step i: a b" lines
    first_error    first i (0-based step) where pred != truth, or None
    recovered      any step after first_error matches truth again

Per-step confidence and features come from a single forward pass over
prompt + completion. Each step line's tokens are located with the tokenizer's
offset mapping, so no re-tokenisation guesswork is involved:

    step_logprob   mean token log-probability of that line
    step_feat      mean last-layer hidden state of that line  -> [H, d]

`step_feat` is pooled *within* a line but not *across* steps, which is exactly
the `[H, d]` input the localizer needs. Pooling across steps is what destroyed
position before.

Baselines reported per horizon, because the localizer must beat them:

    chance           1 / H on "which step broke"
    argmin logprob   point at the least confident step line
    argmax logprob   point at the most confident step line (sanity, should be bad)

The gate on the localizer is whether it beats `argmin logprob` on held-out
tasks. If it ties, confidence already localizes and a learned reader adds
nothing -- the same outcome as the two selection experiments.

Usage:
    python experiments/steplocal.py --horizon 10 --n-train 40 --n-holdout 20
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.arith_pilot import load

HOLDOUT_OFFSET = 10_000_000
STEP_RE = re.compile(r"Step\s*(\d+)\s*:\s*([^\n]*)")


# ---------------------------------------------------------------- corpus ----

def make_problem(rng: random.Random, moves: int, boxes: int = 2,
                 max_move: int = 1, start_max: int = 4,
                 min_total: int = 4) -> tuple[list[str], list[tuple[int, ...]]]:
    """Initial counts plus a valid move sequence, and the true state per step."""
    counts = [rng.randint(0, start_max) for _ in range(boxes)]
    if sum(counts) < min_total:
        counts[rng.randrange(boxes)] += min_total
    states = [tuple(counts)]
    lines = ["Boxes: " + " ".join(map(str, counts))]
    for _ in range(moves):
        src = rng.choice([b for b in range(boxes) if counts[b] >= 1])
        k = rng.randint(1, min(max_move, counts[src]))
        dst = rng.choice([b for b in range(boxes) if b != src])
        counts[src] -= k
        counts[dst] += k
        states.append(tuple(counts))
        lines.append(f"move {k} from box {src + 1} to box {dst + 1}")
    return lines, states


def render(lines: list[str], states: list[tuple[int, ...]]) -> str:
    out = [lines[0]]
    for i, ln in enumerate(lines[1:], 1):
        out.append(ln)
        out.append(f"Step {i}: " + " ".join(map(str, states[i])))
    return "\n".join(out)


def fewshot(boxes: int, max_move: int) -> str:
    """Two worked examples in the target format, generated so they cannot drift.

    Two horizons rather than one so the model cannot infer a fixed line count.
    """
    a, sa = make_problem(random.Random(12345), 2, boxes, max_move)
    b, sb = make_problem(random.Random(999), 4, boxes, max_move)
    return render(a, sa) + "\n\n" + render(b, sb) + "\n\n"


def parse_steps(text: str, boxes: int) -> dict[int, tuple[int, ...]]:
    """Step number -> state, for lines that are exactly `boxes` integers."""
    out: dict[int, tuple[int, ...]] = {}
    for m in STEP_RE.finditer(text):
        step = int(m.group(1))
        payload = m.group(2).strip()
        if re.sub(r"[\d\s,\(\)\[\]\{\}\.]", "", payload):
            continue
        nums = re.findall(r"\d+", payload)
        if len(nums) != boxes or step in out:
            continue
        out[step] = tuple(int(x) for x in nums)
    return out


def first_error(pred: dict[int, tuple[int, ...]], truth: list[tuple[int, ...]],
                horizon: int) -> tuple[int | None, bool]:
    """First step index (0-based) whose emitted state is wrong or missing."""
    fe = None
    for i in range(1, horizon + 1):
        if pred.get(i) != truth[i]:
            fe = i - 1
            break
    if fe is None:
        return None, False
    rec = any(pred.get(i) == truth[i] for i in range(fe + 2, horizon + 1))
    return fe, rec


# ------------------------------------------------------------ generation ----

@torch.no_grad()
def sample_k(model, tok, prompt: str, k: int, device: str, max_new: int,
             temperature: float, top_p: float) -> list[str]:
    enc = tok([prompt], return_tensors="pt").to(device)
    enc.pop("token_type_ids", None)
    gen = model.generate(
        **enc, max_new_tokens=max_new, do_sample=True, temperature=temperature,
        top_p=top_p, num_return_sequences=k, pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id)
    return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                            skip_special_tokens=True)


@torch.no_grad()
def greedy(model, tok, prompt: str, device: str, max_new: int) -> str:
    enc = tok([prompt], return_tensors="pt").to(device)
    enc.pop("token_type_ids", None)
    gen = model.generate(
        **enc, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                            skip_special_tokens=True)[0]


@torch.no_grad()
def step_readout(model, tok, prompt: str, text: str, device: str,
                 horizon: int) -> list[dict]:
    """Per-step logprob and mean hidden state, located by offset mapping."""
    full = prompt + text
    enc = tok(full, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc.pop("token_type_ids", None)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)
    logits = out.logits[0].float()
    hidden = out.hidden_states[-1][0].float()
    ids = enc["input_ids"][0]
    # token i (i >= 1) has logprob from position i-1
    lp_all = torch.log_softmax(logits[:-1], dim=-1).gather(
        -1, ids[1:].unsqueeze(-1)).squeeze(-1).cpu()

    spans: dict[int, tuple[int, int]] = {}
    for m in STEP_RE.finditer(full):
        step = int(m.group(1))
        if 1 <= step <= horizon and step not in spans:
            spans[step] = (m.start(), m.end())

    readout = []
    for step in range(1, horizon + 1):
        span = spans.get(step)
        if span is None:
            readout.append({"step": step, "logprob": None, "feat": None,
                            "tokens": 0})
            continue
        a, b = span
        idx = [i for i, (s, _e) in enumerate(offsets)
               if i >= 1 and a <= s < b]
        if not idx:
            readout.append({"step": step, "logprob": None, "feat": None,
                            "tokens": 0})
            continue
        sel = torch.tensor(idx)
        readout.append({
            "step": step,
            "logprob": float(lp_all[sel - 1].mean()),
            "feat": hidden[sel].mean(0).cpu(),
            "tokens": len(idx),
        })
    return readout


# ----------------------------------------------------------------- stats ----

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def summarize(state: dict, horizon: int, splits: list[str], k: int) -> dict:
    summary = {"horizon": horizon, "splits": {}}
    for split in splits:
        rows = [r for r in state["rows"] if r["split"] == split]
        greedy_rows = [r for r in rows if r["cand"] == 0]
        samp = [r for r in rows if r["cand"] > 0]
        full = [r for r in samp if r["emitted"] == horizon]

        hits: dict[int, int] = {}
        for r in samp:
            hits.setdefault(r["idx"], 0)
            hits[r["idx"]] += int(r["seq_correct"])
        lo, hi = wilson(sum(1 for h in hits.values() if h > 0), len(hits))

        # Unconditional per-step accuracy and conditional (given prefix clean).
        step_acc, cond_acc = [], []
        for i in range(horizon):
            ok = [r for r in full if all(r["step_ok"][:i])]
            step_acc.append(sum(r["step_ok"][i] for r in full)
                            / max(len(full), 1))
            cond_acc.append(sum(r["step_ok"][i] for r in ok) / max(len(ok), 1))

        # Where errors land, over candidates that emitted every step.
        dist = {str(i): 0 for i in range(horizon)}
        dist["none"], dist["missing"] = 0, 0
        for r in samp:
            if r["emitted"] < horizon:
                dist["missing"] += 1
            elif r["first_error"] is None:
                dist["none"] += 1
            else:
                dist[str(r["first_error"])] += 1

        # Localization by confidence, on candidates with an interior error.
        elig = [r for r in full if r["first_error"] is not None
                and r["first_error"] < horizon - 1
                and all(s["logprob"] is not None for s in r["steps"])]
        lp_hit = lp_at_or_after = 0
        for r in elig:
            lps = [s["logprob"] for s in r["steps"]]
            pick = min(range(horizon), key=lambda i: lps[i])
            lp_hit += int(pick == r["first_error"])
            lp_at_or_after += int(pick >= r["first_error"])

        summary["splits"][split] = {
            "tasks": len({r["idx"] for r in rows}),
            "candidates": len(rows),
            "emitted_rate": sum(r["emitted"] for r in rows) / max(len(rows), 1),
            "seq_p1_greedy": (sum(r["seq_correct"] for r in greedy_rows)
                              / max(len(greedy_rows), 1)),
            "seq_p1_sampled": (sum(r["seq_correct"] for r in samp)
                               / max(len(samp), 1)),
            "seq_pass_at_k": (sum(1 for h in hits.values() if h > 0)
                              / max(len(hits), 1)),
            "seq_pass_at_k_ci": [lo, hi],
            "first_error_dist": dist,
            "first_error_missing_frac": dist["missing"] / max(len(samp), 1),
            "first_error_none_frac": dist["none"] / max(len(samp), 1),
            "step_acc": step_acc,
            "cond_step_acc": cond_acc,
            "recovered_frac": (sum(r["recovered"] for r in full
                                   if r["first_error"] is not None)
                               / max(sum(1 for r in full
                                         if r["first_error"] is not None), 1)),
            "localize_eligible": len(elig),
            "localize_chance": 1.0 / horizon,
            "localize_argmin_logprob": lp_hit / max(len(elig), 1),
            "localize_at_or_after": lp_at_or_after / max(len(elig), 1),
        }
    return summary


def print_summary(summary: dict, k: int) -> None:
    h = summary["horizon"]
    for split, s in summary["splits"].items():
        print(f"\n{split}  horizon {h}  tasks {s['tasks']}  "
              f"emitted {s['emitted_rate']:.3f}")
        print(f"  seq p1_greedy {s['seq_p1_greedy']:.3f}  "
              f"p1_sampled {s['seq_p1_sampled']:.3f}  "
              f"pass@{k} {s['seq_pass_at_k']:.3f} "
              f"[{s['seq_pass_at_k_ci'][0]:.3f}, {s['seq_pass_at_k_ci'][1]:.3f}]")
        print("  first_error dist: " + "  ".join(
            f"{key}:{val}" for key, val in s["first_error_dist"].items()))
        print("  step acc      : " + " ".join(
            f"{x:.2f}" for x in s["step_acc"]))
        print("  cond step acc : " + " ".join(
            f"{x:.2f}" for x in s["cond_step_acc"]))
        print(f"  recovered {s['recovered_frac']:.3f}  "
              f"localize n={s['localize_eligible']}  "
              f"chance {s['localize_chance']:.3f}  "
              f"argmin-lp {s['localize_argmin_logprob']:.3f}  "
              f"at-or-after {s['localize_at_or_after']:.3f}")


# ------------------------------------------------------------------ main ----

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="checkpoints/hf/qwen3-0.6b-base")
    ap.add_argument("--arch", default="auto", choices=["dlm", "auto"])
    ap.add_argument("--dtype", default="float32",
                    choices=["float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--boxes", type=int, default=2)
    ap.add_argument("--max-move", type=int, default=1)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-holdout", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=0,
                    help="0 = 14 * horizon + 24")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    args.max_new = args.max_new or 14 * args.horizon + 24
    tag = f"b{args.boxes}m{args.max_move}_h{args.horizon}"
    args.out = args.out or f"results/steplocal_{tag}.pt"
    args.summary = args.summary or f"results/steplocal_{tag}.json"
    out_path = Path(args.out)
    state = {"rows": [], "tasks": [], "config": vars(args)}
    if args.resume and out_path.exists():
        state = torch.load(out_path, weights_only=False)
        print(f"resumed: {len(state['tasks'])} tasks already done")
    done = {(t["split"], t["idx"]) for t in state["tasks"]}

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model, tok = load(Path(args.model).resolve() if Path(args.model).is_dir()
                      else args.model, dtype, args.device, args.arch)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"loaded, {sum(p.numel() for p in model.parameters()):,} params, "
          f"{args.dtype}, horizon={args.horizon}, k={args.k}", flush=True)

    shot = fewshot(args.boxes, args.max_move)
    plan = [("train", args.n_train), ("holdout", args.n_holdout)]
    todo = sum(1 for s, n in plan for i in range(n) if (s, i) not in done)
    print(f"{todo} tasks to build", flush=True)
    t0 = time.time()
    built = 0

    for split, n in plan:
        for idx in range(n):
            if (split, idx) in done:
                continue
            seed = idx if split == "train" else HOLDOUT_OFFSET + idx
            lines, truth = make_problem(random.Random(seed), args.horizon,
                                        args.boxes, args.max_move)
            body = "\n".join(lines)
            prompt = shot + body + "\nStep 1:"

            texts = [greedy(model, tok, prompt, args.device, args.max_new)]
            texts += sample_k(model, tok, prompt, args.k, args.device,
                              args.max_new, args.temperature, args.top_p)

            for ci, text in enumerate(texts):
                pred = parse_steps(text, args.boxes)
                fe, rec = first_error(pred, truth, args.horizon)
                steps = step_readout(model, tok, prompt, text, args.device,
                                     args.horizon)
                step_ok = [int(pred.get(i) == truth[i])
                           for i in range(1, args.horizon + 1)]
                state["rows"].append({
                    "split": split, "idx": idx, "seed": seed, "cand": ci,
                    "text": text,
                    "emitted": len(pred),
                    "pred": {str(k): list(v) for k, v in pred.items()},
                    "first_error": fe, "recovered": int(rec),
                    "seq_correct": int(all(step_ok)),
                    "step_ok": step_ok,
                    "steps": steps,
                })
            state["tasks"].append({
                "split": split, "idx": idx, "seed": seed, "body": body,
                "truth": [list(s) for s in truth], "prompt": prompt,
            })
            built += 1
            if built % args.save_every == 0 or built == todo:
                save(out_path, state)
                rate = built / max(time.time() - t0, 1e-6)
                print(f"  {built}/{todo} tasks  {rate:.3f} task/s  "
                      f"eta {(todo - built) / max(rate, 1e-6) / 60:.1f} min",
                      flush=True)
    save(out_path, state)

    summary = summarize(state, args.horizon, [s for s, _ in plan], args.k)
    summary["config"] = vars(args)
    print_summary(summary, args.k)

    shown = 0
    for r in state["rows"]:
        if shown >= args.show:
            break
        if r["emitted"] == args.horizon and r["first_error"] is not None:
            t = next(x for x in state["tasks"]
                     if x["split"] == r["split"] and x["idx"] == r["idx"])
            print(f"\n--- error at step {r['first_error'] + 1} "
                  f"({r['split']} #{r['idx']}) ---")
            print(t["body"])
            for i, (p, q) in enumerate(zip(t["truth"][1:], range(1, 99)), 1):
                if i > args.horizon:
                    break
                got = r["pred"].get(str(i))
                mark = "ok " if got == p else "BAD"
                print(f"  {mark} step {i}: truth {p}  got {got}")
            shown += 1

    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

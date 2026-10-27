"""Step-by-step arithmetic chains with per-step ground truth.

Why not the box world. `statetrack.py` and `steplocal.py` both failed for the
same reason: neither model available here can apply a single 1-ball move.
Measured on 40 single moves each -- Qwen3-0.6B-Base 22.5%, Qwen2.5-1.5B-Instruct
2.5% raw / 15% chat at 2 boxes. With step 1 already wrong, `first_error` is 0
for every trajectory and localization is undefined, not low. See
findings/2026-10-27-state-tracking-below-floor.md.

Arithmetic chains clear that floor: the same 0.6B scored p1 0.544 on 2-step
chains (results/arith_bench_s2.pt). So this corpus asks the model to emit the
value of each assignment in order, one number per step, conditioned on its own
previous values. That gives three things the earlier corpora could not:

    per-step truth   the generator knows every intermediate value
    first_error      first step whose emitted value is wrong or unparseable
    self-conditioning  step i sees the model's own (possibly wrong) values, so
                     an error propagates exactly as it would in a DAG run

Format. Definitions use `=`, emitted values use `->`, which removes the
ambiguity that made the box-world prompt re-emit `move` lines:

    x = 5
    y = x * 2
    z = y + 1
    x -> 5
    y -> 10
    z ->

Each step is one forward pass: the prompt is the definition block plus the
values emitted so far, and the model emits one integer. Per step this records

    value      parsed integer, or None
    logprob    mean token log-probability of the emitted number
    feat       mean final hidden state of the emitted number  -> [steps, d]

`feat` is pooled within a step but not across steps: the `[H, d]` input a
localizer needs. Pooling across steps is what destroyed position before.

Baselines, because the localizer must beat them rather than chance:

    chance            1 / H
    always step 1     errors are geometrically distributed, so this is not 1/H
    argmin logprob    point at the least confident step

Admission is "the first line is a bare integer", applied identically to greedy
and sampled arms. A wrong value is well formed by construction: `11` and `12`
parse the same.

Usage:
    python experiments/arith_steps.py --steps 5 --n-train 60 --n-holdout 40
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

from experiments.arith_pilot import load, make_problem

HOLDOUT_OFFSET = 10_000_000

FEWSHOT = (
    "x = 2\n"
    "y = x + 3\n"
    "x -> 2\n"
    "y -> 5\n\n"
    "x = 4\n"
    "y = x * 2\n"
    "z = y - 1\n"
    "x -> 4\n"
    "y -> 8\n"
    "z -> 7\n\n"
)

BARE_INT = re.compile(r"^[-+]?\d+$")


def make_chain(rng: random.Random, steps: int) -> tuple[list[str], list[str],
                                                       list[int], int]:
    """Definitions, computed variable names, their values, and the start value.

    Mirrors `arith_pilot.make_problem` draw for draw (same rng call order and
    same op-repair rules), so the final value here equals that function's answer
    for the same seed. The cross-check in scratch/test_arith_steps.py holds that
    invariant.
    """
    names = list("abcdefgh")
    if steps >= len(names):
        raise ValueError(f"steps must be < {len(names)}, got {steps}")
    val = {names[0]: rng.randint(1, 9)}
    start = val[names[0]]
    lines = [f"{names[0]} = {start}"]
    order: list[str] = []
    values: list[int] = []
    for i in range(steps):
        tgt = names[i + 1]
        src = rng.choice([n for n in val])
        op = rng.choice(["+", "-", "*"])
        k = rng.randint(1, 9)
        v = val[src]
        if op == "-" and v - k < 0:
            op = "+"
        if op == "*" and v * k > 60:
            op = "+"
        new = v - k if op == "-" else (v * k if op == "*" else v + k)
        lines.append(f"{tgt} = {src} {op} {k}")
        val[tgt] = new
        order.append(tgt)
        values.append(new)
    return lines, order, values, start


def parse_value(text: str) -> int | None:
    """Admission: the first line is a bare integer."""
    first = text.strip().split("\n")[0].strip()
    return int(first) if BARE_INT.match(first) else None


def build_prompt(lines: list[str], order: list[str], start: int,
                 emitted: list[int | None], step: int,
                 constraint: str | None = None) -> str:
    """Definition block, then the values so far, then the name to fill in.

    `constraint` is an optional line injected immediately before the name to
    fill in, used by the hindsight arm of `oracle_rewind.py`.
    """
    first_name = lines[0].split(" = ")[0]
    vals = [f"{first_name} -> {start}"]
    for j in range(step):
        vals.append(f"{order[j]} -> {emitted[j]}")
    if constraint:
        vals.append(constraint)
    vals.append(f"{order[step]} ->")
    return FEWSHOT + "\n".join(lines) + "\n" + "\n".join(vals)


# ------------------------------------------------------------ generation ----

@torch.no_grad()
def step_value(model, tok, prompt: str, device: str, max_new: int,
               sample: bool, temperature: float, top_p: float) -> str:
    enc = tok([prompt], return_tensors="pt").to(device)
    enc.pop("token_type_ids", None)
    kw = dict(do_sample=True, temperature=temperature, top_p=top_p) if sample \
        else dict(do_sample=False)
    gen = model.generate(
        **enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id, **kw)
    return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                            skip_special_tokens=True)[0]


@torch.no_grad()
def readout(model, tok, prompt: str, text: str, device: str):
    """Mean logprob and mean hidden state of the emitted number."""
    p_ids = tok(prompt)["input_ids"]
    ids = tok(prompt + text)["input_ids"]
    plen = len(p_ids)
    if ids[:plen] != p_ids:
        plen = 0
        for a, b in zip(ids, p_ids):
            if a != b:
                break
            plen += 1
    if plen >= len(ids):
        return None, None
    t = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(t, output_hidden_states=True)
    lp = torch.log_softmax(out.logits[0, plen - 1:len(ids) - 1].float(), dim=-1)
    tgt = t[0, plen:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (float(tok_lp.mean()),
            out.hidden_states[-1][0, plen:].mean(0).float().cpu())


@torch.no_grad()
def batched_step_value(model, tok, prompts: list[str], device: str,
                       max_new: int, sample: bool, temperature: float,
                       top_p: float) -> list[str]:
    """One generation call for several candidates at the same step.

    The tokenizer is left-padding (set by `load`), which is what decoder-only
    generation needs. Batching does not reduce FLOPs on this card, but it removes
    the per-call launch overhead, which dominates when each call emits 1-2 tokens.
    """
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    enc.pop("token_type_ids", None)
    kw = dict(do_sample=True, temperature=temperature, top_p=top_p) if sample \
        else dict(do_sample=False)
    gen = model.generate(
        **enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id, **kw)
    new = gen[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(new, skip_special_tokens=True)


@torch.no_grad()
def batched_readout(model, tok, prompts: list[str], texts: list[str],
                    device: str) -> list[tuple[float | None, torch.Tensor | None]]:
    """Batched logprob and mean hidden state, segmented by character offsets.

    Offset mapping locates the prompt/completion boundary exactly, so a BPE merge
    across it cannot silently shift the span the way a token-count split can.
    """
    full = [p + t for p, t in zip(prompts, texts)]
    old = tok.padding_side
    tok.padding_side = "right"
    try:
        enc = tok(full, return_tensors="pt", padding=True,
                  return_offsets_mapping=True)
    finally:
        tok.padding_side = old
    offsets = enc.pop("offset_mapping")
    enc.pop("token_type_ids", None)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    lp = torch.log_softmax(out.logits.float()[:, :-1], dim=-1).gather(
        -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    hidden = out.hidden_states[-1].float()
    res = []
    for i, p in enumerate(prompts):
        length = int(mask[i].sum())
        off = offsets[i].tolist()
        start = next((j for j in range(length) if off[j][1] > len(p)), None)
        if start is None or start == 0 or start >= length:
            res.append((None, None))
            continue
        res.append((float(lp[i, start - 1:length - 1].mean()),
                    hidden[i, start:length].mean(0).cpu()))
    return res


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
    """Atomic save that survives a transient Windows file lock.

    `Path.replace` fails with PermissionError if any other process (an analysis
    script reading the corpus, a virus scanner) has the target open for even an
    instant. A crash here costs the whole run, so retry before giving up.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    for attempt in range(12):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    tmp.replace(path)


def summarize(state: dict, steps: int, splits: list[str], k: int) -> dict:
    summary = {"steps": steps, "splits": {}}
    for split in splits:
        rows = [r for r in state["rows"] if r["split"] == split]
        greedy_rows = [r for r in rows if r["cand"] == 0]
        samp = [r for r in rows if r["cand"] > 0]
        full = [r for r in samp if r["complete"]]

        hits: dict[int, int] = {}
        for r in samp:
            hits.setdefault(r["idx"], 0)
            hits[r["idx"]] += int(r["chain_correct"])
        lo, hi = wilson(sum(1 for h in hits.values() if h > 0), len(hits))

        step_acc, cond_acc = [], []
        for i in range(steps):
            pref = [r for r in samp if all(r["step_ok"][:i])]
            step_acc.append(sum(r["step_ok"][i] for r in samp)
                            / max(len(samp), 1))
            cond_acc.append(sum(r["step_ok"][i] for r in pref) / max(len(pref), 1))

        dist = {str(i): 0 for i in range(steps)}
        dist["none"], dist["incomplete"] = 0, 0
        for r in samp:
            if not r["complete"]:
                dist["incomplete"] += 1
            elif r["first_error"] is None:
                dist["none"] += 1
            else:
                dist[str(r["first_error"])] += 1

        # Localization, on complete chains with an interior error.
        elig = [r for r in full if r["first_error"] is not None
                and all(s["logprob"] is not None for s in r["steps"])]
        lp_hit = lp_after = first_hit = 0
        for r in elig:
            lps = [s["logprob"] for s in r["steps"]]
            pick = min(range(steps), key=lambda i: lps[i])
            lp_hit += int(pick == r["first_error"])
            lp_after += int(pick >= r["first_error"])
            first_hit += int(r["first_error"] == 0)

        summary["splits"][split] = {
            "tasks": len({r["idx"] for r in rows}),
            "candidates": len(rows),
            "complete_rate": sum(r["complete"] for r in samp)
                             / max(len(samp), 1),
            "chain_p1_greedy": sum(r["chain_correct"] for r in greedy_rows)
                               / max(len(greedy_rows), 1),
            "chain_p1_sampled": sum(r["chain_correct"] for r in samp)
                                / max(len(samp), 1),
            "chain_pass_at_k": sum(1 for h in hits.values() if h > 0)
                               / max(len(hits), 1),
            "chain_pass_at_k_ci": [lo, hi],
            "step_acc": step_acc,
            "cond_step_acc": cond_acc,
            "first_error_dist": dist,
            "first_error_none_frac": dist["none"] / max(len(samp), 1),
            "first_error_incomplete_frac": dist["incomplete"] / max(len(samp), 1),
            "localize_eligible": len(elig),
            "localize_chance": 1.0 / steps,
            "localize_always_first": first_hit / max(len(elig), 1),
            "localize_argmin_logprob": lp_hit / max(len(elig), 1),
            "localize_at_or_after": lp_after / max(len(elig), 1),
        }
    return summary


def print_summary(summary: dict, k: int) -> None:
    h = summary["steps"]
    for split, s in summary["splits"].items():
        print(f"\n{split}  steps {h}  tasks {s['tasks']}  "
              f"complete {s['complete_rate']:.3f}")
        print(f"  chain p1_greedy {s['chain_p1_greedy']:.3f}  "
              f"p1_sampled {s['chain_p1_sampled']:.3f}  "
              f"pass@{k} {s['chain_pass_at_k']:.3f} "
              f"[{s['chain_pass_at_k_ci'][0]:.3f}, {s['chain_pass_at_k_ci'][1]:.3f}]")
        print("  step acc      : " + " ".join(f"{x:.2f}" for x in s["step_acc"]))
        print("  cond step acc : " + " ".join(
            f"{x:.2f}" for x in s["cond_step_acc"]))
        print("  first_error   : " + "  ".join(
            f"{a}:{b}" for a, b in s["first_error_dist"].items()))
        print(f"  localize n={s['localize_eligible']}  "
              f"chance {s['localize_chance']:.3f}  "
              f"always-first {s['localize_always_first']:.3f}  "
              f"argmin-lp {s['localize_argmin_logprob']:.3f}  "
              f"at-or-after {s['localize_at_or_after']:.3f}")


# ------------------------------------------------------------------ main ----

def run_candidates(model, tok, task, args, device) -> dict[int, dict]:
    """All candidates for one task, batched per step.

    Greedy (cand 0) and the k samples are generated in separate calls because
    they need different decoding, but each call covers every live candidate, so a
    task costs 2 calls per step instead of 2 per candidate.
    """
    lines, order, truth, start = task
    rows: dict[int, dict] = {}
    for sample, idxs in ((False, [0]), (True, list(range(1, args.k + 1)))):
        alive = list(idxs)
        emitted: dict[int, list] = {ci: [] for ci in idxs}
        rec: dict[int, list] = {ci: [] for ci in idxs}
        for i in range(args.steps):
            if not alive:
                break
            prompts = [build_prompt(lines, order, start, emitted[ci], i)
                       for ci in alive]
            texts = batched_step_value(model, tok, prompts, device,
                                       args.max_new, sample, args.temperature,
                                       args.top_p)
            reads = batched_readout(model, tok, prompts, texts, device)
            for ci, text, (lp, feat) in zip(alive, texts, reads):
                v = parse_value(text)
                emitted[ci].append(v)
                rec[ci].append({"step": i, "text": text, "value": v,
                                "truth": truth[i], "logprob": lp,
                                "feat": feat, "ok": int(v == truth[i])})
            alive = [ci for ci in alive if emitted[ci][-1] is not None]
        for ci in idxs:
            steps_rec = rec[ci]
            step_ok = [int(s["ok"]) for s in steps_rec]
            complete = len(steps_rec) == args.steps
            fe = (next((i for i, ok in enumerate(step_ok) if not ok), None)
                  if complete else len(step_ok) - 1)
            rows[ci] = {"steps": steps_rec, "emitted": emitted[ci],
                        "truth": truth, "step_ok": step_ok,
                        "complete": int(complete),
                        "chain_correct": int(complete and all(step_ok)),
                        "first_error": fe}
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="checkpoints/hf/qwen3-0.6b-base")
    ap.add_argument("--arch", default="auto", choices=["dlm", "auto"])
    ap.add_argument("--dtype", default="float32",
                    choices=["float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-holdout", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    args.out = args.out or f"results/arith_steps_h{args.steps}.pt"
    args.summary = args.summary or f"results/arith_steps_h{args.steps}.json"
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
          f"{args.dtype}, steps={args.steps}, k={args.k}", flush=True)

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
            lines, order, truth, start = make_chain(random.Random(seed),
                                                    args.steps)

            rows = run_candidates(model, tok,
                                  (lines, order, truth, start), args,
                                  args.device)
            for ci, r in rows.items():
                state["rows"].append({"split": split, "idx": idx, "seed": seed,
                                      "cand": ci, **r})
            state["tasks"].append({"split": split, "idx": idx, "seed": seed,
                                   "lines": lines, "order": order,
                                   "truth": truth, "start": start})
            built += 1
            if built % args.save_every == 0 or built == todo:
                save(out_path, state)
                rate = built / max(time.time() - t0, 1e-6)
                print(f"  {built}/{todo} tasks  {rate:.3f} task/s  "
                      f"eta {(todo - built) / max(rate, 1e-6) / 60:.1f} min",
                      flush=True)
    save(out_path, state)

    summary = summarize(state, args.steps, [s for s, _ in plan], args.k)
    summary["config"] = vars(args)
    print_summary(summary, args.k)

    shown = 0
    for r in state["rows"]:
        if shown >= args.show:
            break
        if r["complete"] and r["first_error"] is not None:
            t = next(x for x in state["tasks"]
                     if x["split"] == r["split"] and x["idx"] == r["idx"])
            print(f"\n--- error at step {r['first_error'] + 1} "
                  f"({r['split']} #{r['idx']}) ---")
            print("\n".join(t["lines"]))
            print("  truth:", t["truth"])
            print("  model:", r["emitted"])
            print("  step logprobs:", [None if s["logprob"] is None
                                       else round(s["logprob"], 3)
                                       for s in r["steps"]])
            shown += 1

    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

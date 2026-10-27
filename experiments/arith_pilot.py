"""Arithmetic/state-tracking pilot on a backbone, for selection experiments.

Purpose: find a task where a candidate can be **wrong and well formed**. The
repair benchmark cannot supply one -- 480 of 480 candidates were either a
correct thesis name or unparseable, so a parser achieved perfect discrimination
and no observer could be distinguished from a regex
(see findings/2026-10-27-benchmark-has-no-plausible-wrong-answers.md).

`17 * 3 = 51` and `17 * 3 = 48` parse identically and only one is right. That
is the property a routing experiment needs.

Reported per difficulty rung (number of chained operations):

  p1_greedy     single deterministic pass
  p1_sampled    single sample at the temperature used for best-of-n
  pass@k        any of k samples correct -- the ceiling for any selector
  wellformed    fraction of continuations that contain an integer at all
  wrong_wellformed  count of answers that parse and are wrong. If this is zero
                    the rung is as useless as the repair corpus.

The acceptance window is fixed before the run: a rung is usable when
p1 in [0.15, 0.50] and pass@k >= p1 + 0.15. A rung that is too easy cannot
show selection, and one at chance has nothing to select.

Usage:
    python experiments/arith_pilot.py --model checkpoints/hf/dual-lm-470m --n 40
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer, LlamaConfig

NUM_RE = re.compile(r"-?\d+")

FEWSHOT = (
    "x = 2\n"
    "y = x + 3\n"
    "y = ?\n"
    "Answer: 5\n\n"
    "x = 4\n"
    "y = x * 2\n"
    "z = y - 1\n"
    "z = ?\n"
    "Answer: 7\n\n"
)


def make_problem(rng: random.Random, steps: int) -> tuple[str, int]:
    """A chain of `steps` assignments with small non-negative values."""
    names = ["a", "b", "c", "d", "e"]
    val = {names[0]: rng.randint(1, 9)}
    lines = [f"{names[0]} = {val[names[0]]}"]
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
    last = names[steps]
    return "\n".join(lines + [f"{last} = ?"]), val[last]


def score(text: str) -> int | None:
    m = NUM_RE.search(text)
    return int(m.group()) if m else None


def load(model_dir: Path, dtype: torch.dtype, device: str):
    sys.path.insert(0, str(model_dir))
    import modeling_dlm_llama as m

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = LlamaConfig.from_pretrained(str(model_dir))
    model = m.DLMLlamaForCausalLM.from_pretrained(
        str(model_dir), config=cfg, torch_dtype=dtype)
    model.to(device).eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompts: list[str], device: str, max_new: int,
             sample: bool, temperature: float, top_p: float, batch: int) -> list[str]:
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        enc.pop("token_type_ids", None)
        gen = model.generate(
            **enc, max_new_tokens=max_new, do_sample=sample,
            temperature=temperature if sample else None,
            top_p=top_p if sample else None,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(new, skip_special_tokens=True))
        print(f"    {min(i + batch, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def run_rung(model, tok, device, steps: int, n: int, k: int, seed: int,
             batch: int, max_new: int, temperature: float) -> dict:
    rng = random.Random(seed + 1000 * steps)
    items = [make_problem(rng, steps) for _ in range(n)]
    prompts = [FEWSHOT + body + "\nAnswer:" for body, _ in items]
    answers = [ans for _, ans in items]

    greedy = generate(model, tok, prompts, device, max_new, False, 0.0, 1.0, batch)
    pred_g = [score(t) for t in greedy]

    sampled = generate(model, tok, prompts, device, max_new, True, temperature, 0.95, batch)
    pred_s = [score(t) for t in sampled]

    hits = [0] * n
    wrong_wf = 0
    wellformed = 0
    # Same prompts as the greedy pass: the samples must differ by decoding, not
    # by problem. Regenerating them with a different seed would score sample s
    # against a different item's answer.
    for _s in range(k):
        out = generate(model, tok, prompts, device, max_new, True, temperature, 0.95, batch)
        for j, t in enumerate(out):
            v = score(t)
            if v is not None:
                wellformed += 1
                if v == answers[j]:
                    hits[j] += 1
                else:
                    wrong_wf += 1

    p1_g = sum(1 for v, a in zip(pred_g, answers) if v == a) / n
    p1_s = sum(1 for v, a in zip(pred_s, answers) if v == a) / n
    passk = sum(1 for h in hits if h > 0) / n
    return {
        "steps": steps,
        "n": n,
        "k": k,
        "p1_greedy": p1_g,
        "p1_sampled": p1_s,
        "pass_at_k": passk,
        "lift": passk - p1_s,
        "wellformed_rate": wellformed / (n * k),
        "wrong_wellformed": wrong_wf,
        "examples": [{"problem": items[i][0], "answer": answers[i],
                      "greedy": greedy[i].strip()[:40],
                      "sampled": sampled[i].strip()[:40]} for i in range(min(3, n))],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="checkpoints/hf/dual-lm-470m")
    ap.add_argument("--n", type=int, default=40, help="problems per rung")
    ap.add_argument("--k", type=int, default=8, help="samples for pass@k")
    ap.add_argument("--steps", default="1,2,3,4", help="rungs, comma separated")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/arith_pilot.json")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model, tok = load(Path(args.model).resolve(), dtype, args.device)
    print(f"loaded, {sum(p.numel() for p in model.parameters()):,} params, {args.dtype}")

    rungs = [int(s) for s in args.steps.split(",")]
    report = {"model": args.model, "dtype": args.dtype, "n": args.n, "k": args.k,
              "temperature": args.temperature, "rungs": {}}

    print(f"\n{'steps':>5} {'p1_greedy':>10} {'p1_samp':>8} {'pass@k':>7} "
          f"{'lift':>6} {'wellfmd':>8} {'wrong_wf':>9}")
    for steps in rungs:
        r = run_rung(model, tok, args.device, steps, args.n, args.k, args.seed,
                     args.batch, args.max_new, args.temperature)
        report["rungs"][steps] = r
        print(f"{steps:>5} {r['p1_greedy']:>10.3f} {r['p1_sampled']:>8.3f} "
              f"{r['pass_at_k']:>7.3f} {r['lift']:>+6.3f} "
              f"{r['wellformed_rate']:>8.3f} {r['wrong_wellformed']:>9}")

    usable = [s for s, r in report["rungs"].items()
              if 0.15 <= r["p1_sampled"] <= 0.50 and r["lift"] >= 0.15
              and r["wrong_wellformed"] > 0]
    report["usable_rungs"] = usable
    print(f"\nusable rungs (p1 in [0.15,0.50], lift >= 0.15, wrong-but-wellformed > 0): "
          f"{usable if usable else 'none'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

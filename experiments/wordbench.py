"""Notebook-and-pen word problems, for testing selection on a fluent-but-wrong generator.

The repair corpus could not test routing because every wrong candidate was
unparseable, so a regex matched the observer
(findings/2026-10-27-benchmark-has-no-plausible-wrong-answers.md). This corpus
fixes that at the scoring rule instead of at the generator:

    A candidate is *admitted* only if its completion is a bare integer.

So every admitted candidate is well formed by construction, and a wrong one is
wrong for arithmetic reasons rather than formatting reasons. The validity
filter is applied identically to every arm downstream, so no arm gets a parser
the others lack.

The task family is a multi-step money problem with a numeric distractor, the
shape that reliably makes small instruct models answer fluently and wrongly:

    A notebook costs $3 and a pen costs $2. I buy 4 notebooks and 5 pens.
    I already own 2 erasers. How much change do I get from $50?

The distractor ("2 erasers") is irrelevant but numeric, so a model that adds
everything gets a plausible wrong answer rather than a malformed one.

Levels:
    0  total cost          (one multiply-add)
    1  change from $50/$100 (multiply-add, then subtract)
    2  item count          (no price arithmetic -- a control rung)

Usage:
    python experiments/wordbench.py --n-train 8 --n-holdout 4 --level 1
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
from experiments._device import resolve as _resolve_device

HOLDOUT_OFFSET = 10_000_000

# (singular, min price, max price)
OBJECTS = [
    ("notebook", 2, 5),
    ("pen", 1, 3),
    ("pencil", 1, 2),
    ("eraser", 1, 2),
    ("folder", 3, 6),
    ("marker", 2, 4),
]

BARE_INT = re.compile(r"^[\s$]*(-?\d+)[\s.$]*$")
ANY_INT = re.compile(r"-?\d+")

SYSTEM = ("You solve arithmetic word problems. Reply with a single integer "
          "and nothing else.")


def _a(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


def make_problem(rng: random.Random, level: int, distractor: bool = True,
                 items: int = 2) -> tuple[str, int]:
    """A money problem, optionally with a numeric distractor that must be ignored."""
    if items == 1:
        (o1, lo1, hi1) = rng.choice(OBJECTS)
        p1 = rng.randint(lo1, hi1)
        a = rng.randint(2, 6)
        total = a * p1
        text = f"{_a(o1).capitalize()} costs ${p1}. I buy {a} {o1}s. "
        if distractor:
            o3 = rng.choice([o for o, _, _ in OBJECTS if o != o1])
            text += f"I already own {rng.randint(1, 5)} {o3}s. "
        if level == 0:
            text += "How much do I spend in total?"
            return text, total
        if level == 1:
            paid = 50 if total < 45 else 100
            return text + f"How much change do I get from ${paid}?", paid - total
        return text + "How many items do I buy in total?", a

    (o1, lo1, hi1), (o2, lo2, hi2) = rng.sample(OBJECTS, 2)
    p1, p2 = rng.randint(lo1, hi1), rng.randint(lo2, hi2)
    a, b = rng.randint(2, 6), rng.randint(2, 7)
    total = a * p1 + b * p2

    text = (f"{_a(o1).capitalize()} costs ${p1} and {_a(o2)} costs ${p2}. "
            f"I buy {a} {o1}s and {b} {o2}s. ")
    if distractor:
        others = [o for o, _, _ in OBJECTS if o not in (o1, o2)]
        o3 = rng.choice(others)
        text += f"I already own {rng.randint(1, 5)} {o3}s. "
    if level == 0:
        text += "How much do I spend in total?"
        answer = total
    elif level == 1:
        paid = 50 if total < 45 else 100
        text += f"How much change do I get from ${paid}?"
        answer = paid - total
    elif level == 2:
        text += "How many items do I buy in total?"
        answer = a + b
    else:
        raise ValueError(f"level must be 0, 1 or 2, got {level}")
    return text, answer


def parse(text: str) -> tuple[int | None, bool]:
    """Return (value, is_bare_integer).

    `is_bare_integer` is the admission rule. A completion that explains itself
    is not admitted, so the benchmark never has to judge prose.
    """
    s = text.strip()
    m = BARE_INT.match(s)
    if m:
        return int(m.group(1)), True
    m = ANY_INT.search(s)
    return (int(m.group()), False) if m else (None, False)


def build_prompt(tok, body: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": body}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    return f"{SYSTEM}\n\n{body}\n\nAnswer: "


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
def score_and_feature(model, tok, prompt: str, texts: list[str], device: str):
    """Batched: one forward for all candidates of a task.

    The prompt is identical across candidates, so right-padding leaves the
    prompt boundary at the same index in every row and the completion tokens
    are simply [plen:length).
    """
    p_ids = tok(prompt)["input_ids"]
    plen = len(p_ids)
    old_side = tok.padding_side
    tok.padding_side = "right"
    try:
        enc = tok([prompt + t for t in texts], return_tensors="pt",
                  padding=True).to(device)
    finally:
        tok.padding_side = old_side
    enc.pop("token_type_ids", None)
    out = model(**enc, output_hidden_states=True)
    logits = out.logits
    hidden = out.hidden_states[-1]
    mask = enc["attention_mask"]

    res = []
    for i, t in enumerate(texts):
        length = int(mask[i].sum())
        if length <= plen:
            res.append((None, None))
            continue
        lg = torch.log_softmax(logits[i, plen - 1:length - 1].float(), dim=-1)
        tgt = enc["input_ids"][i, plen:length]
        lp = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        h = hidden[i, plen:length].mean(0)
        res.append((float(lp.mean()), h.float().cpu()))
    return res


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="checkpoints/hf/qwen2.5-1.5b-instruct")
    ap.add_argument("--arch", default="auto", choices=["dlm", "auto"])
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--items", type=int, default=2, choices=[1, 2],
                    help="how many priced item types the problem asks about")
    ap.add_argument("--no-distractor", action="store_true",
                    help="drop the irrelevant 'I already own N Xs' clause")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-holdout", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--out", default="results/wordbench.pt")
    ap.add_argument("--summary", default="results/wordbench.json")
    ap.add_argument("--show", type=int, default=4,
                    help="how many wrong-but-admitted examples to print")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.device = str(_resolve_device(args.device))

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
          f"{args.dtype}, level={args.level}, k={args.k}", flush=True)

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
            body, answer = make_problem(random.Random(seed), args.level,
                                        distractor=not args.no_distractor,
                                        items=args.items)
            prompt = build_prompt(tok, body)

            texts = [greedy(model, tok, prompt, args.device, args.max_new)]
            texts += sample_k(model, tok, prompt, args.k, args.device,
                              args.max_new, args.temperature, args.top_p)
            feats = score_and_feature(model, tok, prompt, texts, args.device)

            for ci, (text, (lp, feat)) in enumerate(zip(texts, feats)):
                v, clean = parse(text)
                state["rows"].append({
                    "split": split, "idx": idx, "seed": seed, "cand": ci,
                    "text": text, "value": v, "answer": answer,
                    "correct": int(clean and v == answer),
                    "parseable": int(v is not None), "clean": int(clean),
                    "logprob": lp, "feat": feat,
                })
            state["tasks"].append({"split": split, "idx": idx, "seed": seed,
                                   "body": body, "answer": answer,
                                   "prompt": prompt})
            built += 1
            if built % args.save_every == 0 or built == todo:
                save(out_path, state)
                rate = built / max(time.time() - t0, 1e-6)
                print(f"  {built}/{todo} tasks  {rate:.3f} task/s  "
                      f"eta {(todo - built) / max(rate, 1e-6) / 60:.1f} min",
                      flush=True)

    save(out_path, state)

    summary = {"config": vars(args), "splits": {}}
    for split, _n in plan:
        rows = [r for r in state["rows"] if r["split"] == split]
        tasks = [t for t in state["tasks"] if t["split"] == split]
        k = args.k
        greedy_rows = [r for r in rows if r["cand"] == 0]
        samp_rows = [r for r in rows if r["cand"] > 0]
        clean_rows = [r for r in samp_rows if r["clean"]]
        hits = {}
        for r in clean_rows:
            hits.setdefault(r["idx"], 0)
            hits[r["idx"]] += r["correct"]
        summary["splits"][split] = {
            "tasks": len(tasks),
            "candidates": len(rows),
            "clean_rate": sum(r["clean"] for r in rows) / max(len(rows), 1),
            "p1_greedy": sum(r["correct"] for r in greedy_rows) / max(len(greedy_rows), 1),
            "p1_sampled": (sum(r["correct"] for r in clean_rows) / len(clean_rows)
                           if clean_rows else None),
            "pass_at_k": sum(1 for h in hits.values() if h > 0) / max(len(hits), 1),
            "wrong_clean": sum(1 for r in clean_rows if not r["correct"]),
            "tasks_with_no_clean": sum(1 for t in tasks
                                       if not any(r["clean"] and r["idx"] == t["idx"]
                                                  for r in samp_rows)),
        }
        s = summary["splits"][split]
        p1 = "nan" if s["p1_sampled"] is None else f"{s['p1_sampled']:.3f}"
        print(f"{split:<8} tasks {s['tasks']:>3}  clean {s['clean_rate']:.3f}  "
              f"p1_greedy {s['p1_greedy']:.3f}  p1_sampled {p1}  "
              f"pass@{k} {s['pass_at_k']:.3f}  wrong_clean {s['wrong_clean']}  "
              f"no_clean_tasks {s['tasks_with_no_clean']}")

    shown = 0
    for r in state["rows"]:
        if shown >= args.show:
            break
        if r["clean"] and not r["correct"]:
            t = next(x for x in state["tasks"]
                     if x["split"] == r["split"] and x["idx"] == r["idx"])
            print(f"\n--- wrong but admitted ({r['split']} #{r['idx']}) ---")
            print(t["body"])
            print(f"    truth {t['answer']}  model said {r['value']!r} "
                  f"(logprob {r['logprob']})")
            shown += 1
    if shown == 0:
        print("\nno wrong-but-admitted candidate found")

    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

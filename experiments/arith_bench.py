"""Build the arithmetic selection corpus.

The repair benchmark could not test routing because it contained no
wrong-but-well-formed candidate: parseability and correctness agreed 480/480
(findings/2026-10-27-benchmark-has-no-plausible-wrong-answers.md). A chain of
assignments supplies that property by construction -- `a = 7 / b = a + 5 /
b = ?` can be answered `12` or `11` and both are equally well formed.

This script builds the corpus only. `arith_select.py` trains and scores the
selectors.

Per task it records, for each candidate:

    text       the completion
    value      first integer in it, or None
    correct    value == the true answer
    logprob    mean token log-probability, a cheap confidence baseline
    feat       mean-pooled final hidden state over the completion tokens

plus the task body, its seed, and which split it belongs to.

Splits are disjoint by seed construction (train uses 0.., holdout uses
10_000_000..), so no problem can appear in both. Everything is written after
every --save-every tasks, so a crash or power-off costs at most that many.

Usage:
    python experiments/arith_bench.py --n-train 250 --n-holdout 250 --steps 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.arith_pilot import FEWSHOT, load, make_problem, score

HOLDOUT_OFFSET = 10_000_000


def task_seed(split: str, idx: int) -> int:
    return idx if split == "train" else HOLDOUT_OFFSET + idx


@torch.no_grad()
def sample_k(model, tok, prompt: str, k: int, device: str, max_new: int,
             temperature: float, top_p: float) -> list[str]:
    """k independent samples from one prompt, in a single batched call."""
    enc = tok([prompt], return_tensors="pt").to(device)
    enc.pop("token_type_ids", None)
    gen = model.generate(
        **enc, max_new_tokens=max_new, do_sample=True, temperature=temperature,
        top_p=top_p, num_return_sequences=k, pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id)
    new = gen[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(new, skip_special_tokens=True)


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
def score_and_feature(model, tok, prompt: str, text: str, device: str):
    """One forward pass over prompt+text gives both the logprob and the state.

    Using generate's scores would give the logprob but not the hidden state of
    the completion, and a second pass just for features would double the cost.
    """
    p_ids = tok(prompt)["input_ids"]
    ids = tok(prompt + text)["input_ids"]
    plen = len(p_ids)
    if ids[:plen] != p_ids:
        # BPE merged across the boundary; find the real split.
        plen = 0
        for a, b in zip(ids, p_ids):
            if a != b:
                break
            plen += 1
    if plen >= len(ids):
        return None, None
    t = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(t, output_hidden_states=True)
    logits = out.logits[0]
    lp = torch.log_softmax(logits[plen - 1:len(ids) - 1].float(), dim=-1)
    tgt = t[0, plen:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    h = out.hidden_states[-1][0, plen:].mean(0)
    return float(tok_lp.mean()), h.float().cpu()


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="checkpoints/hf/qwen3-0.6b-base")
    ap.add_argument("--arch", default="auto", choices=["dlm", "auto"])
    ap.add_argument("--dtype", default="float32", choices=["float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2, help="chain length")
    ap.add_argument("--k", type=int, default=8, help="samples per task")
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-holdout", type=int, default=250)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--out", default="results/arith_bench.pt")
    ap.add_argument("--summary", default="results/arith_bench.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    state = {"rows": [], "tasks": [], "config": vars(args)}
    if args.resume and out_path.exists():
        state = torch.load(out_path, weights_only=False)
        print(f"resumed: {len(state['tasks'])} tasks already done")
    done = {(t["split"], t["idx"]) for t in state["tasks"]}

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model, tok = load(Path(args.model).resolve() if Path(args.model).is_dir()
                      else args.model, dtype, args.device, args.arch)
    print(f"loaded, {sum(p.numel() for p in model.parameters()):,} params, "
          f"{args.dtype}, steps={args.steps}, k={args.k}")

    plan = [("train", args.n_train), ("holdout", args.n_holdout)]
    todo = sum(1 for s, n in plan for i in range(n) if (s, i) not in done)
    print(f"{todo} tasks to build", flush=True)
    t0 = time.time()
    built = 0

    for split, n in plan:
        for idx in range(n):
            if (split, idx) in done:
                continue
            seed = task_seed(split, idx)
            body, answer = make_problem(random.Random(seed), args.steps)
            prompt = FEWSHOT + body + "\nAnswer:"

            texts = [greedy(model, tok, prompt, args.device, args.max_new)]
            texts += sample_k(model, tok, prompt, args.k, args.device,
                              args.max_new, args.temperature, args.top_p)

            for ci, text in enumerate(texts):
                lp, feat = score_and_feature(model, tok, prompt, text,
                                             args.device)
                v = score(text)
                state["rows"].append({
                    "split": split, "idx": idx, "seed": seed, "cand": ci,
                    "text": text, "value": v, "answer": answer,
                    "correct": int(v is not None and v == answer),
                    "parseable": int(v is not None),
                    "logprob": lp, "feat": feat,
                })
            state["tasks"].append({"split": split, "idx": idx, "seed": seed,
                                   "body": body, "answer": answer,
                                   "prompt": prompt})
            built += 1

            if built % args.save_every == 0 or built == todo:
                save(out_path, state)
                rate = built / max(time.time() - t0, 1e-6)
                eta = (todo - built) / max(rate, 1e-6) / 60
                print(f"  {built}/{todo} tasks  {rate:.2f} task/s  "
                      f"eta {eta:.1f} min", flush=True)

    save(out_path, state)

    # Summary per split, computed on the freshly built rows.
    summary = {"config": vars(args), "splits": {}}
    for split, _n in plan:
        rows = [r for r in state["rows"] if r["split"] == split]
        tasks = [t for t in state["tasks"] if t["split"] == split]
        k = args.k
        greedy_rows = [r for r in rows if r["cand"] == 0]
        samp_rows = [r for r in rows if r["cand"] > 0]
        hits = {}
        for r in samp_rows:
            hits.setdefault(r["idx"], 0)
            hits[r["idx"]] += r["correct"]
        summary["splits"][split] = {
            "tasks": len(tasks),
            "candidates": len(rows),
            "p1_greedy": sum(r["correct"] for r in greedy_rows) / max(len(greedy_rows), 1),
            "p1_sampled": sum(r["correct"] for r in samp_rows) / max(len(samp_rows), 1),
            "pass_at_k": sum(1 for h in hits.values() if h > 0) / max(len(hits), 1),
            "wellformed_rate": sum(r["parseable"] for r in rows) / max(len(rows), 1),
            "wrong_wellformed": sum(1 for r in rows
                                    if r["parseable"] and not r["correct"]),
        }
        s = summary["splits"][split]
        print(f"{split:<8} tasks {s['tasks']:>4}  p1_greedy {s['p1_greedy']:.3f}  "
              f"p1_sampled {s['p1_sampled']:.3f}  pass@{k} {s['pass_at_k']:.3f}  "
              f"wellformed {s['wellformed_rate']:.3f}  "
              f"wrong_wf {s['wrong_wellformed']}")

    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

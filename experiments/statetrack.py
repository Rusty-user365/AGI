"""Long-horizon state tracking: balls between boxes.

Purpose. The short-horizon result was that a head reading hidden states ranks
candidates (AUC 0.682) but does not select better than max-logprob, and adding
logprob to the feature vector changed nothing -- the hidden state was redundant
with confidence (findings/2026-10-27-observer-ranks-but-cannot-select.md).

The one version of the long-horizon argument that this does not settle is
falsifiable: if accuracy falls with horizon while the model's log-probability
stays high on its wrong answers, then confidence stops being a correctness
proxy and a separate observer has room again. This corpus measures exactly
that, per horizon:

    accuracy     exact match on the final counts
    pass@k       any candidate right -- the ceiling for any selector
    logprob AUC  within-task AUC of log-probability for correctness
    wellformed   fraction of completions that are bare integers

If `logprob AUC` stays near 0.65 as horizon grows, confidence still works and
the DAG has the same justification problem it had at length 2. If it falls
toward 0.5 while accuracy falls too, the observer hypothesis is alive again and
worth training.

Difficulty is a knob because exact match punishes every box independently: with
four boxes a model that tracks three perfectly still scores zero. `--boxes 2`
reduces the answer to two integers, which is the setting a 0.6B base model can
actually score above chance on.

A wrong answer is well formed by construction: `3 4` and `3 5` are equally
parseable. Admission is "the first line is N bare integers", applied identically
to every arm downstream.

Usage:
    python experiments/statetrack.py --horizon 30 --boxes 2 --n-train 20
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


def make_problem(rng: random.Random, moves: int, boxes: int = 2,
                 max_move: int = 1) -> tuple[str, tuple[int, ...]]:
    """Initial counts plus a valid move sequence; answer is the final counts."""
    counts = [rng.randint(0, 4) for _ in range(boxes)]
    if sum(counts) < 4:
        counts[rng.randrange(boxes)] += 4
    lines = ["Boxes: " + " ".join(map(str, counts))]
    for _ in range(moves):
        src = rng.choice([b for b in range(boxes) if counts[b] >= 1])
        k = rng.randint(1, min(max_move, counts[src]))
        dst = rng.choice([b for b in range(boxes) if b != src])
        counts[src] -= k
        counts[dst] += k
        lines.append(f"move {k} from box {src + 1} to box {dst + 1}")
    return "\n".join(lines), tuple(counts)


def parse(text: str, boxes: int = 2) -> tuple[tuple[int, ...] | None, bool]:
    """Admission = the *first line* is `boxes` bare integers.

    A base model continues the few-shot pattern after answering, emitting a
    blank line and a new `Boxes:` line. The answer is the first line, so the
    continuation is ignored rather than scored as a formatting failure.
    """
    first = text.strip().split("\n")[0].strip()
    nums = re.findall(r"\d+", first)
    if len(nums) != boxes:
        return None, False
    # Reject anything with words in it: the line must be numbers and separators.
    if re.sub(r"[\d\s,\(\)\[\]\{\}\.]", "", first):
        return None, False
    return tuple(int(x) for x in nums), True


def fewshot(boxes: int, max_move: int) -> str:
    """A worked example, generated rather than hand-written so it cannot drift."""
    body, ans = make_problem(random.Random(12345), 3, boxes, max_move)
    return body + "\nFinal: " + " ".join(map(str, ans)) + "\n\n"


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
    p_ids = tok(prompt)["input_ids"]
    plen = len(p_ids)
    old = tok.padding_side
    tok.padding_side = "right"
    try:
        enc = tok([prompt + t for t in texts], return_tensors="pt",
                  padding=True).to(device)
    finally:
        tok.padding_side = old
    enc.pop("token_type_ids", None)
    out = model(**enc, output_hidden_states=True)
    logits, hidden = out.logits, out.hidden_states[-1]
    mask = enc["attention_mask"]
    res = []
    for i in range(len(texts)):
        length = int(mask[i].sum())
        if length <= plen:
            res.append((None, None))
            continue
        lg = torch.log_softmax(logits[i, plen - 1:length - 1].float(), dim=-1)
        tgt = enc["input_ids"][i, plen:length]
        lp = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        res.append((float(lp.mean()), hidden[i, plen:length].mean(0).float().cpu()))
    return res


def pairwise_auc(scores, labels, tasks):
    num = 0.0
    den = 0
    for t in set(tasks):
        idx = [i for i in range(len(tasks)) if tasks[i] == t]
        s = [scores[i] for i in idx]
        y = [labels[i] for i in idx]
        pos = [a for a, b in zip(s, y) if b]
        neg = [a for a, b in zip(s, y) if not b]
        for p in pos:
            for n in neg:
                num += 1.0 if p > n else (0.5 if p == n else 0.0)
                den += 1
    return (num / den if den else None), den


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
    ap.add_argument("--device", default="auto")
    ap.add_argument("--horizon", type=int, default=30, help="number of moves")
    ap.add_argument("--boxes", type=int, default=2)
    ap.add_argument("--max-move", type=int, default=1)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-holdout", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.device = str(_resolve_device(args.device))

    tag = f"b{args.boxes}m{args.max_move}_h{args.horizon}"
    args.out = args.out or f"results/statetrack_{tag}.pt"
    args.summary = args.summary or f"results/statetrack_{tag}.json"
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
          f"{args.dtype}, boxes={args.boxes}, horizon={args.horizon}, "
          f"k={args.k}", flush=True)

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
            body, answer = make_problem(random.Random(seed), args.horizon,
                                        args.boxes, args.max_move)
            prompt = shot + body + "\nFinal:"

            texts = [greedy(model, tok, prompt, args.device, args.max_new)]
            texts += sample_k(model, tok, prompt, args.k, args.device,
                              args.max_new, args.temperature, args.top_p)
            feats = score_and_feature(model, tok, prompt, texts, args.device)

            for ci, (text, (lp, feat)) in enumerate(zip(texts, feats)):
                v, clean = parse(text, args.boxes)
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

    summary = {"config": vars(args), "horizon": args.horizon,
               "boxes": args.boxes, "splits": {}}
    for split, _n in plan:
        rows = [r for r in state["rows"] if r["split"] == split]
        greedy_rows = [r for r in rows if r["cand"] == 0]
        samp = [r for r in rows if r["cand"] > 0 and r["clean"]]
        hits = {}
        for r in samp:
            hits.setdefault(r["idx"], 0)
            hits[r["idx"]] += r["correct"]
        lp_rows = [r for r in rows if r["logprob"] is not None]
        auc, npairs = pairwise_auc(
            [r["logprob"] for r in lp_rows],
            [bool(r["correct"]) for r in lp_rows],
            [r["idx"] for r in lp_rows])
        summary["splits"][split] = {
            "tasks": len({r["idx"] for r in rows}),
            "candidates": len(rows),
            "clean_rate": sum(r["clean"] for r in rows) / max(len(rows), 1),
            "p1_greedy": (sum(r["correct"] for r in greedy_rows)
                          / max(len(greedy_rows), 1)),
            "p1_sampled": (sum(r["correct"] for r in samp) / len(samp)
                           if samp else None),
            "pass_at_k": (sum(1 for h in hits.values() if h > 0)
                          / max(len(hits), 1)),
            "wrong_clean": sum(1 for r in samp if not r["correct"]),
            "logprob_auc": auc,
            "logprob_pairs": npairs,
        }
        s = summary["splits"][split]
        p1 = "nan" if s["p1_sampled"] is None else f"{s['p1_sampled']:.3f}"
        a = "nan" if s["logprob_auc"] is None else f"{s['logprob_auc']:.3f}"
        print(f"{split:<8} tasks {s['tasks']:>3}  clean {s['clean_rate']:.3f}  "
              f"p1_greedy {s['p1_greedy']:.3f}  p1_sampled {p1}  "
              f"pass@{args.k} {s['pass_at_k']:.3f}  "
              f"wrong_clean {s['wrong_clean']}  logprob_auc {a}")

    shown = 0
    for r in state["rows"]:
        if shown >= args.show:
            break
        if r["clean"] and not r["correct"]:
            t = next(x for x in state["tasks"]
                     if x["split"] == r["split"] and x["idx"] == r["idx"])
            print(f"\n--- wrong but admitted ({r['split']} #{r['idx']}) ---")
            print(t["body"])
            print(f"    truth {t['answer']}  model said {r['value']} "
                  f"(logprob {r['logprob']:.3f})")
            shown += 1

    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

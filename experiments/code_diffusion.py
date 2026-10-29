"""Code repair with a code-trained masked-diffusion model.

Pre-registration: `findings/2026-10-29-code-diffusion-preregistration.md`.

The question is not "is a diffusion model good at code". It is narrower:

    Given the *same* broken body and the *same* failing test, does a model that
    must fill a span consistent with fixed left and right context, and can
    revise that span at every step, repair more problems per forward pass than
    an autoregressive model that continues left to right?

So every repair arm starts from the AR twin's failing body. The body is the
shared input; only the repair mechanism differs. That removes the confound that
would otherwise dominate, because the MDLM fine-tune may simply be a worse coder
than the AR model it was made from.

Two modes:

    --check     load the model, prove it is bidirectional, time one generation
                and one infill. Writes nothing except a log line.
    --arm NAME  run one arm over a set, appending one JSON line per problem so a
                crash or power-off loses at most the problem in flight.

Arms (see the pre-registration for the argument for each):

    mdlm_full           full body from the prompt, one attempt
    mdlm_infill         mask the localized statement, keep both sides, test as
                        right context
    mdlm_infill_nohint  identical but the test is removed (placebo control)

Matching: the unit is a forward pass. A KV-cached AR generation of L tokens
costs L passes; S denoising steps cost S. Each diffusion attempt is given the
number of steps the AR arm spent on the corresponding attempt, and the problem's
total is capped at the AR arm's total. Nothing here estimates that; the AR
harness records the per-attempt length (`gen_lens`).

Usage:
    python experiments/code_diffusion.py --check
    python experiments/code_diffusion.py --arm mdlm_infill \
        --ar-rows results/ar05b/rows.jsonl
    python experiments/code_diffusion.py --compare
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

from experiments.code_bench import (failure_kind, load as load_corpus,
                                    run_tests, truncate)
from experiments.code_loop import feedback_of, localize_cut
from experiments.level2_ladder import mcnemar
from experiments.localizer import wilson

ARMS = ["mdlm_full", "mdlm_infill", "mdlm_infill_nohint"]


# ---------------------------------------------------------------- model load

def load_diffusion(path: Path, dtype, device: str):
    """Load a custom-code masked-diffusion checkpoint.

    The checkpoint's `modeling_qwen2.py` is imported unmodified. It imports
    `TransformersKwargs` from `transformers.utils`, which does not exist in the
    system transformers 4.51.3, so this must run in the 4.57 venv
    (`.venv-tf457`). Patching the model file to fit an older transformers would
    change attention behaviour silently, which is the one thing this run cannot
    afford.
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path))
    kwargs = dict(trust_remote_code=True)
    try:
        model = AutoModelForMaskedLM.from_pretrained(str(path), dtype=dtype,
                                                     **kwargs)
    except TypeError:
        model = AutoModelForMaskedLM.from_pretrained(str(path),
                                                     torch_dtype=dtype, **kwargs)
    model = model.to(device).eval()
    return model, tok


# ---------------------------------------------------------------- tokenizing

def token_ids(tok, text: str):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    return enc["input_ids"], enc["offset_mapping"]


def line_starts(body: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(body):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def statement_span(body: str, cut: int) -> tuple[int, int]:
    """Char span of the statement beginning at line `cut`, plus its block.

    A statement's block ends at the next non-blank line indented no deeper. That
    is the unit a backtrack can reach: the faulty step and whatever it owns, not
    the rest of the function.
    """
    lines = body.splitlines()
    starts = line_starts(body)
    if cut >= len(lines):
        return len(body), len(body)
    indent = len(lines[cut]) - len(lines[cut].lstrip())
    end_line = len(lines)
    for j in range(cut + 1, len(lines)):
        if not lines[j].strip():
            continue
        if (len(lines[j]) - len(lines[j].lstrip())) <= indent:
            end_line = j
            break
    start = starts[cut] if cut < len(starts) else len(body)
    end = starts[end_line] if end_line < len(starts) else len(body)
    return start, max(start, end)


# ---------------------------------------------------------------- denoising

@torch.no_grad()
def denoise(model, x, mask_pos, steps: int, temperature: float, device: str):
    """Fill the masked positions of `x`. One forward pass per step.

    At each step every still-masked position is predicted, and the most
    confident ones are committed so that the remaining masks are spread evenly
    over the remaining steps. With `steps >= n_masked` this is one token per
    step, which is the AR cost model.
    """
    nfe = 0
    for step in range(steps):
        sel = mask_pos.nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            break
        logits = model(x).logits[0]
        nfe += 1
        sub = logits.index_select(0, sel).float()
        if temperature > 0:
            g = -torch.log(-torch.log(torch.rand_like(sub).clamp_min(1e-9)))
            pred = (sub / temperature + g).argmax(-1)
        else:
            pred = sub.argmax(-1)
        conf = sub.softmax(-1).gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        left = max(1, steps - step)
        n = min(sel.numel(), max(1, math.ceil(sel.numel() / left)))
        take = conf.topk(n).indices
        pos = sel[take]
        x[0, pos] = pred[take]
        mask_pos[pos] = False
    return x, nfe


def build_infill(tok, prefix: str, body: str, suffix: str, span, mask_id: int,
                 device: str):
    """Tokenize prefix+body+suffix and mask the tokens covering `span`."""
    text = prefix + body + suffix
    ids, offs = token_ids(tok, text)
    lo, hi = len(prefix) + span[0], len(prefix) + span[1]
    mask_pos = torch.zeros(len(ids), dtype=torch.bool, device=device)
    hit = [i for i, (a, b) in enumerate(offs) if b > lo and a < hi]
    if not hit:
        return None
    for i in hit:
        mask_pos[i] = True
    x = torch.tensor([ids], device=device)
    return x, mask_pos, text


def infill_once(model, tok, prefix, body, suffix, span, steps, temperature,
                device, mask_id):
    built = build_infill(tok, prefix, body, suffix, span, mask_id, device)
    if built is None:
        return body, 0
    x, mask_pos, text = built
    x, nfe = denoise(model, x, mask_pos, steps, temperature, device)
    full = tok.decode(x[0].tolist(), skip_special_tokens=True)
    # Qwen's byte-level BPE round-trips, so decoding the whole sequence gives
    # `text` with only the masked span replaced. Splicing token-by-token would
    # introduce boundary-space artifacts that then get verified as if they were
    # the model's output.
    if full.startswith(prefix):
        out = full[len(prefix):]
    else:                                   # fall back to a character splice
        lo = len(prefix) + span[0]
        out = full[len(prefix):len(prefix) + span[0]] + body[span[0]:]
        _ = lo
    if suffix and out.endswith(suffix):
        out = out[:-len(suffix)]
    return out, nfe


def full_once(model, tok, prompt, n_new, steps, temperature, device, mask_id):
    ids, _ = token_ids(tok, prompt)
    L = len(ids)
    x = torch.tensor([ids + [mask_id] * n_new], device=device)
    mask_pos = torch.zeros(L + n_new, dtype=torch.bool, device=device)
    mask_pos[L:] = True
    x, nfe = denoise(model, x, mask_pos, steps, temperature, device)
    return truncate(tok.decode(x[0, L:].tolist(), skip_special_tokens=True)), nfe


# ---------------------------------------------------------------- arms

def hint_of(res: dict) -> str:
    return "\n# FAILED: " + feedback_of(res)


def run_infill(model, tok, r, args, body0: str, res0: dict, budgets: list[int],
               hint: bool):
    """Attempt up to len(budgets) infills, each costing its matched step count."""
    prompt_lines = len(r["prompt"].splitlines())
    body = body0
    res = res0
    tests = 0
    nfe = 0
    cuts = []
    for steps in budgets[:args.max_attempts]:
        if res["ok"] or steps <= 0:
            break
        cut, _kind = localize_cut(body.splitlines(), res, prompt_lines)
        # cut > 0 is the only real localization (an exception frame inside the
        # body). Everything else is cut = 0, which means "we do not know where
        # the bug is", so the whole body is masked. Masking only the top-level
        # statement block at cut = 0 would be an undocumented localization and
        # would flatter the treatment on exactly the branch that has none.
        span = (0, len(body)) if cut == 0 else statement_span(body, cut)
        suffix = hint_of(res) if hint else ""
        body, used = infill_once(model, tok, r["prompt"], body, suffix, span,
                                 steps, args.temperature, args.device,
                                 tok.mask_token_id)
        nfe += used
        cuts.append(cut)
        res = run_tests(r["prompt"] + body, r["test"], r["entry_point"],
                        args.timeout)
        tests += 1
    return {"ok": res["ok"], "nfe": nfe, "tests": tests, "cuts": cuts,
            "attempts": len(cuts), "body_final": body,
            "kind": None if res0["ok"] else failure_kind(res0["stderr"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/humaneval.jsonl.gz")
    ap.add_argument("--model",
                    default="checkpoints/hf/qwen2.5-coder-0.5b-diffusion-mdlm")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32",
                    choices=["fp32", "fp16", "bf16"],
                    help="fp32 default: the checkpoint is bf16-trained and "
                         "Pascal has no fast fp16, so fp16 buys nothing")
    ap.add_argument("--ar-rows", default="results/ar05b/rows.jsonl",
                    help="AR twin rows; supplies the failing bodies and the "
                         "per-attempt forward-pass budget")
    ap.add_argument("--arm", default=None, choices=ARMS)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="matched to the AR arm's sampling temperature")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out-dir", default="results/diffusion")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[args.dtype]

    if args.compare:
        return compare(root / args.out_dir)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, tok = load_diffusion(model_path, dtype, args.device)
    load_s = time.time() - t0
    if tok.mask_token_id is None:
        raise SystemExit("tokenizer has no mask token; not an MDLM checkpoint")
    print(f"loaded in {load_s:.0f}s  mask_id={tok.mask_token_id}  "
          f"vram={torch.cuda.max_memory_allocated() / 2**20:.0f}MB", flush=True)

    if args.check:
        return check(model, tok, args, root)

    rows = load_corpus(root / args.data)
    if args.limit:
        rows = rows[:args.limit]

    ar_path = Path(args.ar_rows)
    if not ar_path.is_absolute():
        ar_path = root / ar_path
    ar = {json.loads(l)["task_id"]: json.loads(l)
          for l in ar_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.arm}.jsonl"
    done = set()
    if args.resume and out_path.exists():
        done = {json.loads(l)["task_id"] for l in
                out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"resume: {len(done)} problems already done", flush=True)

    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for j, r in enumerate(rows):
            if r["task_id"] in done:
                continue
            seed = args.seed + j
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            rec = {"task_id": r["task_id"], "entry_point": r["entry_point"]}

            if args.arm == "mdlm_full":
                steps = ar[r["task_id"]]["arms"]["greedy"]["gen_len"]
                body, nfe = full_once(model, tok, r["prompt"], steps, steps,
                                      args.temperature, args.device,
                                      tok.mask_token_id)
                res = run_tests(r["prompt"] + body, r["test"],
                                r["entry_point"], args.timeout)
                rec["arm"] = {"ok": res["ok"], "nfe": nfe, "tests": 1,
                              "body": body,
                              "kind": None if res["ok"]
                              else failure_kind(res["stderr"])}
            else:
                g = ar[r["task_id"]]["arms"]["greedy"]
                if g["ok"]:
                    continue        # repair set = problems AR greedy failed
                body0 = g["body"]
                res0 = run_tests(r["prompt"] + body0, r["test"],
                                 r["entry_point"], args.timeout)
                if res0["ok"]:      # nondeterminism guard; should not happen
                    print(f"  !! {r['task_id']} greedy body passes on re-run")
                cgr = ar[r["task_id"]]["arms"].get("cgr", {})
                lens = cgr.get("gen_lens") or []
                budgets = lens[1:] if len(lens) > 1 else [g["gen_len"]]
                arm = run_infill(model, tok, r, args, body0, res0, budgets,
                                 hint=(args.arm != "mdlm_infill_nohint"))
                rec["arm"] = arm
                rec["greedy_kind"] = g["kind"]

            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            a = rec["arm"]
            print(f"  {j + 1:>3}/{len(rows)} {r['task_id']:<16} "
                  f"ok={int(a['ok'])} nfe={a['nfe']} tests={a['tests']} "
                  f"kind={rec.get('greedy_kind')} ({time.time() - t0:.0f}s)",
                  flush=True)

    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
    return compare(out_dir)


# ---------------------------------------------------------------- analysis

def read_arm(path: Path) -> dict:
    return {json.loads(l)["task_id"]: json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()}


def compare(out_dir: Path) -> int:
    files = {p.stem: read_arm(p) for p in sorted(out_dir.glob("*.jsonl"))}
    if not files:
        print("no arm files")
        return 0
    print("\n" + "=" * 78)
    print("DIFFUSION ARMS")
    print("=" * 78)
    print(f"{'arm':<20} {'ok':>7} {'rate':>6} {'95% CI':>16} "
          f"{'nfe':>8} {'nfe/solve':>10} {'tests':>6}")
    for name, recs in files.items():
        n = len(recs)
        ok = sum(int(v["arm"]["ok"]) for v in recs.values())
        nfe = sum(v["arm"]["nfe"] for v in recs.values())
        tests = sum(v["arm"]["tests"] for v in recs.values())
        lo, hi = wilson(ok, n)
        print(f"{name:<20} {ok:>3}/{n:<3} {ok / max(n, 1):>6.3f} "
              f"[{lo:.3f},{hi:.3f}] {nfe:>8} {nfe / max(ok, 1):>10.0f} "
              f"{tests:>6}")

    names = list(files)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            common = sorted(set(files[a]) & set(files[b]))
            if not common:
                continue
            va = [files[a][k]["arm"]["ok"] for k in common]
            vb = [files[b][k]["arm"]["ok"] for k in common]
            m = mcnemar(va, vb)
            print(f"  {a} vs {b} (n={len(common)}): "
                  f"{m['wins']}W/{m['losses']}L p={m['p']:.2e}")

    by_kind: dict[str, dict] = {}
    for name, recs in files.items():
        for k, v in recs.items():
            kind = str(v.get("greedy_kind"))
            row = by_kind.setdefault(kind, {"n": 0, **{n: 0 for n in names}})
            row["n"] += 1
            row[name] += int(v["arm"]["ok"])
    for kind, row in sorted(by_kind.items()):
        print(f"  kind={kind:<10} n={row['n']:<3} " +
              " ".join(f"{n}={row[n]}" for n in names))
    return 0


# ---------------------------------------------------------------- check

def check(model, tok, args, root: Path) -> int:
    """Prove the checkpoint is bidirectional, then time one of each operation."""
    text = ("def add(a, b):\n    total = a - b\n    return total\n"
            "\n# expect add(2, 3) == 5\n")
    ids, _ = token_ids(tok, text)
    x = torch.tensor([ids], device=args.device)
    with torch.no_grad():
        h1 = model.model(input_ids=x).last_hidden_state[0]
        x2 = x.clone()
        x2[0, -1] = int(x2[0, -1]) + 7
        h2 = model.model(input_ids=x2).last_hidden_state[0]
    d_first = (h1[0] - h2[0]).abs().max().item()
    d_last = (h1[-1] - h2[-1]).abs().max().item()
    print(f"bidirectionality: perturbing the last token changes the first "
          f"token's hidden state by {d_first:.4f} (last token: {d_last:.4f})")
    print("VERDICT:", "bidirectional" if d_first > 1e-4 else
          "CAUSAL -- checkpoint is mislabeled, stop")

    rows = load_corpus(root / args.data)
    r = rows[0]
    t0 = time.time()
    body, nfe = full_once(model, tok, r["prompt"], 96, 96, 0.0, args.device,
                          tok.mask_token_id)
    dt = time.time() - t0
    res = run_tests(r["prompt"] + body, r["test"], r["entry_point"],
                    args.timeout)
    print(f"full generation: 96 steps in {dt:.1f}s "
          f"({dt / max(nfe, 1) * 1000:.0f} ms/step) ok={res['ok']}")
    print(body[:300].replace("\n", "\\n"))

    canon = r["canonical_solution"]
    span = statement_span(canon, 1 if canon.count("\n") > 1 else 0)
    t0 = time.time()
    filled, nfe = infill_once(model, tok, r["prompt"], canon, "", span, 32, 0.0,
                              args.device, tok.mask_token_id)
    dt = time.time() - t0
    res = run_tests(r["prompt"] + filled, r["test"], r["entry_point"],
                    args.timeout)
    print(f"infill of canonical span {span}: {dt:.1f}s "
          f"({dt / max(nfe, 1) * 1000:.0f} ms/step) ok={res['ok']}")
    print(f"masked text was: {canon[span[0]:span[1]]!r}")
    print(f"filled text is:  {filled[span[0]:span[1] + 60]!r}")
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

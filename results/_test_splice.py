"""Is the subrange decode exact? That is the whole splice.

Qwen uses byte-level BPE, so decoding a contiguous token range should reproduce
exactly the characters those tokens cover, independently of context. If that
holds, the replacement can be spliced by the covered character range instead of
by a fixed prefix length -- which is what corrupted the indentation when the
first masked token straddled the prompt/body boundary.
"""
from pathlib import Path

from transformers import AutoTokenizer

from experiments.code_bench import load

tok = AutoTokenizer.from_pretrained(
    "checkpoints/hf/qwen2.5-coder-0.5b-diffusion-mdlm")
rows = load(Path("data/humaneval.jsonl.gz"))

bad = 0
total = 0
for r in rows[:20]:
    text = r["prompt"] + r["canonical_solution"]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    n = len(ids)
    for a, b in [(0, n), (1, n), (5, n - 3), (n // 2, n - 1), (n - 5, n)]:
        a, b = max(0, a), min(n, b)
        if a >= b:
            continue
        total += 1
        sub = tok.decode(ids[a:b], skip_special_tokens=True)
        want = text[offs[a][0]:offs[b - 1][1]]
        if sub != want:
            bad += 1
            if bad <= 3:
                print("MISMATCH", r["task_id"], (a, b))
                print("  want", repr(want[:80]))
                print("  got ", repr(sub[:80]))
print("subrange decode failures: %d of %d" % (bad, total))

# And the boundary case that broke: does a token straddle prefix|body?
r = rows[19]
prefix, body = r["prompt"], r["canonical_solution"]
text = prefix + body
enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
straddle = [(i, a, b) for i, (a, b) in enumerate(enc["offset_mapping"])
            if a < len(prefix) < b]
print("tokens straddling prefix|body:", straddle)

"""Verify that ltg/dual-lm-470m loads and that its masked path is bidirectional.

The model card claims one set of weights serves both objectives, switching on
the attention mask: `DLMLlamaForCausalLM` is causal, `DLMLlamaForMaskedLM` is
full bidirectional. The second claim is the one that matters here, because a
span-repair (REP) mechanism is only different from autoregressive regeneration
if a masked position can see tokens to its right.

That claim is enforced by `config.is_causal = False` inside the modeling file,
and whether transformers 4.51.3 honours that flag inside its mask construction
is exactly the kind of version drift that fails silently: the model loads, runs,
produces plausible text, and is simply causal. So it is tested rather than
assumed.

Probe: change a token strictly *after* the position whose prediction is read.
  causal        logits at p must not move
  bidirectional logits at p must move

Usage:
    python scripts/check_dual_lm.py --dir checkpoints/hf/dual-lm-470m
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer, LlamaConfig


def load_class(model_dir: Path, which: str, patch: bool = False,
               attn: str = "sdpa"):
    sys.path.insert(0, str(model_dir))
    import modeling_dlm_llama as m
    from transformers.models.llama import modeling_llama as llama_mod

    cfg = LlamaConfig.from_pretrained(str(model_dir))
    cfg._attn_implementation = attn
    cls = {"causal": m.DLMLlamaForCausalLM, "masked": m.DLMLlamaForMaskedLM}[which]

    if patch:
        # transformers 4.51.3 builds a causal mask unconditionally and ignores
        # config.is_causal, so the masked class is causal in practice. Replacing
        # the mask builder with a full-attention additive mask is the smallest
        # change that restores the behaviour the checkpoint was trained for.
        def full_mask(self, attention_mask, input_tensor, cache_position,
                      past_key_values, output_attentions=False):
            dtype, device = input_tensor.dtype, input_tensor.device
            bsz, q_len = input_tensor.shape[0], input_tensor.shape[1]
            kv_len = q_len
            if past_key_values is not None:
                kv_len = past_key_values.get_seq_length() + q_len
            mask = torch.zeros(bsz, 1, q_len, kv_len, dtype=dtype, device=device)
            if attention_mask is not None and attention_mask.dim() == 2:
                keep = attention_mask[:, None, None, :].to(dtype)
                mask = mask + (1.0 - keep) * torch.finfo(dtype).min
            return mask

        llama_mod.LlamaModel._update_causal_mask = full_mask

    model = cls.from_pretrained(str(model_dir), config=cfg)
    model.eval()
    return model, cfg


def masked_sanity(model, tok, ids: torch.Tensor) -> str:
    """Mask a known word and read the prediction at the position before it."""
    pieces = [tok.decode([int(t)]) for t in ids[0]]
    target = next((i for i, p in enumerate(pieces) if p.strip() == "Paris"), None)
    if target is None:
        return "(target word not found)"
    alt = ids.clone()
    alt[0, target] = tok.convert_tokens_to_ids("<mask>")
    pred = model(input_ids=alt).logits[0, target - 1].argmax()
    return tok.decode([int(pred)]).strip()


@torch.no_grad()
def probe(model, ids: torch.Tensor, read_pos: int, change_pos: int) -> float:
    """Max abs change in logits at read_pos when the token at change_pos moves."""
    base = model(input_ids=ids).logits[0, read_pos]
    alt_ids = ids.clone()
    alt_ids[0, change_pos] = (alt_ids[0, change_pos] + 7919) % 1000 + 100
    alt = model(input_ids=alt_ids).logits[0, read_pos]
    return float((base - alt).abs().max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="checkpoints/hf/dual-lm-470m")
    ap.add_argument("--which", default="masked", choices=["masked", "causal", "both"])
    ap.add_argument("--patch", action="store_true",
                    help="replace LlamaModel._update_causal_mask with a "
                         "full-attention mask (restores bidirectional infill "
                         "on transformers versions that ignore is_causal)")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    args = ap.parse_args()

    d = Path(args.dir).resolve()
    tok = AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)
    text = "The capital of France is Paris and the river through it is the Seine."
    ids = tok(text, return_tensors="pt")["input_ids"]
    print(f"tokens: {ids.shape[1]}  mask_id={tok.convert_tokens_to_ids('<mask>')}")

    # Read the prediction for token 12 from position 11 (MNTP shift), and move a
    # token far to the right of it. Positions 8 and 15 are chosen to sit on real
    # words rather than punctuation.
    read_pos, change_pos = 11, 15

    for which in (["masked", "causal"] if args.which == "both" else [args.which]):
        model, cfg = load_class(d, which, patch=args.patch, attn=args.attn)
        n = sum(p.numel() for p in model.parameters())
        # Control: a token BEFORE the read position must always matter, in both
        # classes. If this is also zero the probe is broken, not the model.
        past = probe(model, ids, read_pos, 3)
        future = probe(model, ids, read_pos, change_pos)
        moved = future > 1e-4
        print(f"\n{which}: params={n:,}  model.config.is_causal="
              f"{getattr(model.config, 'is_causal', None)}  "
              f"attn={getattr(model.config, '_attn_implementation', None)}")
        print(f"  control, token 3 (past)   moves logits at {read_pos}: "
              f"{past > 1e-4} (max abs delta {past:.3e})")
        print(f"  probe,   token {change_pos} (future) moves logits at {read_pos}: "
              f"{moved} (max abs delta {future:.3e})")
        if past <= 1e-4:
            print("  <-- PROBE BROKEN: even a past token has no effect")
        elif which == "masked":
            print(f"  bidirectional: {moved}"
                  + ("" if moved else "  <-- FAIL: masked path is still causal"))
            if moved:
                print(f"  sanity, mask 'Paris' and predict: "
                      f"{masked_sanity(model, tok, ids)!r}")
        else:
            print(f"  causal as expected: {not moved}"
                  + ("" if not moved else "  <-- FAIL: causal path sees the future"))
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

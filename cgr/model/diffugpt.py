"""DiffuGPT-S 127M as the backbone (§3.2).

Why this file exists
--------------------
v0.3 established C1 on a blank canvas: a 173k-parameter masked diffusion LM
trained from scratch on the synthetic corpus. That was enough to test the
*structural* claim, and the paper said plainly what it could not test — the
canvas has no language ability, so its repair numbers are a floor, not a
measurement of the architecture.

This module swaps in a pre-trained diffusion LM without changing the interface
the rest of the system depends on (`forward`, `pooled`,
`masked_denoising_loss`, `infill`, `span_nll`, `token_nll`, `plastic_a/b`).

Three properties of the checkpoint are load-bearing, and are enforced here
rather than assumed:

1. **Shift convention.** DiffuGPT is trained with ``shift: true``: ``logits[i]``
   predicts token ``i + 1``, not token ``i``. This was verified against the
   official implementation (max abs logit difference 2.2e-5, identical top-5).
   Every loss and every likelihood below therefore reads ``logits[i-1]`` for
   token ``i``. Position 0 has no predictor, so it is never a target and never
   re-generated.

2. **Bidirectional attention.** The official loader overwrites GPT-2's causal
   ``attn.bias`` buffer with True. There is no causal mask here at all.

3. **The side channel enters additively and zero-initialised.** The FiLM
   parameters are zero, so at load time the backbone is bit-identical to the
   checkpoint plus an exact identity adapter. That matters for C1: the guard's
   instrument must not be perturbed by the integration itself.

What is *not* claimed here
--------------------------
Out of the box the checkpoint is a competent denoiser of natural text
(15% masking, top-1 0.479) and a weak contiguous infiller (4-token span top-1
0.167, 10-token 0.117, 20-token 0.079), because under the shift convention a
masked run is predicted mostly from the hidden state of a *mask* position.
The rigid template grammar is out of distribution. Fine-tuning on the
synthetic corpus (Stage A) is therefore a precondition for any repair number,
not an optional refinement.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig

#: Checkpoint layout: the transformer lives under `denoise_model.`, the input
#: embedding is stored separately and the LM head is tied to it.
_CKPT_PREFIX = "denoise_model."


def _act(name: str) -> nn.Module:
    if name == "gelu_new":
        return nn.GELU(approximate="tanh")
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unknown activation {name!r}")


class _LoRA(nn.Module):
    """Low-rank update to one projection: ``x @ A^T @ B^T * scale``.

    ``B`` is zero-initialised, so the wrapped projection is bit-identical to the
    pre-trained one at load time. That is not a cosmetic detail: the base
    weights are frozen, and the only thing Stage A is allowed to move is this
    delta. If the delta were non-zero at initialisation, "the pre-trained model
    plus a task adapter" would start from a different function than the
    checkpoint, and no ablation against the base model would be interpretable.

    LoRA here is *not* the plastic substrate. It is the task-adaptation channel,
    trained once in Stage A and then frozen. The plastic substrate
    (`plastic_a`/`plastic_b`) is what `consolidate` may write to during
    interaction, and it is the thing C1 is about. Keeping them separate is what
    lets the paper say "the architecture is gated" without also having to say
    "and nothing else ever learns".
    """

    def __init__(self, d_in: int, d_out: int, rank: int, alpha: float):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, d_in))
        self.b = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.normal_(self.a, std=0.02)
        self.scale = float(alpha) / max(int(rank), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x @ self.a.T) @ self.b.T) * self.scale

    def extra_repr(self) -> str:
        return (f"rank={self.a.shape[0]}, scale={self.scale:.3g}")


class _Block(nn.Module):
    """GPT-2 block, bidirectional, with FiLM and LoRA attached."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, activation: str = "gelu_new",
                 eps: float = 1e-5, lora_rank: int = 8,
                 lora_alpha: float = 16.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.ln_1 = nn.LayerNorm(d_model, eps=eps)
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(d_model, eps=eps)
        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj_mlp = nn.Linear(d_ff, d_model)
        self.mlp_drop = nn.Dropout(dropout)
        self.act = _act(activation)
        # Side channel. Zero weights and zero bias make this an exact identity
        # at load time, so the pre-trained behaviour is preserved bit-for-bit.
        self.film = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        # Task adaptation (PEFT). Attached to the fused qkv projection and to
        # the attention output projection; the MLP is left alone because the
        # format the task needs is an attention/conditioning pattern, and every
        # extra adapter row is a row the frozen-base claim has to account for.
        self.lora_attn = _LoRA(d_model, 3 * d_model, lora_rank, lora_alpha)
        self.lora_proj = _LoRA(d_model, d_model, lora_rank, lora_alpha)

    def forward(self, h: torch.Tensor, cond: torch.Tensor | None,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, D = h.shape
        x = self.ln_1(h)
        qkv = self.c_attn(x) + self.lora_attn(x)
        q, k, v = qkv.split(D, dim=2)
        hd = D // self.n_heads
        q = q.view(B, L, self.n_heads, hd).transpose(1, 2)
        k = k.view(B, L, self.n_heads, hd).transpose(1, 2)
        v = v.view(B, L, self.n_heads, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False,
            dropout_p=self.attn_drop.p if self.training else 0.0)
        a = a.transpose(1, 2).contiguous().view(B, L, D)
        h = h + self.attn_drop(self.c_proj(a) + self.lora_proj(a))
        h = h + self.mlp_drop(self.c_proj_mlp(self.act(self.c_fc(self.ln_2(h)))))
        if cond is not None:
            g, b = self.film(cond).chunk(2, dim=-1)
            h = h * (1.0 + g.unsqueeze(1)) + b.unsqueeze(1)
        return h


class DiffuGPTBackbone(nn.Module):
    """DiffuGPT-S with the CGR side channel and the plastic adapter attached."""

    #: This backbone implements whole-block masking. The canvas model does not,
    #: and `train.py` must not pass mask-policy kwargs to a method that would
    #: ignore them: silently training a different objective is worse than
    #: refusing to start.
    supports_block_mask = True

    def __init__(self, cfg: ModelConfig, mask_id: int, pad_id: int,
                 path: str | Path | None = None, load_weights: bool = True):
        super().__init__()
        self.cfg = cfg
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.max_len = cfg.max_len
        self.eps = 1e-5

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(self.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([
            _Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
                   cfg.activation, self.eps, cfg.lora_rank, cfg.lora_alpha)
            for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=self.eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # weight tying

        # The only modifiable substrate (§3.7). Rank 8, `plastic_b` zero, so
        # the adapter is an exact identity until consolidation writes to it.
        self.plastic_rank = 8
        self.plastic_a = nn.Parameter(
            torch.randn(cfg.d_model, self.plastic_rank) * 0.02)
        self.plastic_b = nn.Parameter(
            torch.zeros(self.plastic_rank, cfg.d_model))

        src = Path(path) if path is not None else Path(cfg.backbone_path)
        if load_weights:
            self.load_pretrained(src)

    # ------------------------------------------------------------------
    # What may be trained
    # ------------------------------------------------------------------
    def lora_parameters(self) -> list[nn.Parameter]:
        """The task-adaptation parameters (Stage A only)."""
        return [p for n, p in self.named_parameters() if ".lora_" in n]

    def freeze_base(self) -> None:
        """Freeze the pre-trained weights, leaving LoRA and plastic trainable.

        This is the PEFT contract: the 127M base is a fixed feature extractor,
        Stage A moves a ~0.4M-parameter low-rank delta (plus the CGR modules),
        and the plastic substrate stays reserved for `consolidate`. It is
        enforced in code rather than by convention because the claim "the
        backbone was not fine-tuned" has to be checkable from the checkpoint.
        """
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.lora_parameters():
            p.requires_grad_(True)
        self.plastic_a.requires_grad_(True)
        self.plastic_b.requires_grad_(True)

    def trainable_names(self) -> list[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def load_pretrained(self, path: str | Path) -> None:
        """Load the DiffuGPT-S safetensors into this module.

        The checkpoint stores GPT-2's Conv1D weights in ``[in, out]`` order;
        ``nn.Linear`` wants ``[out, in]``. That transpose is the single most
        likely silent error in this file, so it is done explicitly and the
        result is checked against the official implementation (see
        ``tests/test_diffugpt.py``).

        The vocabulary is resized by exactly one row for the pad token. The
        pad row is the mean of the pre-trained rows, not random: a random row
        would be an out-of-distribution input wherever padding leaks into an
        attention window.
        """
        from safetensors.torch import load_file

        path = Path(path)
        st = path / "model.safetensors"
        if not st.exists():
            raise FileNotFoundError(
                f"no model.safetensors under {path}; set "
                f"cfg.model.backbone_path to the DiffuGPT-S directory")
        sd = load_file(str(st))

        emb = sd["embed_tokens.weight"]
        n_ckpt, d_ckpt = emb.shape
        if d_ckpt != self.cfg.d_model:
            raise ValueError(
                f"checkpoint d_model={d_ckpt} but config says "
                f"{self.cfg.d_model}; the preset must match the checkpoint")
        if self.cfg.vocab_size < n_ckpt:
            raise ValueError(
                f"config vocab_size={self.cfg.vocab_size} is smaller than the "
                f"checkpoint's {n_ckpt} rows")

        new: dict[str, torch.Tensor] = {"embed_tokens.weight": emb}
        if self.cfg.vocab_size > n_ckpt:
            extra = emb.mean(0, keepdim=True).repeat(
                self.cfg.vocab_size - n_ckpt, 1)
            new["embed_tokens.weight"] = torch.cat([emb, extra], dim=0)
        # The checkpoint holds 1024 positions; the preset deliberately uses a
        # shorter window. Slicing is safe because positions are absolute from
        # the left: truncating the table does not renumber the rows we keep.
        wpe = sd[_CKPT_PREFIX + "wpe.weight"]
        if wpe.shape[0] < self.max_len:
            raise ValueError(
                f"config max_len={self.max_len} exceeds the checkpoint's "
                f"{wpe.shape[0]} positions")
        new["wpe.weight"] = wpe[: self.max_len]
        new["ln_f.weight"] = sd[_CKPT_PREFIX + "ln_f.weight"]
        new["ln_f.bias"] = sd[_CKPT_PREFIX + "ln_f.bias"]
        for i in range(self.cfg.n_layers):
            p = f"{_CKPT_PREFIX}h.{i}."
            q = f"h.{i}."
            for ln in ("ln_1", "ln_2"):
                new[q + ln + ".weight"] = sd[p + ln + ".weight"]
                new[q + ln + ".bias"] = sd[p + ln + ".bias"]
            new[q + "c_attn.weight"] = sd[p + "attn.c_attn.weight"].T
            new[q + "c_attn.bias"] = sd[p + "attn.c_attn.bias"]
            new[q + "c_proj.weight"] = sd[p + "attn.c_proj.weight"].T
            new[q + "c_proj.bias"] = sd[p + "attn.c_proj.bias"]
            new[q + "c_fc.weight"] = sd[p + "mlp.c_fc.weight"].T
            new[q + "c_fc.bias"] = sd[p + "mlp.c_fc.bias"]
            new[q + "c_proj_mlp.weight"] = sd[p + "mlp.c_proj.weight"].T
            new[q + "c_proj_mlp.bias"] = sd[p + "mlp.c_proj.bias"]

        missing, unexpected = self.load_state_dict(new, strict=False)
        allowed_missing = (
            {"lm_head.weight"}  # tied to embed_tokens, never stored separately
            | {f"h.{i}.film.weight" for i in range(self.cfg.n_layers)}
            | {f"h.{i}.film.bias" for i in range(self.cfg.n_layers)}
            | {"plastic_a", "plastic_b"})
        bad = {n for n in missing
               if n not in allowed_missing and ".lora_" not in n}
        if bad or unexpected:
            raise RuntimeError(
                f"checkpoint did not load cleanly: missing={sorted(bad)} "
                f"unexpected={sorted(unexpected)}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def forward(self, ids: torch.Tensor, cond: torch.Tensor | None = None,
                return_hidden: bool = False):
        ids = ids.to(self.device)
        if cond is not None:
            cond = cond.to(self.device)
        B, L = ids.shape
        if L > self.max_len:
            raise ValueError(
                f"sequence has {L} tokens but the backbone's positional "
                f"embedding holds {self.max_len}")
        pos = torch.arange(L, device=ids.device).unsqueeze(0)
        h = self.drop(self.embed_tokens(ids) + self.wpe(pos))

        pad = ids.eq(self.pad_id)
        attn_mask = None
        if pad.any():
            # Additive mask: 0 for real keys, finfo.min for padding. Boolean
            # SDPA masks have "True = keep" semantics, which is the opposite of
            # `nn.MultiheadAttention`'s `key_padding_mask`; an explicit float
            # mask cannot be misread.
            attn_mask = torch.zeros(B, 1, 1, L, dtype=h.dtype, device=h.device)
            attn_mask = attn_mask.masked_fill(
                pad[:, None, None, :], torch.finfo(h.dtype).min)

        for blk in self.h:
            h = blk(h, cond, attn_mask)
        h = self.ln_f(h)
        h = h + (h @ self.plastic_a) @ self.plastic_b
        logits = self.lm_head(h)
        if return_hidden:
            return logits, h
        return logits

    def pooled(self, ids: torch.Tensor,
               cond: torch.Tensor | None = None) -> torch.Tensor:
        ids = ids.to(self.device)
        _, h = self.forward(ids, cond, return_hidden=True)
        m = ids.ne(self.pad_id).unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    # ------------------------------------------------------------------
    # Shift-aware likelihoods
    # ------------------------------------------------------------------
    def _shifted(self, logits: torch.Tensor, ids: torch.Tensor,
                 mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (prediction logits, targets) aligned under the shift.

        ``logits[:, :-1]`` predicts ``ids[:, 1:]``. With ``mask`` (same shape
        as ``ids``) the pair is further restricted to masked *targets*.
        """
        pred = logits[:, :-1]
        tgt = ids[:, 1:]
        if mask is not None:
            sel = mask[:, 1:]
            pred, tgt = pred[sel], tgt[sel]
        return pred, tgt

    def _randint(self, low: int, high: int,
                 generator: torch.Generator | None) -> int:
        """Sample an integer on the generator's device, then move it.

        A CPU generator with a CUDA call is a hard error in torch, so the draw
        happens where the generator lives. This keeps a seed reproducible on
        CPU and GPU, which matters because the masking pattern is part of the
        training objective.
        """
        if high <= low:
            return low
        if generator is not None and generator.device.type == "cuda":
            return int(torch.randint(low, high, (1,),
                                     device=generator.device,
                                     generator=generator).item())
        return int(torch.randint(low, high, (1,), generator=generator).item())

    def _block_span_mask(self, ids: torch.Tensor,
                         generator: torch.Generator | None,
                         min_len: int, max_len: int,
                         sep_id: int | None) -> torch.Tensor:
        """Mask one whole block (or a 15-20 token span) per example.

        The repair benchmark masks a contiguous block, not scattered tokens.
        A model trained only on random masks never has to choose a template
        from scratch, so at inference it blends fragments of several templates
        and fails the strict parser. This mask makes the training operation the
        same shape as the evaluation operation.

        Blocks are the runs between separator tokens. Sequences without
        separators (claim episodes) keep their random-token mask: masking an
        entire short claim would be unconditional generation, which is not the
        operation the repair benchmark measures and would spend capacity on a
        different task.
        """
        B, L = ids.shape
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for b in range(B):
            row = ids[b].tolist()
            n = sum(1 for t in row if t != self.pad_id)
            if n <= 2:
                continue
            # BOS is at 0 and EOS is at n-1; neither is a prediction target.
            seps = ([i for i in range(1, n - 1) if row[i] == sep_id]
                    if sep_id is not None else [])
            if not seps:
                continue
            starts = [1] + [p + 1 for p in seps]
            ends = seps + [n - 1]
            segments = [(s, e) for s, e in zip(starts, ends) if e - s >= 2]
            if not segments:
                continue
            s, e = segments[self._randint(0, len(segments), generator)]
            mask[b, s:e] = True
        return mask

    def masked_denoising_loss(self, ids: torch.Tensor,
                              cond: torch.Tensor | None = None,
                              mask_ratio: torch.Tensor | float | None = None,
                              generator: torch.Generator | None = None,
                              mask_mode: str = "token",
                              block_mask_prob: float = 0.5,
                              block_mask_min: int = 15,
                              block_mask_max: int = 20,
                              sep_id: int | None = None
                              ) -> torch.Tensor:
        """Cross-entropy on masked positions, under the checkpoint's shift.

        The mask ratio is uniform in [0.15, 0.85] per example, as on the
        canvas. That range is aggressive for a 70-token document, and it is
        deliberate: the corpus is small and rigid, so high ratios are what
        force the template structure to be learned rather than copied.

        ``mask_mode="block"`` mixes in whole-block masks: with probability
        ``block_mask_prob`` an example masks one contiguous block instead of
        scattered tokens. The repair benchmark evaluates exactly that operation
        (mask one block, infill it), so training only on random tokens leaves
        the model unable to choose a template from scratch. The mixture keeps
        the denoising signal that the backbone was pre-trained on.
        """
        if mask_mode not in ("token", "block"):
            raise ValueError(f"unknown mask_mode {mask_mode!r}")
        B, L = ids.shape
        ids = ids.to(self.device)
        if cond is not None:
            cond = cond.to(self.device)

        def _rand(shape) -> torch.Tensor:
            """Draw on the generator's device, then move.

            A CPU generator with a CUDA tensor is a hard error in torch, and
            every caller that seeds a run constructs its generator on CPU. The
            values are drawn on the generator's device so a seed produces the
            same mask pattern whether the model is on CPU or GPU — otherwise
            two runs of the same experiment would not be comparable.
            """
            if generator is not None and generator.device != ids.device:
                return torch.rand(shape, generator=generator).to(ids.device)
            return torch.rand(shape, device=ids.device, generator=generator)

        if mask_ratio is None:
            r = _rand((B,))
            mask_ratio = 0.15 + r * 0.7
        if isinstance(mask_ratio, torch.Tensor) and mask_ratio.dim() == 0:
            mask_ratio = mask_ratio.expand(B)
        if isinstance(mask_ratio, (int, float)):
            mask_ratio = torch.full((B,), float(mask_ratio), device=ids.device)

        keep = ids.ne(self.pad_id)
        rand = _rand(ids.shape)
        token_mask = (rand < mask_ratio.unsqueeze(1)) & keep
        # Position 0 has no predictor under the shift convention.
        token_mask[:, 0] = False
        empty = ~token_mask.any(1)
        if empty.any():
            first = keep[:, 1:].float().argmax(1) + 1
            token_mask[empty, first[empty]] = True

        mask = token_mask
        if mask_mode == "block":
            block_mask = self._block_span_mask(
                ids, generator, block_mask_min, block_mask_max, sep_id)
            # A row with no usable span falls back to its token mask rather
            # than producing an empty target set and a NaN loss.
            empty_block = ~block_mask.any(1)
            if empty_block.any():
                block_mask[empty_block] = token_mask[empty_block]
            choose_block = _rand((B,)) < float(block_mask_prob)
            mask = torch.where(choose_block.unsqueeze(1), block_mask, token_mask)

        noisy = torch.where(mask, torch.full_like(ids, self.mask_id), ids)
        logits = self.forward(noisy, cond)
        pred, tgt = self._shifted(logits, ids, mask)
        return F.cross_entropy(pred, tgt)

    @torch.no_grad()
    def token_nll(self, ids: torch.Tensor,
                  cond: torch.Tensor | None = None) -> torch.Tensor:
        """Per-position NLL of batch element 0, under the shift.

        Index ``i`` is the NLL of token ``i`` given everything else. Position 0
        is 0.0 because the checkpoint has no predictor for it; callers must not
        include it in an average (edited spans never do — they start at the
        first content token).
        """
        ids = ids.to(self.device)
        logits = self.forward(ids, cond)
        logp = logits.log_softmax(-1)
        lp = logp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        out = torch.zeros(ids.shape[1], device=ids.device)
        out[1:] = -lp[0]
        return out

    @torch.no_grad()
    def span_nll(self, ids: torch.Tensor, spans: list[tuple[int, int]],
                 cond: torch.Tensor | None = None) -> torch.Tensor:
        nll = self.token_nll(ids, cond)
        vals = [nll[s:e].mean() for s, e in spans if e > s]
        return torch.stack(vals).mean() if vals else torch.zeros(
            (), device=ids.device)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, cond: torch.Tensor | None, length: int,
                 steps: int = 8, temperature: float = 1.0,
                 top_p: float = 0.9, seed: int | None = None) -> torch.Tensor:
        """Unconditional generation. Kept for interface parity.

        Not used by any experiment: every repair path goes through `infill`,
        where the document supplies the context. Generation from nothing is
        reported nowhere, so it is not tuned.
        """
        B = cond.shape[0] if cond is not None else 1
        dev = self.embed_tokens.weight.device
        ids = torch.full((B, length), self.mask_id, dtype=torch.long, device=dev)
        ids[:, 0] = self.pad_id
        return self.infill(ids, [(0, length)], cond=cond, steps=steps,
                           temperature=temperature, top_p=top_p, seed=seed)

    @torch.no_grad()
    def infill(self, ids: torch.Tensor, spans: list[tuple[int, int]],
               cond: torch.Tensor | None = None, steps: int = 6,
               temperature: float = 0.0, top_p: float = 0.9,
               seed: int | None = None, order: str = "left_to_right"
               ) -> torch.Tensor:
        """Mask the given spans and denoise them, leaving other tokens fixed.

        This follows the checkpoint's sampler (DiffuLLaMA `generate_samples`):
        sample a full proposal from the shifted logits, commit a subset of the
        still-masked positions, and repeat. The commit *count* is deterministic
        (ceil(pending / remaining)) rather than Bernoulli, so a run finishes in
        exactly `steps` rounds and two seeds differ only in which positions go
        first.

        `order` is not a style choice. Under this checkpoint's shift
        convention, token `j` is predicted from position `j - 1`, so a token
        whose predecessor is still a mask is predicted from a mask-position
        hidden state. Left-to-right commit keeps every predecessor real and is
        therefore the in-distribution order; the official random order does not
        (it is tuned for unconditional generation from all masks, where no
        predecessor is real either way). `order="random"` is kept so the
        difference can be measured rather than asserted.

        With `order="left_to_right"` the number of rounds is raised to the
        longest span, so each round commits exactly one position. `steps` is
        then a lower bound, not the schedule. This makes decoding cost
        proportional to the number of tokens edited, which is the same quantity
        the experiments match on, so a wider edit is not silently given a
        cheaper or a finer decoder.

        Default `temperature=0.0` (greedy) makes a repair deterministic. On 12
        tasks greedy and `temperature=0.5` were 0.33 and 0.42 parseable — too
        close to separate at that n, so the tie is broken towards the choice
        that removes a variance source, not towards the higher number.

        Position 0 is never masked. Under the shift convention it has no
        predictor, so re-generating it would leave a mask token in the output
        and silently poison every downstream parse.
        """
        if order not in ("left_to_right", "random"):
            raise ValueError(f"unknown commit order {order!r}")
        ids = ids.to(self.device)
        dev = ids.device
        g = torch.Generator(device=dev)
        if seed is not None:
            g.manual_seed(seed)

        out = ids.clone()
        target = torch.zeros_like(out, dtype=torch.bool)
        for s, e in spans:
            target[:, s:e] = True
        target[:, 0] = False
        out[target] = self.mask_id
        pending = target.clone()

        rounds = max(int(steps), 1)
        if order == "left_to_right":
            rounds = max(rounds, int(target.sum(dim=1).max().item()))

        for step in range(rounds):
            if not pending.any():
                break
            logits = self.forward(out, cond)
            prop = self._propose(logits, temperature, top_p, g)
            remaining = max(rounds - step, 1)
            for b in range(out.shape[0]):
                idx = pending[b].nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                k = max(1, math.ceil(idx.numel() / remaining))
                if order == "left_to_right":
                    take = idx[:k]
                else:
                    perm = torch.randperm(idx.numel(), generator=g, device=dev)
                    take = idx[perm[:k]]
                out[b, take] = prop[b, take]
                pending[b, take] = False

        if pending.any():
            # Only reachable if `steps` was 0 or a span was empty; a greedy
            # fill keeps the output parseable instead of leaving mask tokens.
            logits = self.forward(out, cond)
            prop = self._propose(logits, 0.0, 1.0, g)
            out = torch.where(pending, prop, out)
        return out

    def _propose(self, logits: torch.Tensor, temperature: float,
                 top_p: float, g: torch.Generator) -> torch.Tensor:
        """Sample every position, then shift: position j gets logits[j-1]."""
        if temperature is not None and temperature > 0:
            x = logits / max(float(temperature), 1e-6)
            if top_p is not None and 0.0 < float(top_p) < 1.0:
                x = _top_p(x, float(top_p))
            probs = x.softmax(-1)
            flat = probs.reshape(-1, probs.shape[-1])
            samp = torch.multinomial(flat, 1, generator=g).view(probs.shape[:-1])
        else:
            samp = logits.argmax(-1)
        out = samp.clone()
        out[:, 1:] = samp[:, :-1]
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filter, per position, matching DiffuLLaMA's `top_p_logits`."""
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_logits.softmax(-1), dim=-1)
    remove = cumulative > p
    # Keep the first token above the threshold.
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask = mask.scatter(-1, sorted_idx, remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)

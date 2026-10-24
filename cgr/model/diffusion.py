"""Masked-diffusion backbone (§3.2).

Bidirectional attention, random-ratio masking during training, and
confidence-ordered iterative unmasking at inference. Bidirectionality is
what makes revision possible at all: a token can be re-decided in the light
of right-hand context, which an autoregressive decoder cannot do.

The side channel enters only as `cond`, a fixed-width vector applied through
FiLM at every block. There is deliberately **no path from reasoning tokens to
the vocabulary projection** — the reasoner cannot emit the artifact, it can
only bias it. `test_side_channel.py` asserts this structurally.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        # FiLM from the side channel
        self.film = nn.Linear(cfg.d_model, 2 * cfg.d_model)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor | None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ln1(h)
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask,
                         need_weights=False)
        h = h + a
        h = h + self.ff(self.ln2(h))
        if cond is not None:
            g, b = self.film(cond).chunk(2, dim=-1)
            h = h * (1.0 + g.unsqueeze(1)) + b.unsqueeze(1)
        return h


class MaskedDiffusionLM(nn.Module):
    def __init__(self, cfg: ModelConfig, mask_id: int = 1, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        # The only modifiable substrate (§3.7): a low-rank adapter on the
        # hidden state, so a consolidated update can move decisions and not
        # just token logits. `plastic_b` is zero-initialised, so the adapter
        # is an exact identity until consolidation writes to it.
        self.plastic_rank = 8
        self.plastic_a = nn.Parameter(torch.randn(cfg.d_model, self.plastic_rank) * 0.02)
        self.plastic_b = nn.Parameter(torch.zeros(self.plastic_rank, cfg.d_model))

    # ------------------------------------------------------------------
    def forward(self, ids: torch.Tensor, cond: torch.Tensor | None = None,
                return_hidden: bool = False):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0)
        h = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        pad_mask = ids.eq(self.pad_id)
        for blk in self.blocks:
            h = blk(h, cond, key_padding_mask=pad_mask if pad_mask.any() else None)
        h = self.ln_f(h)
        h = h + (h @ self.plastic_a) @ self.plastic_b
        logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits

    def pooled(self, ids: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        _, h = self.forward(ids, cond, return_hidden=True)
        m = ids.ne(self.pad_id).unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    # ------------------------------------------------------------------
    def masked_denoising_loss(self, ids: torch.Tensor,
                              cond: torch.Tensor | None = None,
                              mask_ratio: torch.Tensor | float | None = None,
                              generator: torch.Generator | None = None) -> torch.Tensor:
        """Cross-entropy on masked positions only (MDLM-style)."""
        B, L = ids.shape
        if mask_ratio is None:
            r = torch.rand(B, device=ids.device, generator=generator)
            mask_ratio = self.cfg.__dict__.get("mask_ratio_min", 0.15) + r * 0.7
        if isinstance(mask_ratio, torch.Tensor) and mask_ratio.dim() == 0:
            mask_ratio = mask_ratio.expand(B)
        if isinstance(mask_ratio, float):
            mask_ratio = torch.full((B,), mask_ratio, device=ids.device)

        rand = torch.rand(ids.shape, device=ids.device, generator=generator)
        keep = ids.ne(self.pad_id)
        mask = (rand < mask_ratio.unsqueeze(1)) & keep
        # guarantee at least one masked position per example
        empty = ~mask.any(1)
        if empty.any():
            first = keep.float().argmax(1)
            mask[empty, first[empty]] = True
        noisy = torch.where(mask, torch.full_like(ids, self.mask_id), ids)
        logits = self.forward(noisy, cond)
        loss = F.cross_entropy(logits[mask], ids[mask])
        return loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, cond: torch.Tensor | None, length: int,
                 steps: int = 8, temperature: float = 1.0,
                 seed: int | None = None) -> torch.Tensor:
        dev = self.tok_emb.weight.device
        g = torch.Generator(device=dev)
        if seed is not None:
            g.manual_seed(seed)
        ids = torch.full((cond.shape[0] if cond is not None else 1, length),
                         self.mask_id, dtype=torch.long, device=dev)
        return self._unmask(ids, cond, steps, temperature, g)

    @torch.no_grad()
    def infill(self, ids: torch.Tensor, spans: list[tuple[int, int]],
               cond: torch.Tensor | None = None, steps: int = 6,
               temperature: float = 1.0, seed: int | None = None) -> torch.Tensor:
        """Mask the given spans and denoise them, leaving other tokens fixed."""
        dev = ids.device
        g = torch.Generator(device=dev)
        if seed is not None:
            g.manual_seed(seed)
        out = ids.clone()
        target = torch.zeros_like(out, dtype=torch.bool)
        for s, e in spans:
            target[:, s:e] = True
        out[target] = self.mask_id
        return self._unmask(out, cond, steps, temperature, g, fixed=~target)

    @torch.no_grad()
    def _unmask(self, ids: torch.Tensor, cond: torch.Tensor | None, steps: int,
                temperature: float, g: torch.Generator,
                fixed: torch.Tensor | None = None) -> torch.Tensor:
        ids = ids.clone()
        for _ in range(steps):
            logits = self.forward(ids, cond)
            probs = (logits / max(temperature, 1e-6)).softmax(-1)
            conf, pred = probs.max(-1)
            masked = ids.eq(self.mask_id)
            if not masked.any():
                break
            # unmask a proportional slice, most confident first
            frac = 1.0 / max(steps, 1)
            k = max(1, int(masked.float().sum() / masked.shape[0] * frac))
            for b in range(ids.shape[0]):
                idx = masked[b].nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                c = conf[b, idx]
                take = idx[torch.argsort(c, descending=True)[:k]]
                ids[b, take] = pred[b, take]
        # any residual masks -> argmax fill
        if ids.eq(self.mask_id).any():
            logits = self.forward(ids, cond)
            pred = logits.argmax(-1)
            m = ids.eq(self.mask_id)
            ids[m] = pred[m]
        return ids

    # ------------------------------------------------------------------
    def span_nll(self, ids: torch.Tensor, spans: list[tuple[int, int]],
                 cond: torch.Tensor | None = None) -> torch.Tensor:
        """Mean NLL of the given spans under the model, used for seam/fluency."""
        logits = self.forward(ids, cond)
        logp = logits.log_softmax(-1)
        vals = []
        for s, e in spans:
            tgt = ids[:, s:e]
            lp = logp[:, s:e, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            vals.append(-lp.mean())
        return torch.stack(vals).mean() if vals else torch.zeros((), device=ids.device)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

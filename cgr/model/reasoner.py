"""Latent side channel: coarse-to-fine denoising over thought-blocks (§3.2).

The reasoner produces `n_blocks` latent vectors by denoising from noise, then
pools them into a single steering vector `cond`. It has no vocabulary
projection, so it can only steer the artifact — never become it.

Two properties are tested, not assumed:

1. Structural isolation. `Reasoner` contains no `nn.Linear(*, vocab_size)`.
2. Causal use. `steer_from_blocks` accepts externally supplied blocks, which
   is how the shuffle / constant / none ablations (Exp 3) are implemented
   without touching the backbone.

The "coarse-to-fine" schedule is real but small-scale: noise level k
determines how much of each block is replaced by Gaussian noise before the
denoiser reconstructs it. High k leaves only the block skeleton; low k
leaves the fills.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig


class Reasoner(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, dl, nb = cfg.d_model, cfg.d_latent, cfg.n_blocks
        self.enc = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, dl * nb))
        self.denoiser = nn.Sequential(
            nn.Linear(dl + 1, d), nn.GELU(), nn.Linear(d, dl))
        self.pool = nn.Sequential(
            nn.Linear(dl * nb, d), nn.GELU(), nn.Linear(d, d))
        # verifier proxy q(z) used for gradient guidance (∇_z ℓ in §3.2)
        self.verifier = nn.Sequential(
            nn.Linear(dl * nb, d), nn.GELU(), nn.Linear(d, 1))
        # NOTE: an earlier draft carried a `log_sigma` scale parameter for a
        # Gaussian NLL. The value head regresses a scalar with MSE, so that
        # parameter was never used and never received a gradient. It was
        # removed rather than left in the parameter count. See
        # scripts/check_presets.py, which fails if any parameter is unreachable.

    # ------------------------------------------------------------------
    def encode(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled [B, d_model] -> blocks [B, n_blocks, d_latent]."""
        return self.enc(pooled).view(-1, self.cfg.n_blocks, self.cfg.d_latent)

    def combine(self, blocks: torch.Tensor) -> torch.Tensor:
        """blocks [B, n_blocks, d_latent] -> cond [B, d_model]."""
        return self.pool(blocks.reshape(blocks.shape[0], -1))

    def steer(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = self.encode(pooled)
        return self.combine(blocks), blocks

    def steer_from_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        return self.combine(blocks)

    # ------------------------------------------------------------------
    def denoise(self, blocks: torch.Tensor, level: float,
                eta: float | None = None) -> torch.Tensor:
        """One coarse-to-fine denoising step at noise `level` in [0, 1].

        Optionally applies the guidance term (eta * grad_z q(z)), i.e. the
        discrete stand-in for `w ∇_z ℓ(z; r_l)` in §3.2.
        """
        B, nb, dl = blocks.shape
        noise = torch.randn_like(blocks) * level
        z = blocks + noise
        lev = torch.full((B, nb, 1), level, device=blocks.device)
        delta = self.denoiser(torch.cat([z, lev], dim=-1))
        out = z + delta

        eta = self.cfg.guidance_weight if eta is None else eta
        if eta > 0:
            with torch.enable_grad():
                zz = out.detach().requires_grad_(True)
                q = self.verifier(zz.reshape(B, -1)).sum()
                (grad,) = torch.autograd.grad(q, zz)
            out = out + eta * 0.01 * grad  # small step; guidance is a nudge
        return out

    def denoise_chain(self, blocks: torch.Tensor, steps: int | None = None
                      ) -> torch.Tensor:
        """Coarse-to-fine: level 1 -> 0, then a final clean pass."""
        steps = steps or self.cfg.latent_steps
        z = blocks
        for i in range(steps):
            level = 1.0 - i / max(steps, 1)
            z = self.denoise(z, level)
        return z

    # ------------------------------------------------------------------
    def loss(self, pooled: torch.Tensor, target_score: torch.Tensor | None = None
             ) -> torch.Tensor:
        """Denoising objective + optional verifier regression."""
        blocks = self.encode(pooled).detach()
        total = pooled.new_zeros(())
        for level in (0.9, 0.5, 0.2):
            noisy = blocks + torch.randn_like(blocks) * level
            lev = torch.full((*blocks.shape[:2], 1), level, device=blocks.device)
            pred = noisy + self.denoiser(torch.cat([noisy, lev], dim=-1))
            total = total + torch.nn.functional.mse_loss(pred, blocks)
        if target_score is not None:
            q = self.verifier(blocks.reshape(blocks.shape[0], -1)).squeeze(-1)
            total = total + torch.nn.functional.mse_loss(q, target_score)
        return total

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

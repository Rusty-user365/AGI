"""The seven shared heads (§3.3), with the calibration harness attached.

Each head is a small MLP over a shared trunk. Outputs are returned as a dict
of raw logits/scalars so the training loop and the calibration diagnostics
see the same object.

Note on redundancy: in a deterministic world, V, o and tau are deterministic
functions of the same features and the heads collapse onto each other. To
make the seven heads non-redundant the *data* must be stochastic — outcome
noise, variable completion time, and partial observability. `cgr/scaffold.py`
introduces exactly that; `head_health()` reports the collinearity so a
reviewer can see it rather than take our word for it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig

HEAD_NAMES = ["value", "outcome", "tau", "epistemic", "scope", "source", "disagreement"]


class HeadModule(nn.Module):
    def __init__(self, cfg: ModelConfig, d_in: int | None = None):
        super().__init__()
        self.cfg = cfg
        d_in = d_in or (cfg.d_model * 2 + 8)
        self.trunk = nn.Sequential(
            nn.Linear(d_in, cfg.d_model), nn.GELU(), nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
        )

        def mlp(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_head), nn.GELU(),
                nn.Linear(cfg.d_head, out_dim),
            )

        self.heads = nn.ModuleDict({
            "value": mlp(1),
            "outcome": mlp(cfg.d_outcome),
            "tau": mlp(1),
            "epistemic": mlp(cfg.n_epistemic),
            "scope": mlp(cfg.n_scope),
            "source": mlp(cfg.n_source),
            "disagreement": mlp(1),
        })

    def forward(self, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(feats)
        out = {k: v(h).squeeze(-1) if v[-1].out_features == 1 else v(h)
               for k, v in self.heads.items()}
        return out

    # ------------------------------------------------------------------
    def loss(self, preds: dict[str, torch.Tensor],
             labels: dict[str, torch.Tensor],
             weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        w = weights or {k: 1.0 for k in HEAD_NAMES}
        parts, stats = {}, {}
        for name in HEAD_NAMES:
            p, y = preds[name], labels[name]
            if name == "value":
                l = F.mse_loss(torch.sigmoid(p), y.float())
            elif name == "tau":
                l = F.mse_loss(p, torch.log1p(y.float().clamp(min=0)))
            elif name == "disagreement":
                l = F.mse_loss(torch.sigmoid(p), y.float())
            elif name == "outcome":
                l = F.mse_loss(p, y.float())
            else:
                l = F.cross_entropy(p, y.long())
            parts[name] = l
            stats[name] = float(l.detach())
        total = sum(w[n] * parts[n] for n in HEAD_NAMES)
        stats["total"] = float(total.detach())
        return total, stats

    @torch.no_grad()
    def predict(self, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        p = self.forward(feats)
        return {
            "value": torch.sigmoid(p["value"]),
            "outcome": p["outcome"],
            "tau": F.softplus(p["tau"]),
            "epistemic": p["epistemic"].argmax(-1),
            "epistemic_probs": p["epistemic"].softmax(-1),
            "scope": p["scope"].argmax(-1),
            "scope_probs": p["scope"].softmax(-1),
            "source": p["source"].argmax(-1),
            "disagreement": torch.sigmoid(p["disagreement"]),
        }

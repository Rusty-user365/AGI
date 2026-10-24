"""Eight-action gate with three overrides (§3.4).

Actions
-------
ANS    answer directly
DELIB  deliberate longer (more denoising steps / more side-channel passes)
VER    invoke the interpreter and verify
REP    local repair (renoise one block)
REORD  reorder blocks
REV    global revision (renoise everything)
DEF    defer, record an obligation, revisit later
UPD    commit a verified update to the plastic substrate

The gate itself is a classifier over an oracle policy during Stage A, then
Gumbel-softmax against the same policy with a decaying KL leash in Stage B.
The overrides are *not* learned: they are the safety interlocks, and the
paper should report how often each fires (Exp 5).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig

ACTIONS = ["ANS", "DELIB", "VER", "REP", "REORD", "REV", "DEF", "UPD"]
A = {name: i for i, name in enumerate(ACTIONS)}

# Irreversible actions: taking them commits the system (or the world).
IRREVERSIBLE = {A["ANS"], A["UPD"]}
# Actions that can be undone cheaply after the fact.
REVERSIBLE = {A["REP"], A["REV"], A["DELIB"], A["REORD"], A["VER"]}

DEFAULT_COSTS = {
    A["ANS"]: 0.0, A["DELIB"]: 0.3, A["VER"]: 0.2, A["REP"]: 0.3,
    A["REORD"]: 0.4, A["REV"]: 0.8, A["DEF"]: 0.5, A["UPD"]: 0.6,
}


class Gate(nn.Module):
    def __init__(self, cfg: ModelConfig, d_in: int | None = None):
        super().__init__()
        self.cfg = cfg
        d_in = d_in or (cfg.d_model * 2 + 8)
        self.net = nn.Sequential(
            nn.Linear(d_in, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_head), nn.GELU(),
            nn.Linear(cfg.d_head, cfg.n_actions),
        )
        self.register_buffer(
            "costs", torch.tensor([DEFAULT_COSTS[i] for i in range(cfg.n_actions)]))

    def logits(self, feats: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        return self.net(feats) - lam * self.costs

    def forward(self, feats: torch.Tensor, lam: float = 1.0,
                hard: bool = False, tau: float | None = None):
        lg = self.logits(feats, lam)
        if self.training and not hard:
            tau = self.cfg.gumbel_tau if tau is None else tau
            return F.gumbel_softmax(lg, tau=tau, hard=False), lg
        return lg.argmax(-1), lg

    # ------------------------------------------------------------------
    def decide(self, lg: torch.Tensor, *, surprise: float | None = None,
               value_sigma: float | None = None, tau_err: float = 0.5,
               sigma_hi: float = 0.25) -> tuple[int, dict]:
        """Apply the three overrides to a single example's logits.

        Returns the chosen action and a record of which override fired, so the
        paper can report override frequency instead of asserting the gate
        "selects" among eight actions.
        """
        a = int(lg.argmax())
        info = {"base": ACTIONS[a], "override": None}

        if surprise is not None and surprise > tau_err:
            if a not in (A["DELIB"], A["VER"], A["DEF"]):
                a = A["VER"] if ACTIONS[a] in ("REP", "ANS") else A["DELIB"]
                info["override"] = "surprise"

        if value_sigma is not None and value_sigma > sigma_hi:
            if ACTIONS[a] in ("ANS", "UPD"):
                # reversibility ladder: start with the cheapest reversible
                # action and escalate only on evidence (§3.4)
                a = A["VER"]
                info["override"] = "risk_ladder"

        info["final"] = ACTIONS[a]
        return a, info


def action_name(i: int) -> str:
    return ACTIONS[int(i)]


__all__ = ["ACTIONS", "A", "Gate", "IRREVERSIBLE", "REVERSIBLE",
           "DEFAULT_COSTS", "action_name"]

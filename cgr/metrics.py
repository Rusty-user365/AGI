"""Metrics, with the definitions the paper was missing.

The draft's troubleshooting table already admits "seam metric ≈ 1 always ->
metric is trivial". That happens when a metric has no way to fail. Every
metric here is defined so that a degenerate system scores badly:

  preservation   fraction of untouched tokens that are unchanged, compared
                 block-aligned: the prefix before the first edited token and
                 the suffix after the last, each from its own end. Trivially
                 1.0 for a no-op system, so it is only reported next to
                 `improvement`. A positional comparison would score a length
                 change inside the edited block as damage to the suffix.
  seam           change in mean NLL of the tokens flanking each repaired span,
                 each flank located in its own sequence. A repair that damages
                 its neighbours scores positive (worse). This can fail.
  consistency    exact parse-based document check (see grammar.is_consistent).
  ece            top-1 confidence calibration error, per head.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def ece(conf: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on confidence."""
    conf = conf.float().flatten()
    correct = correct.float().flatten()
    if conf.numel() == 0:
        return float("nan")
    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    n = conf.numel()
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc = correct[m].mean()
        avg = conf[m].mean()
        total += (m.sum().item() / n) * abs(acc - avg)
    return float(total)


def brier(prob: torch.Tensor, target: torch.Tensor) -> float:
    return float(((prob.float() - target.float()) ** 2).mean())


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------

def bootstrap_ci(values, stat=None, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (mean, lo, hi)."""
    vals = [float(v) for v in values]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    stat = stat or (lambda x: sum(x) / len(x))
    rng = random.Random(seed)
    boots = []
    n = len(vals)
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(stat(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return stat(vals), lo, hi


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired difference b - a with a CI. Use for every two-condition claim."""
    assert len(a) == len(b)
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    mean, lo, hi = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    return {"mean_diff": mean, "lo": lo, "hi": hi,
            "significant": (lo > 0) or (hi < 0)}


def minimum_detectable_effect(n: int, sd: float, alpha: float = 0.05,
                              power: float = 0.8) -> float:
    """Rough two-sided MDE for a paired mean. Report this next to null results."""
    z_a = 1.959964
    z_b = 0.841621
    return (z_a + z_b) * sd / math.sqrt(max(n, 1))


# --------------------------------------------------------------------------
# Repair metrics
# --------------------------------------------------------------------------

@dataclass
class RepairScore:
    consistent: bool
    improvement: float          # change in span NLL (positive = better)
    preservation: float         # fraction of untouched tokens unchanged
    seam: float                 # NLL change on flanking tokens (negative = better)
    exact_block_match: bool


def seam_positions(spans: list[tuple[int, int]], length: int) -> list[int]:
    pos = []
    for s, e in spans:
        if s - 1 >= 0:
            pos.append(s - 1)
        if e < length:
            pos.append(e)
    return sorted(set(pos))


@torch.no_grad()
def score_repair(model, orig_ids: torch.Tensor, new_ids: torch.Tensor,
                 spans_orig: list[tuple[int, int]],
                 spans_new: list[tuple[int, int]],
                 consistent: bool) -> RepairScore:
    """Compare a repaired sequence against the original, *block-aligned*.

    A generative edit can change the length of the edited region, so comparing
    the two sequences position by position labels every token after the edit as
    "damaged": the text is fine, it has simply moved. That made `preservation`
    and `seam` measure length change rather than edit quality.

    So the untouched tokens are compared as a prefix (before the first edited
    token) and a suffix (after the last edited token), each aligned from its
    own end. `improvement` and `seam` are measured over each sequence's own
    span offsets. `spans_orig` and `spans_new` must correspond one-to-one and
    in order, which they do when both are derived from the same block layout.
    """
    if len(spans_orig) != len(spans_new):
        raise ValueError("spans_orig and spans_new must correspond one-to-one")

    lo, ln = orig_ids.shape[1], new_ids.shape[1]
    nll_o = _token_nll(model, orig_ids)
    nll_n = _token_nll(model, new_ids)

    mask_o = torch.zeros(lo, dtype=torch.bool)
    mask_n = torch.zeros(ln, dtype=torch.bool)
    for s, e in spans_orig:
        mask_o[s:e] = True
    for s, e in spans_new:
        mask_n[s:e] = True

    improvement = (float(nll_o[mask_o].mean() - nll_n[mask_n].mean())
                   if mask_o.any() and mask_n.any() else 0.0)

    # Untouched regions, aligned from the start and from the end respectively.
    first_o = int(mask_o.nonzero()[0]) if mask_o.any() else lo
    first_n = int(mask_n.nonzero()[0]) if mask_n.any() else ln
    last_o = int(mask_o.nonzero()[-1]) + 1 if mask_o.any() else 0
    last_n = int(mask_n.nonzero()[-1]) + 1 if mask_n.any() else 0
    pre = min(first_o, first_n)
    suf = min(lo - last_o, ln - last_n)
    eq = torch.cat([orig_ids[0, :pre] == new_ids[0, :pre],
                    orig_ids[0, lo - suf:] == new_ids[0, ln - suf:]])
    preservation = float(eq.float().mean()) if eq.numel() else 1.0

    # Seam: each flank of each edited region, located in its own sequence.
    deltas = []
    for (s_o, e_o), (s_n, e_n) in zip(spans_orig, spans_new):
        if s_o - 1 >= 0 and s_n - 1 >= 0:
            deltas.append(float(nll_n[s_n - 1] - nll_o[s_o - 1]))
        if e_o < lo and e_n < ln:
            deltas.append(float(nll_n[e_n] - nll_o[e_o]))
    seam = sum(deltas) / len(deltas) if deltas else 0.0

    same_shape = (orig_ids.shape == new_ids.shape)
    return RepairScore(
        consistent=consistent, improvement=improvement,
        preservation=preservation, seam=seam,
        exact_block_match=bool(same_shape and (orig_ids == new_ids).all()))


@torch.no_grad()
def _token_nll(model, ids: torch.Tensor) -> torch.Tensor:
    """Per-position NLL of batch element 0.

    Delegates to the backbone when it can. The shift convention is a property
    of the backbone — the blank canvas predicts token `i` from `logits[i]`, the
    pre-trained DiffuGPT from `logits[i-1]` — so a convention hard-coded here
    would score one of them against the wrong target and make `improvement`
    and `seam` meaningless.
    """
    if hasattr(model, "token_nll"):
        return model.token_nll(ids)
    logits = model.forward(ids)
    logp = logits.log_softmax(-1)
    lp = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return -lp[0]


# --------------------------------------------------------------------------
# Triage
# --------------------------------------------------------------------------

def marks_banked(order_scores: list[float], budget: int) -> int:
    """Number of marks obtained within a budget under a fixed ordering."""
    return sum(1 for s in order_scores[:budget] if s >= 0.5)


def defer_precision(deferred: list[bool], truly_unverifiable: list[bool]) -> float:
    """Of what we deferred, how much genuinely could not be verified."""
    if not deferred:
        return float("nan")
    hits = sum(1 for d, u in zip(deferred, truly_unverifiable) if d and u)
    return hits / sum(1 for d in deferred if d)


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------

ACTION_COST = {"ANS": 1, "DELIB": 3, "VER": 2, "REP": 3,
               "REORD": 4, "REV": 8, "DEF": 5, "UPD": 6}


def cost_per_correct(actions: list[str], correct: list[bool]) -> float:
    n = sum(1 for c in correct if c)
    if n == 0:
        return float("inf")
    return sum(ACTION_COST[a] for a in actions) / n


def tier_distribution(actions: list[str]) -> dict[str, float]:
    total = max(len(actions), 1)
    out: dict[str, int] = {}
    for a in actions:
        out[a] = out.get(a, 0) + 1
    return {k: v / total for k, v in sorted(out.items())}

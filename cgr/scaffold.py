"""The external scaffolding: the hand-written policy that is the curriculum (§3.8).

This module is deliberately *not* learned. It encodes the decisions an
external scheduler would make — verify before answering, defer when the claim
is empirical, repair locally when the defect is local, revise globally when it
is not — and those decisions become the Stage A / Stage B targets.

Being explicit about this matters for honesty: the paper's claim is that the
scaffolding is the curriculum and internalization is future work. Anyone
reading this file can see exactly what was internalized.

Label noise: `value`, `tau` and `outcome` are observed quantities in a real
deployment, not deterministic functions of the input. We perturb them with a
documented, seeded noise term (`TrainConfig.label_noise`). Without it, the
three regression heads are deterministic functions of the two categorical
heads and collapse into each other. This is a design choice, and it is
reported rather than hidden.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from .data.corpus import Record
from .data.documents import Task
from .data.facts import FAMILIES, SCOPE_NONE, Claim
from .model.gate import A
from .model.interpreter import adjudicate, epistemic_status

EPISTEMIC = {"known": 0, "computable": 1, "check_later": 2, "unknown": 3}
SOURCE = {"weights": 0, "experience": 1, "derived": 2}
SCOPE_INDEX = {f: i for i, f in enumerate(FAMILIES)}


@dataclass
class Episode:
    ids: torch.Tensor
    task: str                     # "repair" | "doc" | "claim"
    labels: dict[str, torch.Tensor]
    action: int
    text: str
    clock: torch.Tensor
    claim: Claim | None = None
    repair: Task | None = None
    record: Record | None = None
    meta: dict = field(default_factory=dict)


def _clock(task: str, remaining: float, deadline: float = 10.0) -> torch.Tensor:
    t_now = 0.0
    elapsed = max(deadline - remaining, 0.0)
    v = torch.tensor([t_now, elapsed, remaining, deadline,
                      1.0 if task == "claim" else 0.0,
                      1.0 if task == "repair" else 0.0,
                      1.0 if task == "doc" else 0.0,
                      remaining / deadline], dtype=torch.float32)
    return v


def _noise(rng: random.Random, scale: float) -> float:
    return rng.gauss(0.0, scale) if scale > 0 else 0.0


def doc_labels(consistent: bool, noise: float, rng: random.Random) -> dict:
    return {
        "value": torch.tensor(1.0 if consistent else 0.55 + _noise(rng, noise)),
        "outcome": torch.tensor([1.0, 0.0, 0.0, 0.0]) + _noise(rng, noise),
        "tau": torch.tensor(max(0.5, 1.0 + _noise(rng, noise))),
        "epistemic": torch.tensor(EPISTEMIC["known"]),
        "scope": torch.tensor(SCOPE_NONE),
        "source": torch.tensor(SOURCE["derived"]),
        "disagreement": torch.tensor(0.0),
    }


def repair_labels(ex: Task, noise: float, rng: random.Random) -> dict:
    # A cross-block defect needs several coordinated edits, so the expected
    # value of a single revision attempt is lower than for a local quality
    # defect. This spread is what keeps the value head from collapsing to a
    # constant; it is a curriculum assumption, not a measurement.
    steps = 2.0 if ex.defect_type == "local" else 4.0
    value = 0.9 if ex.defect_type == "local" else 0.6
    return {
        "value": torch.tensor(value + _noise(rng, noise)),
        "outcome": torch.tensor([1.0, 0.0, 0.0, 0.0]) + _noise(rng, noise),
        "tau": torch.tensor(max(0.5, steps + _noise(rng, noise))),
        "epistemic": torch.tensor(EPISTEMIC["known"]),
        "scope": torch.tensor(SCOPE_NONE),
        "source": torch.tensor(SOURCE["derived"]),
        "disagreement": torch.tensor(0.0),
    }


def claim_labels(c: Claim, noise: float, rng: random.Random) -> dict:
    v = adjudicate(c.spec, c.marker_present)
    stat = epistemic_status(v)
    adj = v.correct is not None
    if adj:
        outcome = [1.0 if v.correct else 0.0, 0.0, 0.0, 0.0]
    else:
        # Non-adjudicable: there is no ground-truth outcome. Labelling it 0.0
        # ("the system was wrong") is a false certainty, and labelling it 1.0
        # (the earlier behaviour, via `v.correct is not None` being True for
        # empirical claims) is what made the value head endorse everything.
        # A flat target over the two used bins says "unknown".
        outcome = [0.5, 0.5, 0.0, 0.0]
    return {
        "value": torch.tensor((1.0 if adj else 0.2) + _noise(rng, noise)),
        "outcome": torch.tensor(outcome) + _noise(rng, noise),
        "tau": torch.tensor(max(0.5, (1.0 if stat == "known" else 4.0)
                                + _noise(rng, noise))),
        "epistemic": torch.tensor(EPISTEMIC[stat]),
        "scope": torch.tensor(SCOPE_INDEX[c.family] if c.marker_present
                              else SCOPE_NONE),
        "source": torch.tensor(SOURCE["derived"] if adj else SOURCE["weights"]),
        "disagreement": torch.tensor(0.0),
    }


def oracle_action(c: Claim | None, ex: Task | None,
                  consistent: bool | None = None) -> int:
    if ex is not None:
        if ex.defect_type == "local":
            return A["REP"]
        if ex.defect_type == "cross_block":
            return A["REV"]
        return A["ANS"]
    if c is not None:
        if c.family == "empirical":
            return A["DEF"]
        if not c.marker_present:
            return A["DELIB"]       # scope unresolved: do not answer yet
        return A["VER"]
    if consistent is not None:
        return A["ANS"]
    return A["DEF"]


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def episodes_from_records(records: list[Record], noise: float = 0.1,
                          seed: int = 0) -> list[Episode]:
    rng = random.Random(seed)
    out = []
    for r in records:
        out.append(Episode(
            ids=torch.tensor(r.ids),
            task="doc",
            labels=doc_labels(True, noise, rng),
            action=oracle_action(None, None, consistent=True),
            text=" ".join(r.blocks),
            clock=_clock("doc", 10.0),
            record=r,
        ))
    return out


def episodes_from_repairs(examples: list[Task], tok, noise: float = 0.1,
                          seed: int = 0) -> list[Episode]:
    from .data.grammar import blocked_spans
    rng = random.Random(seed)
    out = []
    for ex in examples:
        # the *corrupted* document is the observation
        ids, _ = blocked_spans(ex.corrupt_blocks, tok)
        out.append(Episode(
            ids=torch.tensor(ids),
            task="repair",
            labels=repair_labels(ex, noise, rng),
            action=oracle_action(None, ex),
            text=" ".join(ex.corrupt_blocks),
            clock=_clock("repair", 6.0),
            repair=ex,
        ))
    return out


def episodes_from_claims(claims: list[Claim], tok, noise: float = 0.1,
                         seed: int = 0) -> list[Episode]:
    rng = random.Random(seed)
    out = []
    for c in claims:
        out.append(Episode(
            ids=torch.tensor(tok.encode(c.text)),
            task="claim",
            labels=claim_labels(c, noise, rng),
            action=oracle_action(c, None),
            text=c.text,
            clock=_clock("claim", 3.0),
            claim=c,
        ))
    return out


__all__ = ["Episode", "episodes_from_records", "episodes_from_repairs",
           "episodes_from_claims", "oracle_action", "EPISTEMIC", "SOURCE",
           "SCOPE_INDEX"]

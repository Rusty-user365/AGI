"""C1: the structural guard, tested without the editor.

The claim under test is structural, not behavioural:

    a verified-wrong example cannot move the plastic substrate and cannot
    create a factual memory entry, while it still supervises rejection.

C2 (does the guard improve held-out behaviour?) is a different claim and is
not tested here. The existing `test_gradient_masking` in test_cgr.py freezes
everything but the adapter, so it cannot check the rejection path: a frozen
gate has no gradient to be non-zero. These tests leave every parameter
trainable, because a frozen parameter cannot leak and the unguarded control
would then be vacuous.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from cgr.config import Config
from cgr.data.corpus import build_corpus
from cgr.data.facts import make_claim_set
from cgr.model.interpreter import adjudicate
from cgr.model.system import CGRSystem
from cgr.scaffold import episodes_from_claims
from cgr.train import collate

TOL = 1e-8


@pytest.fixture()
def setup():
    cfg = Config.preset("smoke")
    tok, _ = build_corpus(cfg)
    cfg.model.vocab_size = tok.vocab_size
    system = CGRSystem(cfg, tok)
    system.eval()
    # All trainable: a frozen parameter cannot leak, so the unguarded control
    # would prove nothing about the instrument.
    for p in system.parameters():
        p.requires_grad_(True)
    return cfg, tok, system


def _claims(system, n=4):
    pool = make_claim_set(n * 6, seed=11, marker_present=True)
    wrong = [c for c in pool
             if adjudicate(c.spec, c.marker_present).correct is False][:n]
    right = [c for c in pool
             if adjudicate(c.spec, c.marker_present).correct is True][:n]
    assert len(wrong) == n and len(right) == n
    return wrong, right


def _batch(system, cfg, claims):
    return collate(episodes_from_claims(claims, system.tok),
                   system.tok, cfg.model.max_len)


def _plastic_grad(system) -> torch.Tensor:
    parts = []
    for p in (system.backbone.plastic_a, system.backbone.plastic_b):
        parts.append((p.grad if p.grad is not None
                      else torch.zeros_like(p)).reshape(-1))
    return torch.cat(parts)


def _gate_grad_norm(system) -> float:
    return sum(float(p.grad.pow(2).sum()) for p in system.gate.parameters()
               if p.grad is not None) ** 0.5


def _step(system, batch, verified: bool):
    system.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    snap = system.plastic_snapshot()
    opt = torch.optim.SGD(system.parameters(), lr=0.1)
    st = system.consolidate(batch["ids"], batch["clock"], batch["labels"],
                            verified=[verified] * len(batch["ids"]),
                            advantage=[1.0] * len(batch["ids"]), optim=opt)
    return st, snap


# --------------------------------------------------------------------------
# the guarded path
# --------------------------------------------------------------------------

def test_verified_wrong_leaves_substrate_bit_identical(setup):
    cfg, _, system = setup
    wrong, _ = _claims(system)
    st, snap = _step(system, _batch(system, cfg, wrong), verified=False)
    assert st["n_updated"] == 0
    assert st["n_rejected"] == len(wrong)
    assert float(_plastic_grad(system).abs().max()) == 0.0
    assert system.plastic_unchanged(snap)


def test_verified_wrong_supervises_rejection(setup):
    """The check that catches masking everything.

    A implementation that masks the whole example would pass the gradient
    check while teaching the model nothing, including not to reject. The gate
    must still receive gradient from the rejection target.
    """
    cfg, _, system = setup
    wrong, _ = _claims(system)
    st, _ = _step(system, _batch(system, cfg, wrong), verified=False)
    assert st["policy_supervised"] is True
    assert st["stepped"] is True
    assert _gate_grad_norm(system) > TOL


# --------------------------------------------------------------------------
# the controls that validate the instrument
# --------------------------------------------------------------------------

def test_unguarded_control_leaks(setup):
    """Same wrong claims, treated as verified: every guard check must fail."""
    cfg, _, system = setup
    wrong, _ = _claims(system)
    st, snap = _step(system, _batch(system, cfg, wrong), verified=True)
    assert st["n_updated"] == len(wrong)
    assert float(_plastic_grad(system).abs().max()) > TOL
    assert not system.plastic_unchanged(snap)


def test_verified_right_control_updates(setup):
    """The pipeline is not simply broken: a verified-right example updates."""
    cfg, _, system = setup
    _, right = _claims(system)
    st, snap = _step(system, _batch(system, cfg, right), verified=True)
    assert st["n_updated"] == len(right)
    assert float(_plastic_grad(system).abs().max()) > TOL
    assert not system.plastic_unchanged(snap)


# --------------------------------------------------------------------------
# memory guard
# --------------------------------------------------------------------------

def test_unverified_memory_write_is_refused(setup):
    _, _, system = setup
    system.memory.clear()
    eid = system.update_memory("2 + 2 = 5 .", verified=False)
    assert eid == -1
    assert len(system.memory) == 0


def test_verified_memory_write_is_allowed(setup):
    _, _, system = setup
    system.memory.clear()
    eid = system.update_memory("2 + 2 = 4 .", verified=True)
    assert eid >= 0
    assert len(system.memory) == 1


def test_recall_skips_unverified_weight_entries(setup):
    """Defence in depth: a checkpoint written before the guard must not leak."""
    _, _, system = setup
    system.memory.clear()
    d = system.cfg.model.d_model
    system.memory.add("2 + 2 = 5 .", torch.randn(d), scope="none",
                      source="weights", verified=False)
    vec, score, entry = system.memory.recall("2 + 2 = 5 .")
    assert vec is None and score == 0.0 and entry is None

    # An unverified *experience* entry is not an assertion, so it is kept.
    system.memory.add("tried and failed", torch.randn(d), scope="none",
                      source="experience", verified=False)
    vec, _, entry = system.memory.recall("tried and failed")
    assert vec is not None and entry.source == "experience"

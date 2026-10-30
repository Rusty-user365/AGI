"""Fast control tests using callbacks instead of a trained language model."""
from __future__ import annotations

from cgr.control import Action, CommitmentDAG, Controller
from cgr.data.rd_datasets import (make_injection_cases, make_logic_traps,
                                 make_safety_claims)


class FakeForwardHead:
    def __init__(self, surprise: float):
        self.surprise = surprise

    def __call__(self, _draft: str) -> dict[str, float]:
        return {"surprise": self.surprise}


class FakeGate:
    def __init__(self, action: Action = Action.ANS):
        self.action = action

    def propose(self, _draft: str) -> Action:
        return self.action


def fake_turn(controller: Controller, gate: FakeGate,
              forward: FakeForwardHead, draft: str, **kwargs):
    signals = forward(draft)
    return controller.step(draft, proposed=gate.propose(draft),
                           surprise=signals["surprise"], **kwargs)


def test_high_surprise_routes_to_rev_and_restores_verified_checkpoint():
    dag = CommitmentDAG("root answer")
    controller = Controller(dag, surprise_threshold=0.5)
    gate = FakeGate(Action.ANS)

    accepted = fake_turn(controller, gate, FakeForwardHead(0.1),
                         "verified revision", verifier=lambda _: True)
    assert accepted.action == Action.ANS
    assert accepted.artifact == "verified revision"

    dag.checkpoint("unverified speculative draft", verified=False,
                   reason="pre-existing branch")
    revision = fake_turn(controller, gate, FakeForwardHead(0.9),
                         "surprising draft", verifier=lambda _: True)
    assert revision.action == Action.REV
    assert revision.artifact == "verified revision"
    assert dag.current_id == accepted.checkpoint_id
    assert dag.get(accepted.checkpoint_id).verified


def test_verifier_rejection_backtracks_to_previous_checkpoint():
    dag = CommitmentDAG("trusted baseline")
    controller = Controller(dag)
    result = controller.step("bad candidate", proposed=Action.ANS,
                             surprise=0.0, verifier=lambda _: False)
    assert result.action == Action.REV
    assert result.artifact == "trusted baseline"
    assert len(dag) == 2


def test_gate_rev_restores_checkpoint_without_running_verifier():
    dag = CommitmentDAG("previous answer")
    dag.checkpoint("unverified branch", verified=False)
    calls = []
    result = Controller(dag).step(
        "candidate", proposed=Action.REV, surprise=0.0,
        verifier=lambda _: calls.append(True) or True)
    assert result.action == Action.REV
    assert result.artifact == "previous answer"
    assert calls == []


def test_deadline_overrun_defers_without_calling_verifier():
    dag = CommitmentDAG("current draft")
    controller = Controller(dag)
    calls = []

    result = controller.step("late candidate", proposed=Action.ANS,
                             surprise=0.0,
                             verifier=lambda _: calls.append(True) or True,
                             ttc=11, deadline=10)
    assert result.action == Action.DEF
    assert result.artifact == "current draft"
    assert calls == []
    assert len(dag) == 1


def test_logic_traps_are_unique_and_have_distinct_plausible_lures():
    rows = make_logic_traps(5_000, seed=11)
    assert len({row["prompt"] for row in rows}) == 5_000
    for row in rows:
        truth = row["ground_truth"]
        assert truth["cheaper_item_price_cents"] * 2 + truth["difference_cents"] == truth["total_cents"]
        assert truth["cheaper_item_price_cents"] != truth["intuitive_lure_cents"]
        assert row["intuitive_wrong_answer"] != row["verified_correct_answer"]


def test_safety_claims_are_balanced_and_injections_are_labeled():
    claims = make_safety_claims(5_000, seed=5)
    assert sum(row["is_true"] for row in claims) == 2_500
    assert len({tuple(row["operands"]) for row in claims}) == len(claims)
    injections = make_injection_cases(500, seed=7)
    assert len(injections) == 500
    assert all("HACKED" in row["injection_payload"] for row in injections)
    assert all("follow trusted_instruction" in row["expected_behavior"]
               for row in injections)

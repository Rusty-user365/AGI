"""Model-agnostic gate execution and revision checkpoint graph.

This is a CPU-only control prototype. It is not yet wired into CGRSystem's
learned gate or diffusion repair path; callbacks make routing testable without
loading a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Action(str, Enum):
    ANS = "ANS"
    DELIB = "DELIB"
    VER = "VER"
    REP = "REP"
    REORD = "REORD"
    REV = "REV"
    DEF = "DEF"
    UPD = "UPD"


@dataclass
class Checkpoint(Generic[T]):
    node_id: int
    parent_id: int | None
    artifact: T
    verified: bool
    reason: str = ""
    children: list[int] = field(default_factory=list)


class CommitmentDAG(Generic[T]):
    """An append-only tree of draft snapshots with verified-ancestor restore."""

    def __init__(self, initial: T):
        self._nodes: dict[int, Checkpoint[T]] = {
            0: Checkpoint(0, None, initial, True, "initial")
        }
        self._next_id = 1
        self.current_id = 0

    @property
    def current(self) -> Checkpoint[T]:
        return self._nodes[self.current_id]

    def get(self, node_id: int) -> Checkpoint[T]:
        return self._nodes[node_id]

    def checkpoint(self, artifact: T, *, verified: bool,
                   reason: str = "candidate") -> Checkpoint[T]:
        parent = self.current_id
        node = Checkpoint(self._next_id, parent, artifact, verified, reason)
        self._nodes[node.node_id] = node
        self._nodes[parent].children.append(node.node_id)
        self._next_id += 1
        self.current_id = node.node_id
        return node

    def backtrack(self) -> Checkpoint[T]:
        """Restore the nearest verified ancestor, or raise if none exists."""
        node = self.current
        while not node.verified:
            if node.parent_id is None:
                raise RuntimeError("no verified checkpoint is available")
            node = self._nodes[node.parent_id]
        self.current_id = node.node_id
        return node

    def restore(self, node_id: int) -> Checkpoint[T]:
        node = self._nodes[node_id]
        if not node.verified:
            raise ValueError("cannot restore an unverified checkpoint")
        self.current_id = node_id
        return node

    def __len__(self) -> int:
        return len(self._nodes)


@dataclass(frozen=True)
class Decision:
    action: Action
    artifact: object
    checkpoint_id: int
    reason: str


class Controller:
    """Route a proposed action using surprise, deadline, and verifier signals."""

    def __init__(self, dag: CommitmentDAG[T], *, surprise_threshold: float = 0.5):
        if surprise_threshold < 0:
            raise ValueError("surprise_threshold must be non-negative")
        self.dag = dag
        self.surprise_threshold = surprise_threshold

    def step(self, draft: T, *, proposed: Action,
             surprise: float, verifier: Callable[[T], bool],
             ttc: float = 0.0, deadline: float | None = None) -> Decision:
        if ttc < 0 or (deadline is not None and deadline < 0):
            raise ValueError("ttc and deadline must be non-negative")
        if deadline is not None and ttc > deadline:
            return Decision(Action.DEF, self.dag.current.artifact,
                            self.dag.current_id, "deadline exceeded")

        if proposed == Action.REV or surprise > self.surprise_threshold:
            restored = self.dag.backtrack()
            reason = ("gate selected REV; restored verified ancestor"
                      if proposed == Action.REV else
                      "high surprise; restored verified ancestor")
            return Decision(Action.REV, restored.artifact, restored.node_id,
                            reason)

        if proposed == Action.DEF:
            return Decision(Action.DEF, self.dag.current.artifact,
                            self.dag.current_id, "gate deferred")

        is_verified = bool(verifier(draft))
        checkpoint = self.dag.checkpoint(
            draft, verified=is_verified,
            reason="verified candidate" if is_verified else "rejected candidate")
        if not is_verified:
            restored = self.dag.backtrack()
            return Decision(Action.REV, restored.artifact, restored.node_id,
                            "verifier rejected candidate; restored verified ancestor")
        return Decision(proposed, checkpoint.artifact, checkpoint.node_id,
                        "candidate verified and checkpointed")

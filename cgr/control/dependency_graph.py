"""Dependency-aware artifact invalidation; KV suffixes require separate replay."""
from __future__ import annotations

import copy

SKILLS = {"base", "coder", "ui", "cyber_defensive", "fullstack_api_backend_schema", "unknown"}


class DependencyGraph:
    def __init__(self, nodes: list[dict]):
        self.nodes = {}
        for node in nodes:
            key = node["id"]
            if not isinstance(key, str) or not key or key in self.nodes:
                raise ValueError("Node IDs must be unique nonempty strings")
            if node.get("skill") not in SKILLS or not isinstance(node.get("parents"), list):
                raise ValueError("Invalid skill or parents")
            self.nodes[key] = copy.deepcopy(node)
        for node in self.nodes.values():
            if len(set(node["parents"])) != len(node["parents"]) or any(p not in self.nodes for p in node["parents"]):
                raise ValueError("Duplicate or missing dependency")
        self.order = []
        pending = set(self.nodes)
        while pending:
            ready = [key for key in self.nodes if key in pending and set(self.nodes[key]["parents"]) <= set(self.order)]
            if not ready:
                raise ValueError("Cyclic dependency graph")
            self.order.extend(ready)
            pending.difference_update(ready)
        self.artifacts = {}
        self.dirty = set(self.nodes)

    def descendants(self, key: str) -> set[str]:
        if key not in self.nodes:
            raise KeyError(key)
        found = set()
        for candidate in self.order:
            if key in self.nodes[candidate]["parents"] or found.intersection(self.nodes[candidate]["parents"]):
                found.add(candidate)
        return found

    def ancestors(self, key: str) -> set[str]:
        found = set()
        todo = list(self.nodes[key]["parents"])
        while todo:
            parent = todo.pop()
            if parent not in found:
                found.add(parent)
                todo.extend(self.nodes[parent]["parents"])
        return found

    def fault_candidates(self, observed_node: str, failed_contract_nodes: list[str]) -> list[str]:
        """Provenance-constrained suspects, not a proof of root cause.

        failed_contract_nodes must come from independently executed checks,
        never the fixture's hidden injection location.
        """
        scope = self.ancestors(observed_node) | {observed_node}
        failed = scope.intersection(failed_contract_nodes)
        return [key for key in self.order if key in failed and not self.ancestors(key).intersection(failed)]

    def commit(self, key: str, artifact, verifier) -> dict:
        if any(parent in self.dirty for parent in self.nodes[key]["parents"]):
            raise ValueError("Cannot commit with stale/unverified dependencies")
        if not verifier(artifact):
            return {"accepted": False, "invalidated": []}
        invalidated = self.descendants(key) if key in self.artifacts else set()
        self.artifacts[key] = copy.deepcopy(artifact)
        self.dirty.discard(key)
        self.dirty.update(invalidated)
        return {"accepted": True, "invalidated": [n for n in self.order if n in invalidated]}

    def cache_replay_plan(self, key: str, serialized_ids: list[str], token_starts: dict[str, int]) -> dict:
        """Retaining an independent artifact does not retain its later K/V."""
        if len(serialized_ids) != len(set(serialized_ids)) or set(serialized_ids) != set(self.nodes):
            raise ValueError("Serialized sequence must contain every node exactly once")
        starts = [token_starts[n] for n in serialized_ids]
        if any(type(x) is not int or x < 0 for x in starts) or any(b <= a for a, b in zip(starts, starts[1:])):
            raise ValueError("Token boundaries must increase strictly")
        position = {n: i for i, n in enumerate(serialized_ids)}
        if any(position[p] >= position[n] for n in serialized_ids for p in self.nodes[n]["parents"]):
            raise ValueError("Serialized sequence violates dependency order")
        suffix = serialized_ids[position[key]:]
        affected = self.descendants(key) | {key}
        return {"crop_absolute_tokens": token_starts[key], "discard_kv_nodes": suffix,
                "regenerate_artifacts": [n for n in suffix if n in affected],
                "replay_preserved_artifacts": [n for n in suffix if n not in affected]}

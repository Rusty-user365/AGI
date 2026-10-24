"""Additive experience memory with the null property (§3.6).

    output = f(x; W0) + alpha * Read(M, x)

Null property: M = empty => the base output is returned *unchanged*. This is
enforced structurally, not numerically: `recall` returns None when the store
is empty, and the caller skips the addition entirely. Returning a zero vector
and adding it would also be bit-identical for floats, but skipping makes the
intent explicit and survives a later switch to a non-zero-identity read.

The real experiment near this invariant is *rollback*: insert entries, then
remove them, and verify the output is bit-identical again. That tests the
whole read path rather than the arithmetic of adding zero.

Keying uses a fixed external embedding. A deployment would use a pretrained
sentence encoder; we use a deterministic hashed bag-of-words so the codebase
has no model-download dependency. It is fixed (not trained), which is the
property the paper actually needs for transferability.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..config import ModelConfig


class HashEmbedder:
    """Fixed, deterministic, external to the model."""

    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def __call__(self, text: str) -> torch.Tensor:
        v = torch.zeros(self.dim)
        words = text.lower().split()
        if not words:
            return v
        for w in words:
            h = int(hashlib.md5(f"{self.seed}:{w}".encode()).hexdigest()[:8], 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            v[idx] += sign
        return v / v.norm().clamp(min=1e-8)


@dataclass
class Entry:
    key: torch.Tensor
    value: torch.Tensor          # d_model vector
    content: str
    scope: str = "none"
    source: str = "experience"
    verified: bool = False
    justification: str = ""
    answer: str | None = None
    entry_id: int = -1


class SparseMemory(nn.Module):
    def __init__(self, cfg: ModelConfig, embedder: HashEmbedder | None = None):
        super().__init__()
        self.cfg = cfg
        self.embedder = embedder or HashEmbedder(cfg.mem_dim)
        self.entries: list[Entry] = []
        self.alpha = nn.Parameter(torch.zeros(1))  # starts at 0: no opinion
        # NOTE: an earlier draft also had a zero-initialised `to_vocab`
        # projection here (d_model x vocab_size). It was never called, so it
        # contributed ~46k dead parameters to the reported model size. Removed.
        self._next_id = 0
        self.hit_threshold = 0.6

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def add(self, text: str, value: torch.Tensor, *, scope: str = "none",
            source: str = "experience", verified: bool = False,
            justification: str = "", answer: str | None = None) -> int:
        if len(self.entries) >= self.cfg.mem_capacity:
            self.entries.pop(0)
        e = Entry(key=self.embedder(text).detach(),
                  value=value.detach().clone(), content=text, scope=scope,
                  source=source, verified=verified,
                  justification=justification, answer=answer,
                  entry_id=self._next_id)
        self._next_id += 1
        self.entries.append(e)
        return e.entry_id

    def remove(self, entry_id: int) -> bool:
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.entry_id != entry_id]
        return len(self.entries) < before

    def clear(self) -> None:
        self.entries = []

    # ------------------------------------------------------------------
    def recall(self, text: str, k: int = 8):
        """Return (value_vector, score, entry) or (None, 0.0, None) if empty.

        Unverified *factual* entries are skipped. `update_memory` already
        refuses to create them, so this is defence in depth: a checkpoint
        written before the guard existed must not be able to influence the
        output with content the verifier rejected. Experience entries are not
        filtered, because they are not assertions.
        """
        entries = [e for e in self.entries
                   if e.verified or e.source != "weights"]
        if not entries:
            return None, 0.0, None
        q = self.embedder(text)
        keys = torch.stack([e.key for e in entries])
        sims = torch.nn.functional.cosine_similarity(keys, q.unsqueeze(0), dim=-1)
        top = torch.topk(sims, min(k, len(entries)))
        w = top.values.softmax(-1)
        vals = torch.stack([entries[i].value for i in top.indices.tolist()])
        vec = (w.unsqueeze(-1) * vals).sum(0)
        best = int(top.indices[0])
        return vec, float(top.values[0]), entries[best]

    def verified_hit(self, text: str, scope: str | None = None) -> Entry | None:
        """Return a cached verified entry, but only within the same scope.

        Scope matching is not optional. A bag-of-words key is dominated by the
        claim body, so `2 + 2 = 5` and `rounding up to the nearest 5, 2 + 2 = 5`
        are ~0.9 cosine similar. Without the scope check the cache happily
        returns the scoped answer for the bare claim — which is precisely the
        default-scope leakage the architecture is supposed to prevent. This
        was caught by Exp 2 before it could become a reported result.
        """
        if not self.entries:
            return None
        cands = [e for e in self.entries
                 if e.verified and (scope is None or e.scope == scope)]
        if not cands:
            return None
        q = self.embedder(text)
        keys = torch.stack([e.key for e in cands])
        sims = torch.nn.functional.cosine_similarity(keys, q.unsqueeze(0), dim=-1)
        best = int(sims.argmax())
        if float(sims[best]) >= self.hit_threshold:
            return cands[best]
        return None

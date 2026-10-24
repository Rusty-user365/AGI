"""Corpus construction plus thin wrappers over the task generator.

The corruption operators now live in `cgr.data.documents`, where every task
carries an explicit contract:

    The introduction is the authoritative, immutable thesis. All editable
    blocks must be revised to support it.

The two defect types are different in kind, not in severity:

  local        one body block loses specificity/completeness while still
               supporting the authoritative thesis. Coherence is intact; the
               defect is *quality*, and one block needs repair.

  cross_block  the locked intro states T*, while every editable block supports
               a single wrong thesis W. The defect is *coherence*, and one
               block cannot fix it because the others still contradict T*.

This module keeps the old function names as wrappers so that existing callers
keep working, but the objects returned are `Task`s with per-block metadata.
The claim the benchmark supports is conditional: revision scope should follow
the defect's location and dependency structure *and* the available budget.
`repeated_local` is a first-class condition, not a straw man.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..config import Config
from ..tokenizer import Tokenizer
from .documents import (Task, corrupt_cross_block, degrade_local,
                        make_task_set, task_from_blocks)
from .grammar import (CONCLUSION_TEMPLATES, HOLDOUT_COMBOS,
                      THESIS_INDEX, INTRO_TEMPLATES, Doc, blocked_spans,
                      is_consistent, make_doc, parse_blocks)

#: Backwards-compatible name. A `Task` carries the same fields the old
#: `RepairExample` did (`clean_blocks`, `corrupt_blocks`, `target_block`,
#: `defect_type`, `corrupted_block_ids`) but adds the authoritative thesis and
#: per-block metadata, so callers that only read those fields keep working.
RepairExample = Task


@dataclass
class Record:
    doc_id: int
    thesis_id: int
    blocks: list[str]
    ids: list[int]
    spans: list[tuple[int, int]]
    combo: tuple[int, int]
    split: str  # "train" | "val" | "test" | "heldout"


def _record(doc: Doc, tok: Tokenizer, doc_id: int, split: str) -> Record:
    ids, spans = blocked_spans(doc.blocks, tok)
    return Record(doc_id, doc.thesis_id, doc.blocks, ids, spans,
                  doc.combo, split)


def _sample_docs(rng: random.Random, n: int,
                 combos: set[tuple[int, int]]) -> list[Doc]:
    """Draw docs whose (intro, conclusion) template pair is in `combos`."""
    combos = sorted(combos)
    return [make_doc(rng, combo=rng.choice(combos)) for _ in range(n)]


def build_corpus(cfg: Config, tok: Tokenizer | None = None
                 ) -> tuple[Tokenizer, dict[str, list[Record]]]:
    """Generate train/val/test/heldout splits.

    `heldout` uses template combinations never seen in training. This is the
    anti-memorisation split: if heldout and val scores diverge sharply, the
    model learned surface form rather than structure.
    """
    rng = random.Random(cfg.data.seed)
    seen_combos = {(i, j)
                   for i in range(len(INTRO_TEMPLATES))
                   for j in range(len(CONCLUSION_TEMPLATES))} - HOLDOUT_COMBOS

    n_heldout = max(50, cfg.data.n_test // 4)
    train_docs = _sample_docs(rng, cfg.data.n_docs, seen_combos)
    val_docs = _sample_docs(rng, cfg.data.n_val, seen_combos)
    test_docs = _sample_docs(rng, cfg.data.n_test, seen_combos)
    held_docs = _sample_docs(rng, n_heldout, HOLDOUT_COMBOS)

    if tok is None:
        tok = Tokenizer.build([d.render() for d in train_docs])

    splits = {
        "train": [_record(d, tok, i, "train") for i, d in enumerate(train_docs)],
        "val": [_record(d, tok, i, "val") for i, d in enumerate(val_docs)],
        "test": [_record(d, tok, i, "test") for i, d in enumerate(test_docs)],
        "heldout": [_record(d, tok, i, "heldout") for i, d in enumerate(held_docs)],
    }
    return tok, splits


# --------------------------------------------------------------------------
# Corruption (thin wrappers around cgr.data.documents)
# --------------------------------------------------------------------------

def _authoritative(blocks: list[str]) -> int:
    """Read T* off the introduction. A document with no parseable intro has no
    contract and cannot be used as a repair task."""
    p = parse_blocks(blocks)
    if not p.ok or p.thesis is None:
        raise ValueError("document has no parseable introduction")
    tid = THESIS_INDEX.get(p.thesis)
    if tid is None:
        raise ValueError(f"unknown thesis {p.thesis!r}")
    return tid


def corrupt_local(blocks: list[str], rng: random.Random,
                  task_id: int = 0) -> Task:
    """Contract: exactly one body block loses specificity; proposition kept.

    The produced block still parses and still supports T*, so the document
    remains coherent. `degrade_local` validates the produced document and
    raises if the contract is violated, rather than trusting a flag.
    """
    clean = task_from_blocks(blocks, _authoritative(blocks), task_id)
    return degrade_local(clean, rng)


def corrupt_nonlocal(blocks: list[str], rng: random.Random,
                     thesis_id: int, task_id: int = 0) -> Task:
    """Deprecated name for the cross-block defect.

    Kept because old callers and old prose use "non-local". The task it
    returns has `defect == "cross_block"`, the intro locked, and every
    editable block retargeted to one wrong thesis.
    """
    clean = task_from_blocks(blocks, thesis_id, task_id)
    return corrupt_cross_block(clean, rng)


def make_repair_set(records: list[Record], n: int, seed: int = 0
                    ) -> dict[str, list[Task]]:
    """Clean + local + cross-block tasks, one of each defect per document.

    `non_local` is an alias of `cross_block` for backwards compatibility.
    """
    tasks = make_task_set(records, n, seed=seed)
    return {"local": tasks["local"], "cross_block": tasks["cross_block"],
            "non_local": tasks["cross_block"]}


__all__ = [
    "Record", "RepairExample", "Task", "build_corpus", "corrupt_local",
    "corrupt_nonlocal", "make_repair_set", "make_task_set", "is_consistent",
]

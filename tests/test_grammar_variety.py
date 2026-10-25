"""The grammar's two jobs, checked separately.

1. **Round-trip.** Every surface template must render and parse back to the
   same labels. A template that parses as the wrong thesis is worse than a
   template that fails to parse, because the consistency metric would still
   return a number.

2. **Corruption preserves shape.** `degrade_local` and `retarget` must keep the
   block's template. If a corruption also changed the sentence shape, a repair
   model could detect the defect from the style alone, and the experiment would
   measure template matching rather than dependency reasoning.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from cgr.config import Config
from cgr.data.corpus import build_corpus
from cgr.data.documents import (corrupt_cross_block, degrade_local,
                                task_from_blocks)
from cgr.data.grammar import (BODY_TEMPLATES, CONCLUSION_TEMPLATES, THESES,
                              INTRO_TEMPLATES, body_parts, concl_parts,
                              intro_parts, is_consistent, make_doc,
                              parse_block, parse_block_tmpl, render_body,
                              render_concl, render_intro)


def test_every_intro_template_round_trips():
    for i in range(len(INTRO_TEMPLATES)):
        text = render_intro(i, THESES[0][0], 3)
        p = intro_parts(text)
        assert p is not None, f"intro template {i} did not parse: {text!r}"
        assert p[0] == i, f"template index mismatch for {text!r}"
        assert p[1] == THESES[0][0]
        assert p[2] == 3


def test_every_conclusion_template_round_trips():
    for i in range(len(CONCLUSION_TEMPLATES)):
        text = render_concl(i, THESES[1][0], 2)
        p = concl_parts(text)
        assert p is not None, f"conclusion {i} did not parse: {text!r}"
        assert p[0] == i
        assert p[1] == 2, f"count misparsed for {text!r}"
        assert p[2] == THESES[1][0], f"thesis misparsed for {text!r}"


def test_every_body_template_round_trips_without_swapping_fields():
    """The reason and the thesis must not be read from the wrong slot.

    Templates 2 and 3 put them on opposite sides of the comma, so a parser that
    assumes one group order silently reports the reason as the thesis.
    """
    thesis = THESES[2][0]
    reason = THESES[2][1][0]
    for i in range(len(BODY_TEMPLATES)):
        text = render_body(i, 4, reason, thesis)
        p = body_parts(text)
        assert p is not None, f"body {i} did not parse: {text!r}"
        assert p[0] == i
        assert p[1] == reason, f"body {i} reason misparsed: {p[1]!r}"
        assert p[2] == 4
        assert p[3] == thesis, f"body {i} thesis misparsed: {p[3]!r}"


def test_generated_documents_are_consistent_across_all_templates():
    rng = random.Random(0)
    combos, body_tmpls = set(), set()
    for _ in range(300):
        d = make_doc(rng)
        assert is_consistent(d.blocks), d.blocks
        combos.add(d.combo)
        for tmpl, block in zip(d.body_tmpls, d.blocks[1:-1]):
            body_tmpls.add(tmpl)
            assert parse_block_tmpl(block, "body") == tmpl
    # The generator is actually using the variety, not collapsing onto one form.
    assert body_tmpls == set(range(len(BODY_TEMPLATES)))
    assert len(combos) >= len(INTRO_TEMPLATES) + len(CONCLUSION_TEMPLATES)


def test_degrade_preserves_the_surface_template():
    rng = random.Random(1)
    for _ in range(50):
        d = make_doc(rng)
        task = task_from_blocks(d.blocks, d.thesis_id)
        out = degrade_local(task, rng)
        i = out.edited[0]
        assert out.blocks[i].tmpl == task.blocks[i].tmpl, (
            "degrade_local changed the sentence shape; the defect would be "
            "detectable from style alone")
        assert parse_block_tmpl(out.blocks[i].text, "body") == task.blocks[i].tmpl


def test_retarget_preserves_the_surface_template():
    rng = random.Random(2)
    for _ in range(50):
        d = make_doc(rng)
        task = task_from_blocks(d.blocks, d.thesis_id)
        wrong = (d.thesis_id + 1) % len(THESES)
        out = corrupt_cross_block(task, rng, wrong_thesis=wrong)
        for i in out.edited:
            assert out.blocks[i].tmpl == task.blocks[i].tmpl, (
                f"block {i} changed shape under retarget")
            assert out.blocks[i].thesis_id == wrong
        # the content survives: only the proposition changes
        for i in out.edited:
            if out.blocks[i].role == "body":
                assert out.blocks[i].reason == task.blocks[i].reason


def test_unseen_split_uses_theses_absent_from_training():
    cfg = Config.preset("diffugpt")
    # Large enough that every seen thesis appears in train and both held-out
    # theses appear in the unseen split. With 6 theses and a tiny sample the
    # *set* of observed labels is random, so a small n makes the assertion
    # flaky for a reason that has nothing to do with the split logic.
    cfg.data.n_docs = 2_000
    cfg.data.n_val = 200
    cfg.data.n_test = 200
    cfg.data.n_heldout_theses = 2
    _, splits = build_corpus(cfg)
    train_theses = {r.thesis_id for r in splits["train"]}
    unseen_theses = {r.thesis_id for r in splits["unseen"]}
    assert unseen_theses, "unseen split is empty"
    assert not (train_theses & unseen_theses), (
        f"thesis {sorted(train_theses & unseen_theses)} appears in both train "
        f"and the compositional holdout")
    assert unseen_theses == set(range(len(THESES) - 2, len(THESES)))
    assert train_theses == set(range(len(THESES) - 2))


def test_unseen_split_is_empty_without_an_open_vocabulary():
    """The canvas has a word vocabulary built from train, so it cannot encode
    an unseen thesis. The split must be empty rather than silently broken."""
    cfg = Config.preset("smoke")
    cfg.data.n_docs = 20
    cfg.data.n_val = 4
    cfg.data.n_test = 4
    _, splits = build_corpus(cfg)
    assert splits["unseen"] == []


def test_heldout_split_uses_unseen_template_combinations():
    cfg = Config.preset("smoke")
    cfg.data.n_docs = 200
    cfg.data.n_val = 20
    cfg.data.n_test = 20
    _, splits = build_corpus(cfg)
    train_combos = {r.combo for r in splits["train"]}
    held_combos = {r.combo for r in splits["heldout"]}
    assert held_combos
    assert not (train_combos & held_combos), (
        "held-out template combination leaked into training")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

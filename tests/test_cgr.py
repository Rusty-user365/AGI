"""Fast unit tests. No training, no GPU, a few seconds total.

    pytest -q

These cover the invariants and the regressions found while building the suite.
The heavier end-to-end check is `scripts/smoke.py`.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from cgr.config import Config
from cgr.data.corpus import build_corpus, corrupt_local, corrupt_nonlocal
from cgr.data.documents import (Budget, anchor_ok, block_quality, coherence,
                                corrupt_cross_block, degrade_local, make_block,
                                matched_budgets, oracle_repair, quality,
                                retarget, task_from_blocks, task_from_record,
                                validate_cross_block,
                                validate_cross_block_partial, validate_local,
                                validate_task)
from cgr.data.facts import (make_claim, make_claim_set,
                            make_counting_ontology_claims,
                            make_empirical_claims)
from cgr.data.grammar import THESES, THESIS_POOL, blocked_spans, is_consistent, parse_blocks
from cgr.metrics import score_repair
from cgr.model.interpreter import OutputMode, adjudicate, epistemic_status
from cgr.model.system import CGRSystem
from cgr.scaffold import SOURCE, claim_labels, episodes_from_claims
from cgr.tokenizer import Tokenizer
from cgr.train import collate


@pytest.fixture(scope="module")
def setup():
    cfg = Config.preset("smoke")
    tok, splits = build_corpus(cfg)
    cfg.model.vocab_size = tok.vocab_size
    system = CGRSystem(cfg, tok)
    system.eval()
    return cfg, tok, splits, system


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def test_tokenizer_roundtrip():
    tok = Tokenizer.build(["hello world .", "hello there ."])
    ids = tok.encode("hello world .")
    assert tok.decode(ids) == "hello world ."
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id


def test_clean_documents_are_consistent(setup):
    _, _, splits, _ = setup
    assert all(is_consistent(r.blocks) for r in splits["train"][:50])


def test_parser_recovers_structure(setup):
    _, _, splits, _ = setup
    p = parse_blocks(splits["train"][0].blocks)
    assert p.ok
    assert p.n_intro == len(p.body_theses)


def test_local_defect_preserves_coherence_and_loses_quality(setup):
    """The local defect is a *quality* defect, not a coherence defect.

    The old generator replaced the body with unparseable filler, which broke
    `is_consistent` and made the local condition a coherence defect. The
    rebuilt operator keeps the proposition and removes specificity, so the
    document still parses and still supports T*.
    """
    _, _, splits, _ = setup
    r = splits["train"][0]
    ex = corrupt_local(r.blocks, random.Random(0))

    assert ex.defect == "local" and len(ex.edited) == 1
    assert is_consistent(ex.corrupt_blocks)          # still parseable
    assert coherence(ex.blocks, ex.authoritative)["c_edit"] == 1.0
    assert quality(ex.blocks, ex.authoritative) < 1.0
    assert block_quality(ex.blocks[ex.target_block],
                         ex.authoritative) == 0
    assert anchor_ok(r.blocks, ex.corrupt_blocks, [0])
    assert validate_local(task_from_record(r), ex) == []


def test_cross_block_defect_is_not_single_block_solvable(setup):
    """The core of the repair-taxonomy claim, tested on produced documents."""
    _, _, splits, _ = setup
    r = splits["train"][0]
    ex = corrupt_nonlocal(r.blocks, random.Random(0), r.thesis_id)

    assert ex.defect == "cross_block"
    assert anchor_ok(r.blocks, ex.corrupt_blocks, [0])  # intro untouched
    assert not is_consistent(ex.corrupt_blocks)
    assert coherence(ex.blocks, ex.authoritative)["c_edit"] == 0.0
    assert validate_cross_block(task_from_record(r), ex) == []

    # Repairing any single editable block leaves at least one contradiction.
    for i in ex.editable_indices:
        out = ex.texts()
        out[i] = ex.clean_blocks[i]
        coh = coherence(task_from_blocks(out, ex.authoritative).blocks,
                        ex.authoritative)
        assert coh["c_edit"] < 1.0

    # Revising every editable block reaches full coherence, intro preserved.
    out = ex.texts()
    for i in ex.editable_indices:
        out[i] = ex.clean_blocks[i]
    assert coherence(task_from_blocks(out, ex.authoritative).blocks,
                     ex.authoritative)["c_edit"] == 1.0
    assert anchor_ok(ex.texts(), out, [0])


def test_partial_cross_block_is_harder_than_the_full_defect(setup):
    """The third family: a *majority* wrong thesis, one body still correct.

    Two properties make it a genuinely harder distractor rather than the full
    defect with fewer edits:

    1. a majority-vote heuristic over editable blocks picks W, so the locked
       intro is the only cue for T*;
    2. a single local repair still leaves coherence < 1.
    """
    _, _, splits, _ = setup
    rng = random.Random(0)
    for r in splits["train"][:40]:
        ex = corrupt_cross_block(task_from_record(r), rng, partial=True)
        assert ex.defect == "cross_block_partial"
        assert validate_cross_block_partial(task_from_record(r), ex) == []
        assert anchor_ok(r.blocks, ex.corrupt_blocks, [0])
        editable = ex.editable_indices
        assert 2 <= len(ex.edited) < len(editable)
        # strict majority of editable blocks is wrong
        assert len(ex.edited) > len(editable) - len(ex.edited)
        # the conclusion is part of the corruption
        assert len(ex.blocks) - 1 in ex.edited
        coh = coherence(ex.blocks, ex.authoritative)
        assert 0.0 < coh["c_edit"] < 1.0
        # one-block repair cannot reach full coherence
        for i in ex.edited:
            out = ex.texts()
            out[i] = ex.clean_blocks[i]
            fixed = coherence(task_from_blocks(out, ex.authoritative).blocks,
                              ex.authoritative)
            assert fixed["c_edit"] < 1.0
        # the full-defect validator must reject a partial document, and the
        # partial validator must reject the full defect: the two contracts are
        # not interchangeable.
        assert validate_cross_block(task_from_record(r), ex) != []
        full = corrupt_cross_block(task_from_record(r), rng)
        assert validate_cross_block_partial(task_from_record(r), full) != []


def test_validators_can_fail(setup):
    """A contract check that cannot fail is not a check."""
    _, _, splits, _ = setup
    r = splits["train"][0]
    clean = task_from_record(r)
    ex = corrupt_nonlocal(r.blocks, random.Random(0), r.thesis_id)

    # Put one editable block back on T*: the document is no longer "every
    # editable block supports one wrong thesis", so the validator must object.
    bad = task_from_blocks(ex.corrupt_blocks, ex.authoritative,
                           defect="cross_block", edited=ex.edited)
    bad.blocks[1] = make_block(r.blocks[1], "body")
    errs = validate_cross_block(clean, bad)
    assert any("still supports T*" in e for e in errs)

    # Tampered metadata must be caught by re-parsing the text.
    bad2 = task_from_record(r)
    bad2.blocks[1].thesis_id = (r.thesis_id + 1) % len(THESES)
    assert validate_task(bad2) != []


def test_coherence_scorer_on_hand_built_documents():
    """C(D) is a synthetic label-consistency score; pin its arithmetic."""
    T, W = 0, 1
    t, w = THESES[T][0], THESES[W][0]
    r1, r2 = THESIS_POOL[T][0], THESIS_POOL[T][1]
    coherent = [f"this essay argues that {t} with 2 reasons .",
                f"reason 1 : {r1} , so {t} .",
                f"reason 2 : {r2} , so {t} .",
                f"these 2 reasons establish that {t} ."]
    task = task_from_blocks(coherent, T)
    coh = coherence(task.blocks, T)
    assert coh["c_all"] == 1.0 and coh["c_edit"] == 1.0
    assert coh["floor_all"] == 0.25
    assert quality(task.blocks, T) == 1.0

    # One body retargeted: c_all 3/4 (the locked intro still matches), c_edit
    # 2/3 (one of two bodies plus the conclusion is off-thesis).
    incoherent = list(coherent)
    incoherent[1] = f"reason 1 : {r1} , so {w} ."
    bad = task_from_blocks(incoherent, T)
    coh = coherence(bad.blocks, T)
    assert abs(coh["c_all"] - 0.75) < 1e-9
    assert abs(coh["c_edit"] - 2 / 3) < 1e-9
    assert coh["off_thesis_editable"] == [1]
    assert quality(bad.blocks, T) < 1.0

    # Vague but on-thesis: coherence is intact, quality is not. This is the
    # distinction the local condition depends on.
    vague = list(coherent)
    vague[1] = f"reason 1 : this reason is left vague , so {t} ."
    v = task_from_blocks(vague, T)
    assert coherence(v.blocks, T)["c_edit"] == 1.0
    assert quality(v.blocks, T) < 1.0


def test_intro_is_locked_and_cannot_be_retargeted(setup):
    _, _, splits, _ = setup
    r = splits["train"][0]
    with pytest.raises(ValueError):
        retarget(r.blocks[0], "intro", (r.thesis_id + 1) % len(THESES))
    task = task_from_record(r)
    assert task.locked_indices == [0]
    assert "authoritative" in task.contract
    assert task.authoritative_thesis == THESES[r.thesis_id][0]


def test_oracle_solvability_at_matched_budget(setup):
    """Step 3 of the protocol: prove the task is solvable before blaming a model."""
    _, tok, splits, _ = setup
    r = splits["train"][0]
    clean = task_from_record(r)

    def n_tokens(s):
        return len(tok.encode(s, add_bos=False, add_eos=False))

    # local: a one-block oracle repair restores quality and coherence
    local = degrade_local(clean, random.Random(0))
    b = matched_budgets(local, n_tokens)["one_block"]
    out, spend = oracle_repair(local, "one_block", b, n_tokens)
    sc = coherence(task_from_blocks(out, local.authoritative).blocks,
                   local.authoritative)
    assert sc["c_edit"] == 1.0
    assert quality(task_from_blocks(out, local.authoritative).blocks,
                   local.authoritative) == 1.0
    assert anchor_ok(local.texts(), out, [0])

    # cross-block: one block is not enough; global and repeated-local are
    for method, expect_full in (("one_block", False), ("global", True),
                                ("repeated_local", True)):
        cross = corrupt_nonlocal(r.blocks, random.Random(0), r.thesis_id)
        b = matched_budgets(cross, n_tokens)[method]
        out, spend = oracle_repair(cross, method, b, n_tokens)
        coh = coherence(task_from_blocks(out, cross.authoritative).blocks,
                        cross.authoritative)
        assert (coh["c_edit"] == 1.0) is expect_full
        assert anchor_ok(cross.texts(), out, [0])
        assert spend["budget"]["tokens_used"] <= b.max_tokens
        assert spend["budget"]["blocks_used"] <= b.max_blocks


def test_coherence_editable_matches_task_editable_indices(setup):
    """`n_editable` from the scorer must equal the blocks budgets are built for.

    Otherwise a method can be budgeted for k blocks while the metric divides by
    a different k, and the reported spend and the reported score describe
    different documents.
    """
    _, _, splits, _ = setup
    for rec in splits["test"][:10]:
        for task in (task_from_record(rec),
                     corrupt_local(rec.blocks, random.Random(0)),
                     corrupt_nonlocal(rec.blocks, random.Random(0),
                                      rec.thesis_id)):
            coh = coherence(task.blocks, task.authoritative)
            assert coh["n_editable"] == len(task.editable_indices)
            assert coh["n_required"] == len(task.blocks)


def test_budget_flags_overflow_instead_of_crashing():
    """A generated block longer than the allowance is reported, not fatal.

    The old path called `spend`, which raised after the text had already been
    produced: either the run died mid-suite or the overrun went unrecorded.
    """
    b = Budget(1, 10)
    assert b.spend_or_flag(1, 4) is True
    assert b.over_budget is False
    assert b.spend_or_flag(1, 9) is False
    assert b.over_budget is True
    assert b.as_dict()["over_budget"] is True
    with pytest.raises(RuntimeError):
        Budget(1, 10).spend(1, 11)


def test_repair_metrics_align_blocks_not_positions(setup):
    """A longer rewrite must not be scored as damage to everything after it.

    `preservation` used to compare the two sequences position by position, so
    any length change inside the edited block shifted the suffix and made
    untouched tokens look modified. The metric must align the prefix and the
    suffix from their own ends, and it must still be able to fail.
    """
    _, tok, splits, system = setup
    r = splits["test"][0]
    before = list(r.blocks)
    after = list(before)
    after[1] = before[1] + " reason"          # longer block, same surroundings

    o_ids, o_spans = blocked_spans(before, tok)
    n_ids, n_spans = blocked_spans(after, tok)
    assert len(n_ids) > len(o_ids)

    def score(orig, new, os_, ns_):
        return score_repair(
            system.backbone, torch.tensor(orig).unsqueeze(0),
            torch.tensor(new).unsqueeze(0), [os_[1]], [ns_[1]], True)

    sc = score(o_ids, n_ids, o_spans, n_spans)
    assert sc.preservation == 1.0, "length change was scored as damage"

    # Negative control: damage the intro and preservation must drop.
    broken = list(after)
    other = THESES[(r.thesis_id + 1) % len(THESES)][0]
    broken[0] = before[0].replace(THESES[r.thesis_id][0], other)
    assert broken[0] != before[0]
    b_ids, b_spans = blocked_spans(broken, tok)
    sc_bad = score(o_ids, b_ids, o_spans, b_spans)
    assert sc_bad.preservation < 1.0, "preservation cannot detect real damage"


def test_counting_ontology_dispositions():
    """Four claims, four required dispositions, one fixed scenario."""
    claims = make_counting_ontology_claims()
    assert [c.expected for c in claims] == ["reject", "reject", "accept",
                                            "defer"]
    got = []
    for c in claims:
        v = adjudicate(c.spec, c.marker_present)
        got.append("defer" if v.correct is None else
                   ("accept" if v.correct else "reject"))
    assert got == [c.expected for c in claims]
    # The bare "2 + 2 = 5" rejection rests on the default decimal scope.
    assert adjudicate(claims[0].spec, claims[0].marker_present).scope_assumed
    # "there are five" with no counting rule is REFUSE, not a guess.
    assert adjudicate(claims[3].spec, False).mode == OutputMode.REFUSE


# --------------------------------------------------------------------------
# interpreter
# --------------------------------------------------------------------------

def test_base_scope_changes_truth_value():
    """2 + 2 = 10: false in decimal, true in base 4."""
    spec = {"family": "base", "base": 4, "a": 2, "c": 2, "claimed": 10}
    assert adjudicate(spec, marker_present=True).correct is True
    spec10 = dict(spec, base=10)
    assert adjudicate(spec10, marker_present=True).correct is False


def test_rounding_scope_rescues_two_plus_two():
    """2 + 2 = 5: true under round-up-to-5, false by default."""
    spec = {"family": "rounding", "m": 5, "a": 2, "c": 2, "claimed": 5}
    assert adjudicate(spec, marker_present=True).correct is True
    assert adjudicate(spec, marker_present=False).correct is False


def test_bare_claim_is_flagged_assumed():
    spec = {"family": "rounding", "m": 5, "a": 2, "c": 2, "claimed": 5}
    v = adjudicate(spec, marker_present=False)
    assert v.scope_assumed is True
    assert v.provenance.value == "assumed"


def test_counting_scope():
    spec = {"family": "count", "males": 2, "females": 2, "pregnant": 1,
            "include_unborn": True, "claimed": 5}
    assert adjudicate(spec, marker_present=True).correct is True
    spec2 = dict(spec, include_unborn=False, claimed=4)
    assert adjudicate(spec2, marker_present=True).correct is True


def test_empirical_is_refused():
    v = adjudicate({"family": "empirical", "text": "x", "scope_value": 0},
                   marker_present=False)
    assert v.mode == OutputMode.REFUSE
    assert epistemic_status(v) == "check_later"


def test_empirical_is_not_adjudicated_false():
    """Regression: empirical verdicts must carry correct=None, not False.

    Returning False made `correct is not None` true, so the scaffolding
    labelled empirical claims as resolved (value=1.0, source=derived) and the
    value head learned to endorse everything. "Not adjudicable" and
    "adjudicated false" are different states and the type must say so.
    """
    v = adjudicate({"family": "empirical", "text": "x", "scope_value": 0},
                   marker_present=False)
    assert v.correct is None


def test_empirical_labels_are_not_success_labels():
    """The curriculum must not label an unverifiable claim as a success."""
    c = make_empirical_claims(1, seed=0)[0]
    lab = claim_labels(c, noise=0.0, rng=random.Random(0))
    assert float(lab["value"]) < 0.5
    assert float(lab["outcome"][0]) < 1.0
    assert int(lab["source"]) == SOURCE["weights"]


def test_uncertainty_propagation_reduces_to_zero_for_exact():
    v = adjudicate({"family": "count", "males": 2, "females": 2,
                    "pregnant": 0, "include_unborn": False, "claimed": 4},
                   marker_present=True, sigma_frac=0.0)
    assert v.sigma == 0.0 and v.mode == OutputMode.DIRECT


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------

def _batch(setup, n=6):
    cfg, tok, _, system = setup
    claims = make_claim_set(n, seed=3)
    eps = episodes_from_claims(claims, tok)
    return collate(eps, tok, cfg.model.max_len)


def test_null_property(setup):
    cfg, tok, _, system = setup
    b = _batch(setup)
    system.memory.clear()
    system.memory.alpha.data.fill_(0.5)
    with torch.no_grad():
        base = system.forward_episodes(b["ids"], b["clock"], memory_texts=None)
        empty = system.forward_episodes(b["ids"], b["clock"],
                                        memory_texts=["x"] * len(b["ids"]))
    assert torch.equal(base["feats"], empty["feats"])


def test_rollback(setup):
    cfg, tok, _, system = setup
    b = _batch(setup)
    system.memory.clear()
    system.memory.alpha.data.fill_(0.5)
    with torch.no_grad():
        base = system.forward_episodes(b["ids"], b["clock"], memory_texts=None)
    ids = [system.update_memory(f"entry {i}", verified=True) for i in range(5)]
    for i in ids:
        assert system.memory.remove(i)
    with torch.no_grad():
        after = system.forward_episodes(b["ids"], b["clock"],
                                        memory_texts=["entry 1"] * len(b["ids"]))
    assert torch.equal(base["feats"], after["feats"])


def test_gradient_masking(setup):
    cfg, tok, _, system = setup
    b = _batch(setup)
    system.freeze_all_but_plastic()
    opt = torch.optim.SGD(system.plastic_params(), lr=0.1)
    snap = system.plastic_snapshot()
    st = system.consolidate(b["ids"], b["clock"], b["labels"],
                            verified=[False] * len(b["ids"]),
                            advantage=[1.0] * len(b["ids"]), optim=opt)
    assert st["n_updated"] == 0
    assert system.plastic_unchanged(snap)


def test_side_channel_isolation(setup):
    _, _, _, system = setup
    vocab = system.cfg.model.vocab_size
    leaks = [n for n, m in system.reasoner.named_modules()
             if isinstance(m, torch.nn.Linear) and m.out_features == vocab]
    assert leaks == []


def test_side_channel_is_used(setup):
    """Shuffling the steering vector must change the output."""
    cfg, tok, _, system = setup
    b = _batch(setup)
    with torch.no_grad():
        full = system.forward_episodes(b["ids"], b["clock"], "full")
        shuf = system.forward_episodes(b["ids"], b["clock"], "shuffled")
        none = system.forward_episodes(b["ids"], b["clock"], "none")
    assert not torch.equal(full["feats"], shuf["feats"])
    assert not torch.equal(full["feats"], none["feats"])


# --------------------------------------------------------------------------
# regressions
# --------------------------------------------------------------------------

def test_memory_cache_is_scope_safe(setup):
    """Regression: a bare claim must not retrieve a scoped cached answer.

    This was a real bug. `verified_hit` matched on bag-of-words similarity, so
    `2 + 2 = 5` retrieved the cached answer for `rounding up to the nearest 5,
    2 + 2 = 5` — the exact default-scope leakage the architecture exists to
    prevent.
    """
    _, _, _, system = setup
    system.memory.clear()
    scoped = "rounding up to the nearest 5 , 2 + 2 = 5 ."
    bare = "2 + 2 = 5 ."
    system.update_memory(scoped, scope="rounding", verified=True, answer="5")

    assert system.memory.verified_hit(scoped, scope="rounding") is not None
    assert system.memory.verified_hit(bare, scope="none") is None
    assert system.memory.verified_hit(bare, scope="base") is None
    system.memory.clear()


def test_answer_claim_rejects_false_and_defers_empirical(setup):
    _, _, _, system = setup
    system.memory.clear()
    rng = random.Random(0)
    false_claims = []
    for fam in ("base", "mod", "count"):
        for _ in range(10):
            c = make_claim(rng, family=fam, marker_present=True, force_true=False)
            if adjudicate(c.spec, True).correct is False:
                false_claims.append(c)
    assert false_claims
    assert all(system.answer_claim(c).accept is False for c in false_claims)

    from cgr.data.facts import make_empirical_claims
    emp = make_empirical_claims(5, seed=1)
    assert all(system.answer_claim(c).defer for c in emp)


# --------------------------------------------------------------------------
# curriculum contract
# --------------------------------------------------------------------------

def test_oracle_action_covers_delib_and_def():
    """The curriculum must offer every action it expects the gate to learn.

    Exp 5 originally built its eval set from marker-present claims only, so
    DELIB was never the oracle action and the run reported DELIB as a dead
    action. That was a curriculum gap, not a finding. These assertions pin the
    contract so the gap cannot silently return.
    """
    from cgr.model.gate import A
    from cgr.scaffold import oracle_action

    bare = make_claim(random.Random(0), family="base", marker_present=False)
    assert oracle_action(bare, None) == A["DELIB"]
    scoped = make_claim(random.Random(0), family="base", marker_present=True)
    assert oracle_action(scoped, None) == A["VER"]
    emp = make_empirical_claims(1, seed=0)[0]
    assert oracle_action(emp, None) == A["DEF"]

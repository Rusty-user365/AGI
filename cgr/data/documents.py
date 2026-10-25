"""Authoritative-thesis document tasks (the rebuild of the repair substrate).

The previous generator had a loophole that made the non-local experiment
undecidable: it corrupted the intro and the conclusion to two different
theses, and then called "consistency" the target. But a document whose intro
says B and whose bodies say A is only inconsistent *relative to an intended
thesis*. Nothing in the old task said the intro was authoritative, so
"revert the intro to A" was a perfectly valid one-block repair. The task could
not distinguish the intended solution from that alternative.

This module fixes that by making the contract part of the task:

    The introduction is the authoritative, immutable thesis. All editable
    blocks must be revised to support it.

Consequences, all of which are load-bearing:

1. `Task.authoritative` is given, not inferred from whichever solution happens
   to make the document coherent.
2. The intro is `locked`. A repair that edits it violates the contract, and
   `anchor_ok` reports that. The intro's quality is therefore *preserved*, not
   improved; it is not scored as an outcome of local repair.
3. The two defect types are now genuinely different in kind:
     - `local`       one body block loses specificity/completeness but still
                     supports the authoritative thesis. Coherence is intact;
                     the defect is quality. One block needs repair.
     - `cross_block` several editable blocks support a single *wrong* thesis
                     while the locked intro states the authoritative one. The
                     defect is coherence. One block cannot fix it.
4. Every block carries metadata (thesis, role, quality target, locked) and
   every corruption is validated against the document it actually produced.
   There is no `requires_neighbor` flag to trust: `validate_*` re-parses the
   text and checks the contract against what it says.

The claim this supports is conditional and measurable: revision scope should
be selected according to the location and dependency structure of a defect,
*under a budget*. It is not "local repair can never fix a non-local defect" --
`repeated_local` exists precisely to test that, at a matched budget.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .grammar import (BODY_TEMPLATES, CONCLUSION_TEMPLATES, THESES, THESIS_POOL,
                      Block, concl_parts, is_consistent, parse_block,
                      parse_block_tmpl, render_body, render_concl)

ROLES = ("intro", "body", "conclusion")
EDITABLE_ROLES = ("body", "conclusion")


# --------------------------------------------------------------------------
# Blocks and tasks
# --------------------------------------------------------------------------

def make_block(text: str, role: str, *, locked: bool = False,
               quality_target: str = "strong") -> Block:
    """Build a block and derive its metadata *from the text*.

    The metadata is a claim about the text; the validators below check that
    claim against a fresh parse rather than trusting it.
    """
    tid, reason, ridx = parse_block(text, role)
    return Block(text=text, role=role, thesis_id=tid, reason=reason,
                 reason_index=ridx, locked=locked,
                 quality_target=quality_target,
                 tmpl=parse_block_tmpl(text, role))


@dataclass
class Task:
    """One repair task with an explicit, immutable contract."""
    task_id: int
    blocks: list[Block]              # the observation (possibly corrupted)
    authoritative: int               # T*, given by the contract
    defect: str                      # "none" | "local" | "cross_block"
    clean_blocks: list[str]          # ground truth, for the oracle editor
    edited: list[int] = field(default_factory=list)
    operator: str = ""
    meta: dict = field(default_factory=dict)

    # --- compatibility surface used by scaffold / train / exp3 / exp5 -----
    @property
    def corrupt_blocks(self) -> list[str]:
        return [b.text for b in self.blocks]

    @property
    def target_block(self) -> int:
        return self.edited[0] if self.edited else 0

    @property
    def defect_type(self) -> str:
        return self.defect

    @property
    def corrupted_block_ids(self) -> list[int]:
        return list(self.edited)

    @property
    def locked_indices(self) -> list[int]:
        return [i for i, b in enumerate(self.blocks) if b.locked]

    @property
    def editable_indices(self) -> list[int]:
        return [i for i, b in enumerate(self.blocks)
                if b.role in EDITABLE_ROLES and not b.locked]

    def texts(self) -> list[str]:
        return [b.text for b in self.blocks]

    @property
    def contract(self) -> str:
        """The task specification, carried by the task rather than inferred."""
        return ("the introduction is the authoritative, immutable thesis; "
                "all editable blocks must be revised to support it")

    @property
    def authoritative_thesis(self) -> str:
        return THESES[self.authoritative][0]


def task_from_blocks(blocks: list[str], authoritative: int, task_id: int = 0,
                     defect: str = "none", clean_blocks: list[str] | None = None,
                     edited: list[int] | None = None,
                     operator: str = "") -> Task:
    """Roles are positional: block 0 is the intro, the last is the conclusion."""
    n = len(blocks)
    bs = [make_block(t, "intro" if i == 0 else
                     "conclusion" if i == n - 1 else "body",
                     locked=(i == 0))
          for i, t in enumerate(blocks)]
    return Task(task_id=task_id, blocks=bs, authoritative=authoritative,
                defect=defect, clean_blocks=list(clean_blocks or blocks),
                edited=list(edited or []), operator=operator)


def task_from_record(record, task_id: int = 0) -> Task:
    return task_from_blocks(record.blocks, record.thesis_id, task_id,
                            defect="none")


# --------------------------------------------------------------------------
# Coherence and quality
# --------------------------------------------------------------------------

def coherence(blocks: list[Block], authoritative: int) -> dict:
    """C(D) = fraction of required blocks that support the authoritative thesis.

    This is a *synthetic label-consistency* score, not a general semantic
    coherence measure. It is exact because every block's thesis is parseable.

    `c_all` includes the locked intro, so it has a floor of 1/|required|: a
    system that does nothing on a cross-block task scores 1/n, not 0. That
    floor is reported rather than hidden. `c_edit` restricts to editable
    blocks and is the metric that can actually move.
    """
    required = [i for i, b in enumerate(blocks) if b.role in ROLES]
    # Same definition as `Task.editable_indices`, so `n_editable` here always
    # agrees with the number of blocks a budget was computed for.
    editable = [i for i, b in enumerate(blocks)
                if b.role in EDITABLE_ROLES and not b.locked]
    ok_all = [blocks[i].thesis_id == authoritative for i in required]
    ok_edit = [blocks[i].thesis_id == authoritative for i in editable]
    return {
        "c_all": sum(ok_all) / max(len(required), 1),
        "c_edit": sum(ok_edit) / max(len(editable), 1),
        "n_required": len(required),
        "n_editable": len(editable),
        "floor_all": 1.0 / max(len(required), 1),
        "off_thesis": [i for i, ok in zip(required, ok_all) if not ok],
        "off_thesis_editable": [i for i, ok in zip(editable, ok_edit) if not ok],
    }


def block_quality(b: Block, authoritative: int) -> int:
    """1 if the block supports T* *specifically*, 0 otherwise.

    For bodies, "specifically" means the reason text is drawn from T*'s own
    reason pool. A vague placeholder is on-thesis but not specific, which is
    exactly the local defect. Intro and conclusion have no reason slot, so
    being on-thesis is sufficient.
    """
    if b.thesis_id != authoritative:
        return 0
    if b.role != "body":
        return 1
    return int(b.reason in THESIS_POOL[authoritative])


def quality(blocks: list[Block], authoritative: int,
            indices: list[int] | None = None) -> float:
    idx = indices if indices is not None else [
        i for i, b in enumerate(blocks) if b.role == "body"]
    if not idx:
        return 1.0
    return sum(block_quality(blocks[i], authoritative) for i in idx) / len(idx)


def anchor_ok(before: list[str], after: list[str],
              locked: list[int] | None = None) -> bool:
    """The locked blocks must be byte-identical. Editing them is a violation."""
    locked = [0] if locked is None else locked
    if len(before) != len(after):
        return False
    return all(before[i] == after[i] for i in locked)


# --------------------------------------------------------------------------
# Corruption operators, each with a stated contract
# --------------------------------------------------------------------------

def _vague_reason(_reason: str) -> str:
    return "this reason is left vague"


def _no_example(_reason: str) -> str:
    return "the point is stated without a concrete example"


#: A local degradation preserves the supported thesis and the reason slot, and
#: removes specificity. Varied across examples so the defect is not one string.
LOCAL_OPERATORS: dict[str, Callable[[str], str]] = {
    "vague": _vague_reason,
    "no_example": _no_example,
}


def degrade_local(task: Task, rng: random.Random,
                  operator: str | None = None) -> Task:
    """Contract: exactly one body block loses specificity; proposition kept.

    The block still reads `reason i : <something> , so T* .`, so the document
    remains *coherent* (C(D) = 1). What it loses is local quality. This is the
    point of the rebuild: the local condition is a quality defect, not a
    coherence defect, and the old generator conflated them by replacing the
    block with unparseable filler.
    """
    bodies = [i for i, b in enumerate(task.blocks) if b.role == "body"]
    if not bodies:
        raise ValueError("task has no body block to degrade")
    idx = rng.choice(bodies)
    op = operator or rng.choice(sorted(LOCAL_OPERATORS))
    b = task.blocks[idx]
    if b.reason is None or b.reason_index is None or b.thesis_id is None:
        raise ValueError(f"block {idx} is not parseable, cannot degrade it")
    new_reason = LOCAL_OPERATORS[op](b.reason)
    tmpl = b.tmpl if b.tmpl is not None else 0
    text = render_body(tmpl, b.reason_index, new_reason,
                       THESES[b.thesis_id][0])
    blocks = list(task.blocks)
    blocks[idx] = make_block(text, "body", quality_target="vague")
    out = Task(task_id=task.task_id, blocks=blocks,
               authoritative=task.authoritative, defect="local",
               clean_blocks=task.clean_blocks, edited=[idx], operator=op,
               meta=dict(task.meta))
    errs = validate_local(task, out)
    if errs:
        raise AssertionError(f"local corruption violated its contract: {errs}")
    return out


def _concl_parts(text: str) -> tuple[int, int, str]:
    """Return (template_index, n_declared, thesis) for a conclusion block."""
    p = concl_parts(text)
    if p is None:
        raise ValueError(f"unparseable conclusion: {text!r}")
    return p


def retarget(text: str, role: str, new_thesis: int) -> str:
    """Rebuild a block so it supports `new_thesis`, keeping its content.

    Blocks are rebuilt from parsed parts rather than by string replacement.
    `text.replace(old_thesis, new_thesis)` is wrong whenever the thesis phrase
    also occurs inside the reason, and it silently produces a block that no
    longer parses.

    The surface template is preserved. If the corruption also changed the
    sentence shape, a repair model could detect the defect from the style alone
    and the experiment would measure template matching rather than dependency
    reasoning.
    """
    t = THESES[new_thesis][0]
    if role == "intro":
        raise ValueError("the introduction is locked and must never be retargeted")
    if role == "body":
        tmpl, reason, ridx, _ = body_parts_or_raise(text)
        return render_body(tmpl, ridx, reason, t)
    tmpl, n, _ = _concl_parts(text)
    return render_concl(tmpl, t, n)


def body_parts_or_raise(text: str) -> tuple[int, str, int, str]:
    from .grammar import body_parts
    p = body_parts(text)
    if p is None:
        raise ValueError(f"unparseable body: {text!r}")
    return p


def corrupt_cross_block(task: Task, rng: random.Random,
                        wrong_thesis: int | None = None,
                        k: int | None = None,
                        partial: bool = False) -> Task:
    """Contract: the locked intro is untouched; editable blocks support W != T*.

    Default `k` is *all* editable blocks, so the document is maximally
    inconsistent and a one-block repair leaves |editable|-1 contradictions.
    A single wrong thesis W is used throughout, so the defect is "the document
    argues for the wrong thesis" and not "the document is a random mess".

    `partial=True` produces the harder distractor. A *strict majority* of the
    editable blocks (always including the conclusion) support W, while the
    remaining editable blocks still support T*. The majority constraint is the
    point: with the wrong thesis on more editable blocks than the right one, a
    "keep the majority thesis" heuristic sides with the corruption, and only
    the locked intro identifies T*. That makes the intro anchor load-bearing
    rather than decorative, and it is checked by the validator rather than
    left to chance.

    `partial` still edits at least two blocks, so a single local repair cannot
    restore coherence. A one-block mismatch would be a local defect wearing a
    cross-block label, which is exactly the conflation this module removed.
    """
    editable = task.editable_indices
    if len(editable) < 2:
        raise ValueError("cross-block defect needs at least two editable blocks")
    T = task.authoritative
    if wrong_thesis is None:
        wrong_thesis = rng.choice([i for i in range(len(THESES)) if i != T])
    if wrong_thesis == T:
        raise ValueError("wrong thesis must differ from the authoritative one")
    if partial:
        if len(editable) < 3:
            raise ValueError(
                "partial cross-block needs at least three editable blocks: "
                "two wrong (a strict majority) and at least one right")
        lo = len(editable) // 2 + 1     # strict majority
        hi = len(editable) - 1          # at least one block stays correct
        k = rng.randint(lo, hi) if k is None else min(max(k, lo), hi)
    else:
        k = len(editable) if k is None else min(k, len(editable))
    if k < 2:
        raise ValueError(
            "cross_block needs at least two edited blocks; a single-block "
            "mismatch is a local defect, not a cross-block one")
    chosen = set(editable)
    # always include the conclusion when k allows, so the document's own
    # summary is wrong rather than only some bodies
    concl = len(task.blocks) - 1
    if k < len(editable):
        chosen = {concl} if concl in editable else set()
        bodies = [i for i in editable if i != concl]
        rng.shuffle(bodies)
        chosen |= set(bodies[:max(0, k - len(chosen))])
    blocks = list(task.blocks)
    for i in sorted(chosen):
        blocks[i] = make_block(retarget(blocks[i].text, blocks[i].role,
                                        wrong_thesis),
                               blocks[i].role, quality_target="off_thesis")
    defect = "cross_block_partial" if partial else "cross_block"
    out = Task(task_id=task.task_id, blocks=blocks, authoritative=T,
               defect=defect, clean_blocks=task.clean_blocks,
               edited=sorted(chosen),
               operator=f"retarget->{wrong_thesis}", meta=dict(task.meta))
    errs = (validate_cross_block_partial(task, out) if partial
            else validate_cross_block(task, out))
    if errs:
        raise AssertionError(
            f"{defect} corruption violated its contract: {errs}")
    return out


# --------------------------------------------------------------------------
# Contract validators: re-parse the document, never trust the flags
# --------------------------------------------------------------------------

def _metadata_truthful(task: Task) -> list[str]:
    errs = []
    for i, b in enumerate(task.blocks):
        tid, reason, ridx = parse_block(b.text, b.role)
        if tid != b.thesis_id:
            errs.append(f"block {i}: metadata thesis {b.thesis_id} != parsed {tid}")
        if reason != b.reason:
            errs.append(f"block {i}: metadata reason != parsed reason")
        if ridx != b.reason_index:
            errs.append(f"block {i}: metadata reason index != parsed index")
    return errs


def validate_local(clean: Task, corrupt: Task) -> list[str]:
    errs = _metadata_truthful(corrupt)
    if len(clean.blocks) != len(corrupt.blocks):
        return errs + ["block count changed"]
    if corrupt.defect != "local":
        errs.append(f"defect label is {corrupt.defect!r}, expected 'local'")
    if len(corrupt.edited) != 1:
        errs.append(f"expected exactly one edited block, got {corrupt.edited}")
    if not anchor_ok(clean.texts(), corrupt.texts(), clean.locked_indices):
        errs.append("locked intro changed")
    if is_consistent(clean.texts()) and not is_consistent(corrupt.texts()):
        errs.append("local degradation broke structural consistency")
    coh = coherence(corrupt.blocks, corrupt.authoritative)
    if coh["c_edit"] != 1.0:
        errs.append(f"local degradation changed the thesis (c_edit={coh['c_edit']})")
    idx = corrupt.target_block
    if corrupt.blocks[idx].role != "body":
        errs.append("local degradation did not target a body block")
    if block_quality(corrupt.blocks[idx], corrupt.authoritative) != 0:
        errs.append("degraded block still scores as specific")
    if clean.blocks[idx].text == corrupt.blocks[idx].text:
        errs.append("degradation did not change the block")
    others = [i for i, b in enumerate(corrupt.blocks) if b.role == "body" and i != idx]
    if quality(corrupt.blocks, corrupt.authoritative, others) != 1.0:
        errs.append("degradation damaged a block it was not supposed to touch")
    return errs


def validate_cross_block(clean: Task, corrupt: Task) -> list[str]:
    errs = _metadata_truthful(corrupt)
    if len(clean.blocks) != len(corrupt.blocks):
        return errs + ["block count changed"]
    if corrupt.defect != "cross_block":
        errs.append(f"defect label is {corrupt.defect!r}, expected 'cross_block'")
    if not anchor_ok(clean.texts(), corrupt.texts(), clean.locked_indices):
        errs.append("locked intro changed: the contract was violated")
    intro = corrupt.blocks[0]
    if intro.thesis_id != corrupt.authoritative:
        errs.append("intro no longer states the authoritative thesis")
    if len(corrupt.edited) < 2:
        errs.append(f"cross-block defect edited only {len(corrupt.edited)} block(s)")
    editable = corrupt.editable_indices
    if any(corrupt.blocks[i].thesis_id == corrupt.authoritative for i in editable):
        errs.append("at least one editable block still supports T*")
    wrong = {corrupt.blocks[i].thesis_id for i in editable}
    if len(wrong) != 1:
        errs.append(f"editable blocks support {len(wrong)} different theses, expected 1")
    if corrupt.authoritative in wrong:
        errs.append("wrong thesis equals the authoritative thesis")
    # the *content* (reason text) must survive: only the proposition changes
    for i in corrupt.edited:
        if corrupt.blocks[i].role == "body":
            if corrupt.blocks[i].reason != clean.blocks[i].reason:
                errs.append(f"block {i}: reason changed, only the thesis should")
    coh = coherence(corrupt.blocks, corrupt.authoritative)
    if coh["c_edit"] != 0.0:
        errs.append(f"cross-block c_edit is {coh['c_edit']}, expected 0")
    return errs


def validate_cross_block_partial(clean: Task, corrupt: Task) -> list[str]:
    """The harder cross-block defect: W is the *majority*, T* the intro.

    Separate from `validate_cross_block` rather than a flag on it, because the
    two contracts differ in a way that matters: the full defect asserts that
    *every* editable block is wrong, while this one asserts that some are still
    right. Merging them would let a full defect validate as partial (and vice
    versa), and the whole point of the extra family is that the two are not
    interchangeable.
    """
    errs = _metadata_truthful(corrupt)
    if len(clean.blocks) != len(corrupt.blocks):
        return errs + ["block count changed"]
    if corrupt.defect != "cross_block_partial":
        errs.append(f"defect label is {corrupt.defect!r}, "
                    f"expected 'cross_block_partial'")
    if not anchor_ok(clean.texts(), corrupt.texts(), clean.locked_indices):
        errs.append("locked intro changed: the contract was violated")
    intro = corrupt.blocks[0]
    if intro.thesis_id != corrupt.authoritative:
        errs.append("intro no longer states the authoritative thesis")

    editable = corrupt.editable_indices
    edited = set(corrupt.edited)
    if not edited:
        return errs + ["partial cross-block edited no blocks"]
    if not edited <= set(editable):
        errs.append(f"edited blocks {sorted(edited - set(editable))} are not "
                    f"editable")
    if len(edited) < 2:
        errs.append(f"partial cross-block edited only {len(edited)} block(s); "
                    f"a single block is a local defect")
    if len(edited) >= len(editable):
        errs.append("partial cross-block edited every editable block; that is "
                    "the full cross-block defect")
    kept = [i for i in editable if i not in edited]
    if not kept:
        errs.append("partial cross-block left no editable block supporting T*")
    if len(edited) <= len(kept):
        errs.append(
            f"wrong thesis is on {len(edited)} of {len(editable)} editable "
            f"blocks, not a strict majority; the hard distractor requires that "
            f"a majority-vote heuristic sides with the corruption")

    concl = len(corrupt.blocks) - 1
    if concl not in edited:
        errs.append("partial cross-block did not edit the conclusion; the "
                    "document's own summary would still support T*")

    wrong = {corrupt.blocks[i].thesis_id for i in edited}
    if len(wrong) != 1:
        errs.append(f"edited blocks support {len(wrong)} different theses, "
                    f"expected 1")
    if corrupt.authoritative in wrong:
        errs.append("wrong thesis equals the authoritative thesis")

    for i in kept:
        if corrupt.blocks[i].thesis_id != corrupt.authoritative:
            errs.append(f"unedited editable block {i} does not support T*")
        if corrupt.blocks[i].text != clean.blocks[i].text:
            errs.append(f"unedited editable block {i} changed")
    # the *content* (reason text) must survive: only the proposition changes
    for i in corrupt.edited:
        if corrupt.blocks[i].role == "body":
            if corrupt.blocks[i].reason != clean.blocks[i].reason:
                errs.append(f"block {i}: reason changed, only the thesis should")

    coh = coherence(corrupt.blocks, corrupt.authoritative)
    expected = len(kept) / len(editable) if editable else 0.0
    if abs(coh["c_edit"] - expected) > 1e-9:
        errs.append(f"partial c_edit is {coh['c_edit']}, expected {expected}")
    if not (0.0 < coh["c_edit"] < 1.0):
        errs.append(f"partial c_edit is {coh['c_edit']}, expected strictly "
                    f"between 0 and 1")
    return errs


def validate_task(task: Task) -> list[str]:
    errs = _metadata_truthful(task)
    if task.defect == "none":
        if not is_consistent(task.texts()):
            errs.append("clean task is not structurally consistent")
        if coherence(task.blocks, task.authoritative)["c_edit"] != 1.0:
            errs.append("clean task is not fully coherent")
    return errs


# --------------------------------------------------------------------------
# Budgets
# --------------------------------------------------------------------------

@dataclass
class Budget:
    """A revision budget in blocks and tokens.

    Both are tracked because they are not interchangeable: a global revision
    rewrites many blocks in one call, repeated local repair rewrites the same
    number of tokens over several calls. "Matched budget" means matched in
    both, and the realized spend is always reported.
    """
    max_blocks: int
    max_tokens: int
    blocks_used: int = 0
    tokens_used: int = 0
    #: Set when a method produced more text than its allowance. The edit is
    #: kept and scored (it cannot be un-made), but the row is reported as a
    #: budget violation rather than silently accepted or crashing the run.
    over_budget: bool = False

    def can_spend(self, n_blocks: int, n_tokens: int) -> bool:
        return (self.blocks_used + n_blocks <= self.max_blocks
                and self.tokens_used + n_tokens <= self.max_tokens)

    def spend(self, n_blocks: int, n_tokens: int) -> None:
        if not self.can_spend(n_blocks, n_tokens):
            raise RuntimeError(
                f"budget exceeded: {self.blocks_used}+{n_blocks} blocks, "
                f"{self.tokens_used}+{n_tokens} tokens")
        self.blocks_used += n_blocks
        self.tokens_used += n_tokens

    def spend_or_flag(self, n_blocks: int, n_tokens: int) -> bool:
        """Spend if it fits; otherwise record a violation and spend anyway.

        Returns True if the spend fit the allowance. A generated block can be
        longer than the corrupted one, and refusing to spend after the text
        has already been produced would either crash a long run or hide the
        violation. Reporting it is the honest third option.
        """
        fits = self.can_spend(n_blocks, n_tokens)
        if not fits:
            self.over_budget = True
        self.blocks_used += n_blocks
        self.tokens_used += n_tokens
        return fits

    def as_dict(self) -> dict:
        return {"max_blocks": self.max_blocks, "max_tokens": self.max_tokens,
                "blocks_used": self.blocks_used,
                "tokens_used": self.tokens_used,
                "over_budget": self.over_budget}

    def exhausted(self) -> bool:
        return (self.blocks_used >= self.max_blocks
                or self.tokens_used >= self.max_tokens)


def matched_budgets(task: Task, n_tokens: Callable[[str], int]) -> dict:
    """Budgets for the three revision methods on one task.

    `global` and `repeated_local` get the *same* allowance (all editable
    blocks, all their tokens). `one_block` gets one block. This is what makes
    the comparison about scope rather than about permission to edit more text.

    Per-block cost is the larger of the clean and the corrupted text, because
    a corruption can change a block's length and a method must not be refused
    a budget that its own observation requires.
    """
    editable = task.editable_indices

    def cost(i: int) -> int:
        return max(n_tokens(task.clean_blocks[i]), n_tokens(task.blocks[i].text))

    g_tokens = sum(cost(i) for i in editable)
    tgt = task.target_block
    return {
        "one_block": Budget(1, cost(tgt)),
        "global": Budget(len(editable), g_tokens),
        "repeated_local": Budget(len(editable), g_tokens),
    }


# --------------------------------------------------------------------------
# Oracle editor: proves the task is solvable before any model is blamed
# --------------------------------------------------------------------------

def oracle_repair(task: Task, method: str, budget: Budget,
                  n_tokens: Callable[[str], int]) -> tuple[list[str], dict]:
    """Deterministic editor with access to ground truth.

    This is the control from step 3 of the protocol. If the oracle cannot
    reach the target under the budget, the *task or the budget* is wrong and
    no model result on it means anything. If it can, a model failure is a
    model failure.
    """
    out = task.texts()
    changed: list[int] = []
    if method == "one_block":
        i = task.target_block
        cost = n_tokens(task.clean_blocks[i])
        if budget.can_spend(1, cost):
            out[i] = task.clean_blocks[i]
            budget.spend(1, cost)
            changed.append(i)
    elif method == "global":
        for i in task.editable_indices:
            cost = n_tokens(task.clean_blocks[i])
            if not budget.can_spend(1, cost):
                break
            out[i] = task.clean_blocks[i]
            budget.spend(1, cost)
            changed.append(i)
    elif method == "repeated_local":
        # Worst block first, one call per block, never twice. The detector is
        # ground truth, which biases *in favour* of repeated local repair:
        # if it still loses to global, the loss is not a detection problem.
        def badness(i: int) -> tuple[int, int]:
            b = task.blocks[i]
            return (b.thesis_id == task.authoritative, block_quality(
                b, task.authoritative))
        order = sorted(task.editable_indices, key=badness)
        for i in order:
            cost = n_tokens(task.clean_blocks[i])
            if not budget.can_spend(1, cost):
                break
            out[i] = task.clean_blocks[i]
            budget.spend(1, cost)
            changed.append(i)
            blocks = [make_block(t, "intro" if j == 0 else
                                 "conclusion" if j == len(out) - 1 else "body",
                                 locked=(j == 0))
                      for j, t in enumerate(out)]
            if coherence(blocks, task.authoritative)["c_edit"] == 1.0:
                break
    else:
        raise ValueError(method)
    return out, {"blocks_changed": len(changed), "changed": changed,
                 "budget": budget.as_dict()}


# --------------------------------------------------------------------------
# Task sets
# --------------------------------------------------------------------------

def make_task_set(records: list, n: int, seed: int = 0,
                  fits: Callable[[list[str]], bool] | None = None,
                  stats: dict | None = None) -> dict[str, list[Task]]:
    """Clean tasks plus one local and one cross-block corruption of each.

    `fits` is an optional predicate on a document (e.g. "fits the model's
    context"). It is applied to the clean, local and cross-block variants
    *together*: a record is used only if all three fit, so the two defect
    conditions are drawn from the same documents and the comparison stays
    paired. `stats` is filled with how many documents were scanned and
    skipped, because that is a selection effect and must be reported.
    """
    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)
    out: dict[str, list[Task]] = {"none": [], "local": [], "cross_block": [],
                                  "cross_block_partial": []}
    skipped = 0
    scanned = 0
    for k, rec in enumerate(pool):
        if all(len(v) >= n for v in out.values()):
            break
        scanned = k + 1
        clean = task_from_record(rec, task_id=k)
        loc = degrade_local(clean, rng)
        cross = corrupt_cross_block(clean, rng)
        partial = corrupt_cross_block(clean, rng, partial=True)
        if fits is not None and not all(fits(t) for t in
                                        (clean.texts(), loc.texts(),
                                         cross.texts(), partial.texts())):
            skipped += 1
            continue
        if len(out["none"]) < n:
            out["none"].append(clean)
        if len(out["local"]) < n:
            out["local"].append(loc)
        if len(out["cross_block"]) < n:
            out["cross_block"].append(cross)
        if len(out["cross_block_partial"]) < n:
            out["cross_block_partial"].append(partial)
    if stats is not None:
        stats["scanned"] = scanned
        stats["skipped"] = skipped
    return out


__all__ = [
    "Block", "Task", "Budget", "ROLES", "EDITABLE_ROLES", "LOCAL_OPERATORS",
    "make_block", "task_from_blocks", "task_from_record", "coherence",
    "block_quality", "quality", "anchor_ok", "degrade_local", "retarget",
    "corrupt_cross_block", "validate_local", "validate_cross_block",
    "validate_cross_block_partial", "validate_task", "matched_budgets",
    "oracle_repair", "make_task_set",
]

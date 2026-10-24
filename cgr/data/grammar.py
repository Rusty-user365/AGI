"""Template grammar for synthetic essays.

Design constraints that matter for the paper:

1. Every document must be *exactly parseable*, so consistency is a
   deterministic function and not a human judgement.
2. Surface form must vary enough that memorisation is visible. Documents
   are keyed by a (thesis, reason-count, template-combination) tuple so we
   can hold out whole combinations and measure the train/test gap.
3. The corruption operators must produce defects whose repair is
   *verifiable by code*.

The format is rigid on purpose:

    this essay argues that <T> with <N> reasons .
    reason <i> : <R> , so <T> .
    ...
    these <N> reasons establish that <T> .
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Thesis inventory. Each thesis owns a reason pool, so a document is
# internally coherent rather than a random bag of sentences.
# --------------------------------------------------------------------------

THESES: list[tuple[str, list[str]]] = [
    ("remote work raises output", [
        "commuting hours become working hours",
        "quiet environments reduce interruption",
        "written records replace meetings",
        "hiring widens beyond one city",
        "handovers become explicit",
    ]),
    ("street trees cool cities", [
        "canopy blocks direct sunlight",
        "evaporation lowers surface heat",
        "shade reduces road temperature",
        "leaves filter particulate matter",
        "green space lowers measured stress",
    ]),
    ("slow feedback harms learning", [
        "errors are corrected late",
        "practice repeats the mistake",
        "motivation falls without signal",
        "diagnosis arrives after the exam",
        "revision windows close early",
    ]),
    ("open data improves policy", [
        "outside checks find errors",
        "models can be reproduced",
        "assumptions become visible",
        "comparisons across regions become possible",
        "audit trails survive staff turnover",
    ]),
    ("vertical farms waste energy", [
        "lighting replaces free sunlight",
        "cooling loads run all year",
        "crop choice narrows to greens",
        "capital cost per hectare is high",
        "backup power is mandatory",
    ]),
    ("peer review slows discovery", [
        "queues add months per round",
        "reviewers are unpaid and scarce",
        "novel claims attract suspicion",
        "rebuttals reward detail over insight",
        "resubmission restarts the clock",
    ]),
]

#: thesis_id -> that thesis's own reason pool. Used to decide whether a body
#: block is *specific* about its thesis or merely on-thesis but vague.
THESIS_POOL: dict[int, list[str]] = {i: pool for i, (_, pool) in enumerate(THESES)}
#: thesis text -> thesis_id. Parsing maps text back onto an integer label so
#: every coherence check is exact rather than a string comparison.
THESIS_INDEX: dict[str, int] = {t: i for i, (t, _) in enumerate(THESES)}

INTRO_TEMPLATES = [
    "this essay argues that {T} with {N} reasons .",
    "we argue that {T} , and offer {N} reasons .",
    "the claim is that {T} ; {N} reasons follow .",
]

CONCLUSION_TEMPLATES = [
    "these {N} reasons establish that {T} .",
    "we have shown that {T} on the strength of {N} reasons .",
    "the {N} reasons given support {T} .",
]

FILLER = "this reason is left vague ."

# Held-out template combinations. Documents whose (intro, conclusion)
# combination is in this set never appear in training.
HOLDOUT_COMBOS: set[tuple[int, int]] = {(2, 0), (0, 2), (1, 1)}


@dataclass
class Doc:
    thesis_id: int
    thesis: str
    n_reasons: int
    reasons: list[str]
    intro_tmpl: int
    concl_tmpl: int
    blocks: list[str] = field(default_factory=list)

    @property
    def combo(self) -> tuple[int, int]:
        return (self.intro_tmpl, self.concl_tmpl)

    def render(self) -> str:
        return " ".join(self.blocks)

    def spoof(self) -> str:
        """Surface form used for split keys only."""
        return f"{self.thesis_id}|{self.n_reasons}|{self.intro_tmpl}|{self.concl_tmpl}"


def make_doc(rng: random.Random,
             thesis_id: int | None = None,
             n_reasons: int | None = None,
             combo: tuple[int, int] | None = None) -> Doc:
    if thesis_id is None:
        thesis_id = rng.randrange(len(THESES))
    thesis, pool = THESES[thesis_id]
    if n_reasons is None:
        n_reasons = rng.randint(2, 4)
    if combo is None:
        combo = (rng.randrange(len(INTRO_TEMPLATES)),
                 rng.randrange(len(CONCLUSION_TEMPLATES)))
    reasons = rng.sample(pool, n_reasons)

    intro = INTRO_TEMPLATES[combo[0]].format(T=thesis, N=n_reasons)
    bodies = [f"reason {i + 1} : {reasons[i]} , so {thesis} ."
              for i in range(n_reasons)]
    concl = CONCLUSION_TEMPLATES[combo[1]].format(T=thesis, N=n_reasons)
    return Doc(thesis_id, thesis, n_reasons, reasons, combo[0], combo[1],
               [intro] + bodies + [concl])


# --------------------------------------------------------------------------
# Exact parsers. Consistency is defined by these, so they must be strict.
# --------------------------------------------------------------------------

_INTRO_RE = re.compile(
    r"(?:this essay argues that|we argue that|the claim is that) (.+?)(?: with| , and offer| ;) (\d+) reasons")
_CONCL_RES = [
    re.compile(r"these (\d+) reasons establish that (.+?) \."),
    re.compile(r"we have shown that (.+?) on the strength of (\d+) reasons \."),
    re.compile(r"the (\d+) reasons given support (.+?) \."),
]
_BODY_RE = re.compile(r"reason (\d+) : (.+?) , so (.+?) \.")


@dataclass
class Block:
    """A document block plus the labels that make a defect checkable.

    The metadata is a *claim about the text*. `documents._metadata_truthful`
    re-parses the text and rejects any task whose metadata disagrees with what
    the text actually says, so the labels cannot drift away from the data.

    role          "intro" | "body" | "conclusion"
    thesis_id     the thesis this block actually supports (None if unparseable)
    reason        body only: the reason phrase
    reason_index  body only: the declared reason number
    locked        True for the authoritative intro: editing it breaks the task
    quality_target "strong" (specific) | "vague" | "off_thesis"
    """

    text: str
    role: str
    thesis_id: int | None = None
    reason: str | None = None
    reason_index: int | None = None
    locked: bool = False
    quality_target: str = "strong"


def parse_block(text: str, role: str) -> tuple[int | None, str | None,
                                              int | None]:
    """Return (thesis_id, reason, reason_index) for one block.

    Unparseable input returns `(None, ...)` rather than raising, because the
    repair experiments must be able to score a *model-produced* document that
    may be garbage. Callers that require a valid block check for None.
    """
    if role == "intro":
        m = _INTRO_RE.search(text)
        if not m:
            return None, None, None
        return THESIS_INDEX.get(m.group(1)), None, None
    if role == "body":
        m = _BODY_RE.search(text)
        if not m:
            return None, None, None
        return THESIS_INDEX.get(m.group(3)), m.group(2), int(m.group(1))
    if role == "conclusion":
        for i, rx in enumerate(_CONCL_RES):
            m = rx.search(text)
            if not m:
                continue
            thesis = m.group(1) if i == 1 else m.group(2)
            return THESIS_INDEX.get(thesis), None, None
        return None, None, None
    raise ValueError(f"unknown role {role!r}")


@dataclass
class Parse:
    ok: bool
    thesis: str | None = None
    n_intro: int | None = None
    body_theses: list[str] = field(default_factory=list)
    body_indices: list[int] = field(default_factory=list)
    n_declared_concl: int | None = None
    concl_thesis: str | None = None
    reason_texts: list[str] = field(default_factory=list)


def parse_blocks(blocks: list[str]) -> Parse:
    if len(blocks) < 3:
        return Parse(ok=False)
    m = _INTRO_RE.search(blocks[0])
    if not m:
        return Parse(ok=False)
    thesis, n_intro = m.group(1), int(m.group(2))

    body_theses, body_idx, reasons = [], [], []
    for b in blocks[1:-1]:
        bm = _BODY_RE.search(b)
        if bm:
            body_idx.append(int(bm.group(1)))
            reasons.append(bm.group(2))
            body_theses.append(bm.group(3))
        else:
            # filler or unparseable body -> counts as an invalid reason slot
            body_idx.append(-1)
            reasons.append(None)

    n_concl, concl_thesis = None, None
    for i, rx in enumerate(_CONCL_RES):
        cm = rx.search(blocks[-1])
        if not cm:
            continue
        if i == 1:  # "we have shown that T on the strength of N reasons"
            concl_thesis, n_concl = cm.group(1), int(cm.group(2))
        else:
            n_concl, concl_thesis = int(cm.group(1)), cm.group(2)
        break

    return Parse(True, thesis, n_intro, body_theses, body_idx,
                 n_concl, concl_thesis, reasons)


def is_consistent(blocks: list[str]) -> bool:
    """Document-level consistency: counts, indices, and theses must all agree."""
    p = parse_blocks(blocks)
    if not p.ok:
        return False
    n_body = len(p.body_theses)
    if p.n_intro != n_body:
        return False
    if p.n_declared_concl != n_body:
        return False
    if p.body_indices != list(range(1, n_body + 1)):
        return False
    if any(r is None for r in p.reason_texts):
        return False
    theses = set(p.body_theses) | {p.thesis, p.concl_thesis}
    if len(theses) != 1:
        return False
    if any(t is None or t == "" for t in theses):
        return False
    return True


def blocked_spans(blocks: list[str], tok) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenise a document and return (ids, per-block (start, end) spans).

    Layout: [BOS] b0 [SEP] b1 [SEP] ... [SEP] bk [EOS]
    """
    ids = [tok.bos_id]
    spans = []
    for i, b in enumerate(blocks):
        if i > 0:
            ids.append(tok.sep_id)
        start = len(ids)
        ids.extend(tok.encode(b, add_bos=False, add_eos=False))
        spans.append((start, len(ids)))
    ids.append(tok.eos_id)
    return ids, spans

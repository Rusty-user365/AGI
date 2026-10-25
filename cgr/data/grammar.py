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

#: Surface templates. Varied clause order and lexical framing reduce a model's
#: ability to use one fixed sentence shape as a shortcut. Each is still exactly
#: parseable so the verifier stays deterministic. This is controlled template
#: variation, not open-ended paraphrasing or a claim of natural-language scope.
INTRO_TEMPLATES = [
    "this essay argues that {T} with {N} reasons .",
    "we argue that {T} , and offer {N} reasons .",
    "the claim is that {T} ; {N} reasons follow .",
    "this paper will show that {T} in {N} reasons .",
    "{T} is the central claim here , supported by {N} reasons .",
    "we take the position that {T} ; {N} points support it .",
    "the case for {T} rests on {N} reasons .",
    "here we defend {T} through {N} reasons .",
]

CONCLUSION_TEMPLATES = [
    "these {N} reasons establish that {T} .",
    "we have shown that {T} on the strength of {N} reasons .",
    "the {N} reasons given support {T} .",
    "so {T} , as the {N} reasons above show .",
    "the argument for {T} rests on {N} reasons .",
    "taken together , {N} points support {T} .",
    "on these {N} grounds , we conclude {T} .",
    "thus , the evidence favors {T} ; {N} reasons support this .",
]

BODY_TEMPLATES = [
    "reason {i} : {R} , so {T} .",
    "reason {i} : because {R} , {T} .",
    "reason {i} : {T} , since {R} .",
    "point {i} : given {R} , {T} follows .",
    "point {i} : evidence from {R} supports {T} .",
    "point {i} : {T} follows from {R} .",
    "point {i} : the observation that {R} backs {T} .",
    "point {i} : we see {T} in {R} .",
]

FILLER = "this reason is left vague ."

#: Reasons that are *on-thesis but not specific*. A body containing one of
#: these is a local quality defect, not a coherence defect.
VAGUE_REASONS = {
    "this reason is left vague",
    "the point is stated without a concrete example",
}

# Held-out (intro, conclusion) combinations. Documents whose combination is in
# this set never appear in training, so a model that memorised surface form is
# visible on the held-out split.
HOLDOUT_COMBOS: set[tuple[int, int]] = {(4, 0), (0, 4), (3, 1), (1, 3), (2, 2)}


@dataclass
class Doc:
    thesis_id: int
    thesis: str
    n_reasons: int
    reasons: list[str]
    intro_tmpl: int
    concl_tmpl: int
    blocks: list[str] = field(default_factory=list)
    #: One surface template per body, so the document is not a single repeated
    #: sentence shape. Empty for docs built before the templates were varied.
    body_tmpls: list[int] = field(default_factory=list)

    @property
    def combo(self) -> tuple[int, int]:
        return (self.intro_tmpl, self.concl_tmpl)

    def render(self) -> str:
        return " ".join(self.blocks)

    def spoof(self) -> str:
        """Surface form used for split keys only."""
        return (f"{self.thesis_id}|{self.n_reasons}|{self.intro_tmpl}|"
                f"{self.concl_tmpl}|{'-'.join(map(str, self.body_tmpls))}")


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
    body_tmpls = [rng.randrange(len(BODY_TEMPLATES)) for _ in range(n_reasons)]

    intro = INTRO_TEMPLATES[combo[0]].format(T=thesis, N=n_reasons)
    bodies = [render_body(body_tmpls[i], i + 1, reasons[i], thesis)
              for i in range(n_reasons)]
    concl = CONCLUSION_TEMPLATES[combo[1]].format(T=thesis, N=n_reasons)
    return Doc(thesis_id, thesis, n_reasons, reasons, combo[0], combo[1],
               [intro] + bodies + [concl], body_tmpls)


# --------------------------------------------------------------------------
# Exact parsers. Consistency is defined by these, so they must be strict.
# --------------------------------------------------------------------------

#: (pattern, thesis_group, n_group) per intro template. One pattern per
#: template rather than one tolerant pattern for all of them, because the
#: parser's return value includes *which* template was used: `retarget` and
#: `degrade_local` re-render a block, and re-rendering it with a different
#: surface form would make the corruption detectable from style alone.
_INTRO_RES = [
    (re.compile(r"this\s+essay\s+argues\s+that\s+(.+?)\s+with\s+"
                r"(\d+)\s+reasons"), 1, 2),
    (re.compile(r"we\s+argue\s+that\s+(.+?)\s*,\s*and\s+offer\s+"
                r"(\d+)\s+reasons"), 1, 2),
    (re.compile(r"the\s+claim\s+is\s+that\s+(.+?)\s*;\s*"
                r"(\d+)\s+reasons\s+follow"), 1, 2),
    (re.compile(r"this\s+paper\s+will\s+show\s+that\s+(.+?)\s+in\s+"
                r"(\d+)\s+reasons"), 1, 2),
    (re.compile(r"(.+?)\s+is\s+the\s+central\s+claim\s+here\s*,\s*"
                r"supported\s+by\s+(\d+)\s+reasons"), 1, 2),
    (re.compile(r"we\s+take\s+the\s+position\s+that\s+(.+?)\s*;\s*"
                r"(\d+)\s+points\s+support\s+it"), 1, 2),
    (re.compile(r"the\s+case\s+for\s+(.+?)\s+rests\s+on\s+"
                r"(\d+)\s+reasons"), 1, 2),
    (re.compile(r"here\s+we\s+defend\s+(.+?)\s+through\s+"
                r"(\d+)\s+reasons"), 1, 2),
]
_CONCL_RES = [
    (re.compile(r"these\s+(\d+)\s+reasons\s+establish\s+that\s+(.+?)\s*\."),
     1, 2),
    (re.compile(r"we\s+have\s+shown\s+that\s+(.+?)\s+on\s+the\s+strength\s+of"
                r"\s+(\d+)\s+reasons\s*\."), 2, 1),
    (re.compile(r"the\s+(\d+)\s+reasons\s+given\s+support\s+(.+?)\s*\."),
     1, 2),
    (re.compile(r"so\s+(.+?)\s*,\s*as\s+the\s+(\d+)\s+reasons\s+above\s+show"
                r"\s*\."), 2, 1),
    (re.compile(r"the\s+argument\s+for\s+(.+?)\s+rests\s+on\s+(\d+)\s+reasons"
                r"\s*\."), 2, 1),
    (re.compile(r"taken\s+together\s*,?\s+(\d+)\s+points\s+support\s+"
                r"(.+?)\s*\."), 1, 2),
    (re.compile(r"on\s+these\s+(\d+)\s+grounds\s*,?\s+we\s+conclude\s+"
                r"(.+?)\s*\."), 1, 2),
    (re.compile(r"thus\s*,?\s+the\s+evidence\s+favors\s+(.+?)\s*;\s*"
                r"(\d+)\s+reasons\s+support\s+this\s*\."), 2, 1),
]
#: (pattern, reason_group, thesis_group) per body template. Templates 2 and 3
#: put the thesis and the reason on opposite sides of the comma, so the group
#: order is not the same for every pattern; naming the groups by position here
#: is what keeps `parse_block` from silently reading the reason as the thesis.
_BODY_RES = [
    (re.compile(r"reason\s+(\d+)\s*:\s*(.+?)\s*,\s*so\s+(.+?)\s*\."), 2, 3),
    (re.compile(r"reason\s+(\d+)\s*:\s*because\s+(.+?)\s*,\s*(.+?)\s*\."),
     2, 3),
    (re.compile(r"reason\s+(\d+)\s*:\s*(.+?)\s*,\s*since\s+(.+?)\s*\."), 3, 2),
    (re.compile(r"point\s+(\d+)\s*:\s*given\s+(.+?)\s*,\s*"
                r"(.+?)\s+follows\s*\."), 2, 3),
    (re.compile(r"point\s+(\d+)\s*:\s*evidence\s+from\s+(.+?)\s+"
                r"supports\s+(.+?)\s*\."), 2, 3),
    (re.compile(r"point\s+(\d+)\s*:\s*(.+?)\s+follows\s+from\s+"
                r"(.+?)\s*\."), 3, 2),
    (re.compile(r"point\s+(\d+)\s*:\s*the\s+observation\s+that\s+"
                r"(.+?)\s+backs\s+(.+?)\s*\."), 2, 3),
    (re.compile(r"point\s+(\d+)\s*:\s*we\s+see\s+(.+?)\s+in\s+"
                r"(.+?)\s*\."), 3, 2),
]


def render_intro(tmpl: int, thesis: str, n: int) -> str:
    return INTRO_TEMPLATES[tmpl].format(T=thesis, N=n)


def render_concl(tmpl: int, thesis: str, n: int) -> str:
    return CONCLUSION_TEMPLATES[tmpl].format(T=thesis, N=n)


def render_body(tmpl: int, idx: int, reason: str, thesis: str) -> str:
    return BODY_TEMPLATES[tmpl].format(i=idx, R=reason, T=thesis)


def intro_parts(text: str) -> tuple[int, str, int] | None:
    """Return (template_index, thesis_text, n_declared) or None."""
    for i, (rx, tg, ng) in enumerate(_INTRO_RES):
        m = rx.search(text)
        if m:
            return i, _norm(m.group(tg)), int(m.group(ng))
    return None


def body_parts(text: str) -> tuple[int, str, int, str] | None:
    """Return (template_index, reason, reason_index, thesis_text) or None."""
    for i, (rx, rg, tg) in enumerate(_BODY_RES):
        m = rx.search(text)
        if m:
            return i, _norm(m.group(rg)), int(m.group(1)), _norm(m.group(tg))
    return None


def concl_parts(text: str) -> tuple[int, int, str] | None:
    """Return (template_index, n_declared, thesis_text) or None."""
    for i, (rx, ng, tg) in enumerate(_CONCL_RES):
        m = rx.search(text)
        if m:
            return i, int(m.group(ng)), _norm(m.group(tg))
    return None


def _norm(s: str | None) -> str | None:
    """Collapse whitespace so a captured thesis matches `THESIS_INDEX`.

    The parsers are whitespace-tolerant because a *model-produced* block may
    put the space on the wrong side of a comma (``"sunlight, so"``) or drop it
    before the full stop. The template's canonical spacing is a property of the
    generator, not of every string that is semantically the same block, and
    rejecting a correct repair over a space would make the consistency metric
    measure punctuation instead of content.
    """
    if s is None:
        return None
    return re.sub(r"\s+", " ", s).strip()


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
    #: Surface template this block was rendered with. Needed to re-render the
    #: block after a corruption (degrade/retarget) without changing its shape,
    #: which is what keeps the corruption a content edit rather than a style
    #: edit that the model could detect for free.
    tmpl: int | None = None


def parse_block(text: str, role: str) -> tuple[int | None, str | None,
                                              int | None]:
    """Return (thesis_id, reason, reason_index) for one block.

    Unparseable input returns `(None, ...)` rather than raising, because the
    repair experiments must be able to score a *model-produced* document that
    may be garbage. Callers that require a valid block check for None.
    """
    if role == "intro":
        p = intro_parts(text)
        if p is None:
            return None, None, None
        return THESIS_INDEX.get(p[1]), None, None
    if role == "body":
        p = body_parts(text)
        if p is None:
            return None, None, None
        return THESIS_INDEX.get(p[3]), p[1], p[2]
    if role == "conclusion":
        p = concl_parts(text)
        if p is None:
            return None, None, None
        return THESIS_INDEX.get(p[2]), None, None
    raise ValueError(f"unknown role {role!r}")


def parse_block_tmpl(text: str, role: str) -> int | None:
    """Template index of a block, or None when it does not parse."""
    if role == "intro":
        p = intro_parts(text)
    elif role == "body":
        p = body_parts(text)
    elif role == "conclusion":
        p = concl_parts(text)
    else:
        raise ValueError(f"unknown role {role!r}")
    return None if p is None else p[0]


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
    ip = intro_parts(blocks[0])
    if ip is None:
        return Parse(ok=False)
    thesis, n_intro = ip[1], ip[2]

    body_theses, body_idx, reasons = [], [], []
    for b in blocks[1:-1]:
        bp = body_parts(b)
        if bp:
            body_idx.append(bp[2])
            reasons.append(bp[1])
            body_theses.append(bp[3])
        else:
            # filler or unparseable body -> counts as an invalid reason slot
            body_idx.append(-1)
            reasons.append(None)

    n_concl, concl_thesis = None, None
    cp = concl_parts(blocks[-1])
    if cp is not None:
        n_concl, concl_thesis = cp[1], cp[2]

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

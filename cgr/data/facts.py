"""Scope-indexed claims with exactly adjudicable truth.

This replaces the draft's motivating example. The draft's version

    2 + 2 = 5, where 2 males and 2 females are counted, and one female is pregnant

does not work: 2 males + 2 females is 4 people, and a pregnancy is not a
person, so the claim is simply false and a reviewer will say so in the first
paragraph. Worse, adjudicating it requires world knowledge that the exact
interpreter does not have, which undercuts the "binding exact evidence"
claim in §3.5.

We therefore use four scope families, each of which is *mechanically
adjudicable* by the interpreter:

  base      "in base 4, 2 + 2 = 10"           true (2+2=4 = 10 in base 4)
  rounding  "rounding up to the nearest 5, 2 + 2 = 5"   true
  mod       "modulo 3, 2 + 2 = 1"            true
  count     "counting including the unborn, ... there are 5 individuals"  true

The rounding family rescues "2 + 2 = 5" honestly: it is true under a stated
rounding scope and false under the default. The claim is therefore not
"the model can do arithmetic" but "the model represents *which scope
applies*", which is the actual thesis.

`marker_present` controls whether the scope prefix appears in the text. The
marker-stripped variants are the adversarial split: if scope classification
accuracy collapses when the surface marker is removed, the scope head was
reading a token, not a scope.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

FAMILIES = ["base", "rounding", "mod", "count"]
SCOPE_NONE = 4  # index used when a claim carries no explicit scope

# Families/values deliberately excluded from training so we can measure
# whether the scope head generalises beyond the scopes it has seen.
HELDOUT_SCOPES = {("base", 8), ("rounding", 10), ("mod", 7)}


@dataclass
class Claim:
    family: str
    spec: dict
    text: str
    marker: str | None
    marker_present: bool
    heldout_scope: bool = False
    #: For hand-built probe sets: the disposition the task specification
    #: requires ("accept" | "reject" | "defer"). None for generated claims.
    expected: str | None = None
    note: str = ""

    @property
    def scope_key(self) -> tuple[str, int]:
        return (self.family, self.spec.get("scope_value", 0))

    def scope_index(self) -> int:
        if not self.marker_present:
            return SCOPE_NONE
        return FAMILIES.index(self.family)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _base_repr(n: int, base: int) -> str:
    if base == 10:
        return str(n)
    if n == 0:
        return "0"
    digits = "0123456789"
    out = []
    while n:
        out.append(digits[n % base])
        n //= base
    return "".join(reversed(out))


def _marker_text(spec: dict) -> str:
    fam = spec["family"]
    if fam == "base":
        return f"in base {spec['base']}"
    if fam == "rounding":
        return f"rounding up to the nearest {spec['m']}"
    if fam == "mod":
        return f"modulo {spec['m']}"
    if fam == "count":
        return ("counting including the unborn" if spec["include_unborn"]
                else "counting distinct people")
    raise ValueError(fam)


def _body_text(spec: dict) -> str:
    fam = spec["family"]
    if fam in ("base", "rounding", "mod"):
        return f"{spec['a']} + {spec['c']} = {spec['claimed']} ."
    if fam == "count":
        return (f"there are {spec['males']} males and {spec['females']} females "
                f"and one is pregnant , so there are {spec['claimed']} individuals .")
    if fam == "empirical":
        return spec["text"]
    raise ValueError(fam)


def render(spec: dict, marker_present: bool) -> tuple[str, str | None]:
    body = _body_text(spec)
    if marker_present:
        m = _marker_text(spec)
        return f"{m} , {body}", m
    return body, None


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def _true_and_false_values(spec: dict) -> tuple[int, int]:
    """Return (true_value, a_plausible_wrong_value)."""
    fam = spec["family"]
    if fam == "base":
        s = spec["a"] + spec["c"]
        return s, s + 1
    if fam == "rounding":
        m = spec["m"]
        s = spec["a"] + spec["c"]
        tv = ((s + m - 1) // m) * m
        # a "false" twin must actually be false: if the sum is already a
        # multiple of m, the unrounded value equals the rounded one.
        fv = s if s != tv else tv + m
        return tv, fv
    if fam == "mod":
        return (spec["a"] + spec["c"]) % spec["m"], (spec["a"] + spec["c"]) % spec["m"] + 1
    if fam == "count":
        n = spec["males"] + spec["females"] + (spec["pregnant"] if spec["include_unborn"] else 0)
        return n, n - 1
    raise ValueError(fam)


def make_claim(rng: random.Random, family: str | None = None,
               marker_present: bool = True, allow_heldout: bool = False,
               force_true: bool | None = None) -> Claim:
    family = family or rng.choice(FAMILIES)

    if family == "base":
        base = rng.choice([2, 4, 8, 10])
        spec = {"family": "base", "base": base, "scope_value": base,
                "a": rng.randint(1, 5), "c": rng.randint(1, 5)}
        true_v = spec["a"] + spec["c"]
        correct = _base_repr(true_v, base)
        if force_true is None:
            force_true = rng.random() < 0.5
        if force_true:
            spec["claimed"] = int(correct)
        else:
            # a wrong digit-string that is still well formed
            spec["claimed"] = int(_base_repr(true_v + rng.randint(1, 3), base))

    elif family == "rounding":
        m = rng.choice([2, 5, 10])
        spec = {"family": "rounding", "m": m, "scope_value": m,
                "a": rng.randint(1, 5), "c": rng.randint(1, 5)}
        tv, fv = _true_and_false_values(spec)
        if force_true is None:
            spec["claimed"] = rng.choice([tv, fv])
        else:
            spec["claimed"] = tv if force_true else fv

    elif family == "mod":
        m = rng.choice([2, 3, 5, 7])
        spec = {"family": "mod", "m": m, "scope_value": m,
                "a": rng.randint(1, 8), "c": rng.randint(1, 8)}
        tv, fv = _true_and_false_values(spec)
        spec["claimed"] = tv if force_true is True else (
            fv if force_true is False else rng.choice([tv, fv]))

    elif family == "count":
        incl = rng.random() < 0.5
        spec = {"family": "count", "include_unborn": bool(incl),
                "unit_defined": True,
                "scope_value": int(incl),
                "males": rng.randint(1, 4), "females": rng.randint(1, 4),
                "pregnant": 1}
        # Some counts are observed rather than stated exactly. That is the
        # only source of "computable but uncertain" claims, and without it the
        # epistemic head has no positive examples of that class.
        if rng.random() < 0.25:
            spec["noisy"] = 0.15
        tv, fv = _true_and_false_values(spec)
        if force_true is None:
            spec["claimed"] = rng.choice([tv, fv])
        else:
            spec["claimed"] = tv if force_true else fv

    else:
        raise ValueError(f"unknown family {family!r}")

    text, marker = render(spec, marker_present)
    heldout = (family, spec.get("scope_value", 0)) in HELDOUT_SCOPES
    return Claim(family, spec, text, marker, marker_present, heldout)


def make_claim_set(n: int, seed: int = 0, marker_present: bool = True,
                   balanced: bool = True,
                   empirical_frac: float = 0.2) -> list[Claim]:
    """Mix adjudicable claims with empirical (non-adjudicable) ones.

    The empirical share is not padding: without it every claim is
    "known", the epistemic head is a constant, and its calibration number is
    meaningless. It also gives the DEF action real training examples.
    """
    rng = random.Random(seed)
    out = []
    n_emp = int(n * empirical_frac)
    for i in range(n - n_emp):
        fam = FAMILIES[i % len(FAMILIES)]
        out.append(make_claim(rng, family=fam, marker_present=marker_present,
                              force_true=(i % 2 == 0) if balanced else None))
    if n_emp:
        out.extend(make_empirical_claims(n_emp, seed=seed + 7))
    rng.shuffle(out)
    return out


def make_empirical_claims(n: int, seed: int = 0) -> list[Claim]:
    """Claims the interpreter cannot adjudicate -> epistemic status CHECK_LATER.

    Generated from a slot grammar rather than a three-item list so that a
    held-out control set can be disjoint from the treated set. Without that,
    the absorption test in Exp 2 has no control.
    """
    rng = random.Random(seed)
    subjects = ["this policy", "the new schedule", "the subsidy",
                "the pilot program", "the merger", "the tariff"]
    verbs = ["raised", "reduced", "increased", "lowered", "stabilised"]
    objects = ["measured output", "staff turnover", "planting rates",
               "absenteeism", "cost overruns", "export volume"]
    times = ["last year", "in that office", "across the region",
             "during the trial", "after the reform"]
    out = []
    for i in range(n):
        text = (f"{rng.choice(subjects)} {rng.choice(verbs)} "
                f"{rng.choice(objects)} {rng.choice(times)} .")
        spec = {"family": "empirical", "text": text, "scope_value": 0}
        out.append(Claim("empirical", spec, text, None, False, False))
    return out


def make_counting_ontology_claims() -> list[Claim]:
    """The four specified claims over one fixed counting scenario.

    Scenario (stated in the task, not inferred): **four born people plus
    exactly one fetus**. The point is scope binding, not arithmetic that
    changes with context. The four claims pin the four required dispositions:

    | claim                                      | disposition |
    |--------------------------------------------|-------------|
    | `2 + 2 = 5`                                | reject      |
    | "there are five born people"               | reject      |
    | "there are five entities, counting the fetus" | accept   |
    | "there are five" (no counting rule)        | defer       |

    The first rejection is adjudicated under the *default* scope, so its
    provenance is `assumed` rather than `exact`. The last is not adjudicable at
    all until the counting unit is supplied: `unit_defined=False` makes the
    interpreter REFUSE instead of silently assuming "born people", which is the
    default-scope leakage the architecture exists to prevent.
    """
    return [
        Claim("base",
              {"family": "base", "base": 10, "scope_value": 10,
               "a": 2, "c": 2, "claimed": 5},
              "2 + 2 = 5 .", None, False, False,
              expected="reject",
              note="ordinary arithmetic gives four; rejection rests on the "
                   "default decimal scope, so provenance is assumed"),
        Claim("count",
              {"family": "count", "unit_defined": True, "unit": "born people",
               "include_unborn": False, "scope_value": 0,
               "males": 2, "females": 2, "pregnant": 1, "claimed": 5},
              "there are five born people .", "born people", True, False,
              expected="reject",
              note="there are four born people; the unit is stated, so this "
                   "is an exact rejection"),
        Claim("count",
              {"family": "count", "unit_defined": True,
               "unit": "entities counting the fetus",
               "include_unborn": True, "scope_value": 1,
               "males": 2, "females": 2, "pregnant": 1, "claimed": 5},
              "there are five entities , counting the fetus .",
              "counting the fetus", True, False,
              expected="accept",
              note="five entities under the stated counting convention"),
        Claim("count",
              {"family": "count", "unit_defined": False, "scope_value": 0,
               "males": 2, "females": 2, "pregnant": 1, "claimed": 5},
              "there are five .", None, False, False,
              expected="defer",
              note="counting unit undefined: 'five what?' has no truth value "
                   "until a rule is supplied"),
    ]


__all__ = ["Claim", "FAMILIES", "SCOPE_NONE", "HELDOUT_SCOPES",
           "make_claim", "make_claim_set", "make_empirical_claims",
           "make_counting_ontology_claims", "render"]

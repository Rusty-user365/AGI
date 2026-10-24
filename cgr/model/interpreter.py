"""Exact interpreter with provenance and uncertainty propagation (§3.5).

Design point that matters for the paper: the interpreter is what makes a
claim *adjudicable*. Every scope family in `cgr.data.facts` has a
deterministic semantics, so "verified-wrong" and "verified-right" are
machine-checkable with no human labels.

Provenance is not decoration. A claim whose scope marker is absent is
evaluated under the *default* scope, and that default is an assumption
about the speaker, not a fact about the world. We record it as
Provenance.ASSUMED rather than EXACT. This is the mechanism behind the
default-scope-leakage metric: the system can reject the bare claim and
still report that its rejection rests on an assumption.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Provenance(str, Enum):
    EXACT = "exact"
    ESTIMATED = "estimated"
    ASSUMED = "assumed"


class OutputMode(str, Enum):
    DIRECT = "direct"
    INTERVAL = "interval"
    REFUSE = "refuse"


@dataclass
class Value:
    value: float
    provenance: Provenance
    sigma: float = 0.0
    note: str = ""


@dataclass
class Verdict:
    correct: bool | None          # None => not adjudicable
    value: float | None
    claimed: float | None
    provenance: Provenance
    sigma: float
    mode: OutputMode
    scope_assumed: bool
    note: str = ""


# --------------------------------------------------------------------------
# Uncertainty propagation:  Sigma_y = J Sigma_v J^T   (§3.5)
# --------------------------------------------------------------------------

def propagate(f, x: list[float], sigma: list[float]) -> tuple[float, float]:
    """First-order propagation through a scalar function.

    J is built by central differences; Sigma_v is diagonal (inputs are
    treated as independent, which is the assumption stated in the paper).
    """
    x = [float(v) for v in x]
    n = len(x)
    jac = []
    for i in range(n):
        h = 1e-5 * max(1.0, abs(x[i]))
        xp, xm = list(x), list(x)
        xp[i] += h
        xm[i] -= h
        jac.append((f(xp) - f(xm)) / (2 * h))
    var = sum((j * s) ** 2 for j, s in zip(jac, sigma))
    return float(f(x)), float(math.sqrt(max(var, 0.0)))


# --------------------------------------------------------------------------
# Adjudication
# --------------------------------------------------------------------------

DEFAULT_SCOPE = {"base": 10, "rounding": 1, "mod": None, "count": False}


def _base_to_int(digits: int, base: int) -> int:
    """Interpret the *decimal-written* claimed number as a numeral in `base`."""
    s = str(int(digits))
    try:
        return int(s, base)
    except ValueError:
        return -1  # digit illegal in this base -> definitely wrong


def adjudicate(spec: dict, marker_present: bool = True,
               sigma_frac: float = 0.0) -> Verdict:
    """Adjudicate a claim.

    If the scope marker is absent, the claim is evaluated under the *default*
    scope (decimal, no rounding, no modulus, distinct people) and the
    provenance is ASSUMED rather than EXACT. That asymmetry is the whole point
    of the default-scope-leakage metric: a bare "2 + 2 = 5" is rejected, but
    the rejection rests on an assumption about the speaker, and the system
    says so instead of hiding it.
    """
    fam = spec["family"]
    assumed = not marker_present

    if fam == "empirical":
        # `correct` must be None, not False. None means "not adjudicable";
        # False means "adjudicated and wrong". Conflating them made the
        # scaffolding label empirical claims as successfully resolved
        # (value=1.0, source=derived), which is what saturated the value head in
        # the first version of the suite. The REFUSE mode is what carries the
        # "cannot answer" information here.
        return Verdict(None, None, None, Provenance.ESTIMATED, float("inf"),
                       OutputMode.REFUSE, True,
                       "no exact semantics; requires external evidence")

    if fam == "base":
        base = spec["base"] if marker_present else DEFAULT_SCOPE["base"]
        s = spec["a"] + spec["c"]
        claimed = int(spec["claimed"])
        # the claimed numeral, read in the effective base, must equal a+c
        claimed_val = _base_to_int(claimed, base)
        sigma = sigma_frac * abs(s)
        mode = OutputMode.INTERVAL if sigma > 0 else OutputMode.DIRECT
        scope_note = "" if marker_present else " (scope assumed: decimal)"
        return Verdict(claimed_val == s, float(s), float(claimed_val),
                       Provenance.ASSUMED if assumed else Provenance.EXACT,
                       sigma, mode, assumed,
                       f"a+c={s} in base {base}; claimed numeral reads "
                       f"{claimed_val}{scope_note}")

    if fam == "rounding":
        m = spec["m"] if marker_present else 1  # default: no rounding
        s = spec["a"] + spec["c"]
        tv = ((s + m - 1) // m) * m
        return Verdict(int(spec["claimed"]) == tv, float(tv), float(spec["claimed"]),
                       Provenance.ASSUMED if assumed else Provenance.EXACT,
                       0.0, OutputMode.DIRECT, assumed,
                       f"ceil({s}/{m})*{m} = {tv}"
                       + ("" if marker_present else " (scope assumed: no rounding)"))

    if fam == "mod":
        s = spec["a"] + spec["c"]
        if marker_present:
            tv = s % spec["m"]
            note = f"({s}) mod {spec['m']} = {tv}"
        else:
            tv = s  # default: no modulus
            note = f"({s}) exact = {tv} (scope assumed: no modulus)"
        return Verdict(int(spec["claimed"]) == tv, float(tv), float(spec["claimed"]),
                       Provenance.ASSUMED if assumed else Provenance.EXACT,
                       0.0, OutputMode.DIRECT, assumed, note)

    if fam == "count":
        # A count is only adjudicable once the counting *unit* is fixed. "there
        # are five" over four born people and one fetus has no truth value
        # until a rule says whether the fetus is counted; assuming a rule is
        # exactly the default-scope leakage this system is built to avoid.
        # So the interpreter refuses instead of picking the default.
        if not spec.get("unit_defined", True):
            return Verdict(None, None, float(spec.get("claimed", 0)),
                           Provenance.ASSUMED, float("inf"),
                           OutputMode.REFUSE, True,
                           "counting unit undefined; clarification required "
                           "before adjudication")
        incl = bool(spec.get("include_unborn")) if marker_present else False
        n = spec["males"] + spec["females"]
        if incl:
            n += spec.get("pregnant", 0)
        sigma = 0.0
        # Provenance tracks the weakest link in *scope* determination; sigma
        # tracks numeric uncertainty separately. A bare claim therefore stays
        # ASSUMED even when its count is also estimated.
        prov = Provenance.ASSUMED if assumed else Provenance.EXACT
        sigma_frac = spec.get("noisy", sigma_frac)
        if sigma_frac > 0:  # noisy observation of the crowd
            _, sigma = propagate(lambda v: v[0], [float(n)], [sigma_frac * n])
            if not assumed:
                prov = Provenance.ESTIMATED
        mode = OutputMode.INTERVAL if sigma > 0 else OutputMode.DIRECT
        if sigma > 2.0:
            mode = OutputMode.REFUSE
        return Verdict(int(spec["claimed"]) == n, float(n), float(spec["claimed"]),
                       prov, sigma, mode, assumed,
                       f"{spec['males']}+{spec['females']}"
                       f"{'+unborn' if spec.get('include_unborn') else ''} = {n}")

    raise ValueError(f"unknown family {fam!r}")


def _repr_in_base(n: int, base: int) -> str:
    if n == 0:
        return "0"
    digits = "0123456789"
    out = []
    while n:
        out.append(digits[n % base])
        n //= base
    return "".join(reversed(out))


def epistemic_status(verdict: Verdict) -> str:
    """Map an adjudication onto the four epistemic classes of §3.3.

    known       exact, and the scope is stated
    computable  exact once computed, but carries estimation uncertainty
    check_later no exact semantics; needs external evidence
    unknown     adjudicable in principle, but the scope is not stated, so any
                answer rests on an assumption
    """
    if verdict.mode == OutputMode.REFUSE:
        return "check_later"
    if verdict.correct is None:
        return "unknown"
    if verdict.scope_assumed:
        return "unknown"
    if verdict.sigma > 0:
        return "computable"
    return "known"


__all__ = ["Provenance", "OutputMode", "Value", "Verdict", "adjudicate",
           "propagate", "epistemic_status", "DEFAULT_SCOPE"]

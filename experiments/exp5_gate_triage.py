"""Experiment 5 — the gate: usage, overrides, routing, and triage.

The draft specifies eight actions and tests roughly four. This measures:

  1. action histogram          does the gate ever use all eight? (If REORD is
                               never selected, that is a finding.)
  2. gate accuracy vs oracle   how far from the scaffolding policy
  3. override frequency        how often each safety interlock fires
  4. cost per correct answer   against the oracle policy
  5. triage                    sequential vs triage vs random vs hardest-first

The triage test is deliberately weak: it uses *predicted* tau as the ordering
key. If predicted tau is uninformative, triage cannot beat random, and that is
the honest result. The draft's "prompted baseline may match — report either
way" applies here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random

import torch

from _common import arg_parser, banner, get_system, table, write_result
from cgr.data.corpus import build_corpus, make_repair_set
from cgr.data.facts import make_claim_set
from cgr.metrics import bootstrap_ci, cost_per_correct, tier_distribution
from cgr.model.gate import ACTIONS
from cgr.scaffold import (episodes_from_claims, episodes_from_records,
                          episodes_from_repairs)
from cgr.train import collate

PREDICTION = {
    "histogram": "ANS/VER/REP/REV/DEF dominate; REORD and UPD absent because "
                 "the curriculum never offers them; DELIB appears only if "
                 "marker-stripped claims are in the eval set",
    "gate_accuracy": "> 0.8 against the scaffolding policy",
    "overrides": "surprise fires rarely; risk ladder fires on low-confidence claims",
    "triage": "triage >= sequential if predicted tau is informative; "
              "triage ~ random if not",
}


def main() -> int:
    ap = arg_parser("Experiment 5: gate usage and triage")
    args = ap.parse_args()
    n = args.n or 60

    banner("Experiment 5 — gate usage, overrides, triage")
    system, cfg = get_system(args.preset, seed=args.seed,
                             checkpoint=args.checkpoint)
    system.eval()
    tok, splits = build_corpus(cfg)

    # ------------------------------------------------------------------
    # build a mixed evaluation set
    # ------------------------------------------------------------------
    recs = splits["test"][:n]
    # Both marker-present and marker-stripped claims must be in the eval set.
    # The stripped ones are the only examples where the scaffolding policy
    # chooses DELIB, so an eval set built from marker-present claims alone
    # would report "DELIB is dead" when in fact DELIB was never offered a
    # single example. That was a real bug in the first version of this script.
    claims = (make_claim_set(n // 2, seed=args.seed + 1, marker_present=True)
              + make_claim_set(n - n // 2, seed=args.seed + 11,
                               marker_present=False))
    reps = make_repair_set(splits["test"], n // 2, seed=args.seed + 2)
    eps = (episodes_from_records(recs, noise=0.0, seed=0)
           + episodes_from_claims(claims, tok, noise=0.0, seed=1)
           + episodes_from_repairs(reps["local"], tok, noise=0.0, seed=2)
           + episodes_from_repairs(reps["cross_block"], tok, noise=0.0, seed=3))
    batch = collate(eps, tok, cfg.model.max_len)

    with torch.no_grad():
        out = system.forward_episodes(batch["ids"], batch["clock"])
        preds = system.heads.predict(out["feats"])
        logits = out["action_logits"]

    oracle = batch["actions"]
    chosen = logits.argmax(-1)
    acc = float((chosen == oracle).float().mean())

    hist = tier_distribution([ACTIONS[int(a)] for a in chosen])
    oracle_hist = tier_distribution([ACTIONS[int(a)] for a in oracle])

    # ------------------------------------------------------------------
    # per-condition diversity. A gate that answers ANS to everything can look
    # "diverse" over a mixed eval set only because the oracle is diverse. Split
    # by input type and report the distribution *within* each condition.
    # ------------------------------------------------------------------
    bounds = [("clean doc", 0, len(recs)),
              ("claim", len(recs), len(recs) + len(claims)),
              ("repair: local", len(recs) + len(claims),
               len(recs) + len(claims) + len(reps["local"])),
              ("repair: cross_block", len(recs) + len(claims) + len(reps["local"]),
               len(eps))]
    cond_rows, cond_json = [], {}
    for name, a, b in bounds:
        if b <= a:
            continue
        sl = slice(a, b)
        mh = tier_distribution([ACTIONS[int(x)] for x in chosen[sl]])
        oh = tier_distribution([ACTIONS[int(x)] for x in oracle[sl]])
        cond_json[name] = {"n": b - a,
                           "acc": float((chosen[sl] == oracle[sl]).float().mean()),
                           "model": mh, "oracle": oh}
        cond_rows.append({
            "condition": name, "n": b - a,
            "acc": cond_json[name]["acc"],
            "model_top": max(mh, key=mh.get),
            "model_top_share": max(mh.values()),
            "model_actions": len(mh), "oracle_actions": len(oh)})
    table(cond_rows, ["condition", "n", "acc", "model_top",
                      "model_top_share", "model_actions", "oracle_actions"])

    # The meaningful diversity question is whether the gate separates the two
    # defect types, not whether it emits many labels somewhere in the mix.
    def share(name, action):
        return cond_json.get(name, {}).get("model", {}).get(action, 0.0)

    separates_defects = (abs(share("repair: local", "REP")
                             - share("repair: cross_block", "REP")) > 0.1
                         or abs(share("repair: local", "REV")
                                - share("repair: cross_block", "REV")) > 0.1)

    # ------------------------------------------------------------------
    # overrides (per-example uncertainty = normalised action entropy)
    # ------------------------------------------------------------------
    surprise = (preds["outcome"][:, 0] - batch["labels"]["outcome"][:, 0]).abs()
    p_act = logits.softmax(-1)
    entropy = -(p_act * (p_act + 1e-9).log()).sum(-1) / torch.log(
        torch.tensor(float(cfg.model.n_actions)))
    overrides = {"surprise": 0, "risk_ladder": 0, "none": 0}
    final_actions = []
    for i in range(len(eps)):
        a, info = system.gate.decide(logits[i], surprise=float(surprise[i]),
                                     value_sigma=float(entropy[i]))
        overrides[info["override"] or "none"] += 1
        final_actions.append(ACTIONS[a])
    n_ov = max(sum(overrides.values()), 1)
    overrides = {k: v / n_ov for k, v in overrides.items()}

    # ------------------------------------------------------------------
    # cost
    # ------------------------------------------------------------------
    correct = (preds["outcome"][:, 0] > 0.5)
    cost_model = cost_per_correct(final_actions, correct.tolist())
    cost_oracle = cost_per_correct([ACTIONS[int(a)] for a in oracle],
                                   correct.tolist())

    # ------------------------------------------------------------------
    # triage simulation
    # ------------------------------------------------------------------
    budget = max(5, len(eps) // 5)
    # Ground truth comes from the labels, never from the model's own value head.
    # The first version scored orderings with `preds["value"] > 0.5`, and the
    # value head endorses ~100% of items, so every ordering scored exactly
    # `budget` and triage looked tied with random for a reason that had nothing
    # to do with tau. Scoring against labels makes the comparison real.
    truth_solvable = (batch["labels"]["value"] > 0.5).tolist()
    tau = preds["tau"].tolist()
    rng = random.Random(args.seed)

    def marks(order):
        return sum(1 for i in order[:budget] if truth_solvable[i])

    given = list(range(len(eps)))
    triage = sorted(given, key=lambda i: tau[i])
    hardest = sorted(given, key=lambda i: -tau[i])
    rand_marks = []
    for _ in range(500):
        o = given[:]
        rng.shuffle(o)
        rand_marks.append(marks(o))

    triage_marks = marks(triage)
    given_marks = marks(given)
    hardest_marks = marks(hardest)
    r_mean, r_lo, r_hi = bootstrap_ci(rand_marks)

    # Effect size and a permutation test. `marks` is bounded: the best any
    # ordering can do is reach every solvable item inside the budget. With a
    # skewed solvable/unsolvable mix that ceiling is only a mark or two above
    # random, so "triage > random" can be true and still be meaningless.
    # Report the gain as a fraction of the attainable gain, plus a p-value.
    n_solvable = sum(truth_solvable)
    max_marks = min(budget, n_solvable)
    gain = triage_marks - r_mean
    gain_max = max(max_marks - r_mean, 1e-9)
    normalized_gain = gain / gain_max
    p_perm = (1 + sum(1 for m in rand_marks if m >= triage_marks)) / (
        1 + len(rand_marks))
    # Does the arrival order itself start with the easy items? If sequential
    # already scores the maximum, the "sequential" baseline is an artefact of
    # how the eval set was concatenated, not a property of the policy.
    sequential_is_artifact = given_marks >= max_marks

    # ------------------------------------------------------------------
    # head health (§4.4): variance collapse, calibration, collinearity
    # ------------------------------------------------------------------
    from cgr.metrics import ece

    def cat_health(name: str) -> dict:
        probs = preds[f"{name}_probs"]
        conf, pred = probs.max(-1)
        correct = (pred == batch["labels"][name]).float()
        return {"ece": ece(conf, correct),
                "acc": float(correct.mean()),
                "conf_mean": float(conf.mean()),
                "conf_sd": float(conf.std())}

    health = {
        "scope": cat_health("scope"),
        "epistemic": cat_health("epistemic"),
        "value": {"sd": float(preds["value"].std()),
                  "mean": float(preds["value"].mean())},
        "tau": {"sd": float(preds["tau"].std()),
                "mean": float(preds["tau"].mean())},
    }
    # inter-head collinearity: correlation of the two heads' confidence
    conf_scope, _ = preds["scope_probs"].max(-1)
    conf_epi, _ = preds["epistemic_probs"].max(-1)
    if conf_scope.numel() > 2:
        a = conf_scope - conf_scope.mean()
        b = conf_epi - conf_epi.mean()
        denom = (a.norm() * b.norm()).clamp(min=1e-8)
        health["scope_epistemic_conf_corr"] = float((a * b).sum() / denom)
    else:
        health["scope_epistemic_conf_corr"] = float("nan")

    # ------------------------------------------------------------------
    rows = [{"action": a,
             "model": hist.get(a, 0.0),
             "oracle": oracle_hist.get(a, 0.0)}
            for a in ACTIONS]
    table(rows, ["action", "model", "oracle"])
    print(f"\n  gate accuracy vs oracle: {acc:.3f}")
    print(f"  overrides: {overrides}")
    print(f"  cost/correct: model {cost_model:.2f}  oracle {cost_oracle:.2f}")
    print(f"  head health: {health}")

    triage_rows = [
        {"ordering": "sequential", "marks": given_marks},
        {"ordering": "triage (tau asc)", "marks": triage_marks},
        {"ordering": "random (mean)", "marks": r_mean},
        {"ordering": "hardest first", "marks": hardest_marks},
    ]
    table(triage_rows, ["ordering", "marks"])
    print(f"  budget = {budget} of {len(eps)}; random 95% CI [{r_lo:.1f}, {r_hi:.1f}]")
    print(f"  attainable max = {max_marks}; gain {gain:+.2f} of "
          f"{gain_max:.2f} ({normalized_gain:.0%}); permutation p = {p_perm:.3f}")

    # Distinguish two very different kinds of "dead" action:
    #   - absent from the oracle histogram -> the curriculum never offers it,
    #     so there is no training signal and the gate cannot learn it;
    #   - present in the oracle histogram but never predicted -> the gate
    #     failed to learn an action that was available.
    # Collapsing these into one list would let a curriculum gap masquerade as a
    # modelling result (and vice versa).
    never_offered = [a for a in ACTIONS if oracle_hist.get(a, 0.0) < 0.01]
    offered_not_learned = [a for a in ACTIONS
                           if oracle_hist.get(a, 0.0) >= 0.01
                           and hist.get(a, 0.0) < 0.01]
    unused = never_offered + offered_not_learned
    notes = []
    if never_offered:
        notes.append(f"Never offered by the curriculum: {never_offered}. "
                     f"These have no training signal; report the effective "
                     f"action space, not the nominal eight.")
    if offered_not_learned:
        notes.append(f"Offered but never predicted: {offered_not_learned}. "
                     f"This is a gate learning failure, not a curriculum gap.")
    if p_perm < 0.05 and normalized_gain >= 0.2:
        notes.append(
            f"Triage beats random (triage {triage_marks} vs random "
            f"{r_mean:.1f}, CI [{r_lo:.1f}, {r_hi:.1f}]; gain {gain:+.2f} of an "
            f"attainable {gain_max:.2f} = {normalized_gain:.0%}; permutation "
            f"p={p_perm:.3f}). Predicted tau carries usable ordering "
            f"information.")
    else:
        notes.append(
            f"Triage does not establish an advantage over random (triage "
            f"{triage_marks}, random mean {r_mean:.1f}, CI [{r_lo:.1f}, "
            f"{r_hi:.1f}]; gain {gain:+.2f} of an attainable {gain_max:.2f} = "
            f"{normalized_gain:.0%}; permutation p={p_perm:.3f}; budget "
            f"{budget} of {len(eps)}, solvable base rate "
            f"{n_solvable / max(len(eps), 1):.2f}). Report the null, and note "
            f"the ceiling: with this mix the metric can only move by "
            f"{gain_max:.1f} marks.")
    if sequential_is_artifact:
        notes.append(
            f"'Sequential' scores the maximum ({given_marks}) because the eval "
            f"set is concatenated docs-then-claims; it is an artefact of "
            f"arrival order, not evidence about any policy. Compare triage to "
            f"random only.")
    if float(preds["value"].mean()) > 0.95:
        notes.append(f"The value head is saturated (mean {float(preds['value'].mean()):.3f}). "
                     f"Ordering is therefore scored against labels, not against "
                     f"the model's own value prediction; the saturation itself "
                     f"is a head-health finding.")
    if acc < 0.8:
        notes.append(f"Gate accuracy {acc:.2f} is below the 0.8 target; the "
                     f"gate has not internalised the scaffolding policy.")
    if not separates_defects:
        notes.append(
            "The gate does not separate local from cross-block repairs: the "
            "share of REP or REV differs by less than 0.10 between the two "
            "repair conditions. A gate that cannot tell the defect types apart "
            "cannot choose a revision scope on the evidence, so any triage "
            "advantage would have to come from a different signal. Report this "
            "as a gate failure, separately from the oracle taxonomy result "
            "(exp4), which does not use the learned gate.")

    write_result("exp5_gate_triage", {
        "experiment": "exp5_gate_triage",
        "preset": args.preset, "seed": args.seed, "n": len(eps),
        "preregistered_prediction": PREDICTION,
        "conditions": {
            "histogram_model": hist,
            "histogram_oracle": oracle_hist,
            "by_condition": cond_json,
            "gate_accuracy": acc,
            "overrides": overrides,
            "cost_per_correct_model": cost_model,
            "cost_per_correct_oracle": cost_oracle,
            "triage": {"budget": budget, "sequential": given_marks,
                       "triage": triage_marks, "random_mean": r_mean,
                       "random_ci": [r_lo, r_hi], "hardest": hardest_marks,
                       "max_attainable": max_marks, "gain": gain,
                       "normalized_gain": normalized_gain,
                       "permutation_p": p_perm,
                       "sequential_is_arrival_artifact":
                           bool(sequential_is_artifact),
                       "truth_solvable_rate":
                           sum(truth_solvable) / max(len(eps), 1)},
        },
        "summary": {
            "unused_actions": unused,
            "never_offered": never_offered,
            "offered_not_learned": offered_not_learned,
            "triage_beats_random": bool(p_perm < 0.05 and normalized_gain >= 0.2),
            "triage_normalized_gain": normalized_gain,
            "triage_truth_source": "labels.value>0.5 (not model prediction)",
            "gate_matches_scaffold": bool(acc >= 0.8),
            "separates_defects": bool(separates_defects),
            "head_health": health,
        },
        "notes": notes,
    })
    print("\n" + "\n".join(f"  ! {x}" for x in notes) if notes else "\n  no anomalies")
    return 0


if __name__ == "__main__":
    sys.exit(main())

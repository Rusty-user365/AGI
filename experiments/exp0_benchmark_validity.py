"""Experiment 0 — benchmark validity gate.

Runs on CPU in seconds, before any GPU time is spent. It asks one question:

    Can the paper's tasks be solved without the architecture?

If a linear probe on hashed surface features matches the scope head, then the
scope head is reading a token, not a scope, and §5.2's scope claim is
unsupported. If an n-gram model explains the document distribution, then the
repair experiment is measuring memorisation.

This is the experiment the draft was missing, and it is the cheapest one.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from _common import arg_parser, banner, get_system, table, write_result
from cgr.baselines import (MajorityBaseline, NgramLM, RidgeClassifier,
                           hashed_bigrams, probe_report)
from cgr.data.corpus import build_corpus
from cgr.data.facts import (FAMILIES, HELDOUT_SCOPES, SCOPE_NONE, make_claim,
                            make_claim_set)
from cgr.model.interpreter import adjudicate, epistemic_status
from cgr.scaffold import EPISTEMIC, SCOPE_INDEX, episodes_from_claims
from cgr.train import collate

PREDICTION = {
    "scope_probe_accuracy": ">= 0.90 on marker-present claims",
    "scope_probe_on_stripped": "<= chance (0.20), because the cue is absent",
    "ngram_vs_model_gap": "small; the corpus is template-generated",
    "interpretation": "if the probe matches the model, the scope head adds "
                      "nothing and the claim must be re-scoped to 'the model "
                      "learns a surface marker', not 'the model represents scope'",
}


def scope_label(c) -> int:
    """True scope family, regardless of whether the marker is present.

    For the *stripped* condition this is the right target: the question is
    whether the content alone reveals the scope. (Separately, the system's
    own head is trained to emit SCOPE_NONE when the marker is absent — "I do
    not know the scope" is a distinct and correct answer, and it is scored in
    exp2, not here.)

    Empirical claims have no scope family at all, so they map to SCOPE_NONE.
    """
    return SCOPE_INDEX.get(c.family, SCOPE_NONE)


def epistemic_label(c) -> int:
    v = adjudicate(c.spec, c.marker_present)
    return EPISTEMIC[epistemic_status(v)]


def main() -> int:
    ap = arg_parser("Experiment 0: can trivial baselines solve the tasks?")
    args = ap.parse_args()
    n = args.n or 2000

    banner("Experiment 0 — benchmark validity")
    print(f"preset={args.preset}  n={n}")

    # ------------------------------------------------------------------
    # Claims: scope + epistemic probes
    # ------------------------------------------------------------------
    train_c = make_claim_set(n, seed=101, marker_present=True)
    test_c = make_claim_set(n, seed=202, marker_present=True)
    stripped_c = make_claim_set(n, seed=202, marker_present=False)

    y_tr = [scope_label(c) for c in train_c]
    y_te = [scope_label(c) for c in test_c]
    y_st = [scope_label(c) for c in stripped_c]

    scope_probe = probe_report([c.text for c in train_c], y_tr,
                               [c.text for c in test_c], y_te,
                               n_classes=5)
    stripped_probe = probe_report([c.text for c in train_c], y_tr,
                                  [c.text for c in stripped_c], y_st,
                                  n_classes=5)

    e_tr = [epistemic_label(c) for c in train_c]
    e_te = [epistemic_label(c) for c in test_c]
    epi_probe = probe_report([c.text for c in train_c], e_tr,
                             [c.text for c in test_c], e_te, n_classes=4)

    # held-out scopes: unseen (family, value) pairs
    held_c = [c for c in test_c if (c.family, c.spec.get("scope_value", 0))
              in HELDOUT_SCOPES]
    held_report = None
    if len(held_c) >= 20:
        Xtr = hashed_bigrams([c.text for c in train_c])
        Xhe = hashed_bigrams([c.text for c in held_c])
        clf = RidgeClassifier(5).fit(Xtr, y_tr)
        held_report = {"n": len(held_c),
                       "probe_acc": clf.accuracy(Xhe, [scope_label(c) for c in held_c])}

    rows = [
        {"task": "scope (marker present)", "probe": scope_probe["probe_acc"],
         "majority": scope_probe["majority_acc"], "n": scope_probe["n_test"]},
        {"task": "scope (marker stripped)", "probe": stripped_probe["probe_acc"],
         "majority": stripped_probe["majority_acc"], "n": stripped_probe["n_test"]},
        {"task": "epistemic", "probe": epi_probe["probe_acc"],
         "majority": epi_probe["majority_acc"], "n": epi_probe["n_test"]},
    ]
    if held_report:
        rows.append({"task": "scope (held-out values)",
                     "probe": held_report["probe_acc"],
                     "majority": float("nan"), "n": held_report["n"]})
    table(rows, ["task", "probe", "majority", "n"])

    # ------------------------------------------------------------------
    # Documents: is the corpus explained by an n-gram model?
    # ------------------------------------------------------------------
    from cgr.config import Config
    cfg = Config.preset(args.preset)
    tok, splits = build_corpus(cfg)
    train_texts = [" ".join(r.blocks) for r in splits["train"]]
    test_texts = [" ".join(r.blocks) for r in splits["test"]]
    held_texts = [" ".join(r.blocks) for r in splits["heldout"]]

    ng = NgramLM(order=2).fit(train_texts)
    ng_test = sum(ng.logprob(t) for t in test_texts) / max(len(test_texts), 1)
    ng_held = sum(ng.logprob(t) for t in held_texts) / max(len(held_texts), 1)

    # unique-template coverage: how many test docs are exact duplicates of a
    # training doc?
    train_set = set(train_texts)
    dup_rate = sum(1 for t in test_texts if t in train_set) / max(len(test_texts), 1)

    print(f"\n  n-gram logprob/test-doc: {ng_test:.3f}")
    print(f"  n-gram logprob/heldout-doc: {ng_held:.3f}")
    print(f"  exact duplicate rate test->train: {dup_rate:.3f}")

    # ------------------------------------------------------------------
    # Optional: the trained model's own heads, for comparison
    # ------------------------------------------------------------------
    model_report = None
    from _common import ckpt_dir
    ckpt = ckpt_dir(args.preset, args.seed) / "stage_b.pt"
    if ckpt.exists():
        system, _ = get_system(args.preset, seed=args.seed,
                           checkpoint=args.checkpoint)
        eps = episodes_from_claims(test_c, system.tok)
        batch = collate(eps, system.tok, cfg.model.max_len)
        with torch.no_grad():
            out = system.forward_episodes(batch["ids"], batch["clock"])
            preds = system.heads.predict(out["feats"])
        scope_acc = (preds["scope"] == batch["labels"]["scope"]).float().mean().item()
        epi_acc = (preds["epistemic"] == batch["labels"]["epistemic"]).float().mean().item()
        model_report = {"scope_acc": scope_acc, "epistemic_acc": epi_acc}
        print(f"\n  model scope head: {scope_acc:.3f}  (probe {scope_probe['probe_acc']:.3f})")
        print(f"  model epistemic head: {epi_acc:.3f}  (probe {epi_probe['probe_acc']:.3f})")

    # ------------------------------------------------------------------
    # verdicts
    # ------------------------------------------------------------------
    notes = []
    if scope_probe["probe_acc"] >= 0.90:
        notes.append(
            f"Scope is surface-solvable: a linear probe reaches "
            f"{scope_probe['probe_acc']:.2f}. The scope head cannot be claimed "
            f"to represent scope; at best it learns the marker token.")
    if stripped_probe["probe_acc"] <= 0.35:
        notes.append(
            "Without the marker the cue is information-theoretically absent, "
            "so the stripped condition measures the default-scope prior, not "
            "scope inference. Report it as such.")
    if dup_rate > 0.05:
        notes.append(
            f"{dup_rate:.1%} of test documents are exact duplicates of training "
            f"documents. Deduplicate before reporting repair results.")

    if model_report:
        gap = model_report["scope_acc"] - scope_probe["probe_acc"]
        notes.append(
            f"Model scope head is {gap:+.3f} relative to the probe; a negative "
            f"gap means the architecture is worse than a linear probe.")
    if epi_probe["probe_acc"] >= 0.90:
        notes.append(
            f"Epistemic status is also surface-solvable (probe "
            f"{epi_probe['probe_acc']:.2f}). The head is recovering template "
            f"form, not epistemic state.")

    write_result("exp0_benchmark_validity", {
        "experiment": "exp0_benchmark_validity",
        "preset": args.preset, "seed": args.seed, "n": n,
        "preregistered_prediction": PREDICTION,
        "conditions": {
            "scope_probe": scope_probe,
            "scope_probe_stripped": stripped_probe,
            "epistemic_probe": epi_probe,
            "heldout_scope_probe": held_report,
            "ngram_logprob_test": ng_test,
            "ngram_logprob_heldout": ng_held,
            "exact_duplicate_rate": dup_rate,
            "model": model_report,
        },
        "summary": {
            "scope_surface_solvable": scope_probe["probe_acc"] >= 0.90,
            "benchmark_usable": dup_rate <= 0.05,
        },
        "notes": notes,
    })
    print("\n" + "\n".join(f"  ! {x}" for x in notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Experiment 1 — design invariants, reported as invariants.

These are not empirical claims and must not appear in the results tables as
if they were. They are properties of the implementation, verified over many
inputs, and they include a *negative control* so the test is shown to have
power.

  1. Null property     M = empty  =>  bit-identical output
  2. Rollback          insert then remove => bit-identical
  3. Gradient masking  verified-wrong => plastic substrate bit-identical
  4. Isolation         the side channel has no path to the vocabulary

The negative control (1b) installs a deliberately leaky memory and shows the
null test fails. Without it, "100/100 bit-identical" is unfalsifiable.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from _common import arg_parser, banner, get_system, table, write_result
from cgr.data.corpus import build_corpus
from cgr.train import build_episodes, collate

PREDICTION = {
    "null_property": "100/100 bit-identical",
    "rollback": "100/100 bit-identical",
    "negative_control": "the leaky memory MUST fail the null test",
    "gradient_masking": "0 updates on verified-wrong, across 50 repeats",
    "isolation": "no Linear layer in the reasoner projects to vocab",
}


def bit_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a, b)


def main() -> int:
    ap = arg_parser("Experiment 1: design invariants")
    args = ap.parse_args()
    n = args.n or 100

    banner("Experiment 1 — design invariants")
    system, cfg = get_system(args.preset, seed=args.seed,
                             checkpoint=args.checkpoint)
    system.eval()
    tok, splits = build_corpus(cfg)

    # a batch of n distinct inputs
    eps = build_episodes(cfg, tok, splits, n_claims=n, n_repairs=n // 2,
                         seed=args.seed)[:n]
    batch = collate(eps, tok, cfg.model.max_len)
    x, clock = batch["ids"], batch["clock"]
    d = cfg.model.d_model

    # alpha is zero after training in most runs; force it non-zero so the null
    # test exercises the skip path rather than multiplication by zero.
    system.memory.alpha.data.fill_(0.5)

    # ------------------------------------------------------------------
    # 1. null property
    # ------------------------------------------------------------------
    system.memory.clear()
    with torch.no_grad():
        base = system.forward_episodes(x, clock, memory_texts=None)
    results = {}
    for name, fn in [
        ("empty_memory", lambda: system.forward_episodes(
            x, clock, memory_texts=["x y z"] * len(x))),
        ("empty_memory_repeated", lambda: system.forward_episodes(
            x, clock, memory_texts=["x y z"] * len(x))),
    ]:
        with torch.no_grad():
            out = fn()
        ok = bit_identical(base["feats"], out["feats"]) and \
            bit_identical(base["preds"]["value"], out["preds"]["value"])
        results[name] = {"bit_identical": ok, "n": len(x)}

    # ------------------------------------------------------------------
    # 1b. negative control: is the read path inert, or is the guard doing
    #     the work? With a non-empty store the same read path must move the
    #     output. If it does not, the null property is vacuous.
    # ------------------------------------------------------------------
    real_recall = system.memory.recall

    def leaky_recall(text, k=8):
        return torch.randn(d) * 1e-3, 1.0, None

    system.memory.clear()
    # Deliberate non-factual entry: the guard refuses unverified writes, and
    # this control needs the store to be non-empty so the recall path is
    # exercised at all.
    dummy = system.update_memory("dummy entry", verified=False,
                                 allow_unverified=True)
    system.memory.recall = leaky_recall
    with torch.no_grad():
        leaky = system.forward_episodes(x, clock, memory_texts=["x y z"] * len(x))
    leak_detected = not bit_identical(base["feats"], leaky["feats"])
    system.memory.recall = real_recall
    system.memory.clear()

    # ------------------------------------------------------------------
    # 2. rollback
    # ------------------------------------------------------------------
    system.memory.clear()
    added = [system.update_memory(f"memory entry number {i}", verified=True)
             for i in range(20)]
    with torch.no_grad():
        populated = system.forward_episodes(x, clock,
                                            memory_texts=["memory entry number 3"] * len(x))
    for eid in added:
        system.memory.remove(eid)
    with torch.no_grad():
        after = system.forward_episodes(x, clock,
                                        memory_texts=["memory entry number 3"] * len(x))
    rollback_ok = bit_identical(base["feats"], after["feats"])
    memory_has_effect = not bit_identical(base["feats"], populated["feats"])

    # ------------------------------------------------------------------
    # 3. gradient masking, repeated
    # ------------------------------------------------------------------
    system.freeze_all_but_plastic()
    opt = torch.optim.SGD(system.plastic_params(), lr=0.05)
    repeats = 50
    unchanged = 0
    for _ in range(repeats):
        snap = system.plastic_snapshot()
        system.consolidate(x, clock, batch["labels"],
                           verified=[False] * len(x),
                           advantage=[1.0] * len(x), optim=opt)
        unchanged += int(system.plastic_unchanged(snap))

    # positive control: same call with verified=True must move it
    snap = system.plastic_snapshot()
    system.consolidate(x, clock, batch["labels"], verified=[True] * len(x),
                       advantage=[1.0] * len(x), optim=opt)
    masking_has_power = not system.plastic_unchanged(snap)

    # ------------------------------------------------------------------
    # 4. structural isolation
    # ------------------------------------------------------------------
    vocab = cfg.model.vocab_size
    leaks = [n_ for n_, m in system.reasoner.named_modules()
             if isinstance(m, torch.nn.Linear) and m.out_features == vocab]

    # ------------------------------------------------------------------
    rows = [
        {"invariant": "null property", "result": results["empty_memory"]["bit_identical"],
         "n": len(x), "note": "M=empty, alpha=0.5"},
        {"invariant": "null (repeat)", "result": results["empty_memory_repeated"]["bit_identical"],
         "n": len(x), "note": "deterministic"},
        {"invariant": "negative control", "result": leak_detected,
         "n": len(x), "note": "non-empty store MUST move output"},
        {"invariant": "rollback", "result": rollback_ok, "n": len(x),
         "note": "insert 20, remove 20"},
        {"invariant": "memory has effect", "result": memory_has_effect,
         "n": len(x), "note": "populated != empty"},
        {"invariant": "gradient masking", "result": unchanged == repeats,
         "n": repeats, "note": f"{unchanged}/{repeats} unchanged"},
        {"invariant": "masking has power", "result": masking_has_power,
         "n": 1, "note": "verified-right DOES move it"},
        {"invariant": "isolation", "result": len(leaks) == 0, "n": 1,
         "note": f"{len(leaks)} vocab projections in reasoner"},
    ]
    table(rows, ["invariant", "result", "n", "note"])

    passed = all(r["result"] for r in rows)
    write_result("exp1_invariants", {
        "experiment": "exp1_invariants",
        "preset": args.preset, "seed": args.seed, "n": len(x),
        "preregistered_prediction": PREDICTION,
        "conditions": results,
        "summary": {
            "null_property": results["empty_memory"]["bit_identical"],
            "negative_control_detected_leak": leak_detected,
            "rollback": rollback_ok,
            "memory_has_effect": memory_has_effect,
            "gradient_masking_repeats_unchanged": unchanged,
            "masking_has_power": masking_has_power,
            "reasoner_vocab_leaks": leaks,
            "all_passed": passed,
        },
        "notes": [
            "These are design invariants, not empirical findings. They belong "
            "in a 'Design invariants' subsection, not in §5's results tables.",
            "The null property is guaranteed by the code path that skips the "
            "memory read when the store is empty. The negative control shows "
            "the test detects a leak, so the guarantee is not vacuous.",
        ],
    })
    print(f"\n  all invariants passed: {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

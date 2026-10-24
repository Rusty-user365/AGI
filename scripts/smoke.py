"""End-to-end smoke test on the `smoke` preset. Must pass before anything else.

Checks, in order:
  1. corpus builds and is exactly parseable / consistent
  2. defects break consistency, and non-local defects defeat a one-block budget
  3. model forward shapes
  4. training runs and loss decreases
  5. null property: empty memory is bit-identical
  6. rollback: insert then remove is bit-identical
  7. gradient masking: verified-wrong never moves the plastic substrate
  8. side channel is structurally isolated from the vocabulary
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cgr.config import Config
from cgr.data.corpus import (build_corpus, corrupt_local, corrupt_nonlocal,
                             make_repair_set)
from cgr.data.documents import (anchor_ok, coherence, quality,
                                task_from_blocks)
from cgr.data.facts import make_claim_set
from cgr.data.grammar import is_consistent, parse_blocks
from cgr.model.interpreter import adjudicate
from cgr.model.system import CGRSystem
from cgr.train import build_episodes, collate, train

FAILED = []


def check(name: str, cond: bool, extra: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {extra}")
    if not cond:
        FAILED.append(name)


def main() -> int:
    t0 = time.time()
    cfg = Config.preset("smoke")
    print(f"\n== corpus ==")
    tok, splits = build_corpus(cfg)
    cfg.model.vocab_size = max(cfg.model.vocab_size, tok.vocab_size)
    print(f"  vocab={tok.vocab_size} train={len(splits['train'])} "
          f"val={len(splits['val'])} heldout={len(splits['heldout'])}")

    bad = [r for r in splits["train"][:200] if not is_consistent(r.blocks)]
    check("clean docs parse as consistent", len(bad) == 0, f"({len(bad)} bad)")

    p = parse_blocks(splits["train"][0].blocks)
    check("parser recovers thesis and count",
          p.ok and p.thesis is not None and p.n_intro == len(p.body_theses),
          f"(n={p.n_intro}, bodies={len(p.body_theses)})")

    # --- defects -------------------------------------------------------
    import random
    rng = random.Random(0)
    r = splits["train"][0]
    loc = corrupt_local(r.blocks, rng)
    coh = coherence(loc.blocks, loc.authoritative)
    check("local defect keeps coherence, loses quality",
          is_consistent(loc.corrupt_blocks) and coh["c_edit"] == 1.0
          and quality(loc.blocks, loc.authoritative) < 1.0,
          f"(c_edit={coh['c_edit']:.2f}, "
          f"quality={quality(loc.blocks, loc.authoritative):.2f})")
    nl = corrupt_nonlocal(r.blocks, rng, r.thesis_id)
    check("cross-block defect is incoherent under T*",
          is_consistent(r.blocks) and not is_consistent(nl.corrupt_blocks)
          and coherence(nl.blocks, nl.authoritative)["c_edit"] == 0.0)
    check("cross-block corruption preserves the locked intro",
          anchor_ok(r.blocks, nl.corrupt_blocks, [0]))
    # a one-block budget cannot fix the cross-block case
    fixed_one = nl.texts()
    for i in nl.editable_indices:
        fixed_one[i] = nl.clean_blocks[i]
        c1 = coherence(task_from_blocks(fixed_one, nl.authoritative).blocks,
                       nl.authoritative)
        check(f"single edit of block {i} leaves contradictions",
              c1["c_edit"] < 1.0, f"(c_edit={c1['c_edit']:.2f})")
        fixed_one = nl.texts()
    fixed_all = nl.texts()
    for i in nl.editable_indices:
        fixed_all[i] = nl.clean_blocks[i]
    check("revising all editable blocks restores coherence",
          coherence(task_from_blocks(fixed_all, nl.authoritative).blocks,
                    nl.authoritative)["c_edit"] == 1.0
          and anchor_ok(nl.texts(), fixed_all, [0]))

    # --- interpreter ---------------------------------------------------
    print(f"\n== interpreter ==")
    claims = make_claim_set(40, seed=1)
    verd = [adjudicate(c.spec, c.marker_present) for c in claims]
    adjudicable = [v for c, v in zip(claims, verd) if c.family != "empirical"]
    empirical = [v for c, v in zip(claims, verd) if c.family == "empirical"]
    check("all scoped claims are adjudicable",
          all(v.correct is not None for v in adjudicable))
    check("empirical claims are not adjudicable (correct is None)",
          all(v.correct is None and v.mode.value == "refuse" for v in empirical))
    base4 = {"family": "base", "base": 4, "a": 2, "c": 2, "claimed": 10}
    check("2+2=10 is true in base 4", adjudicate(base4, True).correct is True)
    base10 = {"family": "base", "base": 10, "a": 2, "c": 2, "claimed": 10}
    check("2+2=10 is false in base 10", adjudicate(base10, True).correct is False)
    rnd = {"family": "rounding", "m": 5, "a": 2, "c": 2, "claimed": 5}
    check("2+2=5 is true under round-up-to-5", adjudicate(rnd, True).correct is True)
    check("2+2=5 is false in default scope", adjudicate(rnd, False).correct is False)
    cnt = {"family": "count", "males": 2, "females": 2, "pregnant": 1,
           "include_unborn": True, "claimed": 5}
    check("count incl. unborn: 5 is true", adjudicate(cnt, True).correct is True)
    bare = adjudicate(rnd, False)
    check("bare claim is flagged as scope-assumed", bare.scope_assumed is True)

    # --- model ---------------------------------------------------------
    print(f"\n== model ==")
    system = CGRSystem(cfg, tok)
    params = system.n_params()
    print(f"  params: {params}")
    eps = build_episodes(cfg, tok, splits, n_claims=80, n_repairs=40, seed=0)
    batch = collate(eps[:8], tok, cfg.model.max_len)
    out = system.forward_episodes(batch["ids"], batch["clock"], hard=True)
    check("features have the expected width",
          out["feats"].shape == (8, system.feat_dim),
          f"{tuple(out['feats'].shape)}")
    check("all seven heads produce outputs",
          set(out["preds"].keys()) == {"value", "outcome", "tau", "epistemic",
                                       "scope", "source", "disagreement"})
    check("gate produces 8 action logits", out["action_logits"].shape == (8, 8))

    # --- null property -------------------------------------------------
    print(f"\n== invariants ==")
    x = batch["ids"]
    clock = batch["clock"]
    system.eval()
    # alpha is learned and starts at zero ("no opinion"). For the invariant
    # tests we force it non-zero, so the null property is testing the skip
    # path rather than the arithmetic of multiplying by zero.
    system.memory.alpha.data.fill_(0.5)
    with torch.no_grad():
        base = system.forward_episodes(x, clock, memory_texts=None)
        system.memory.clear()
        empty = system.forward_episodes(x, clock, memory_texts=["a b c"] * 8)
    check("null property: empty memory is bit-identical",
          torch.equal(base["feats"], empty["feats"]))

    # rollback
    ids_added = []
    for i in range(4):
        ids_added.append(system.update_memory(f"doc number {i}", verified=True))
    with torch.no_grad():
        with_mem = system.forward_episodes(x, clock,
                                           memory_texts=["doc number 0"] * 8)
    for eid in ids_added:
        system.memory.remove(eid)
    with torch.no_grad():
        after = system.forward_episodes(x, clock, memory_texts=["doc number 0"] * 8)
    check("rollback: removing all entries restores bit-identity",
          torch.equal(base["feats"], after["feats"]))
    check("memory actually changes output when populated",
          not torch.equal(base["feats"], with_mem["feats"]))

    # --- structural isolation -----------------------------------------
    vocab = cfg.model.vocab_size
    leaks = [n for n, mod in system.reasoner.named_modules()
             if isinstance(mod, torch.nn.Linear) and mod.out_features == vocab]
    check("side channel has no path to the vocabulary", len(leaks) == 0,
          str(leaks))

    # --- gradient masking ---------------------------------------------
    print(f"\n== plasticity ==")
    system.freeze_all_but_plastic()
    opt = torch.optim.SGD(system.plastic_params(), lr=0.1)
    snap = system.plastic_snapshot()
    stats = system.consolidate(batch["ids"], batch["clock"], batch["labels"],
                               verified=[False] * 8, advantage=[1.0] * 8, optim=opt)
    check("verified-wrong: no update (gradient masked)",
          system.plastic_unchanged(snap) and stats["n_updated"] == 0)

    stats2 = system.consolidate(batch["ids"], batch["clock"], batch["labels"],
                                verified=[True] * 8, advantage=[1.0] * 8, optim=opt)
    check("verified-right + advantage: substrate updates",
          not system.plastic_unchanged(snap) and stats2["n_updated"] == 8)

    stats3 = system.consolidate(batch["ids"], batch["clock"], batch["labels"],
                                verified=[True] * 8, advantage=[0.0] * 8, optim=opt)
    check("below advantage threshold: no update", stats3["n_updated"] == 0)

    # --- training ------------------------------------------------------
    print(f"\n== training (smoke) ==")
    summary = train(cfg, "checkpoints/smoke", verbose=True)
    first = summary["history"]["stage_a"][0]["loss"]
    last = summary["history"]["stage_a"][-1]["loss"]
    check("stage A loss decreases", last < first, f"({first:.3f} -> {last:.3f})")

    print(f"\n{time.time() - t0:.1f}s")
    if FAILED:
        print(f"\nFAILED: {FAILED}")
        return 1
    print("\nall smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

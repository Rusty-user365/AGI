"""Experiment 4 — revision scope under a matched budget.

The task contract is explicit and is part of the data, not inferred from
whichever solution happens to look coherent:

    The introduction is the authoritative, immutable thesis T*. All editable
    blocks must be revised to support it.

Defect types are different in kind:

  local                one body block loses specificity/completeness but still
                       supports T*. Coherence is intact; the defect is quality.
  cross_block          the locked intro states T*, while every editable block
                       supports a single wrong thesis W. The defect is
                       coherence.
  cross_block_partial  the harder distractor: W is on a *strict majority* of
                       the editable blocks (always including the conclusion)
                       while at least one body still supports T*. A
                       majority-vote heuristic therefore sides with the
                       corruption, and only the locked intro identifies T*.

Methods, all measured against the same contract:

  one_block       renoise the single block a detector flags
  global          renoise every editable block (the intro is never masked)
  repeated_local  one local edit at a time, worst block first, until the
                  budget is exhausted or c_edit reaches 1

`global` and `repeated_local` receive the *same* block and token allowance;
`one_block` receives one block. That is what makes the comparison about scope
rather than about permission to edit more text.

Protocol (frozen before measurement):

  1. oracle solvability gate. A deterministic editor with ground truth must
     reach the target under each method's budget. If it cannot, the task or
     the budget is wrong and no model number is interpretable.
  2. model grid over defect x method, with the realized budget reported.
  3. gate-selected condition: does the trained gate choose a strategy that is
     actually useful, or does it merely pick the one that edits more text?

Metrics (all reported together, because polish is not success):

  anchor_preserved  the locked intro is byte-identical
  quality           fraction of body blocks that are specific about T*
  C(D)              synthetic label-consistency: mean over required blocks of
                    1[thesis(b) == T*]. `c_all` includes the locked intro, so a
                    no-op on a cross-block task floors at 1/n; `c_edit` is over
                    editable blocks and is the metric that can move.
  blocks_changed, tokens_spent   realized revision budget
  improvement, preservation, seam   backbone NLL-based edit quality

The defensible claim is conditional: revision scope should follow the location
and dependency structure of the defect *and* the available budget. It is not
"local repair can never fix a non-local defect" -- `repeated_local` is run
specifically to test that.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# `_common` lives next to the experiment scripts, not in the package. Adding
# the directory explicitly is what lets `experiments/exp8_...py` import this
# module as `experiments.exp4_repair_taxonomy`; without it the import fails
# with `ModuleNotFoundError: _common` even though the script works when run
# directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from _common import arg_parser, banner, get_system, table, write_result
from cgr.data.corpus import build_corpus
from cgr.data.documents import (anchor_ok, block_quality, coherence,
                                make_task_set, matched_budgets, oracle_repair,
                                quality, task_from_blocks)
from cgr.data.grammar import blocked_spans, is_consistent
from cgr.metrics import bootstrap_ci, paired_bootstrap, score_repair
from cgr.model.gate import ACTIONS

METHODS = ("one_block", "global", "repeated_local")
DEFECTS = ("local", "cross_block", "cross_block_partial")

PREDICTION = {
    "local/one_block": "c_edit -> 1.0 and quality -> 1.0 at a one-block budget",
    "local/global": "no better than one_block; the extra scope is wasted",
    "cross_block/one_block": "c_edit stays < 1.0: at least one contradiction "
                             "remains because the other editable blocks are "
                             "untouched",
    "cross_block/global": "c_edit > cross_block/one_block; revising all "
                          "editable blocks can restore label consistency",
    "cross_block/repeated_local": "MEASURED, not assumed. With an oracle "
                                   "detector and a matched budget it may also "
                                   "reach 1.0; if so the claim is about cost "
                                   "and scope, not possibility",
    "cross_block_partial/one_block": "c_edit stays < 1.0 as well: the wrong "
                                     "thesis is on a strict majority of "
                                     "editable blocks, so fixing one leaves "
                                     "the rest wrong",
    "cross_block_partial/global": "c_edit = 1.0: the locked intro is the only "
                                  "block that identifies T*, so a revision "
                                  "must take it as given",
    "cross_block_partial/repeated_local": "MEASURED. May reach 1.0 if the "
                                          "budget allows one edit per wrong "
                                          "block; if so the difference from "
                                          "global is cost, not possibility",
    "anchor": "preserved in every condition; editing the intro is a contract "
              "violation, not a repair",
    "gate": "the trained gate's choice is at least as good as always-one-block "
            "and is reported next to always-global",
}

# Oracle gate expectations. These are checked on the *produced* document, not
# on the task's own flags.
ORACLE_EXPECT = {
    ("local", "one_block"): {"c_edit": 1.0, "quality": 1.0},
    ("local", "global"): {"c_edit": 1.0, "quality": 1.0},
    ("local", "repeated_local"): {"c_edit": 1.0, "quality": 1.0},
    ("cross_block", "one_block"): {"c_edit_lt": 1.0},
    ("cross_block", "global"): {"c_edit": 1.0, "quality": 1.0},
    ("cross_block", "repeated_local"): {"c_edit": 1.0, "quality": 1.0},
    ("cross_block_partial", "one_block"): {"c_edit_lt": 1.0},
    ("cross_block_partial", "global"): {"c_edit": 1.0, "quality": 1.0},
    ("cross_block_partial", "repeated_local"): {"c_edit": 1.0, "quality": 1.0},
}


# --------------------------------------------------------------------------
# Scoring a produced document
# --------------------------------------------------------------------------

def score_document(task, texts: list[str],
                   specificity_defined: bool = True) -> dict:
    """Score the *actual* document. Roles are positional, block 0 is locked.

    Two families of numbers, because they answer different questions:

      c_edit / quality   document-level coherence. For a local defect a no-op
                         already scores c_edit 1.0 (the degraded block keeps the
                         thesis) and quality (k-1)/k (the untouched siblings are
                         still specific), so these cannot say whether the
                         *edited* block was repaired.
      c_target / q_target  the same scores restricted to the blocks the task
                         actually edited. For a local defect that is one block;
                         for a cross-block defect it is every editable block.
                         This is the only pair that can attribute a repair to
                         the editor rather than to the siblings it left alone.

    `specificity_defined=False` is for one case only: a local defect on the
    `unseen` split. The degraded block's reason was replaced, and the thesis's
    reason pool never appeared in training, so a model cannot know which
    strings count as specific. A repair could still score by copying a sibling
    block's reason, but that measures evidence reuse rather than specificity,
    and reporting 0.0 would read as "the editor cannot repair this" when the
    task is under-specified for it. The value is therefore reported as
    undefined (None) with the reason attached, not as zero.
    """
    blocks = task_from_blocks(texts, task.authoritative).blocks
    coh = coherence(blocks, task.authoritative)
    bodies = [i for i, b in enumerate(blocks) if b.role == "body"]
    edited = task.edited or [task.target_block]
    c_target = sum(1 for i in edited
                   if blocks[i].thesis_id == task.authoritative) / len(edited)
    if specificity_defined:
        q_target = sum(block_quality(blocks[i], task.authoritative)
                       for i in edited) / len(edited)
        q_all = quality(blocks, task.authoritative)
        q_target_block = quality(blocks, task.authoritative,
                                 [task.target_block] if bodies else [])
    else:
        q_target = q_all = q_target_block = None
    return {
        "anchor": bool(anchor_ok(task.texts(), texts, task.locked_indices)),
        "c_all": coh["c_all"],
        "c_edit": coh["c_edit"],
        "floor_all": coh["floor_all"],
        "c_target": c_target,
        "q_target": q_target,
        "quality": q_all,
        "quality_target": q_target_block,
        "consistent": bool(is_consistent(texts)),
        "off_thesis_editable": len(coh["off_thesis_editable"]),
        "n_editable": coh["n_editable"],
    }


def n_tokens_of(tok):
    return lambda s: len(tok.encode(s, add_bos=False, add_eos=False))


# --------------------------------------------------------------------------
# Model editor under a budget
# --------------------------------------------------------------------------

def run_model(system, task, method: str, budget, n_tok, tok,
              steps: int = 6, seed: int = 0,
              use_reasoner: bool = False,
              reasoning_mode: str | None = None) -> tuple[list[str], list[int]]:
    """Apply one method to one task without exceeding its budget.

    `repeated_local` uses a ground-truth detector (worst block first). That
    deliberately favours local repair: if it still loses to global revision,
    the loss is a scope effect, not a detection artifact.

    `reasoning_mode` selects what drives the FiLM side channel: `none` (cond=0),
    `full` (per-document reasoner vector) or `learned` (the checkpoint's own
    `const_cond`). The last one is the in-distribution path for a checkpoint
    trained with `cond_mode=learned`; without it that control is evaluated
    outside the distribution it was trained in.
    """
    out = task.texts()
    changed: list[int] = []

    if method == "one_block":
        i = task.target_block
        if budget.can_spend(1, n_tok(out[i])):
            new = system.repair(out, i, mode="local", steps=steps, seed=seed,
                                use_reasoner=use_reasoner,
                                reasoning_mode=reasoning_mode)
            budget.spend_or_flag(1, n_tok(new[i]))
            out, changed = new, [i]

    elif method == "global":
        idx = task.editable_indices
        cost = sum(n_tok(out[i]) for i in idx)
        if budget.can_spend(len(idx), cost):
            new = system.repair(out, 0, mode="editable", steps=steps, seed=seed,
                                use_reasoner=use_reasoner,
                                reasoning_mode=reasoning_mode)
            budget.spend_or_flag(len(idx), sum(n_tok(new[i]) for i in idx))
            out, changed = new, list(idx)

    elif method == "repeated_local":
        def badness(i: int) -> tuple[int, int]:
            b = task.blocks[i]
            return (b.thesis_id == task.authoritative,
                    block_quality(b, task.authoritative))

        for i in sorted(task.editable_indices, key=badness):
            if not budget.can_spend(1, n_tok(out[i])):
                break
            new = system.repair(out, i, mode="local", steps=steps, seed=seed,
                                use_reasoner=use_reasoner,
                                reasoning_mode=reasoning_mode)
            budget.spend_or_flag(1, n_tok(new[i]))
            out, changed = new, changed + [i]
            if coherence(task_from_blocks(out, task.authoritative).blocks,
                         task.authoritative)["c_edit"] == 1.0:
                break

    else:
        raise ValueError(method)
    return out, changed


def gate_choice(system, task, tok) -> str:
    """The trained gate's action on the corrupted document."""
    ids, _ = blocked_spans(task.texts(), tok)
    t = torch.tensor(ids).unsqueeze(0)
    clock = torch.tensor([[0.0, 0.0, 6.0, 10.0, 0.0, 1.0, 0.0, 0.6]])
    with torch.no_grad():
        out = system.forward_episodes(t, clock)
    return ACTIONS[int(out["action_logits"][0].argmax())]


def edit_quality(system, task, before: list[str], after: list[str],
                 tok, spans_idx: list[int]) -> dict:
    """NLL-based improvement / preservation / seam for the produced edit.

    Both span lists are passed so the comparison is block-aligned; a rewrite
    that changes a block's length must not be scored as damage to everything
    after it. Spans are matched by block index, so the two lists must have the
    same length, which they do because roles are positional.
    """
    orig_ids, orig_spans = blocked_spans(before, tok)
    new_ids, new_spans = blocked_spans(after, tok)
    idx = [i for i in spans_idx if i < len(orig_spans) and i < len(new_spans)]
    o = torch.tensor(orig_ids).unsqueeze(0)
    nw = torch.tensor(new_ids).unsqueeze(0)
    sc = score_repair(system.backbone, o, nw, [orig_spans[i] for i in idx],
                      [new_spans[i] for i in idx], is_consistent(after))
    return {"improvement": sc.improvement, "preservation": sc.preservation,
            "seam": sc.seam}


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def summarise(rows: list[dict]) -> dict:
    n = max(len(rows), 1)

    def mean(k):
        vals = [r[k] for r in rows if r.get(k) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "anchor": mean("anchor"), "quality": mean("quality"),
        "c_all": mean("c_all"), "c_edit": mean("c_edit"),
        "c_target": mean("c_target"), "q_target": mean("q_target"),
        "noop_c_target": mean("noop_c_target"),
        "noop_q_target": mean("noop_q_target"),
        "floor_all": mean("floor_all"),
        "blocks_changed": mean("blocks_changed"),
        "tokens_spent": mean("tokens_spent"),
        "consistent": mean("consistent"),
        "improvement": mean("improvement"),
        "preservation": mean("preservation"),
        "seam": mean("seam"),
        "over_budget": mean("over_budget"),
        "n": len(rows),
        # how many rows actually contributed a specificity score; < n only on
        # the local/unseen condition, where the target is under-specified.
        "n_quality": len([r for r in rows if r.get("quality") is not None]),
    }


def check_expect(agg: dict, expect: dict) -> bool:
    for k, v in expect.items():
        if k == "c_edit_lt" and not (agg["c_edit"] < v):
            return False
        if k in ("c_edit", "quality") and abs(agg[k] - v) > 1e-9:
            return False
    return True


def main() -> int:
    ap = arg_parser("Experiment 4: revision scope under a matched budget")
    args = ap.parse_args()
    n = args.n or 40

    banner("Experiment 4 — revision scope under a matched budget")
    system, cfg = get_system(args.preset, seed=args.seed,
                             checkpoint=args.checkpoint)
    system.eval()
    # Use the checkpoint's tokenizer, not a freshly built one: for the word
    # tokenizer the vocabulary is derived from the training documents, so a
    # corpus rebuilt after a grammar change would disagree with the weights.
    # For the pre-trained BPE tokenizer this is a no-op.
    tok, splits = build_corpus(cfg, tok=system.tok)
    n_tok = n_tokens_of(tok)

    selection: dict = {}
    split = getattr(args, "split", "test")
    records = splits.get(split) or []
    if not records:
        raise SystemExit(
            f"split {split!r} is empty. `unseen` requires "
            f"DataConfig.n_heldout_theses > 0 and an open-vocabulary "
            f"tokenizer (the `diffugpt` preset sets both). The blank canvas "
            f"builds a word vocabulary from train and cannot encode an unseen "
            f"thesis, so it has no unseen split by construction.")
    tasks = make_task_set(
        records, n, seed=args.seed,
        fits=lambda texts: len(blocked_spans(texts, tok)[0])
        <= cfg.model.max_len,
        stats=selection)
    selection["split"] = split
    n_used = {d: len(tasks[d]) for d in DEFECTS}
    n_dropped = {d: n - n_used[d] for d in DEFECTS}

    def score(task, texts):
        """Specificity is under-specified for a local defect on unseen theses.

        The degraded block's reason was replaced and the thesis's reason pool
        never appeared in training, so the model cannot know which strings are
        specific. See `score_document` for why this is reported as undefined
        rather than zero.
        """
        return score_document(
            task, texts,
            specificity_defined=not (split == "unseen"
                                     and task.defect == "local"))

    # ------------------------------------------------------------------
    # 1. oracle solvability gate — before any model number is reported
    # ------------------------------------------------------------------
    oracle_rows: dict[str, list[dict]] = {}
    oracle_agg: dict[str, dict] = {}
    for defect in DEFECTS:
        for method in METHODS:
            rows = []
            for task in tasks[defect]:
                budget = matched_budgets(task, n_tok)[method]
                out, spend = oracle_repair(task, method, budget, n_tok)
                sc = score_document(task, out)
                sc["blocks_changed"] = spend["blocks_changed"]
                sc["tokens_spent"] = spend["budget"]["tokens_used"]
                rows.append(sc)
            oracle_rows[f"{defect}/{method}"] = rows
            oracle_agg[f"{defect}/{method}"] = summarise(rows)

    gate = {}
    for key, expect in ORACLE_EXPECT.items():
        gate[f"{key[0]}/{key[1]}"] = check_expect(oracle_agg[f"{key[0]}/{key[1]}"],
                                                  expect)
    gate["anchor_preserved_all"] = all(
        a["anchor"] == 1.0 for a in oracle_agg.values())
    oracle_solvable = all(gate.values())

    print("\n  oracle solvability gate (deterministic editor, ground truth)")
    table([{"condition": k, **v} for k, v in oracle_agg.items()],
          ["condition", "anchor", "c_edit", "c_all", "quality",
           "blocks_changed", "tokens_spent", "n"])
    print("  gate: " + ", ".join(f"{k}={'ok' if v else 'FAIL'}"
                                 for k, v in gate.items()))

    # ------------------------------------------------------------------
    # 2. model grid
    # ------------------------------------------------------------------
    model_rows: dict[str, list[dict]] = {}
    model_agg: dict[str, dict] = {}
    for defect in DEFECTS:
        for method in METHODS:
            rows = []
            for task in tasks[defect]:
                budget = matched_budgets(task, n_tok)[method]
                before = task.texts()
                out, changed = run_model(system, task, method, budget, n_tok,
                                         tok, seed=args.seed)
                sc = score(task, out)
                noop = score(task, before)
                sc["noop_c_target"] = noop["c_target"]
                sc["noop_q_target"] = noop["q_target"]
                sc["blocks_changed"] = len(changed)
                sc["tokens_spent"] = budget.tokens_used
                sc["over_budget"] = budget.over_budget
                sc.update(edit_quality(system, task, before, out, tok,
                                       task.editable_indices))
                rows.append(sc)
            model_rows[f"{defect}/{method}"] = rows
            model_agg[f"{defect}/{method}"] = summarise(rows)

    # gate-selected condition: the gate picks REP (one block) or REV (all
    # editable). A no-op choice (ANS/DEF) spends nothing and is scored as-is.
    gate_rows: dict[str, list[dict]] = {}
    gate_agg: dict[str, dict] = {}
    for defect in DEFECTS:
        rows, choices = [], {}
        for task in tasks[defect]:
            a = gate_choice(system, task, tok)
            method = {"REP": "one_block", "REV": "global"}.get(a, "none")
            before = task.texts()
            if method == "none":
                out, changed, spent, over = before, [], 0, False
            else:
                budget = matched_budgets(task, n_tok)[method]
                out, changed = run_model(system, task, method, budget, n_tok,
                                         tok, seed=args.seed)
                spent, over = budget.tokens_used, budget.over_budget
            sc = score(task, out)
            noop = score(task, before)
            sc["noop_c_target"] = noop["c_target"]
            sc["noop_q_target"] = noop["q_target"]
            sc["blocks_changed"] = len(changed)
            sc["tokens_spent"] = spent
            sc["over_budget"] = over
            sc.update(edit_quality(system, task, before, out, tok,
                                   task.editable_indices))
            rows.append(sc)
            choices[a] = choices.get(a, 0) + 1
        gate_rows[defect] = rows
        gate_agg[defect] = summarise(rows)
        gate_agg[defect]["choices"] = choices

    print("\n  model grid")
    rows = [{"condition": k, **v} for k, v in model_agg.items()]
    rows += [{"condition": f"{d}/gate_selected", **gate_agg[d]}
             for d in DEFECTS]
    table(rows, ["condition", "anchor", "c_edit", "c_all", "c_target",
                 "q_target", "quality", "blocks_changed", "tokens_spent",
                 "consistent", "improvement", "preservation", "seam", "n"])
    noop_rows = [{"condition": f"{d}/no_op",
                  "c_target": model_agg[f"{d}/one_block"]["noop_c_target"],
                  "q_target": model_agg[f"{d}/one_block"]["noop_q_target"]}
                 for d in DEFECTS]
    print("\n  no-op baseline on the edited blocks (what doing nothing scores)")
    table(noop_rows, ["condition", "c_target", "q_target"])

    # ------------------------------------------------------------------
    # 3. paired comparisons on c_edit
    # ------------------------------------------------------------------
    def paired(defect, a, b):
        x = [r["c_edit"] for r in model_rows[f"{defect}/{a}"]]
        y = [r["c_edit"] for r in model_rows[f"{defect}/{b}"]]
        return paired_bootstrap(x, y)

    comps = {
        "local/global_minus_one_block": paired("local", "one_block", "global"),
        "cross_block/global_minus_one_block": paired("cross_block",
                                                     "one_block", "global"),
        "cross_block/repeated_local_minus_global": paired(
            "cross_block", "repeated_local", "global"),
        "cross_block/repeated_local_minus_one_block": paired(
            "cross_block", "one_block", "repeated_local"),
        "cross_block_partial/global_minus_one_block": paired(
            "cross_block_partial", "one_block", "global"),
        "cross_block_partial/repeated_local_minus_global": paired(
            "cross_block_partial", "repeated_local", "global"),
        "cross_block_partial/repeated_local_minus_one_block": paired(
            "cross_block_partial", "one_block", "repeated_local"),
        "local/repeated_local_minus_global": paired(
            "local", "repeated_local", "global"),
    }
    interaction = (comps["cross_block/global_minus_one_block"]["mean_diff"]
                   - comps["local/global_minus_one_block"]["mean_diff"])

    # ------------------------------------------------------------------
    # 4. infill sanity: can the backbone restore a masked clean block?
    # ------------------------------------------------------------------
    # Two different questions, and conflating them was a real bug in the
    # first version of this check:
    #
    #   copy fidelity   does the regenerated block equal the original
    #                   byte-for-byte? A masked-diffusion infill regenerates
    #                   from the mask, so this is ~0 at *any* training budget
    #                   even when the model is competent. It is a diagnostic,
    #                   not a pass gate.
    #   task ability    does the regenerated block parse and support the
    #                   document's thesis? This is the property the repair
    #                   grid actually needs, and it can be high.
    #
    # The no-op baseline is reported alongside because a clean block already
    # parses: the metric can be high without the model doing anything, so the
    # absolute number is not evidence on its own. What the sanity check rules
    # out is degenerate output that never parses, which is what a floor looks
    # like.
    from cgr.data.grammar import parse_block

    sanity_n = min(20, len(records))
    exact = parse_ok = thesis_ok = 0
    for r in records[:sanity_n]:
        idx = 1
        new_blocks = system.repair(r.blocks, idx, mode="local", steps=6,
                                   seed=args.seed)
        gen = new_blocks[idx]
        exact += int(gen.strip() == r.blocks[idx].strip())
        tid, _, _ = parse_block(gen, "body")
        parse_ok += int(tid is not None)
        thesis_ok += int(tid == r.thesis_id)
    infill_exact = exact / max(sanity_n, 1)
    infill_parse = parse_ok / max(sanity_n, 1)
    infill_on_thesis = thesis_ok / max(sanity_n, 1)

    # The no-op baseline on the same (clean) blocks: parse rate is 1.0 by
    # construction, which is why the gate below is on *on-thesis* output and
    # why the absolute number is reported with this baseline attached.
    noop_parse = noop_thesis = 0
    for r in records[:sanity_n]:
        tid, _, _ = parse_block(r.blocks[1], "body")
        noop_parse += int(tid is not None)
        noop_thesis += int(tid == r.thesis_id)
    noop_parse /= max(sanity_n, 1)
    noop_thesis /= max(sanity_n, 1)

    # ------------------------------------------------------------------
    # 5. interpretation, written as conditions that can fail
    # ------------------------------------------------------------------
    local_gap = comps["local/global_minus_one_block"]
    cross_gap = comps["cross_block/global_minus_one_block"]
    rep_vs_glob = comps["cross_block/repeated_local_minus_global"]
    rep_vs_one = comps["cross_block/repeated_local_minus_one_block"]

    scope_conditional = (
        cross_gap["mean_diff"] > 0 and cross_gap["lo"] > 0
        and local_gap["mean_diff"] <= abs(cross_gap["mean_diff"]))
    # The editor can only be compared if it can produce on-thesis text at all,
    # and repeated local can only be said to "match" global if global actually
    # moved the metric. Without this guard, a floor of zeros reads as
    # agreement. The gate is on task ability, not byte-identical copy: see the
    # comment in section 4.
    informative = infill_on_thesis >= 0.2
    global_moved = model_agg["cross_block/global"]["c_edit"] > 0.05
    repeated_local_suffices = (
        informative and global_moved and rep_vs_glob["mean_diff"] >= -0.05)

    notes = []
    if selection.get("skipped"):
        notes.append(
            f"Context filter: {selection['skipped']} of "
            f"{selection['scanned']} scanned documents were skipped because "
            f"a clean, local, cross-block or partial-cross-block variant "
            f"exceeds the model's {cfg.model.max_len}-token context. The "
            f"filter is applied to all variants together, so the defect "
            f"conditions stay paired. Kept: {n_used}. This is a selection "
            f"effect and is reported.")
    if any(v > 0 for v in n_dropped.values()):
        notes.append(
            f"The {split} pool was exhausted before reaching n={n} per "
            f"condition: {n_used}. Report the realized n, not the requested n.")
    if split == "unseen" and tasks["local"]:
        notes.append(
            "Specificity (q_target/quality) is undefined for local defects on "
            "the unseen split and is reported as None, not 0. The degraded "
            "block's reason was replaced and the unseen thesis's reason pool "
            "never appeared in training, so no model could know which strings "
            "are specific. Coherence (c_target/c_edit) is still defined, and "
            "the oracle gate below still scores specificity because it has "
            "ground truth.")
    if not oracle_solvable:
        failed = [k for k, v in gate.items() if not v]
        notes.append(
            f"ORACLE GATE FAILED ({failed}). The task or the budget is broken; "
            f"the model grid below is not interpretable. Do not report it as a "
            f"model result.")
    if not (gate["cross_block/one_block"]
            and gate["cross_block_partial/one_block"]):
        notes.append(
            "A cross-block generator is not actually non-local: a one-block "
            "oracle repair restored full coherence. The defect definition must "
            "be fixed before the comparison means anything.")
    if not informative:
        notes.append(
            f"Infill sanity: the backbone produces a parseable, on-thesis block "
            f"only {infill_on_thesis:.0%} of the time (parses "
            f"{infill_parse:.0%}; no-op baseline parses {noop_parse:.0%} and is "
            f"on-thesis {noop_thesis:.0%}; byte-identical copy "
            f"{infill_exact:.0%}). Below the 0.2 ability gate, every repair "
            f"number is at the floor; the scope comparison is NOT decidable "
            f"with this editor. The oracle gate above shows the task itself is "
            f"solvable, so this is a model failure, not a task failure. Report "
            f"the floor and do not interpret the grid or the gate-selected "
            f"row.")
    elif infill_exact < 0.2:
        notes.append(
            f"Infill sanity passes on task ability ({infill_on_thesis:.0%} "
            f"parseable and on-thesis, no-op on-thesis {noop_thesis:.0%}) but "
            f"not on byte-identical copy ({infill_exact:.0%}). Copy fidelity is "
            f"the wrong target for a generative infill: the model regenerates "
            f"the span instead of copying it. Both numbers are reported; the "
            f"grid is interpreted against the ability number only.")

    # The model grid must be read on the blocks the task actually edited.
    # Document-level c_edit/quality include siblings the method never touched,
    # so a no-op scores 1.0 / (k-1)/k on a local defect and a degenerate edit
    # can inherit that credit. This is the same class of bug as the earlier
    # preservation artefact: a metric that cannot isolate the change it claims
    # to measure.
    def beats(a, k, nk):
        x, y = a.get(k), a.get(nk)
        return x is not None and y is not None and x > y + 1e-9

    beats_noop = {}
    for defect in DEFECTS:
        for method in METHODS:
            a = model_agg[f"{defect}/{method}"]
            beats_noop[f"{defect}/{method}"] = (
                beats(a, "c_target", "noop_c_target")
                or beats(a, "q_target", "noop_q_target"))
    if not any(beats_noop.values()):
        notes.append(
            "On the edited blocks, no method beats doing nothing (c_target and "
            "q_target are at or below the no-op baseline in every condition). "
            "The document-level c_edit/quality numbers therefore come from "
            "blocks the method left untouched, not from a repair. Do not read "
            "the model grid as an editor result.")
    elif not all(beats_noop[f"{d}/{m}"]
                 for d in ("cross_block", "cross_block_partial")
                 for m in METHODS):
        notes.append(
            "The editor beats no-op on the edited blocks for at least one "
            "condition, but not for every cross-block method. Read the "
            "cross-block rows on c_target/q_target, not on c_edit alone.")
    if model_agg["cross_block/one_block"]["anchor"] < 1.0:
        notes.append("Anchor violated by one-block repair; check the detector.")
    if model_agg["cross_block/global"]["anchor"] < 1.0:
        notes.append(
            "Anchor violated by global revision. The `editable` mode should "
            "never mask block 0; a violation means the locked-intro contract "
            "is not being honoured and the row is not a valid revision.")
    over = {k: v["over_budget"] for k, v in model_agg.items()
            if v["over_budget"] > 0}
    if over:
        notes.append(
            f"Budget violations: {over} (fraction of tasks where the generated "
            f"text exceeded the matched allowance). The edits are scored as "
            f"produced, because they cannot be un-made, but a method that "
            f"overruns its budget is not being compared at a matched budget. "
            f"Treat those rows as upper bounds on that method.")

    if informative:
        if scope_conditional:
            notes.append(
                f"Scope follows the defect: global minus one-block on c_edit is "
                f"{cross_gap['mean_diff']:+.2f} "
                f"[{cross_gap['lo']:+.2f}, {cross_gap['hi']:+.2f}] for "
                f"cross-block and {local_gap['mean_diff']:+.2f} "
                f"[{local_gap['lo']:+.2f}, {local_gap['hi']:+.2f}] for local "
                f"(interaction {interaction:+.2f}). Extra scope helps only "
                f"where the defect is distributed.")
        else:
            notes.append(
                f"The predicted scope interaction is absent (cross-block "
                f"global-one-block {cross_gap['mean_diff']:+.2f} "
                f"[{cross_gap['lo']:+.2f}, {cross_gap['hi']:+.2f}], local "
                f"{local_gap['mean_diff']:+.2f} "
                f"[{local_gap['lo']:+.2f}, {local_gap['hi']:+.2f}], "
                f"interaction {interaction:+.2f}). Report the null.")

        if repeated_local_suffices:
            notes.append(
                f"Repeated local repair reaches the global-revision level at a "
                f"matched budget (c_edit difference "
                f"{rep_vs_glob['mean_diff']:+.2f} "
                f"[{rep_vs_glob['lo']:+.2f}, {rep_vs_glob['hi']:+.2f}] vs "
                f"global; {rep_vs_one['mean_diff']:+.2f} vs a single one-block "
                f"edit). The defensible claim is therefore about the *cost and "
                f"coordination* of scope, not that local repair cannot fix a "
                f"cross-block defect. Report this rather than the stronger "
                f"claim.")
        elif global_moved:
            notes.append(
                f"Repeated local repair does not reach global revision at a "
                f"matched budget (c_edit {rep_vs_glob['mean_diff']:+.2f} "
                f"[{rep_vs_glob['lo']:+.2f}, {rep_vs_glob['hi']:+.2f}], but "
                f"{rep_vs_one['mean_diff']:+.2f} vs one block). Under this "
                f"editor and budget, scope is load-bearing beyond the number "
                f"of edits.")

        pg = comps["cross_block_partial/global_minus_one_block"]
        prg = comps["cross_block_partial/repeated_local_minus_global"]
        notes.append(
            f"Hard-distractor condition (wrong thesis on a strict majority of "
            f"editable blocks, at least one body still supporting T*): global "
            f"minus one-block c_edit is {pg['mean_diff']:+.2f} "
            f"[{pg['lo']:+.2f}, {pg['hi']:+.2f}], repeated-local minus global "
            f"{prg['mean_diff']:+.2f} [{prg['lo']:+.2f}, {prg['hi']:+.2f}]. "
            f"Because the wrong thesis is the majority, a majority-vote "
            f"heuristic sides with the corruption; only the locked intro "
            f"identifies T*.")

        notes.append(
            f"Gate-selected choices (cross-block): "
            f"{gate_agg['cross_block']['choices']}; (partial): "
            f"{gate_agg['cross_block_partial']['choices']}. Compare its c_edit "
            f"{gate_agg['cross_block']['c_edit']:.2f} against always-one-block "
            f"{model_agg['cross_block/one_block']['c_edit']:.2f} and "
            f"always-global {model_agg['cross_block/global']['c_edit']:.2f}. "
            f"Selection is only useful if it beats the cheaper fixed policy.")

    write_result("exp4_repair_taxonomy", {
        "experiment": "exp4_repair_taxonomy",
        "preset": args.preset, "seed": args.seed, "n": n, "split": split,
        "n_used": n_used, "n_dropped_context_filter": n_dropped,
        "selection": selection,
        "preregistered_prediction": PREDICTION,
        "contract": {
            "authoritative_thesis": "given per task (T*), not inferred",
            "intro": "locked; editing it is a contract violation",
            "editable": "body and conclusion blocks",
            "cross_block_partial": "a strict majority of editable blocks "
                                   "(always including the conclusion) support "
                                   "the wrong thesis; at least one body still "
                                   "supports T*",
            "budgets": "one_block = 1 block; global and repeated_local get "
                       "identical block and token allowances",
            "c_all_floor": "1/n_required, because the locked intro always "
                           "matches T*",
            "coherence": "synthetic label-consistency, not semantic coherence",
            "specificity_on_unseen_local": "undefined (None), not 0: the "
                                           "degraded block's reason was "
                                           "replaced and the unseen thesis's "
                                           "reason pool never appeared in "
                                           "training",
        },
        "oracle_gate": {"conditions": oracle_agg, "checks": gate,
                        "solvable": bool(oracle_solvable)},
        "conditions": [{"condition": k, **v} for k, v in model_agg.items()],
        "gate_selected": {d: gate_agg[d] for d in DEFECTS},
        "summary": {
            "infill_exact_match": infill_exact,
            "infill_parse_rate": infill_parse,
            "infill_on_thesis_rate": infill_on_thesis,
            "noop_parse_rate": noop_parse,
            "noop_on_thesis_rate": noop_thesis,
            "comparisons_c_edit": comps,
            "scope_interaction": interaction,
            "scope_follows_defect": bool(scope_conditional),
            "informative": bool(informative),
            "model_beats_noop_on_target": bool(any(beats_noop.values())),
            "beats_noop_by_condition": beats_noop,
            "global_moved": bool(global_moved),
            "repeated_local_matches_global": bool(repeated_local_suffices),
            "prediction_holds": bool(oracle_solvable and informative
                                     and scope_conditional),
        },
        "notes": notes,
    })
    print("\n" + "\n".join(f"  ! {x}" for x in notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build the paper's tables from results/*.json.

The output (`results/paper_tables.md`) is generated, never hand-edited. If a
number appears in the paper it must be traceable to a JSON file produced by a
script in `experiments/`.

Multi-seed runs write `results/<name>_s1.json`, `_s2.json`, ... (seed 0 keeps
the unsuffixed name). When more than one seed is present, every table gains a
`seed` column instead of quietly reporting one run as if it were the result.
That is the whole point: a paper that claims three seeds must show three.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_all(name: str) -> list[tuple[int, dict]]:
    """Return [(seed, payload)] for one experiment, seed 0 first."""
    runs = []
    for p in sorted(RESULTS.glob(f"{name}*.json")):
        stem = p.stem
        if stem == name:
            seed = 0
        elif stem.startswith(name + "_s"):
            try:
                seed = int(stem.split("_s")[-1])
            except ValueError:
                continue
        else:
            continue
        runs.append((seed, json.loads(p.read_text(encoding="utf-8"))))
    return sorted(runs, key=lambda r: r[0])


def fmt(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        if v != v:
            return "n/a"
        if v == int(v) and abs(v) < 1e6:
            return f"{v:.3f}"
        return f"{v:.4g}"
    return str(v)


def table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_(no data)_\n"
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def note_lines(runs: list[tuple[int, dict]], multi: bool) -> list[str]:
    lines = []
    for seed, payload in runs:
        for n in payload.get("notes", []):
            lines.append(f"- [s{seed}] {n}" if multi else f"- {n}")
    return lines


def main() -> int:
    md = ["# Generated results tables",
          "",
          "Every number below is read from `results/*.json`. Do not edit.",
          "",
          "A `seed` column appears when more than one seed is present. Seeds "
          "present: "
          + ", ".join(str(s) for s in sorted({
              s for name in ("exp0_benchmark_validity", "exp1_invariants",
                             "exp2_scope_and_pressure", "exp3_shuffle",
                             "exp4_repair_taxonomy", "exp5_gate_triage",
                             "exp6_structural_guard")
              for s, _ in load_all(name)})) + ".",
          ""]

    # ------------------------------------------------------------------
    e0 = load_all("exp0_benchmark_validity")
    multi = len(e0) > 1
    md += ["## Table 0 — Benchmark validity (must pass before anything else)", ""]
    if e0:
        rows = []
        for seed, d in e0:
            c = d["conditions"]
            base = {"seed": seed} if multi else {}
            rows += [
                {**base, "task": "scope (marker present)",
                 "probe": c["scope_probe"]["probe_acc"],
                 "majority": c["scope_probe"]["majority_acc"]},
                {**base, "task": "scope (marker stripped)",
                 "probe": c["scope_probe_stripped"]["probe_acc"],
                 "majority": c["scope_probe_stripped"]["majority_acc"]},
                {**base, "task": "epistemic",
                 "probe": c["epistemic_probe"]["probe_acc"],
                 "majority": c["epistemic_probe"]["majority_acc"]},
            ]
        cols = (["seed"] if multi else []) + ["task", "probe", "majority"]
        md.append(table(rows, cols))
        for seed, d in e0:
            md.append(f"- [s{seed}] exact duplicate rate test→train: "
                      f"{fmt(d['conditions']['exact_duplicate_rate'])}"
                      if multi else
                      f"- exact duplicate rate test→train: "
                      f"{fmt(d['conditions']['exact_duplicate_rate'])}")
        md += note_lines(e0, multi)
        md.append("")
    else:
        md.append("_(not run)_\n")

    # ------------------------------------------------------------------
    e1 = load_all("exp1_invariants")
    multi = len(e1) > 1
    md += ["## Table 1 — Design invariants (not empirical claims)", ""]
    if e1:
        rows = []
        for seed, d in e1:
            s = d["summary"]
            base = {"seed": seed} if multi else {}
            rows += [
                {**base, "invariant": "null property (M=empty)",
                 "result": s["null_property"]},
                {**base, "invariant": "negative control detects leak",
                 "result": s["negative_control_detected_leak"]},
                {**base, "invariant": "rollback restores bit-identity",
                 "result": s["rollback"]},
                {**base, "invariant": "memory has an effect when populated",
                 "result": s["memory_has_effect"]},
                {**base, "invariant": "gradient masking (50 repeats)",
                 "result": s["gradient_masking_repeats_unchanged"] == 50},
                {**base, "invariant": "masking has power",
                 "result": s["masking_has_power"]},
                {**base, "invariant": "side channel isolated from vocab",
                 "result": len(s["reasoner_vocab_leaks"]) == 0},
            ]
        md.append(table(rows, (["seed"] if multi else [])
                        + ["invariant", "result"]))
        md.append("")
    else:
        md.append("_(not run)_\n")

    # ------------------------------------------------------------------
    e2 = load_all("exp2_scope_and_pressure")
    multi = len(e2) > 1
    md += ["## Table 2 — Scope, pressure, and what the mask is for", ""]
    if e2:
        rows = []
        for seed, d in e2:
            c = d["conditions"]
            base = {"seed": seed} if multi else {}
            rows += [
                {**base, "condition": "A wrong, before update",
                 "metric": "not accepted", "value": c["A"]["not_accepted_before"]},
                {**base, "condition": "A wrong, after update",
                 "metric": "not accepted", "value": c["A"]["not_accepted_after"]},
                {**base, "condition": "B guarded", "metric": "delta NLL",
                 "value": c["B"]["nll_delta_guarded"]},
                {**base, "condition": "B unguarded", "metric": "NLL diff-in-diff",
                 "value": c["B"]["nll_diff_in_diff"]},
                {**base, "condition": "B unguarded", "metric": "value diff-in-diff",
                 "value": c["B"]["value_diff_in_diff"]},
                {**base, "condition": "C scoped-true", "metric": "accept rate",
                 "value": c["C"]["accept_rate"]},
                {**base, "condition": "C scoped-true", "metric": "scope stored",
                 "value": c["C"]["scope_stored_rate"]},
                {**base, "condition": "C bare-false", "metric": "reject rate",
                 "value": c["C"]["bare_reject_rate"]},
                {**base, "condition": "C bare-false", "metric": "assumed flag",
                 "value": c["C"]["bare_assumed_flag_rate"]},
                {**base, "condition": "B head health", "metric": "value mean",
                 "value": c["B"].get("value_mean")},
                {**base, "condition": "B head health", "metric": "value sd",
                 "value": c["B"].get("value_sd")},
                {**base, "condition": "B head health", "metric": "endorse rate",
                 "value": c["B"]["endorse_before"]},
                {**base, "condition": "D empirical", "metric": "defer rate",
                 "value": c["D"]["defer_rate"]},
                {**base, "condition": "E counting ontology",
                 "metric": "interpreter matches spec",
                 "value": c.get("E", {}).get("interpreter_matches_spec")},
                {**base, "condition": "E counting ontology",
                 "metric": "system matches spec",
                 "value": c.get("E", {}).get("system_matches_spec")},
                {**base, "condition": "E undefined unit", "metric": "defers",
                 "value": c.get("E", {}).get("undefined_unit_defers")},
            ]
        md.append(table(rows, (["seed"] if multi else [])
                        + ["condition", "metric", "value"]))
        md += note_lines(e2, multi)
        md.append("")
    else:
        md.append("_(not run)_\n")

    # ------------------------------------------------------------------
    e3 = load_all("exp3_shuffle")
    multi = len(e3) > 1
    md += ["## Table 3 — Side-channel causality", ""]
    if e3:
        rows = []
        for seed, d in e3:
            full = d["summary"]["losses"]["full"]
            base = {"seed": seed} if multi else {}
            for k, v in d["conditions"].items():
                rows.append({**base, "condition": k,
                             "loss": v["denoise_loss"],
                             "vs full": v["denoise_loss"] - full,
                             "repair consistency":
                                 d["summary"]["repair_consistency"].get(k)})
        md.append(table(rows, (["seed"] if multi else [])
                        + ["condition", "loss", "vs full",
                           "repair consistency"]))
        for seed, d in e3:
            s = d["summary"]
            md.append(f"- [s{seed}] full vs shuffled: "
                      f"{fmt(s['full_vs_shuffled_mean_diff'])}, 95% CI "
                      f"[{fmt(s['full_vs_shuffled_ci'][0])}, "
                      f"{fmt(s['full_vs_shuffled_ci'][1])}], MDE {fmt(s['mde'])}"
                      if multi else
                      f"- full vs shuffled: {fmt(s['full_vs_shuffled_mean_diff'])}, "
                      f"95% CI [{fmt(s['full_vs_shuffled_ci'][0])}, "
                      f"{fmt(s['full_vs_shuffled_ci'][1])}], MDE {fmt(s['mde'])}")
            md.append(f"- [s{seed}] positive control carries information: "
                      f"{fmt(s['channel_carries_information'])}"
                      if multi else
                      f"- positive control carries information: "
                      f"{fmt(s['channel_carries_information'])}")
        md += note_lines(e3, multi)
        md.append("")
    else:
        md.append("_(not run)_\n")

    # ------------------------------------------------------------------
    e4 = load_all("exp4_repair_taxonomy")
    multi = len(e4) > 1
    md += ["## Table 4 — Revision scope under a matched budget", ""]
    if e4:
        rows = []
        for seed, d in e4:
            base = {"seed": seed} if multi else {}
            rows += [{**base, **r} for r in d["conditions"]]
            for defect, g in d.get("gate_selected", {}).items():
                rows.append({**base, **g, "condition": f"{defect}/gate_selected"})
        md.append(table(rows, (["seed"] if multi else [])
                        + ["condition", "anchor", "c_edit", "c_all",
                           "quality", "blocks_changed", "tokens_spent",
                           "consistent", "improvement", "preservation", "seam",
                           "n"]))
        for seed, d in e4:
            s = d["summary"]
            pre = f"[s{seed}] " if multi else ""
            md.append(f"- {pre}oracle solvability gate: "
                      f"{fmt(d.get('oracle_gate', {}).get('solvable'))}")
            md.append(f"- {pre}infill sanity (exact block restore): "
                      f"{fmt(s.get('infill_exact_match'))}")
            md.append(f"- {pre}informative (infill >= 0.2): "
                      f"{fmt(s.get('informative'))}")
            for k, v in s.get("comparisons_c_edit", {}).items():
                md.append(f"- {pre}{k}: c_edit diff {fmt(v['mean_diff'])} "
                          f"(95% CI [{fmt(v['lo'])}, {fmt(v['hi'])}])")
            md.append(f"- {pre}scope interaction: "
                      f"{fmt(s.get('scope_interaction'))}; prediction holds: "
                      f"{fmt(s.get('prediction_holds'))}")
            md.append("")
        md += note_lines(e4, multi)
        md.append("")
    else:
        md.append("_(not run)_\n")

    # ------------------------------------------------------------------
    e5 = load_all("exp5_gate_triage")
    multi = len(e5) > 1
    md += ["## Table 5 — Gate usage, overrides, triage", ""]
    if e5:
        rows = []
        for seed, d in e5:
            c = d["conditions"]
            base = {"seed": seed} if multi else {}
            actions = (c["histogram_model"].keys()
                       | c["histogram_oracle"].keys())
            rows += [{**base, "action": a,
                      "model": c["histogram_model"].get(a, 0.0),
                      "oracle": c["histogram_oracle"].get(a, 0.0)}
                     for a in actions]
        md.append(table(sorted(rows, key=lambda r: (r["action"], r["seed"]))
                        if multi else sorted(rows, key=lambda r: r["action"]),
                        (["seed"] if multi else []) + ["action", "model",
                                                       "oracle"]))
        for seed, d in e5:
            c = d["conditions"]
            pre = f"[s{seed}] " if multi else ""
            md.append(f"- {pre}gate accuracy vs scaffolding: "
                      f"{fmt(c['gate_accuracy'])}\n")
            md.append(f"- {pre}overrides: {c['overrides']}\n")
            md.append(f"- {pre}cost per correct: model "
                      f"{fmt(c['cost_per_correct_model'])}, oracle "
                      f"{fmt(c['cost_per_correct_oracle'])}\n")
            by = c.get("by_condition") or {}
            if by:
                cond_rows = []
                for name, v in by.items():
                    model = v.get("model", {})
                    oracle = v.get("oracle", {})
                    top = max(model, key=model.get) if model else ""
                    cond_rows.append({
                        "condition": name, "n": v.get("n"),
                        "acc": v.get("acc"), "model_top": top,
                        "model_top_share": model.get(top, 0.0),
                        "model_actions": len(model),
                        "oracle_actions": len(oracle)})
                md.append(f"- {pre}gate action diversity **within** each "
                          f"condition (a mixed eval set can look diverse only "
                          f"because the oracle is diverse):\n")
                md.append(table(cond_rows, ["condition", "n", "acc",
                                            "model_top", "model_top_share",
                                            "model_actions", "oracle_actions"]))
                md.append(f"- {pre}gate separates local from cross-block: "
                          f"{fmt(d['summary'].get('separates_defects'))}\n")
            t = c["triage"]
            md.append(table([{"ordering": "sequential", "marks": t["sequential"]},
                             {"ordering": "triage", "marks": t["triage"]},
                             {"ordering": "random", "marks": t["random_mean"]},
                             {"ordering": "hardest first", "marks": t["hardest"]}],
                            ["ordering", "marks"]))
        md += note_lines(e5, multi)
        md.append("")
    else:
        md.append("_(not run)_\n")

    # ------------------------------------------------------------------
    e6 = load_all("exp6_structural_guard")
    multi = len(e6) > 1
    md += ["## Table 6 — Structural guard (C1), independent of the editor", "",
           "C1 is the structural claim: a verified-wrong example cannot move "
           "the plastic substrate and cannot create a factual memory entry. "
           "It does not depend on the learned editor. C2 (does the guard "
           "improve held-out behaviour?) is a separate claim and is "
           "unresolved.", ""]
    if e6:
        rows = []
        for seed, d in e6:
            base = {"seed": seed} if multi else {}
            for check, verdict in d["verdicts"].items():
                rows.append({**base, "check": check, "verdict": verdict})
            for name, c in d["controls"].items():
                if name == "unguarded":
                    rows.append({**base, "check": "control: unguarded leaks",
                                 "verdict": fmt(c["A_fails"] and c["B_fails"]
                                                and c["C_fails"])})
                else:
                    rows.append({**base, "check": "control: verified-right updates",
                                 "verdict": fmt(c["gradient_flows"]
                                                and c["parameters_change"]
                                                and c["entry_created"])})
        md.append(table(rows, (["seed"] if multi else [])
                        + ["check", "verdict"]))
        for seed, d in e6:
            s = d["summary"]
            pre = f"[s{seed}] " if multi else ""
            md.append(f"- {pre}C1 pass: {fmt(s['c1_pass'])}; ambiguous: "
                      f"{fmt(s['c1_ambiguous'])}; instrument valid: "
                      f"{fmt(s['instrument_valid'])}\n")
            ex = d.get("extra_checks", {})
            if "A2_holds" in ex:
                md.append(f"- {pre}diagnostic A2 (not pre-registered): mixed "
                          f"batch vs verified-right-only max gradient diff "
                          f"{fmt(ex['A2_mixed_vs_right_only_max_grad_diff'])} "
                          f"(holds: {fmt(ex['A2_holds'])})\n")
            b2 = ex.get("B2_adamw_wd")
            if b2:
                md.append(f"- {pre}diagnostic B2 (not pre-registered): AdamW "
                          f"weight decay moves the parameter when grad is a "
                          f"zero tensor: {fmt(b2['grad_zeros_moves'])}; when "
                          f"grad is None: {fmt(b2['grad_none_moves'])}\n")
        md += note_lines(e6, multi)
        md.append("")
    else:
        md.append("_(not run)_\n")

    # ------------------------------------------------------------------
    e7 = load_all("exp7_editor_floor")
    multi = len(e7) > 1
    md += ["## Table 7 — Editor floor: training budget vs capacity", "",
           "Two questions were conflated by the earlier byte-identical gate. "
           "`on_thesis` is clean-block infill ability; `local_*` and `cross_*` "
           "are scored on the blocks the task actually edits, with the no-op "
           "baseline shown. Copy fidelity (`exact`) is a diagnostic only.",
           "",
           "**Seed 0 only, n=10 tasks per defect.** This is a budget curve, "
           "not a multi-seed result; the non-monotonicity between 750 and 1500 "
           "steps is not interpreted.", ""]
    if e7:
        rows = []
        for seed, d in e7:
            base = {"seed": seed} if multi else {}
            for c in d["conditions"]:
                rows.append({**base, "budget": c["budget"],
                             "opt_steps": c["opt_steps"],
                             "exact": c["exact"],
                             "on_thesis": c["on_thesis"],
                             "specific": c["specific"],
                             "local_c": c["local_c_edit"],
                             "local_q": c["local_quality"],
                             "local_noop_c": c["local_noop_c_edit"],
                             "local_noop_q": c["local_noop_quality"],
                             "cross_c": c["cross_c_edit"],
                             "cross_q": c["cross_quality"],
                             "cross_noop_c": c["cross_noop_c_edit"]})
        md.append(table(rows, (["seed"] if multi else [])
                        + ["budget", "opt_steps", "exact", "on_thesis",
                           "specific", "local_c", "local_q", "local_noop_c",
                           "local_noop_q", "cross_c", "cross_q",
                           "cross_noop_c"]))
        for seed, d in e7:
            s = d["summary"]
            pre = f"[s{seed}] " if multi else ""
            md.append(f"- {pre}ability rises with budget: "
                      f"{fmt(s['ability_rises_with_budget'])}; specificity "
                      f"rises: {fmt(s['specificity_rises'])}; local repair "
                      f"rises: {fmt(s['local_repair_rises'])}; cross-block "
                      f"repair rises: {fmt(s['cross_repair_rises'])}\n")
        md += note_lines(e7, multi)
        md.append("")
    else:
        md.append("_(not run)_\n")

    out = RESULTS / "paper_tables.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

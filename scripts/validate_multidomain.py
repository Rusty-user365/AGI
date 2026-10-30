"""Generate public fixtures, self-test their validators, or grade model artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgr.control.dependency_graph import DependencyGraph
from cgr.control.multidomain_validation import scenarios, graph_nodes, coder_verdict, cyber_verdict, ui_verdict, golden_card

GOOD_CODE = "def clamp(value, lower, upper):\n    return max(lower, min(value, upper))"
BAD_CODE = "def clamp(value, lower, upper):\n    return value"
GOOD_QUERY = {"sql": "SELECT id,name FROM users WHERE name = ?", "params": ["$input"]}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate():
    data = {"version": 1, "purpose": "public_integration_fixtures_not_independent_holdout_tasks",
            "independent_task_families": 1, "scenarios": scenarios(), "graph": graph_nodes()}
    path = ROOT / "data" / "validation" / "cgr_multidomain_v1.json"
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError("Frozen fixture file differs; create a new suite version")
    write_json(path, data)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def self_test():
    rows = []
    for row in scenarios():
        graph = DependencyGraph(graph_nodes())
        for node in graph.order:
            graph.commit(node, BAD_CODE if node == "6" else "artifact:" + node, lambda _: True)
        snapshot = dict(graph.artifacts)
        # Late integration failure triggers rechecking ancestor contracts.
        failed_contracts = ["6"] if not coder_verdict(row, graph.artifacts["6"])["passed"] else []
        suspects = graph.fault_candidates("20", failed_contracts)
        rejected = graph.commit("6", BAD_CODE, lambda code: coder_verdict(row, code)["passed"])
        rejection_atomic = graph.artifacts == snapshot and not rejected["accepted"]
        repaired = graph.commit("6", GOOD_CODE, lambda code: coder_verdict(row, code)["passed"])
        plan = graph.cache_replay_plan("6", list(map(str, range(1, 21))), {str(n): n * 10 for n in range(1, 21)})
        invalidated = set(graph.dirty)
        recomputed = []
        for node in graph.order:
            if node in graph.dirty:
                graph.commit(node, "fixture-recomputed:" + node, lambda _: True)
                recomputed.append(node)
        safe = cyber_verdict(GOOD_QUERY)["passed"]
        unsafe = cyber_verdict({"sql": "SELECT id,name FROM users WHERE name = ? OR 1=1", "params": ["$input"]})["passed"]
        card = golden_card(row)
        ui_good = ui_verdict(row, card)["passed"]
        ui_bad = ui_verdict(row, card.replace('for="amount"', 'for="wrong"'))["passed"]
        rows.append({"scenario": row["id"], "observed_failure_node": "20", "fault_candidates": suspects,
            "bad_patch_rejected_atomically": rejection_atomic,
            "affected_descendants": repaired["invalidated"],
            "fixture_descendants_recomputed": recomputed,
            "all_artifacts_current_after_recompute": not graph.dirty,
            "independent_artifacts_preserved": all(graph.artifacts[n] == snapshot[n] and n not in graph.dirty for n in ("8", "12", "16")),
            "cache_replay_plan": plan, "cyber_positive_pass": safe, "cyber_negative_rejected": not unsafe,
            "ui_positive_pass": ui_good, "ui_negative_rejected": not ui_bad,
            "passed": suspects == ["6"] and rejection_atomic and repaired["accepted"] and
                "20" in invalidated and not graph.dirty and set(recomputed) == invalidated and
                all(graph.artifacts[n] == snapshot[n] for n in ("8", "12", "16")) and
                plan["replay_preserved_artifacts"] == ["8", "12", "16"] and safe and not unsafe and ui_good and not ui_bad})
    return {"measurement_type": "validator_and_graph_fixture_self_test", "model_generations": 0,
            "scenarios": rows, "passed": all(row["passed"] for row in rows),
            "note": "Known fixtures validate the harness. Recompute uses fixture artifacts and cache boundaries are synthetic: only the replay plan is checked. No multi-adapter inference, React behavior, or learned localization is measured."}


def evaluate(candidates):
    rows = []
    for task in scenarios():
        provided = candidates.get(task["id"], {})
        for skill, validator in (("coder", lambda c: coder_verdict(task, c)),
                                 ("cyber_defensive", cyber_verdict), ("ui", lambda c: ui_verdict(task, c))):
            candidate = provided.get(skill)
            rows.append({"scenario": task["id"], "skill": skill,
                         "status": "not_run" if candidate is None else "evaluated",
                         "verdict": None if candidate is None else validator(candidate)})
    return {"measurement_type": "supplied_candidate_validation", "rows": rows,
            "note": "Generation provenance and token-budget accounting must be supplied by the inference harness; this command only grades artifacts."}


def adapter_metadata(specification):
    skill, folder = specification.split("=", 1)
    if skill not in {"coder", "cyber_defensive", "ui"}:
        raise ValueError("Supported registry skills: coder, cyber_defensive, ui")
    path = Path(folder)
    try:
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        manifest = json.loads((path.parent / "run_manifest.json").read_text(encoding="utf-8"))
        weights = path / "adapter_model.safetensors"
        checks = {"base_id": config.get("base_model_name_or_path") == manifest.get("base_model_id") == "Qwen/Qwen3-0.6B",
                  "base_revision": manifest.get("base_revision") == "c1899de289a04d12100db370d81485cdf75e47ca",
                  "lora": config.get("peft_type") == "LORA",
                  "rank": config.get("r") == manifest.get("lora_rank"),
                  "alpha": config.get("lora_alpha") == manifest.get("lora_alpha"),
                  "targets": set(config.get("target_modules", [])) == set(manifest.get("target_modules_requested", [])) and bool(config.get("target_modules")),
                  "weights_present": weights.is_file() and weights.stat().st_size > 0}
        return {"skill": skill, "path": str(path), "checks": checks, "metadata_pass": all(checks.values()),
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest() if checks["weights_present"] else None,
                "remaining": ["tokenizer/template compatibility", "tensor integrity and load/switch test", "task quality", "Bridge tests"]}
    except (OSError, ValueError) as exc:
        return {"skill": skill, "path": str(path), "metadata_pass": False, "reason": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--adapter", action="append", default=[])
    args = parser.parse_args()
    suite, digest = generate()
    report = {"suite": str(suite), "suite_sha256": digest,
              "pending": ["cyber and UI trained artifacts", "multi-adapter model trials", "independent task evaluation"]}
    if args.self_test:
        report["self_test"] = self_test()
    if args.candidates:
        report["candidate_results"] = evaluate(json.loads(args.candidates.read_text(encoding="utf-8")))
    if args.adapter:
        report["adapter_metadata"] = [adapter_metadata(item) for item in args.adapter]
    write_json(ROOT / "results" / "multidomain_validation_readiness.json", report)
    template = ROOT / "data" / "validation" / "candidate_template.json"
    if not template.exists():
        write_json(template, {row["id"]: {"coder": None, "cyber_defensive": None, "ui": None} for row in scenarios()})
    print(json.dumps({"suite_sha256": digest, "self_test_passed": report.get("self_test", {}).get("passed"),
                      "model_generations": 0, "report": "results/multidomain_validation_readiness.json"}))
    if args.self_test and not report["self_test"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Verifier-driven specialist switching on one frozen Qwen/PEFT backbone.

Integration harness for ADAPTER_ROUTING_PLAN.md. Reuses CommitmentDAG and
Controller; the arithmetic localizer and the legacy learned CGR gate are not
silently treated as trained code routers. Skills come from declared task nodes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from .adapter_tasks import task_sequence
from .commitment import Action, CommitmentDAG, Controller

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results" / "cgr_adapter_switching_latest.json"


def verify_candidate(node: dict, text: str, truncated: bool = False) -> dict:
    if truncated:
        return {"passed": False, "tests": [], "reason": "Token budget exhausted; candidate rejected"}
    if "expected_text" in node:
        passed = text.strip() == node["expected_text"]
        return {"passed": passed, "tests": [], "reason": "Exact reply matched" if passed else "Exact reply mismatch"}
    try:
        completed = subprocess.run(
            [sys.executable, "-I", str(ROOT / "scripts" / "adapter_task_verifier.py")],
            input=json.dumps({"text": text, "function": node["function"], "cases": node["cases"]}),
            text=True, capture_output=True, timeout=3, check=False,
        )
        if completed.returncode != 0:
            return {"passed": False, "tests": [], "reason": "Verifier subprocess failed"}
        return json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        return {"passed": False, "tests": [], "reason": "Verifier exceeded 3-second limit"}
    except (ValueError, OSError):
        return {"passed": False, "tests": [], "reason": "Verifier could not evaluate candidate"}


class AdapterSwitchRuntime:
    def __init__(self, runner):
        self.runner = runner
        self.torch = runner._torch
        self.model = runner._model
        self.tok = runner._tokenizer
        self.device = next(self.model.parameters()).device
        self.events: list[dict] = []
        self.route = None
        self.switches = 0
        self.tolerance = 1e-4 if runner._dtype_name == "float32" else 0.125

    def _sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _timed(self, callback):
        self._sync()
        start = time.perf_counter()
        value = callback()
        self._sync()
        return value, (time.perf_counter() - start) * 1000

    @contextmanager
    def _policy(self, skill: str):
        self._sync()
        switch_start = time.perf_counter()
        if skill == "coder":
            self.model.set_adapter("coder", inference_mode=True)
            self.model.base_model.enable_adapter_layers()
            context = self.torch.inference_mode()
        else:
            context = self.model.disable_adapter()
        with context, self.torch.inference_mode():
            # Inspect actual LoRA layer state; active_adapters alone still names
            # coder inside PEFT's disable_adapter() context.
            layers = [module for module in self.model.modules()
                      if hasattr(module, "lora_A") and hasattr(module, "disable_adapters")]
            enabled = skill == "coder"
            if not layers or any(bool(layer.disable_adapters) == enabled for layer in layers):
                raise AssertionError("Unexpected enabled/disabled LoRA state")
            if enabled and any(list(layer.active_adapters) != ["coder"] for layer in layers):
                raise AssertionError("Wrong specialist is active")
            if self.route != skill:
                if self.route is not None:
                    self.switches += 1
                self.route = skill
                self._sync()
                self.events.append({"event": "adapter_state", "active": skill,
                                    "inspected_lora_layers": len(layers),
                                    "selection_and_assertion_ms": (time.perf_counter() - switch_start) * 1000})
            yield

    def _forward(self, ids, cache=None):
        past = cache.get_seq_length() if cache is not None else 0
        mask = self.torch.ones((1, past + ids.shape[1]), dtype=self.torch.long, device=self.device)
        return self.model(input_ids=ids, past_key_values=cache, attention_mask=mask,
                          use_cache=True, return_dict=True, logits_to_keep=1)

    def _tensor(self, ids):
        return self.torch.tensor([ids], dtype=self.torch.long, device=self.device)

    def _chat(self, messages, generation_prompt):
        return self.tok.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=generation_prompt, enable_thinking=False)

    def _snapshot(self, cache):
        return [(layer.keys.detach().clone(), layer.values.detach().clone()) for layer in cache.layers]

    def _same_prefix(self, cache, snapshot, length):
        return cache.get_seq_length() == length and len(cache.layers) == len(snapshot) and all(
            self.torch.equal(layer.keys, key[..., :length, :]) and
            self.torch.equal(layer.values, value[..., :length, :])
            for layer, (key, value) in zip(cache.layers, snapshot))

    def _crop(self, cache, boundary, snapshot):
        cache.crop(boundary)
        passed = self._same_prefix(cache, snapshot, boundary)
        if not passed:
            raise AssertionError("Rollback changed retained base prefix K/V")
        return passed

    def _decode(self, prefix_ids, clean_cache, skill, cap, base_logits):
        working = copy.deepcopy(clean_cache)
        # Re-evaluate the last header token under the selected policy. The
        # earlier verified prefix stays base-produced by deliberate design.
        generated = []
        eos = self.tok.eos_token_id
        with self._policy(skill):
            if skill == "base":
                logits = base_logits
            else:
                working.crop(len(prefix_ids) - 1)
                output = self._forward(self._tensor(prefix_ids[-1:]), working)
                working, logits = output.past_key_values, output.logits[:, -1, :]
            for _ in range(cap):
                token = int(logits.argmax(-1).item())
                generated.append(token)
                output = self._forward(self._tensor([token]), working)
                working, logits = output.past_key_values, output.logits[:, -1, :]
                if token == eos:
                    break
        return {
            "text": self.tok.decode(generated, skip_special_tokens=True).strip(),
            "token_ids": generated, "tokens": len(generated),
            "truncated": not generated or generated[-1] != eos,
            "cache": working,
        }

    def _continuation(self, logits, cache, cap=4):
        ids = []
        with self._policy("base"):
            for _ in range(cap):
                token = int(logits.argmax(-1).item())
                ids.append(token)
                output = self._forward(self._tensor([token]), cache)
                cache, logits = output.past_key_values, output.logits[:, -1, :]
                if token == self.tok.eos_token_id:
                    break
        return ids

    def _bridge(self, prefix_ids, prefix_cache, patch_ids):
        # Specialist working cache is never reused here. No transient repair
        # instructions are committed; only exact accepted patch IDs are replayed.
        with self._policy("base"):
            bridge, bridge_ms = self._timed(lambda: self._forward(
                self._tensor(patch_ids), copy.deepcopy(prefix_cache)))
            fresh, reference_ms = self._timed(lambda: self._forward(self._tensor(prefix_ids + patch_ids)))
            left = bridge.logits[:, -1, :].float()
            right = fresh.logits[:, -1, :].float()
            finite = bool(self.torch.isfinite(left).all() and self.torch.isfinite(right).all())
            delta = float((left - right).abs().max().item())
            argmax = bool(self.torch.equal(left.argmax(-1), right.argmax(-1)))
            continuation = self._continuation(left, copy.deepcopy(bridge.past_key_values))
            reference = self._continuation(right, fresh.past_key_values)
        correct_length = bridge.past_key_values.get_seq_length() == len(prefix_ids) + len(patch_ids)
        passed = finite and delta <= self.tolerance and argmax and continuation == reference and correct_length
        return bridge.past_key_values, {
            "passed": passed, "finite_logits": finite, "max_abs_logit_delta": delta,
            "absolute_tolerance": self.tolerance, "argmax_matches": argmax,
            "continuation_matches": continuation == reference,
            "continuation_token_ids": continuation, "reference_token_ids": reference,
            "cache_length_matches": correct_length, "patch_prefill_ms": bridge_ms,
            "fresh_reference_ms": reference_ms,
            "resume_source": "bridge output logits; final patch token is not fed twice",
        }

    def _attempt(self, node, prefix_ids, cache, skill, cap, base_logits, injected=False):
        if injected:
            ids = self.tok.encode(node["injected_candidate"], add_special_tokens=False) + [self.tok.eos_token_id]
            with self._policy("base"):
                output, elapsed = self._timed(lambda: self._forward(self._tensor(ids), copy.deepcopy(cache)))
            result = {"text": node["injected_candidate"], "token_ids": ids, "tokens": len(ids),
                      "truncated": False, "cache": output.past_key_values}
        else:
            result, elapsed = self._timed(lambda: self._decode(prefix_ids, cache, skill, cap, base_logits))
        result.update({"generation_ms": elapsed, "policy": skill,
            "source": "injected_negative_control" if injected else "model_generation",
            "verifier": verify_candidate(node, result["text"], result["truncated"])})
        return result

    @staticmethod
    def _public(attempt):
        return {key: value for key, value in attempt.items() if key != "cache"}

    def run(self, selection, mode, cap, seed):
        if version("transformers") != "4.57.6":
            raise RuntimeError("Switching harness requires Transformers 4.57.6 absolute-length crop API")
        self.torch.manual_seed(seed)
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)
        nodes = task_sequence(selection)
        initial_ids = self._chat([{"role": "system", "content": "Follow each user task. For coding tasks provide only the requested Python function."}], False)
        with self._policy("base"):
            initial = self._forward(self._tensor(initial_ids))
        committed_ids, cache = initial_ids, initial.past_key_values
        dag = CommitmentDAG(list(initial_ids))
        controller = Controller(dag)
        rows = []
        total_start = time.perf_counter()
        for node in nodes:
            self.runner._stage = "switching: " + node["title"]
            if node["required_skill"] not in ("base", "coder"):
                rows.append({"node_id": node["id"], "title": node["title"], "status": "abstained",
                             "route_reason": "Unknown/unavailable skill; no silent coder fallback"})
                continue
            previous_ids, previous_cache = list(committed_ids), copy.deepcopy(cache)
            header = self._chat([{"role": "user", "content": node["prompt"]}], True)
            prefix_ids = committed_ids + header
            with self._policy("base"):
                prefix = self._forward(self._tensor(header), cache)
            clean = prefix.past_key_values
            snapshot = self._snapshot(clean)
            skill = node["required_skill"]
            injected = "injected_candidate" in node
            # Planned route gets a base-only diagnostic on the same prefix.
            baseline = self._attempt(node, prefix_ids, clean, "base", cap, prefix.logits[:, -1, :], injected)
            rollback = self._crop(baseline["cache"], len(prefix_ids), snapshot)
            baseline["cache"] = None
            row = {"node_id": node["id"], "title": node["title"], "required_skill": skill,
                   "route_source": "declared_node_metadata", "verifier_kind": "pure_function_cases" if skill == "coder" else "exact_text",
                   "prompt": node["prompt"], "prefix_boundary_tokens": len(prefix_ids),
                   "rollback_prefix_kv_exact": rollback, "base_attempt": self._public(baseline)}
            should_specialize = skill == "coder" and (mode == "planned" or not baseline["verifier"]["passed"] or injected)
            if should_specialize:
                failed_decision = None
                if not baseline["verifier"]["passed"]:
                    failed_decision = controller.step(prefix_ids + baseline["token_ids"], proposed=Action.VER,
                        surprise=0, verifier=lambda _: False)
                chosen = self._attempt(node, prefix_ids, clean, "coder", cap, prefix.logits[:, -1, :])
                chosen["cache"] = None  # never bridge specialist-created K/V
                row["specialist_attempt"] = self._public(chosen)
                row["route_reason"] = "Planned coder node" if mode == "planned" and not injected else "Verifier failed at this node; bounded coder repair"
                row["repair_calls"] = 1
                if failed_decision:
                    row["failed_base_action"] = failed_decision.action.value
            else:
                chosen = baseline
                row["route_reason"] = "Base task" if skill == "base" else "Base already passed; specialist not needed"
                row["repair_calls"] = 0
            row["selected_policy"] = chosen["policy"]
            row["task_verified"] = chosen["verifier"]["passed"]
            if not self._same_prefix(clean, snapshot, len(prefix_ids)):
                raise AssertionError("Generation modified the preserved base cache")
            row["preserved_base_cache_exact"] = True
            bridge_ok = False
            if chosen["verifier"]["passed"]:
                next_cache, coherence = self._bridge(prefix_ids, clean, chosen["token_ids"])
                row["bridge"] = coherence
                bridge_ok = coherence["passed"]
            decision = controller.step(prefix_ids + chosen["token_ids"], proposed=Action.ANS,
                surprise=0, verifier=lambda _: chosen["verifier"]["passed"] and bridge_ok)
            row.update({"controller_action": decision.action.value, "controller_reason": decision.reason,
                        "checkpoint_id": decision.checkpoint_id, "accepted": decision.action == Action.ANS})
            if row["accepted"]:
                committed_ids, cache = list(decision.artifact), next_cache
                row["status"] = "committed"
            else:
                committed_ids, cache = previous_ids, previous_cache
                if committed_ids != decision.artifact:
                    raise AssertionError("DAG and cache rollback disagree")
                if cache.get_seq_length() != len(committed_ids):
                    raise AssertionError("Restored cache and committed token boundary disagree")
                row["status"] = "rolled_back"
            row["committed_tokens_after"] = len(committed_ids)
            rows.append(row)
        # Persistent model state is disabled when returning to the other UI.
        with self._policy("base"):
            pass
        self.model.base_model.disable_adapter_layers()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        measured = [row for row in rows if "accepted" in row]
        specialist = [row for row in measured if "specialist_attempt" in row]
        result = {
            "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": "CGR verifier-driven adapter switching integration",
            "selection": selection, "mode": mode, "max_new_tokens": cap, "seed": seed,
            "base_model_id": "Qwen/Qwen3-0.6B", "base_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "adapter_path": str(self.runner.adapter_path),
            "adapter_weights_sha256": hashlib.sha256((self.runner.adapter_path / "adapter_model.safetensors").read_bytes()).hexdigest(),
            "adapter_config_sha256": hashlib.sha256((self.runner.adapter_path / "adapter_config.json").read_bytes()).hexdigest(),
            "tokenizer_chat_template_sha256": hashlib.sha256(self.tok.chat_template.encode()).hexdigest(),
            "runtime": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
            "device": str(self.device), "dtype": self.runner._dtype_name,
            "cache_type": type(cache).__name__,
            "peak_vram_gb": self.torch.cuda.max_memory_allocated(self.device) / 1024**3 if self.device.type == "cuda" else None,
            "nodes": rows, "events": self.events,
            "dag": [{"node_id": index, "parent_id": dag.get(index).parent_id,
                     "verified": dag.get(index).verified, "tokens": len(dag.get(index).artifact),
                     "reason": dag.get(index).reason} for index in range(len(dag))],
            "summary": {"accepted_nodes": sum(row["accepted"] for row in measured),
                        "attempted_nodes": len(measured), "specialist_calls": len(specialist),
                        "verified_adapter_transitions": self.switches,
                        "all_rollbacks_exact": all(row["rollback_prefix_kv_exact"] for row in measured),
                        "all_bridges_passed": all(row.get("bridge", {}).get("passed", False) for row in measured if row["task_verified"]),
                        "base_task_passes": sum(row["base_attempt"]["verifier"]["passed"] for row in specialist),
                        "specialist_task_passes": sum(row["specialist_attempt"]["verifier"]["passed"] for row in specialist),
                        "elapsed_seconds": time.perf_counter() - total_start},
            "limits": ["Disclosed integration tasks; no held-out quality or learned-router claim.",
                       "Only local coder checkpoint-1000 is available; other domains abstain.",
                       "Routing/localization uses explicit task nodes and verifier failures, not a trained latent probe.",
                       "Base-only diagnostics share the exact prefix; they are not an independent full-trajectory benchmark.",
                       "Injected failure control is labelled and excluded from quality claims.",
                       "Diagnostic fresh-base prefill/continuation checks add overhead; no production speedup claim."],
        }
        directory = ROOT / "results" / "cgr_adapter_switching"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / (run_id + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary = RESULT.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(RESULT)
        return result

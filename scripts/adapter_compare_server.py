"""Local, real-inference A/B runner for the CGR base/adapter studio.

The server loads one exact base checkpoint, attaches one local PEFT adapter,
then compares adapter-disabled and adapter-enabled generations sequentially.
It binds to localhost by default and serves the studio page from /.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HTML_PATH = ROOT / "notebooks" / "cgr_multi_adapter_studio.html"
DEFAULT_ADAPTER = (
    ROOT
    / "notebooks"
    / "coder_qwen3_0.6b_q_mlp_r16_v1___Coder Adapter"
    / "checkpoint-1000"
)
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ADAPTER_NAME = "coder"
RESULT_PATH = ROOT / "results" / "coder_base_vs_adapter_checkpoint_1000.json"


class CompareRunner:
    def __init__(self, adapter_path: Path, requested_device: str) -> None:
        self.adapter_path = adapter_path.expanduser().resolve()
        self.requested_device = requested_device
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._adapter_config: dict[str, Any] = {}
        self._device = "cuda" if requested_device == "auto" else requested_device
        self._dtype_name = "not loaded"
        self._stage = "ready"

    def status(self) -> dict[str, Any]:
        config_path = self.adapter_path / "adapter_config.json"
        weights_path = self.adapter_path / "adapter_model.safetensors"
        available = config_path.is_file() and weights_path.is_file()
        config: dict[str, Any] = {}
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

        device = self._device
        hardware = "PyTorch unavailable"
        if find_spec("torch") is not None:
            try:
                import torch

                if self.requested_device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._device = device
                if device.startswith("cuda") and not torch.cuda.is_available():
                    hardware = "CUDA requested but unavailable"
                elif device.startswith("cuda"):
                    hardware = torch.cuda.get_device_name(0)
                else:
                    hardware = "CPU"
            except Exception as exc:  # status should still load without torch
                hardware = f"PyTorch status error: {type(exc).__name__}"

        return {
            "base_model_id": BASE_MODEL_ID,
            "base_revision": BASE_REVISION,
            "adapter_name": "Coder",
            "adapter_checkpoint": self.adapter_path.name,
            "adapter_path": str(self.adapter_path),
            "adapter_available": available,
            "adapter_parent_model": config.get("base_model_name_or_path"),
            "rank": config.get("r"),
            "target_modules": config.get("target_modules", []),
            "device": f"{device} · {hardware}",
            "model_loaded": self._model is not None,
            "dependencies": {
                package: find_spec(package) is not None
                for package in ("torch", "transformers", "peft", "accelerate", "safetensors")
            },
            "dtype": self._dtype_name,
            "stage": self._stage,
        }

    def _load(self) -> None:
        missing = [
            package
            for package in ("torch", "transformers", "peft", "accelerate", "safetensors")
            if find_spec(package) is None
        ]
        if missing:
            raise RuntimeError(
                "Missing Python dependencies: " + ", ".join(missing)
                + ". Install scripts/requirements-adapter-compare.txt in the Python environment you use to start the server."
            )

        config_path = self.adapter_path / "adapter_config.json"
        weights_path = self.adapter_path / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"Adapter checkpoint is incomplete: {self.adapter_path}")

        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
        self._adapter_config = adapter_config
        adapter_parent = adapter_config.get("base_model_name_or_path")
        if adapter_parent != BASE_MODEL_ID:
            raise ValueError(
                f"Adapter expects {adapter_parent!r}, while this runner is pinned to {BASE_MODEL_ID!r}."
            )
        manifest_path = self.adapter_path.parent / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Adapter training run_manifest.json is required to verify its exact base revision")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("base_model_id") != BASE_MODEL_ID or manifest.get("base_revision") != BASE_REVISION:
            raise ValueError("Adapter training manifest does not match the pinned base model/revision")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        if self.requested_device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but torch.cuda.is_available() is false.")
        device = torch.device(self._device)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self._dtype_name = str(dtype).replace("torch.", "")

        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID,
            revision=BASE_REVISION,
            trust_remote_code=False,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            revision=BASE_REVISION,
            torch_dtype=dtype,
            device_map={"": str(device)},
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        base.config.use_cache = True
        model = PeftModel.from_pretrained(
            base,
            str(self.adapter_path),
            adapter_name=ADAPTER_NAME,
            is_trainable=False,
            device_map={"": str(device)},
        )
        model.set_adapter(ADAPTER_NAME, inference_mode=True)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model

    def _inputs(self, prompt: str):
        tokenizer = self._tokenizer
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            encoded = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
        else:
            encoded = tokenizer(prompt, return_tensors="pt")

        device = next(self._model.parameters()).device
        return {key: value.to(device) for key, value in encoded.items()}

    def _generate(self, inputs, body: dict[str, Any], adapter_enabled: bool, adapter_name: str = ADAPTER_NAME) -> dict[str, Any]:
        torch = self._torch
        model = self._model
        cuda_active = self._device.startswith("cuda") and torch.cuda.is_available()
        seed = body["seed"]
        torch.manual_seed(seed)
        if cuda_active:
            torch.cuda.manual_seed_all(seed)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        generation_args: dict[str, Any] = {
            "max_new_tokens": body["max_new_tokens"],
            "do_sample": body["do_sample"],
            "use_cache": True,
            "pad_token_id": self._tokenizer.pad_token_id
            if self._tokenizer.pad_token_id is not None
            else self._tokenizer.eos_token_id,
        }
        if body["do_sample"]:
            generation_args["temperature"] = body["temperature"]
            generation_args["top_p"] = body["top_p"]
        if body.get("progress_label"):
            from transformers import StoppingCriteria, StoppingCriteriaList
            owner = self
            initial_length = inputs["input_ids"].shape[-1]
            class Progress(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    owner._stage = f"{body['progress_label']}: {input_ids.shape[-1] - initial_length} tokens"
                    return False
            generation_args["stopping_criteria"] = StoppingCriteriaList([Progress()])

        if adapter_enabled:
            model.set_adapter(adapter_name, inference_mode=True)
            model.base_model.enable_adapter_layers()
            adapter_context = nullcontext()
        else:
            adapter_context = model.disable_adapter()

        with adapter_context:
            with torch.inference_mode():
                if cuda_active:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                sequences = model.generate(**inputs, **generation_args)
                if cuda_active:
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

        prompt_tokens = inputs["input_ids"].shape[-1]
        generated_ids = sequences[0, prompt_tokens:]
        generated_tokens = int(generated_ids.numel())
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        eos_id = self._tokenizer.eos_token_id
        eos_ids = set(eos_id if isinstance(eos_id, (list, tuple)) else [eos_id])
        finished_with_eos = generated_tokens > 0 and int(generated_ids[-1]) in eos_ids
        peak_vram_gb = None
        if cuda_active:
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        return {
            "text": text,
            "latency_seconds": elapsed,
            "generated_tokens": generated_tokens,
            "tokens_per_second": generated_tokens / elapsed if elapsed > 0 else 0.0,
            "peak_vram_gb": peak_vram_gb,
            "finish_reason": "eos" if finished_with_eos else "max_tokens_or_other_stop",
        }

    def compare(self, body: dict[str, Any]) -> dict[str, Any]:
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if len(prompt) > 12000:
            raise ValueError("prompt is too long (maximum 12,000 characters)")

        try:
            max_new_tokens = int(body.get("max_new_tokens", 160))
            seed = int(body.get("seed", 42))
            do_sample = bool(body.get("do_sample", False))
            temperature = float(body.get("temperature", 0.7))
            top_p = float(body.get("top_p", 0.9))
        except (TypeError, ValueError) as exc:
            raise ValueError("generation settings must be numeric") from exc
        if not 8 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be between 8 and 512")
        if not 0 <= seed <= 2147483647:
            raise ValueError("seed must be between 0 and 2,147,483,647")
        if not 0.1 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.1 and 2.0")
        if not 0.1 <= top_p <= 1.0:
            raise ValueError("top_p must be between 0.1 and 1.0")

        request = {
            "prompt": prompt.strip(),
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        }
        with self._lock:
            try:
                if self._model is None:
                    self._stage = "loading model"
                    self._load()
                inputs = self._inputs(request["prompt"])
                prompt_tokens = int(inputs["input_ids"].shape[-1])
                self._stage = "generating base"
                base_result = self._generate(inputs, request, adapter_enabled=False)
                self._stage = "generating adapter"
                adapter_result = self._generate(inputs, request, adapter_enabled=True)
            finally:
                self._stage = "ready"

            result = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "base_model_id": BASE_MODEL_ID,
                "base_revision": BASE_REVISION,
                "adapter_name": "Coder",
                "adapter_path": str(self.adapter_path),
                "adapter_rank": self._adapter_config.get("r"),
                "adapter_target_modules": self._adapter_config.get("target_modules", []),
                "device": f"{self._device} ({self._dtype_name})",
                "prompt": request["prompt"],
                "prompt_tokens": prompt_tokens,
                "settings": {
                    "max_new_tokens": max_new_tokens,
                    "seed": seed,
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else None,
                    "top_p": top_p if do_sample else None,
                },
                "base": base_result,
                "adapter": adapter_result,
                "runtime": {
                    package: version(package)
                    for package in ("torch", "transformers", "peft", "accelerate")
                },
                "cpu_threads": self._torch.get_num_threads(),
                "timing_note": "Generation only; excludes model loading. Base runs first; timings are not a warmed, order-balanced benchmark.",
            }
            RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = RESULT_PATH.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            temporary.replace(RESULT_PATH)
            return result

    def switching(self, body: dict[str, Any]) -> dict[str, Any]:
        from cgr.control.adapter_tasks import task_sequence
        from cgr.control.adapter_runtime import AdapterSwitchRuntime

        selection = body.get("task", "switch_suite")
        task_sequence(selection)
        mode = body.get("mode", "planned")
        if mode not in ("planned", "repair"):
            raise ValueError("mode must be planned or repair")
        cap = int(body.get("max_new_tokens", 160))
        seed = int(body.get("seed", 42))
        if not 16 <= cap <= 256 or not 0 <= seed <= 2147483647:
            raise ValueError("Use 16-256 new tokens and a seed between 0 and 2147483647")
        with self._lock:
            try:
                if self._model is None:
                    self._stage = "loading model"
                    self._load()
                return AdapterSwitchRuntime(self).run(selection, mode, cap, seed)
            finally:
                if self._model is not None:
                    self._model.base_model.disable_adapter_layers()
                self._stage = "ready"

    def webpage_baseline(self) -> dict[str, Any]:
        protocol_path = ROOT / "data" / "validation" / "webpage_design_v1.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        output_dir = ROOT / "results" / "webpage_design_v1"
        with self._lock:
            if (output_dir / "base_result.json").exists():
                return json.loads((output_dir / "base_result.json").read_text(encoding="utf-8"))
            try:
                if self._model is None:
                    self._stage = "loading model"
                    self._load()
                inputs = self._inputs(protocol["prompt"])
                self._stage = "generating webpage with all adapters disabled"
                self._model.base_model.disable_adapter_layers()
                layers = [m for m in self._model.modules() if hasattr(m, "lora_A") and hasattr(m, "disable_adapters")]
                if not layers or not all(m.disable_adapters for m in layers):
                    raise RuntimeError("Base-only adapter state assertion failed")
                result = self._generate(inputs, {**protocol["settings"], "progress_label": "base webpage generation"}, adapter_enabled=False)
                text = result["text"]
                match = re.search(r"```(?:html)?\s*\n(.*?)```", text, flags=re.S | re.I)
                page = match.group(1) if match else text
                result.update({"protocol": protocol, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                    "created_at": datetime.now(timezone.utc).isoformat(), "device": self._device,
                    "dtype": self._dtype_name, "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                    "prompt_token_ids": inputs["input_ids"][0].tolist(),
                    "runtime": {p: version(p) for p in ("torch", "transformers", "peft", "accelerate")},
                    "adapters_disabled": True, "inspected_lora_layers": len(layers),
                    "html_extraction": "outer markdown fence removed" if match else "unchanged model text",
                    "html_complete_marker": "</html>" in page.lower(),
                    "html_sha256": hashlib.sha256(page.encode("utf-8")).hexdigest(),
                    "ui_adapter_status": "training; not run"})
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "base_raw.txt").write_text(text, encoding="utf-8")
                (output_dir / "base.html").write_text(page, encoding="utf-8")
                result["html_hash_encoding"] = "html_sha256 hashes decoded text as UTF-8 with LF newlines"
                result["html_file_sha256"] = hashlib.sha256((output_dir / "base.html").read_bytes()).hexdigest()
                (output_dir / "base_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                return result
            finally:
                self._stage = "ready"

    def webpage_ui(self) -> dict[str, Any]:
        folder = ROOT / "notebooks" / "CGR_UI_Adapter_Training" / "ui_qwen3_0.6b_q_mlp_r16_v1" / "final_adapter"
        directory = ROOT / "results" / "webpage_design_v1"
        baseline = json.loads((directory / "base_result.json").read_text(encoding="utf-8"))
        protocol = baseline["protocol"]
        manifest = json.loads((folder.parent / "run_manifest.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))
        if manifest.get("base_model_id") != BASE_MODEL_ID or manifest.get("base_revision") != BASE_REVISION or config.get("base_model_name_or_path") != BASE_MODEL_ID:
            raise ValueError("UI adapter parent/revision does not match the frozen baseline")
        if set(config.get("target_modules", [])) != set(manifest.get("target_modules_requested", [])) or config.get("r") != manifest.get("lora_rank") or config.get("lora_alpha") != manifest.get("lora_alpha"):
            raise ValueError("UI adapter config and training manifest disagree")
        with self._lock:
            if (directory / "ui_result.json").exists():
                return json.loads((directory / "ui_result.json").read_text(encoding="utf-8"))
            try:
                if self._model is None:
                    self._stage = "loading model"
                    self._load()
                inputs = self._inputs(protocol["prompt"])
                if inputs["input_ids"][0].tolist() != baseline["prompt_token_ids"]:
                    raise ValueError("Prompt token IDs differ from the frozen baseline")
                from transformers import AutoTokenizer
                trained_tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
                trained_ids = trained_tokenizer.apply_chat_template([{"role": "user", "content": protocol["prompt"]}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
                if trained_ids != baseline["prompt_token_ids"]:
                    raise ValueError("UI training tokenizer does not reproduce the frozen prompt")
                if "ui" not in self._model.peft_config:
                    self._stage = "loading final UI adapter"
                    self._model.load_adapter(str(folder), adapter_name="ui", is_trainable=False, device_map={"": self._device})
                self._model.set_adapter("ui", inference_mode=True)
                self._model.base_model.enable_adapter_layers()
                layers = [m for m in self._model.modules() if hasattr(m, "lora_A") and hasattr(m, "disable_adapters")]
                if len(layers) != 112 or any(m.disable_adapters or list(m.active_adapters) != ["ui"] or "ui" not in m.lora_A for m in layers):
                    raise RuntimeError("UI-only LoRA activation assertion failed")
                ui_tensors = [p for n, p in self._model.named_parameters() if ".ui." in n]
                if not ui_tensors or not all(self._torch.isfinite(p).all().item() for p in ui_tensors):
                    raise ValueError("UI adapter has missing or non-finite weights")
                self._model.eval()
                result = self._generate(inputs, {**protocol["settings"], "progress_label": "UI adapter webpage generation"}, True, adapter_name="ui")
                text = result["text"]
                match = re.search(r"```(?:html)?\s*\n(.*?)```", text, flags=re.S | re.I)
                page = match.group(1) if match else text
                result.update({"protocol": protocol, "protocol_sha256": baseline["protocol_sha256"],
                    "created_at": datetime.now(timezone.utc).isoformat(), "device": self._device, "dtype": self._dtype_name,
                    "prompt_tokens": len(trained_ids), "prompt_token_ids": trained_ids,
                    "prompt_tokens_match_baseline": True, "training_tokenizer_matches_baseline": True,
                    "runtime": {p: version(p) for p in ("torch", "transformers", "peft", "accelerate")},
                    "active_adapter": "ui", "adapter_path": str(folder), "adapter_training_step": 464,
                    "adapter_weights_sha256": hashlib.sha256((folder / "adapter_model.safetensors").read_bytes()).hexdigest(),
                    "inspected_lora_layers": len(layers), "adapter_tensor_count": len(ui_tensors), "all_adapter_tensors_finite": True,
                    "prefix_policy": "full UI-adapter prefill; fresh cache", "cpu_threads": self._torch.get_num_threads(),
                    "html_extraction": "outer markdown fence removed" if match else "unchanged model text",
                    "html_complete_marker": "</html>" in page.lower(), "html_sha256": hashlib.sha256(page.encode()).hexdigest(),
                    "html_hash_encoding": "html_sha256 hashes decoded text as UTF-8 with LF newlines"})
                (directory / "ui_raw.txt").write_text(text, encoding="utf-8")
                (directory / "ui.html").write_text(page, encoding="utf-8")
                result["html_file_sha256"] = hashlib.sha256((directory / "ui.html").read_bytes()).hexdigest()
                (directory / "ui_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                return result
            finally:
                if self._model is not None:
                    self._model.base_model.disable_adapter_layers()
                self._stage = "ready"


class StudioHandler(BaseHTTPRequestHandler):
    runner: CompareRunner

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _json(self, data: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler method name
        route = urlparse(self.path).path
        if route in ("/webpage-test", "/webpage-test/base", "/webpage-test/ui", "/webpage-test/ui-source", "/webpage-test/source", "/api/webpage/result"):
            directory = ROOT / "results" / "webpage_design_v1"
            if route == "/api/webpage/result":
                try:
                    result = json.loads((directory / "base_result.json").read_text(encoding="utf-8"))
                    ui_result = json.loads((directory / "ui_result.json").read_text(encoding="utf-8")) if (directory / "ui_result.json").is_file() else None
                    self._json({"result": result, "ui_result": ui_result})
                except OSError:
                    self._json({"result": None})
                return
            path = ROOT / "notebooks" / "webpage_design_test.html" if route == "/webpage-test" else directory / ("ui.html" if route in ("/webpage-test/ui", "/webpage-test/ui-source") else "base.html")
            if not path.is_file():
                self._json({"error": "Page generation is not complete yet"}, 404)
                return
            payload = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8" if route.endswith("source") else "text/html; charset=utf-8")
            if route in ("/webpage-test/base", "/webpage-test/ui"):
                self.send_header("Content-Security-Policy", "sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; form-action 'none'; base-uri 'none'")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        if route == "/api/status":
            self._json(self.runner.status())
            return
        if route == "/api/switching/tasks":
            from cgr.control.adapter_tasks import catalog
            self._json({"tasks": catalog(), "available_specialists": ["coder"],
                        "registry_status": "candidate; integration testing only"})
            return
        if route == "/api/switching/history":
            from cgr.control.adapter_runtime import RESULT
            records = []
            directory = RESULT.parent / "cgr_adapter_switching"
            for path in sorted(directory.glob("*.json"), reverse=True)[:100]:
                try:
                    result = json.loads(path.read_text(encoding="utf-8"))
                    if result.get("adapter_path") == str(self.runner.adapter_path) and result.get("base_revision") == BASE_REVISION:
                        records.append({key: result[key] for key in ("run_id", "created_at", "selection", "mode", "summary")})
                except (OSError, ValueError, KeyError):
                    continue
            self._json({"runs": records})
            return
        if route == "/api/switching/result":
            from cgr.control.adapter_runtime import RESULT
            run_id = parse_qs(urlparse(self.path).query).get("run_id", [""])[0]
            if not re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{8}", run_id):
                self._json({"error": "Invalid run ID"}, 400)
                return
            try:
                result = json.loads((RESULT.parent / "cgr_adapter_switching" / (run_id + ".json")).read_text(encoding="utf-8"))
                if result.get("adapter_path") != str(self.runner.adapter_path) or result.get("base_revision") != BASE_REVISION:
                    raise ValueError("Run belongs to another experiment configuration")
            except (OSError, ValueError):
                self._json({"error": "Run not found for this model/adapter"}, 404)
                return
            self._json({"run": result})
            return
        if route == "/api/switching/latest":
            from cgr.control.adapter_runtime import RESULT
            try:
                result = json.loads(RESULT.read_text(encoding="utf-8"))
                if result.get("adapter_path") != str(self.runner.adapter_path) or result.get("base_revision") != BASE_REVISION:
                    result = None
            except (OSError, ValueError):
                result = None
            self._json({"run": result})
            return
        if route == "/api/latest":
            try:
                result = json.loads(RESULT_PATH.read_text(encoding="utf-8-sig"))
                if not isinstance(result, dict) or not all(key in result for key in ("base", "adapter", "settings", "prompt")):
                    result = None
                if result and (result.get("base_revision") != BASE_REVISION or result.get("adapter_path") != str(self.runner.adapter_path)):
                    result = None
            except (OSError, ValueError):
                result = None
            self._json({"comparison": result})
            return
        if route in ("/", "/index.html", "/switching"):
            try:
                payload = ((ROOT / "notebooks" / "cgr_adapter_switching.html") if route == "/switching" else HTML_PATH).read_bytes()
            except OSError as exc:
                self._json({"error": f"Could not read studio page: {exc}"}, 500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        self._json({"error": "Not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler method name
        route = urlparse(self.path).path
        if route not in ("/api/compare", "/api/switching/run", "/api/webpage/base", "/api/webpage/ui"):
            self._json({"error": "Not found"}, 404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 < content_length <= 20000:
                raise ValueError("request body must be between 1 and 20,000 bytes")
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            if route == "/api/webpage/base":
                self._json(self.runner.webpage_baseline())
            elif route == "/api/webpage/ui":
                self._json(self.runner.webpage_ui())
            else:
                self._json(self.runner.switching(body) if route == "/api/switching/run" else self.runner.compare(body))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            print("Comparison failed:")
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = CompareRunner(args.adapter_path, args.device)
    handler = type("BoundStudioHandler", (StudioHandler,), {"runner": runner})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    print(f"CGR Adapter A/B Lab: http://{args.host}:{args.port}")
    print(f"Base: {BASE_MODEL_ID} @ {BASE_REVISION}")
    print(f"Adapter: {runner.adapter_path}")
    print("Model loading is deferred until the first comparison.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local comparison server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

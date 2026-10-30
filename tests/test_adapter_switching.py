"""Verifier and commitment failure cases; real cache tests run in the harness."""
from __future__ import annotations

from cgr.control.adapter_runtime import verify_candidate
from cgr.control.adapter_tasks import TASKS, task_sequence
from cgr.control import Action, CommitmentDAG, Controller


def test_wrong_patch_cannot_be_committed():
    node = TASKS["rollback_control"]
    verdict = verify_candidate(node, node["injected_candidate"])
    assert not verdict["passed"]
    dag = CommitmentDAG([10, 20])
    result = Controller(dag).step([10, 20, 99], proposed=Action.ANS,
                                  surprise=0, verifier=lambda _: verdict["passed"])
    assert result.action == Action.REV
    assert result.artifact == [10, 20]
    assert not dag.get(1).verified


def test_correct_patch_passes_but_truncated_patch_does_not():
    node = TASKS["clamp"]
    code = "def clamp(value, lower, upper):\n    return max(lower, min(value, upper))"
    assert verify_candidate(node, code)["passed"]
    assert not verify_candidate(node, code, truncated=True)["passed"]


def test_imports_and_dunder_access_are_rejected():
    node = TASKS["rollback_control"]
    for code in ("import os\ndef add(a,b): return a+b",
                 "def add(a,b): return a.__class__",
                 "def add(a,b): return open('file').read()"):
        assert not verify_candidate(node, code)["passed"]


def test_nonterminating_candidate_times_out():
    assert not verify_candidate(TASKS["rollback_control"],
        "def add(a,b):\n    while True:\n        pass")["passed"]


def test_sequence_contains_repeated_base_and_specialist_boundaries():
    nodes = task_sequence("switch_suite")
    assert [node["required_skill"] for node in nodes] == ["base", "coder", "base", "coder", "base"]
    assert task_sequence("unknown_control")[0]["required_skill"] == "unknown"


def tiny_runtime():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from types import SimpleNamespace
    from cgr.control.adapter_runtime import AdapterSwitchRuntime
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128)
    model = get_peft_model(Qwen3ForCausalLM(config),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "gate_proj"], task_type="CAUSAL_LM"),
        adapter_name="coder").eval()
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.1)
    tokenizer = SimpleNamespace(eos_token_id=63, decode=lambda ids, **_: str(ids))
    return AdapterSwitchRuntime(SimpleNamespace(_torch=torch, _model=model,
        _tokenizer=tokenizer, _dtype_name="float32"))


def test_real_cache_rollback_and_bridge_resume_with_tiny_qwen():
    import copy
    import pytest
    runtime = tiny_runtime()
    prefix = [1, 2, 3, 4]
    with runtime._policy("base"):
        output = runtime._forward(runtime._tensor(prefix))
    snapshot = runtime._snapshot(output.past_key_values)
    candidate = runtime._decode(prefix, output.past_key_values, "base", 3, output.logits[:, -1, :])
    assert runtime._crop(candidate["cache"], len(prefix), snapshot)
    with runtime._policy("coder"):
        specialist = runtime._forward(runtime._tensor([5, 6]), copy.deepcopy(output.past_key_values))
    base_cache, checks = runtime._bridge(prefix, output.past_key_values, [5, 6])
    assert checks["passed"] and checks["continuation_matches"]
    assert base_cache.get_seq_length() == 6
    assert runtime._same_prefix(output.past_key_values, snapshot, len(prefix))
    corrupted = copy.deepcopy(output.past_key_values)
    with runtime.torch.inference_mode():
        corrupted.layers[0].keys[..., 0, :] += 1
    with pytest.raises(AssertionError):
        runtime._crop(corrupted, len(prefix), snapshot)

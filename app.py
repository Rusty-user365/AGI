"""Local CPU chat UI for GPT-2, DiffuGPT-S, and a CGR checkpoint.

Select one model per turn. The conversation history is shared when switching
models. This app performs inference only; it does not train or update memory.
C1 is an update-time guard, not an output filter. The optional DAG controls are
manual test signals, not learned model behavior.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")
_CACHE: dict[tuple, object] = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class LoadedCGR:
    model: object
    cfg: object


def _load_gpt2(model_id: str):
    key = ("gpt2", model_id)
    with _CACHE_LOCK:
        if key not in _CACHE:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model.to(DEVICE).eval()
            _CACHE[key] = (tokenizer, model)
        return _CACHE[key]


def _load_diffugpt(model_dir: str):
    key = ("diffugpt", str(Path(model_dir).resolve()))
    with _CACHE_LOCK:
        if key not in _CACHE:
            from cgr.config import Config
            from cgr.model.diffugpt import DiffuGPTBackbone
            from cgr.tokenizer import GPT2Tokenizer

            cfg = Config.preset("diffugpt")
            cfg.model.backbone_path = model_dir
            tokenizer = GPT2Tokenizer(model_dir)
            model = DiffuGPTBackbone(cfg.model, tokenizer.mask_id,
                                     tokenizer.pad_id, path=model_dir,
                                     load_weights=True)
            model.to(DEVICE).eval()
            _CACHE[key] = (tokenizer, model, cfg)
        return _CACHE[key]


def _load_cgr(checkpoint: str):
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    key = ("cgr", str(checkpoint_path), checkpoint_path.stat().st_mtime_ns)
    with _CACHE_LOCK:
        if key not in _CACHE:
            from cgr.train import load_system
            model, cfg = load_system(checkpoint_path)
            model.to(DEVICE).eval()
            _CACHE[key] = LoadedCGR(model, cfg)
        return _CACHE[key]


def _prefix_and_span(tokenizer, prompt: str, max_len: int,
                     requested_tokens: int):
    mask_count = min(max(int(requested_tokens), 4), max_len - 2)
    prefix = tokenizer.encode(prompt.rstrip() + "\n",
                              add_bos=True, add_eos=False)
    prefix = prefix[:max_len - mask_count]
    if not prefix:
        prefix = [tokenizer.bos_id]
    start = len(prefix)
    ids = torch.tensor([prefix + [tokenizer.mask_id] * mask_count],
                       dtype=torch.long, device=DEVICE)
    return ids, (start, start + mask_count)


def _chat_prompt(history: list[dict], message: str) -> str:
    """Render recent turns as a plain transcript shared by all models."""
    lines = []
    for item in (history or [])[-12:]:
        role = item.get("role", "user")
        content = item.get("content", "")
        if not isinstance(content, str):
            continue
        label = "Assistant" if role == "assistant" else "User"
        lines.append(f"{label}: {content}")
    lines.extend((f"User: {message.strip()}", "Assistant:"))
    return "\n".join(lines)


@torch.inference_mode()
def _generate_gpt2(prompt: str, token_budget: int, temperature: float,
                   model_id: str) -> tuple[str, str]:
    tokenizer, model = _load_gpt2(model_id.strip())
    context_limit = max(16, int(model.config.n_positions) - token_budget)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=min(900, context_limit))
    input_ids = encoded["input_ids"].to(DEVICE)
    kwargs = {"max_new_tokens": token_budget,
              "pad_token_id": tokenizer.pad_token_id,
              "eos_token_id": tokenizer.eos_token_id}
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
    else:
        kwargs.update(do_sample=False)
    generated = model.generate(input_ids, **kwargs)
    text = tokenizer.decode(generated[0, input_ids.shape[1]:],
                            skip_special_tokens=True).strip()
    return text, f"GPT-2 ({model_id}) · causal continuation · CPU"


@torch.inference_mode()
def _generate_diffugpt(prompt: str, token_budget: int,
                       model_dir: str) -> tuple[str, str]:
    tokenizer, model, cfg = _load_diffugpt(model_dir.strip())
    ids, span = _prefix_and_span(tokenizer, prompt, cfg.model.max_len,
                                 token_budget)
    output = model.infill(ids, [span], steps=token_budget, seed=0,
                          order=cfg.model.infill_order)
    text = tokenizer.decode(output[0, span[0]:span[1]].tolist()).strip()
    return text, "DiffuGPT-S · masked-span infilling · CPU · greedy"


@torch.inference_mode()
def _generate_cgr(prompt: str, token_budget: int, checkpoint: str,
                  expected_answer: str, previous_checkpoint: str,
                  controller_surprise: float, ttc: float,
                  deadline: float) -> tuple[str, str]:
    from cgr.control import Action, CommitmentDAG, Controller

    loaded = _load_cgr(checkpoint)
    system, cfg = loaded.model, loaded.cfg
    tokenizer = system.tok
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    prompt_ids = prompt_ids[:max(2, cfg.model.max_len - token_budget)]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    clock = torch.tensor([[0.0, 0.0, 3.0, 10.0, 1.0, 0.0, 0.0, 0.3]],
                         dtype=torch.float32, device=DEVICE)
    features = system.forward_episodes(ids, clock, memory_texts=[prompt],
                                       memory_enabled=True, hard=True)
    heads = system.heads.predict(features["feats"])
    _, gate_info = system.gate.decide(features["action_logits"][0])
    fill_ids, span = _prefix_and_span(tokenizer, prompt, cfg.model.max_len,
                                      token_budget)
    output = system.backbone.infill(
        fill_ids, [span], cond=features["cond"], steps=token_budget,
        seed=0, order=cfg.model.infill_order)
    candidate = tokenizer.decode(output[0, span[0]:span[1]].tolist()).strip()

    expected = (expected_answer or "").strip()
    surprise = float(controller_surprise or 0)
    time_to_completion = float(ttc or 0)
    time_deadline = float(deadline if deadline is not None else 60)
    should_run_controller = (bool(expected) or surprise > 0.5
                             or time_to_completion > time_deadline)
    controller_status = "DAG controller not run; candidate remains unverified."
    response = candidate
    if should_run_controller:
        dag = CommitmentDAG((previous_checkpoint or "").strip())
        controller = Controller(dag, surprise_threshold=0.5)
        exact_verifier = lambda text: bool(expected) and text.strip() == expected
        decision = controller.step(
            candidate, proposed=Action(gate_info["final"]),
            surprise=surprise, verifier=exact_verifier,
            ttc=time_to_completion, deadline=time_deadline)
        response = (f"**Candidate**\n\n{candidate}\n\n"
                    f"**After controller**\n\n{decision.artifact}")
        controller_status = (
            f"DAG: {decision.action.value} ({decision.reason}); "
            f"checkpoint #{decision.checkpoint_id}.")

    epi_names = ["known", "computable", "check_later", "unknown"]
    epi_id = int(heads["epistemic"][0])
    diagnostics = (
        f"Gate: {gate_info['final']} (raw {gate_info['base']}) · "
        f"value {float(heads['value'][0]):.3f} · "
        f"disagreement {float(heads['disagreement'][0]):.3f} · "
        f"epistemic {epi_names[epi_id]}. {controller_status} "
        "C1 is an update-time guard; no update was attempted.")
    return response, diagnostics


def chat_turn(message: str, history: list[dict], model_choice: str,
              max_new_tokens: int, temperature: float, gpt2_id: str,
              diffugpt_dir: str, cgr_checkpoint: str, expected_answer: str,
              previous_checkpoint: str, controller_surprise: float,
              ttc: float, deadline: float):
    if not message or not message.strip():
        return history or [], "", "Enter a message to send."
    history = list(history or [])
    transcript = _chat_prompt(history, message)
    token_budget = int(max_new_tokens)
    try:
        if model_choice == "GPT-2":
            response, status = _generate_gpt2(
                transcript, token_budget, temperature, gpt2_id)
        elif model_choice == "DiffuGPT-S":
            response, status = _generate_diffugpt(
                transcript, token_budget, diffugpt_dir)
        else:
            response, status = _generate_cgr(
                transcript, token_budget, cgr_checkpoint, expected_answer,
                previous_checkpoint, controller_surprise, ttc, deadline)
    except Exception as e:
        response = f"Could not run {model_choice}: {type(e).__name__}: {e}"
        status = f"{model_choice} failed. Check its dependencies, model path, and checkpoint."
    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": f"[{model_choice}] {response}"},
    ])
    return history, "", status


CSS = """
.gradio-container { max-width: 1050px !important; margin: 0 auto !important; }
#conversation { border: 1px solid var(--border-color-primary); border-radius: 16px; }
#send-row { align-items: end; }
"""

with gr.Blocks(title="CGR Chat") as demo:
    gr.Markdown("# CGR Lab\nChat with one selected model at a time.")
    with gr.Row():
        model_choice = gr.Dropdown(
            choices=["GPT-2", "DiffuGPT-S", "CGR"], value="CGR",
            label="Model", scale=2)
        max_new_tokens = gr.Slider(4, 48, value=16, step=1,
                                   label="Response tokens", scale=2)
        temperature = gr.Slider(0, 1.2, value=0.7, step=0.05,
                                label="GPT-2 temperature", scale=2)
    chatbot = gr.Chatbot(height=590, elem_id="conversation",
                         show_label=False)
    with gr.Row(elem_id="send-row"):
        message = gr.Textbox(label="Message", placeholder="Message CGR Lab",
                             lines=2, scale=8, show_label=False)
        send = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("New chat", scale=1)
    status = gr.Markdown("Select a model above. Each turn runs only that model.")
    with gr.Accordion("Model paths and CGR controller tests", open=False):
        gpt2_id = gr.Textbox(label="GPT-2 model ID or local directory",
                             value="gpt2")
        diffugpt_dir = gr.Textbox(label="DiffuGPT-S directory",
                                  value=str(ROOT / "DiffuGPT-S"))
        cgr_checkpoint = gr.Textbox(
            label="CGR checkpoint",
            value=str(ROOT / "checkpoints" / "diffugpt_5k_e10_cgr" / "stage_b.pt"))
        expected_answer = gr.Textbox(
            label="Exact expected CGR response (optional)",
            placeholder="Used only when CGR is selected; exact string match.")
        previous_checkpoint = gr.Textbox(
            label="Previous verified checkpoint text (optional)")
        with gr.Row():
            controller_surprise = gr.Slider(
                0, 1, value=0, step=0.05,
                label="Test surprise signal (> 0.50 routes REV)")
            ttc = gr.Number(value=0, label="Test time-to-completion")
            deadline = gr.Number(value=60, label="Test deadline")

    inputs = [message, chatbot, model_choice, max_new_tokens, temperature,
              gpt2_id, diffugpt_dir, cgr_checkpoint, expected_answer,
              previous_checkpoint, controller_surprise, ttc, deadline]
    outputs = [chatbot, message, status]
    send.click(chat_turn, inputs=inputs, outputs=outputs)
    message.submit(chat_turn, inputs=inputs, outputs=outputs)
    clear.click(lambda: ([], "", "New conversation."), outputs=outputs)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", share=False,
        theme=gr.themes.Soft(), css=CSS)

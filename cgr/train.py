"""Training: Stage A (supervised warmup) and Stage B (live gate).

Stage A trains backbone + reasoner + heads + gate jointly against the
scaffolding's labels. Stage B replaces argmax with Gumbel-softmax and adds a
KL leash to the scaffolding policy, which is annealed.

Stage C (RL on verified outcomes) is specified in the paper and not run. It is
also not implemented here, deliberately: shipping an untested RL loop and
calling it "specified" is how the paper got into trouble in the first place.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Config
from .data.corpus import build_corpus, make_repair_set
from .data.facts import make_claim_set
from .model.gate import A, ACTIONS
from .model.system import CGRSystem
from .scaffold import (episodes_from_claims, episodes_from_records,
                       episodes_from_repairs)
from .tokenizer import TokenizerLike, tokenizer_from_spec


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------

def collate(episodes, tok: TokenizerLike, max_len: int) -> dict:
    L = min(max(e.ids.numel() for e in episodes), max_len)
    ids = torch.full((len(episodes), L), tok.pad_id, dtype=torch.long)
    for i, e in enumerate(episodes):
        n = min(e.ids.numel(), L)
        ids[i, :n] = e.ids[:n]
    clock = torch.stack([e.clock for e in episodes])
    labels = {k: torch.stack([e.labels[k] for e in episodes])
              for k in episodes[0].labels}
    actions = torch.tensor([e.action for e in episodes], dtype=torch.long)
    return {"ids": ids, "clock": clock, "labels": labels, "actions": actions}


def build_episodes(cfg: Config, tok: TokenizerLike, splits: dict,
                   n_claims: int | None = None, n_repairs: int | None = None,
                   seed: int = 0):
    n_claims = n_claims or len(splits["train"])
    n_repairs = n_repairs or max(200, len(splits["train"]) // 2)
    noise = 0.1

    docs = episodes_from_records(splits["train"], noise=noise, seed=seed)
    claims_pos = make_claim_set(n_claims // 2, seed=seed + 1, marker_present=True)
    claims_neg = make_claim_set(n_claims // 2, seed=seed + 2, marker_present=False)
    claims = episodes_from_claims(claims_pos + claims_neg, tok, noise=noise,
                                  seed=seed + 3)
    repairs = make_repair_set(splits["train"], n_repairs // 2, seed=seed + 4)
    reps = []
    for kind in ("local", "cross_block", "cross_block_partial"):
        reps += episodes_from_repairs(repairs[kind], tok, noise=noise, seed=seed + 5)

    episodes = docs + claims + reps
    rng = random.Random(seed)
    rng.shuffle(episodes)
    return episodes


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------

def _atomic_torch_save(obj, path: Path) -> None:
    """Write a checkpoint without leaving a truncated file on power loss.

    `torch.save` to the final path means an interrupted write destroys the only
    recovery point. Writing to a sibling temp file and renaming is atomic on
    the same volume, so a crash leaves either the old checkpoint or the new one.
    """
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_text(text: str, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@torch.no_grad()
def side_channel_stats(backbone, reasoner, reasoner_ref, probe_ids,
                       probe_cond) -> dict:
    """Evidence that the conditioning path is wired, not just present.

    `film_max_abs`/`film_l2` say whether FiLM has left its zero init.
    `reasoner_l2_delta` says whether the reasoner has moved since the run
    started. `cond_probe_delta` is the decisive one: it is the change in the
    backbone's output when `cond` is replaced by a random vector, so it is zero
    for a frozen map, a detached vector, or a zero multiplier -- the three ways
    this path has silently died before -- and non-zero only when the vector
    genuinely reaches the computation.
    """
    films = [m.film.weight.detach() for m in backbone.h]
    stats = {
        "film_max_abs": max(float(f.abs().max()) for f in films),
        "film_l2": float(torch.sqrt(sum((f * f).sum() for f in films))),
        "reasoner_l2_delta": (
            float(torch.sqrt(sum(((p.detach() - r) ** 2).sum()
                                 for p, r in zip(reasoner.parameters(),
                                                 reasoner_ref))))
            if reasoner_ref else 0.0),
    }
    was_training = backbone.training
    backbone.eval()
    plain = backbone(probe_ids)
    probed = backbone(probe_ids, probe_cond)
    backbone.train(was_training)
    stats["cond_probe_delta"] = float((plain - probed).abs().max())
    stats["side_channel_live"] = stats["cond_probe_delta"] > 0.0
    return stats


def pick_device(name: str | None = None) -> torch.device:
    """Resolve a device name. `None`/"auto" means CUDA when it is available.

    Kept in one place because the choice is recorded in the result files: a
    number produced on a 4 GB card and a number produced on CPU are the same
    model but not the same run, and the paper has to say which one it was.
    """
    if name in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(cfg: Config, out_dir: str | Path = "checkpoints",
          verbose: bool = True, device: str | None = None,
          objective: str = "cgr", log_every: int = 100,
          resume: str | None = None, cond_mode: str = "reasoner",
          overwrite: bool = False) -> dict:
    if objective not in ("cgr", "backbone"):
        raise ValueError("objective must be 'cgr' or 'backbone'")
    if objective == "backbone" and not getattr(cfg.train, "freeze_backbone", False):
        raise ValueError("the backbone baseline requires a frozen pretrained base")
    if cond_mode not in ("reasoner", "learned", "none"):
        raise ValueError("cond_mode must be 'reasoner', 'learned' or 'none'")
    if objective == "backbone" and cond_mode != "none":
        # The denoising-only control has no reasoner and no side-channel loss,
        # so a cond_mode other than "none" would be silently ignored.
        raise ValueError("the backbone objective does not use a conditioning "
                         "path; use cond_mode='none'")
    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train_summary.json").exists() and not overwrite:
        # A completed run is expensive and is the evidence for whatever the
        # paper claims. Writing a different experiment into the same directory
        # would destroy it silently, so this refuses unless asked explicitly.
        raise FileExistsError(
            f"{out} already holds a completed run (train_summary.json). "
            f"Choose another --out, or pass --overwrite to replace it.")

    tok, splits = build_corpus(cfg)
    # Size the vocabulary to the tokenizer. For the corpus-built word
    # vocabulary this is the real type count (a 512-row table for 183 words
    # would be 90% padding, and the model could emit ids the tokenizer cannot
    # decode). For GPT-2 it is the fixed 50258 rows, and the assignment is a
    # no-op that keeps the checkpoint honest about what it was built for.
    cfg.model.vocab_size = tok.vocab_size
    episodes = build_episodes(cfg, tok, splits, seed=cfg.train.seed)

    dev = pick_device(device if device is not None
                      else getattr(cfg.train, "device", "cpu"))
    system = CGRSystem(cfg, tok).to(dev)
    if resume:
        ck = torch.load(Path(resume), map_location="cpu", weights_only=False)
        # Checkpoints written before the matched control existed have no
        # `const_cond`. That key is *new by design* and is expected to be
        # absent; anything else missing is a real mismatch and must fail.
        missing, unexpected = system.load_state_dict(ck["model"], strict=False)
        allowed = {k for k in missing if k.endswith("const_cond")}
        leftover = sorted(set(missing) - allowed)
        if unexpected or leftover:
            raise RuntimeError(
                f"checkpoint {resume} does not match the model: "
                f"missing={leftover[:5]} unexpected={sorted(unexpected)[:5]}")
        if verbose:
            note = (f"; initialised {len(allowed)} new side-channel key(s) "
                    f"from their init" if allowed else "")
            print(f"[train] resumed weights from {resume} "
                  f"(optimizer state is reset){note}")
    # PEFT: the pre-trained base stays frozen. Only the low-rank adapters, the
    # plastic substrate and the CGR modules are trainable. Doing this before the
    # optimiser is built keeps the optimiser's parameter groups honest.
    if getattr(cfg.train, "freeze_backbone", False):
        system.freeze_backbone_base()
        if objective == "backbone":
            # Matched task-adaptation control: only LoRA learns from the same
            # synthetic token stream, with the denoising objective. CGR heads,
            # reasoner, gate, side channel and consolidation adapter are frozen.
            for p in system.parameters():
                p.requires_grad_(False)
            for p in system.lora_params():
                p.requires_grad_(True)
        else:
            # A frozen or zero-initialised side channel trains happily and is
            # indistinguishable from a working one by loss alone, so check the
            # contract before the first step rather than after the run.
            system.backbone.assert_side_channel_live()
        n_train = sum(p.numel() for p in system.parameters() if p.requires_grad)
        n_base = system.n_params()["backbone"]
        if verbose:
            print(f"[train] frozen pre-trained base; trainable={n_train:,} "
                  f"of {n_base:,} backbone params "
                  f"(LoRA rank {cfg.model.lora_rank})")
    opt = torch.optim.AdamW(system.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)

    # Learning-rate schedule. `constant` is the default and is what every
    # v0.3 number was produced with; changing it retroactively would make the
    # committed tables unreproducible. Fine-tuning a pre-trained backbone is
    # the one case where warmup is not optional, so the diffugpt preset opts in.
    sched = None
    if getattr(cfg.train, "lr_schedule", "constant") != "constant":
        steps_per_epoch = math.ceil(len(episodes) / cfg.train.batch_size)
        total = max(steps_per_epoch * (cfg.train.stage_a_epochs
                                       + cfg.train.stage_b_epochs), 1)
        warm = max(int(cfg.train.warmup), 1)

        def _lr_lambda(step: int) -> float:
            if step < warm:
                return (step + 1) / warm
            prog = (step - warm) / max(total - warm, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    # A crash (host restart, power loss, driver reset) must not force the whole
    # schedule to restart. The checkpoint carries the weights; this file carries
    # how much of the schedule is already done. Without it, `--resume` would
    # re-run 4A+1B from epoch 0 and a resumed run would silently train for more
    # epochs than configured. A mid-epoch checkpoint loses at most the steps
    # since the last save (500), because the partially completed epoch is simply
    # re-run.
    hist = {"stage_a": [], "stage_b": []}
    done_a = done_b = 0
    prior_elapsed = 0.0
    progress_path = out / "train_progress.json"
    if resume and progress_path.exists():
        try:
            prior = json.loads(progress_path.read_text(encoding="utf-8"))
            hist = {"stage_a": list(prior.get("history", {}).get("stage_a", [])),
                    "stage_b": list(prior.get("history", {}).get("stage_b", []))}
            done_a, done_b = len(hist["stage_a"]), len(hist["stage_b"])
            prior_elapsed = float(prior.get("elapsed_s", 0.0) or 0.0)
            if verbose:
                print(f"[train] resuming after {done_a} Stage-A and "
                      f"{done_b} Stage-B completed epochs")
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            hist = {"stage_a": [], "stage_b": []}
            done_a = done_b = 0
            if verbose:
                print(f"[train] could not read {progress_path} ({exc}); "
                      f"starting the schedule from epoch 0")
    t0 = time.time()

    # The masking distribution is part of the objective, not a sampler detail:
    # the repair benchmark masks a whole contiguous block, so a token-only
    # training mask teaches a different operation. Both the CGR run and the
    # LoRA-only control receive the same mask policy.
    mask_kwargs = {
        "mask_mode": getattr(cfg.train, "mask_mode", "token"),
        "block_mask_prob": getattr(cfg.train, "block_mask_prob", 0.5),
        "block_mask_min": getattr(cfg.train, "block_mask_min", 15),
        "block_mask_max": getattr(cfg.train, "block_mask_max", 20),
        "sep_id": tok.sep_id,
    }
    if not getattr(system.backbone, "supports_block_mask", False):
        # The canvas backbone has no block-mask implementation. Passing the
        # kwargs would raise; dropping them silently would train a different
        # objective than the one the command line asked for. So refuse unless
        # the request is the default token mask.
        if mask_kwargs["mask_mode"] != "token":
            raise ValueError(
                f"backbone {type(system.backbone).__name__} does not support "
                f"mask_mode={mask_kwargs['mask_mode']!r}")
        mask_kwargs = {}

    # Where the conditioning vector comes from. "reasoner" is the CGR arm: a
    # per-document steering vector. "learned" is the matched control: the same
    # FiLM side channel and the same parameter count, driven by one learned
    # vector for every example, so the conditioning carries no information
    # about the input. "none" disables the side channel.
    reasoning_mode = {"reasoner": "full", "learned": "learned",
                      "none": "none"}[cond_mode]

    # ------------------------------------------------------------------
    # Side-channel instrumentation
    # ------------------------------------------------------------------
    # A dead conditioning path trains to a plausible loss and is invisible in
    # every aggregate number: that is exactly how it survived five hours of
    # training and a gate run. The run therefore records the evidence that the
    # wire is live, at every checkpoint:
    #
    #   film_max_abs / film_l2   is FiLM still at its zero init?
    #   film_grad_max            does the loss reach FiLM at all?
    #   reasoner_l2_delta        has the reasoner moved since the run started?
    #   cond_probe_delta         does perturbing `cond` change the output?
    #
    # The last one is decisive. Magnitudes can be non-zero while `cond` is
    # detached; only a changed output proves the vector is in the graph.
    has_side = hasattr(system.backbone, "const_cond")
    probe_ids = probe_cond = None
    ref = {"reasoner": []}
    if has_side:
        with torch.no_grad():
            probe_text = ("this essay argues that street trees cool cities "
                          "with 3 reasons .")
            probe_ids = torch.tensor(
                tok.encode(probe_text, add_bos=False, add_eos=False)
                [:cfg.model.max_len], device=dev).unsqueeze(0)
            g = torch.Generator().manual_seed(cfg.train.seed)
            probe_cond = torch.randn(1, cfg.model.d_model, device=dev,
                                     generator=g)
        ref["reasoner"] = [p.detach().clone()
                           for p in system.reasoner.parameters()]

    diag = {"step": 0, "stage": None, "epoch": None,
            "film_max_abs": 0.0, "film_l2": 0.0, "film_grad_max": 0.0,
            "reasoner_l2_delta": 0.0, "cond_abs_mean": 0.0,
            "cond_probe_delta": None, "side_channel_live": False,
            "cond_mode": cond_mode}

    @torch.no_grad()
    def measure_diagnostics(cond_value: torch.Tensor | None = None) -> None:
        """Refresh the side-channel numbers. Cheap: one short forward."""
        if not has_side:
            return
        diag.update(side_channel_stats(system.backbone, system.reasoner,
                                       ref["reasoner"], probe_ids,
                                       probe_cond))
        if cond_value is not None:
            diag["cond_abs_mean"] = float(cond_value.detach().abs().mean())

    def save_checkpoint() -> None:
        """Save the current weights so an interrupted run can be resumed.

        A 5k run is hours long and the host can restart. Saving only at the end
        throws away every completed epoch; saving after each epoch costs a few
        seconds and makes `--resume <out>/stage_b.pt` an honest continuation.
        """
        state = {k: v.detach().cpu() for k, v in system.state_dict().items()}
        _atomic_torch_save({"model": state,
                            "cfg": cfg.__dict__,
                            "tokenizer": tok.spec(),
                            "n_params": system.n_params()},
                           out / "stage_b.pt")
        tok.save(out / "tokenizer.json")
        cfg.to_json(out / "config.json")
        _atomic_text(json.dumps(
            {"history": hist, "n_episodes": len(episodes),
             "elapsed_s": prior_elapsed + (time.time() - t0)}, indent=2),
            out / "train_progress.json")
        # Written next to the weights so an interrupted run can be audited:
        # if `side_channel_live` is false at the last checkpoint, the run did
        # not exercise the conditioning path no matter what the loss says.
        _atomic_text(json.dumps(diag, indent=2), out / "train_diagnostics.json")

    def run_epoch(epoch: int, stage: str) -> dict:
        epoch_t0 = time.perf_counter()
        system.train()
        rng = random.Random(cfg.train.seed + epoch)
        order = list(range(len(episodes)))
        rng.shuffle(order)
        agg = {"loss": 0.0, "n": 0, "gate_acc": 0.0, "gate_n": 0}
        last_cond = None
        for i in range(0, len(order), cfg.train.batch_size):
            batch = collate([episodes[j] for j in order[i:i + cfg.train.batch_size]],
                            tok, cfg.model.max_len)
            batch = {k: (v.to(dev) if torch.is_tensor(v)
                         else {kk: vv.to(dev) for kk, vv in v.items()})
                     for k, v in batch.items()}
            if objective == "backbone":
                loss = system.backbone.masked_denoising_loss(
                    batch["ids"], **mask_kwargs)
            else:
                out_f = system.forward_episodes(
                    batch["ids"], batch["clock"],
                    reasoning_mode=reasoning_mode,
                    memory_texts=None, hard=(stage == "A"))
                denoise_loss = system.backbone.masked_denoising_loss(
                    batch["ids"], out_f["cond"], **mask_kwargs)
                last_cond = out_f["cond"]
                r_loss = system.reasoner.loss(out_f["pooled"],
                                              batch["labels"]["value"])
                h_loss, _ = system.heads.loss(out_f["preds"], batch["labels"],
                                              cfg.train.head_weights)
                if stage == "A":
                    g_loss = F.cross_entropy(out_f["action_logits"], batch["actions"])
                else:
                    tau = max(0.5, cfg.model.gumbel_tau * (1.0 - epoch / max(cfg.train.stage_b_epochs, 1)))
                    soft, _ = system.gate(out_f["feats"], hard=False, tau=tau)
                    oracle = F.one_hot(batch["actions"], cfg.model.n_actions).float()
                    g_loss = F.kl_div((soft + 1e-8).log(), oracle,
                                      reduction="batchmean") * cfg.train.kl_leash
                loss = denoise_loss + r_loss + h_loss + g_loss

            opt.zero_grad()
            loss.backward()
            if objective == "cgr" and agg["n"] == 0:
                # One backward is enough to prove the conditioning path is in
                # the graph. A zero gradient here means `cond` was detached or
                # multiplied by a frozen zero map -- a failure that no loss
                # value, and no final accuracy, would ever reveal.
                g = system.backbone.h[0].film.weight.grad
                if g is None or float(g.abs().max()) == 0.0:
                    raise RuntimeError(
                        "the side channel received no gradient on the first "
                        "step; cond_mode=%r is not reaching the backbone"
                        % cond_mode)
            torch.nn.utils.clip_grad_norm_(
                [p for p in system.parameters() if p.requires_grad],
                cfg.train.grad_clip)
            # Captured before the step: after `opt.step()` the gradient is still
            # there, but capturing here makes it unambiguous that this is the
            # gradient the step used.
            if has_side:
                grads = [m.film.weight.grad for m in system.backbone.h]
                diag["film_grad_max"] = max(
                    (float(g.abs().max()) for g in grads if g is not None),
                    default=0.0)
            opt.step()
            if sched is not None:
                sched.step()
            diag["step"] = agg["n"] + 1
            diag["stage"] = stage
            diag["epoch"] = epoch

            if objective == "cgr":
                with torch.no_grad():
                    pred_a = out_f["action_logits"].argmax(-1)
                    agg["gate_acc"] += (pred_a == batch["actions"]).float().mean().item()
                    agg["gate_n"] += 1
            agg["loss"] += float(loss.detach())
            agg["n"] += 1
            # Periodic recovery point. A host restart or a driver reset should
            # not throw away an hour of work; the epoch boundary is too coarse.
            if agg["n"] % 500 == 0:
                measure_diagnostics(last_cond)
                save_checkpoint()
            if (verbose and log_every > 0
                    and agg["n"] % log_every == 0):
                print(f"[{stage} {epoch}] step {agg['n']}/{len(range(0, len(order), cfg.train.batch_size))} "
                      f"loss={float(loss.detach()):.4f}", flush=True)
        agg["loss"] /= max(agg["n"], 1)
        agg["gate_acc"] = (agg["gate_acc"] / agg["gate_n"]
                           if agg["gate_n"] else None)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        agg["elapsed_s"] = time.perf_counter() - epoch_t0
        agg["step_s"] = agg["elapsed_s"] / max(agg["n"], 1)
        agg["epoch"] = epoch
        agg["stage"] = stage
        measure_diagnostics(last_cond)
        if verbose and has_side:
            print(f"[diag {stage} {epoch}] film_max={diag['film_max_abs']:.3e} "
                  f"film_grad={diag['film_grad_max']:.3e} "
                  f"reasoner_delta={diag['reasoner_l2_delta']:.3e} "
                  f"cond_probe={diag['cond_probe_delta']:.3e} "
                  f"live={diag['side_channel_live']}", flush=True)
        return agg

    for ep in range(done_a, cfg.train.stage_a_epochs):
        h = run_epoch(ep, "A")
        hist["stage_a"].append(h)
        save_checkpoint()
        if verbose:
            gate_acc = (f"{h['gate_acc']:.3f}" if h["gate_acc"] is not None
                        else "n/a")
            print(f"[A {ep}] loss={h['loss']:.4f} gate_acc={gate_acc}")
    for ep in range(done_b, cfg.train.stage_b_epochs):
        h = run_epoch(ep, "B")
        hist["stage_b"].append(h)
        save_checkpoint()
        if verbose:
            gate_acc = (f"{h['gate_acc']:.3f}" if h["gate_acc"] is not None
                        else "n/a")
            print(f"[B {ep}] loss={h['loss']:.4f} gate_acc={gate_acc}")

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        cuda_memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(dev),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(dev),
        }
    else:
        cuda_memory = None

    # Saved on CPU so a checkpoint trained on a GPU loads on a CPU-only runner
    # without a `map_location` dance, and so the file is device-agnostic.
    # Reuse the same atomic path as the periodic save: the final checkpoint is
    # the one most likely to be written while the host is shutting down.
    save_checkpoint()

    summary = {"elapsed_s": time.time() - t0,
               "elapsed_s_total": prior_elapsed + (time.time() - t0),
               "params": system.n_params(),
               "history": hist, "vocab": tok.vocab_size,
               "device": str(dev), "n_episodes": len(episodes),
               "freeze_backbone": bool(getattr(cfg.train, "freeze_backbone",
                                               False)),
               "lora_rank": int(getattr(cfg.model, "lora_rank", 0)),
               "objective": objective,
               "cond_mode": cond_mode,
               "side_channel": diag,
               "resumed_from": str(resume) if resume else None,
               "prior_epochs": {"stage_a": done_a, "stage_b": done_b},
               "cuda_memory": cuda_memory}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2),
                                            encoding="utf-8")
    if verbose:
        print(json.dumps(summary["params"], indent=2))
    return summary


def load_system(path: str | Path) -> tuple[CGRSystem, Config]:
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg_path = Path(path).parent / "config.json"
    cfg = Config.from_json(cfg_path) if cfg_path.exists() else Config.preset("smoke")
    raw = ck["tokenizer"]
    # Checkpoints written before the DiffuGPT integration stored the bare
    # `itos` list; newer ones store a spec dict that names the vocabulary.
    tok = tokenizer_from_spec(raw if isinstance(raw, dict)
                              else {"kind": "word", "itos": raw})
    # `load_weights=False`: every parameter is about to be overwritten from the
    # checkpoint, so the original pre-trained download is not required here.
    system = CGRSystem(cfg, tok, load_weights=False)
    # Checkpoints written before the matched control existed have no
    # `const_cond`. It is new by design, so it is allowed to be absent and
    # keeps its init; every other mismatch is a real error. Strict loading here
    # would make every older checkpoint unevaluable, which is how a schema
    # addition turns into a lost result.
    missing, unexpected = system.load_state_dict(ck["model"], strict=False)
    leftover = sorted({k for k in missing if not k.endswith("const_cond")})
    if unexpected or leftover:
        raise RuntimeError(
            f"checkpoint {path} does not match the model: "
            f"missing={leftover[:5]} unexpected={sorted(unexpected)[:5]}")
    system.eval()
    return system, cfg


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--docs", type=int, default=None,
                    help="override DataConfig.n_docs for a short run")
    ap.add_argument("--stage-a-epochs", type=int, default=None)
    ap.add_argument("--stage-b-epochs", type=int, default=None)
    ap.add_argument("--device", default=None,
                    help="cpu, cuda, or auto (defaults to preset setting)")
    ap.add_argument("--objective", choices=["cgr", "backbone"], default="cgr",
                    help="CGR training or the matched LoRA denoising baseline")
    ap.add_argument("--mask-mode", choices=["token", "block"], default=None,
                    help="training mask distribution; 'block' matches the "
                         "repair benchmark's whole-block infill")
    ap.add_argument("--block-mask-prob", type=float, default=None,
                    help="probability of a whole-block mask under --mask-mode block")
    ap.add_argument("--resume", default=None,
                    help="stage_b.pt to initialise weights from; optimizer state "
                         "is reset, so this is fine-tuning, not exact continuation")
    ap.add_argument("--cond-mode", choices=["reasoner", "learned", "none"],
                    default="reasoner",
                    help="conditioning source: the reasoner's per-document "
                         "vector (cgr), a learned per-batch constant with the "
                         "same parameter count (matched control), or no side "
                         "channel at all")
    ap.add_argument("--log-every", type=int, default=100,
                    help="print training progress every N batches; 0 disables")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into an --out directory that already "
                         "holds a completed run")
    args = ap.parse_args()
    cfg = Config.preset(args.preset)
    if args.docs is not None:
        cfg.data.n_docs = args.docs
    if args.stage_a_epochs is not None:
        cfg.train.stage_a_epochs = args.stage_a_epochs
    if args.stage_b_epochs is not None:
        cfg.train.stage_b_epochs = args.stage_b_epochs
    if args.mask_mode is not None:
        cfg.train.mask_mode = args.mask_mode
    if args.block_mask_prob is not None:
        cfg.train.block_mask_prob = args.block_mask_prob
    train(cfg, args.out, device=args.device, objective=args.objective,
          log_every=args.log_every, resume=args.resume,
          cond_mode=args.cond_mode, overwrite=args.overwrite)

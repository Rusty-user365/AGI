"""Assembly: the CGR system (§3.1-3.7).

State:
    s_t = (g, h_1:t, M_t, T_t)

`g` is the goal latent (here: the pooled + side-channel representation),
`h_1:t` the compressed history (the backbone hidden state), `M_t` the
experience memory, `T_t` the clock.

Nothing in this file updates weights during interaction. `consolidate` is the
only mutator, it is called offline, and it refuses to write when an outcome is
not verified or its advantage is below threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..data.facts import Claim
from ..data.grammar import blocked_spans
from ..tokenizer import TokenizerLike
from .diffugpt import DiffuGPTBackbone
from .diffusion import MaskedDiffusionLM
from .gate import ACTIONS, A, Gate
from .heads import HeadModule
from .interpreter import OutputMode, adjudicate, epistemic_status
from .memory import SparseMemory
from .reasoner import Reasoner

REASONING_MODES = ("full", "shuffled", "constant", "none")


def build_backbone(cfg: Config, tok, load_weights: bool = True) -> nn.Module:
    """Select the sequence model.

    Both backbones expose the same six methods, which is the whole contract the
    rest of the system relies on: `forward`, `pooled`,
    `masked_denoising_loss`, `infill`, `span_nll`, `token_nll`, plus the
    `plastic_a`/`plastic_b` adapter and `mask_id`/`pad_id`. Adding a backbone
    that does not implement all of them is a wiring bug, not a variant.

    `load_weights=False` skips reading the pre-trained file. It exists for
    `load_system`, which immediately overwrites every parameter with a trained
    checkpoint and must not require the original download to be present.
    """
    kind = getattr(cfg.model, "backbone", "canvas")
    if kind == "canvas":
        return MaskedDiffusionLM(cfg.model, tok.mask_id, tok.pad_id)
    if kind == "diffugpt":
        return DiffuGPTBackbone(cfg.model, tok.mask_id, tok.pad_id,
                                path=cfg.model.backbone_path,
                                load_weights=load_weights)
    raise ValueError(f"unknown backbone {kind!r}")


@dataclass
class Decision:
    action: str
    accept: bool | None
    defer: bool
    verdict: object | None
    epistemic: str
    scope: int
    confidence: float
    override: str | None
    provenance: str | None
    detail: str = ""


class CGRSystem(nn.Module):
    def __init__(self, cfg: Config, tok: TokenizerLike,
                 load_weights: bool = True):
        super().__init__()
        self.cfg = cfg
        self.tok = tok
        m = cfg.model
        self.backbone = build_backbone(cfg, tok, load_weights=load_weights)
        self.reasoner = Reasoner(m)
        self.heads = HeadModule(m)
        self.gate = Gate(m)
        self.memory = SparseMemory(m)
        self.advantage_threshold = 0.5

    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """Where the system's parameters live.

        Callers build tensors from Python lists (tokenised text), so every
        public entry point moves its inputs here instead of asking the caller
        to know the device. Without that, a GPU-trained checkpoint could only
        be used by an experiment that duplicated the device logic.
        """
        return next(self.parameters()).device

    @property
    def feat_dim(self) -> int:
        return self.cfg.model.d_model * 2 + 8

    def n_params(self) -> dict[str, int]:
        def n(mod):
            return sum(p.numel() for p in mod.parameters())
        return {
            "backbone": n(self.backbone), "reasoner": n(self.reasoner),
            "heads": n(self.heads), "gate": n(self.gate),
            "memory": n(self.memory),
            "trainable": sum(p.numel() for p in self.parameters()
                             if p.requires_grad),
            "total": sum(p.numel() for p in self.parameters()),
        }

    # ------------------------------------------------------------------
    def freeze_backbone_base(self) -> None:
        """PEFT switch: freeze the pre-trained weights, keep the adapters live.

        The blank canvas has no pre-trained base, so there is nothing to
        protect and the call is a no-op there. On a pre-trained backbone this
        is the difference between "we fine-tuned 127M parameters on 4k
        synthetic documents" and "we trained a low-rank delta and the CGR
        modules"; the paper needs the second, and the checkpoint has to be able
        to prove which one happened.
        """
        if hasattr(self.backbone, "freeze_base"):
            self.backbone.freeze_base()

    def lora_params(self) -> list[torch.nn.Parameter]:
        if hasattr(self.backbone, "lora_parameters"):
            return self.backbone.lora_parameters()
        return []

    # ------------------------------------------------------------------
    def encode(self, ids: torch.Tensor, reasoning_mode: str = "full",
               denoise: bool = False, cond_override: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Two-pass steering.

        Pass 1  the backbone reads the artifact with no side channel, giving a
                representation the reasoner can work from.
        Pass 2  the reasoner's pooled steering vector conditions the backbone,
                and the artifact representation is recomputed.

        This is what makes the side channel *causal* rather than decorative:
        the steering vector is a function of the input, but the second pass is
        a different computation because of it. `cond_override` lets an
        experiment substitute a steering vector computed elsewhere (the
        positive control in Exp 3).
        """
        pooled0 = self.backbone.pooled(ids)
        blocks = self.reasoner.encode(pooled0)

        if reasoning_mode == "learned":
            # Matched control: the same FiLM side channel, driven by one learned
            # vector for the whole batch instead of a per-document steering
            # vector. Identical parameters, identical losses; the only thing
            # removed is the input-dependent information. `cond_override` takes
            # precedence so a caller can substitute a specific vector.
            if not hasattr(self.backbone, "const_cond"):
                raise ValueError(
                    f"reasoning_mode='learned' needs a backbone with a "
                    f"conditioning side channel; "
                    f"{type(self.backbone).__name__} has none")
            cond = (cond_override if cond_override is not None
                    else self.backbone.const_cond.unsqueeze(0).expand(
                        ids.shape[0], -1))
            pooled = self.backbone.pooled(ids, cond)
            return pooled, cond, blocks

        if reasoning_mode == "none" and cond_override is None:
            return pooled0, None, blocks

        if cond_override is not None:
            cond = cond_override
        else:
            if reasoning_mode == "shuffled" and blocks.shape[0] > 1:
                blocks = blocks.roll(1, dims=0)
            elif reasoning_mode == "constant":
                blocks = torch.zeros_like(blocks)
            if denoise:
                blocks = self.reasoner.denoise_chain(blocks)
            cond = self.reasoner.combine(blocks)

        pooled = self.backbone.pooled(ids, cond)
        return pooled, cond, blocks

    def build_features(self, ids: torch.Tensor, clock: torch.Tensor,
                       reasoning_mode: str = "full",
                       memory_texts: list[str] | None = None,
                       memory_enabled: bool = True,
                       denoise: bool = False,
                       cond_override: torch.Tensor | None = None) -> dict:
        ids = ids.to(self.device)
        clock = clock.to(self.device)
        pooled, cond, blocks = self.encode(ids, reasoning_mode, denoise,
                                           cond_override)
        mem_vec = None
        mem_scores = torch.zeros(ids.shape[0], device=ids.device)
        mem_entries = [None] * ids.shape[0]
        if memory_enabled and memory_texts is not None and len(self.memory) > 0:
            vecs = []
            any_hit = False
            for i, t in enumerate(memory_texts):
                v, s, e = self.memory.recall(t)
                mem_scores[i] = s
                mem_entries[i] = e
                if v is None:
                    vecs.append(torch.zeros(self.cfg.model.d_model, device=ids.device))
                else:
                    vecs.append(v.to(ids.device))
                    any_hit = True
            if any_hit:
                mem_vec = torch.stack(vecs)
                pooled = pooled + self.memory.alpha * mem_vec
        if cond is None:
            cond = torch.zeros(ids.shape[0], self.cfg.model.d_model, device=ids.device)
        feats = torch.cat([pooled, cond, clock], dim=-1)
        return {"feats": feats, "pooled": pooled, "cond": cond, "blocks": blocks,
                "mem_vec": mem_vec, "mem_scores": mem_scores,
                "mem_entries": mem_entries}

    # ------------------------------------------------------------------
    def forward_episodes(self, ids: torch.Tensor, clock: torch.Tensor,
                         reasoning_mode: str = "full",
                         memory_texts: list[str] | None = None,
                         memory_enabled: bool = True, lam: float = 1.0,
                         hard: bool = True,
                         cond_override: torch.Tensor | None = None) -> dict:
        f = self.build_features(ids, clock, reasoning_mode, memory_texts,
                                memory_enabled, cond_override=cond_override)
        preds = self.heads(f["feats"])
        action, logits = self.gate(f["feats"], lam=lam, hard=hard)
        f.update({"preds": preds, "action": action, "action_logits": logits})
        return f

    # ------------------------------------------------------------------
    def answer_claim(self, claim: Claim, reasoning_mode: str = "full",
                     use_memory: bool = True) -> Decision:
        """Full propose -> verify -> commit loop for one claim.

        Authority order matters and is deliberate:

          1. the interpreter, which can REFUSE (no exact semantics) — the
             system then defers regardless of what the gate wanted;
          2. an unresolved scope, which forces deliberation under the default
             scope and records that the default was assumed;
          3. the gate, which chooses among the remaining actions.

        This ordering is what makes rejection of adjudicable false claims
        independent of the weights, and it is why Exp 2 reports the gate as
        non-load-bearing for those claims.
        """
        ids = torch.tensor(self.tok.encode(claim.text)).unsqueeze(0)
        clock = torch.tensor([[0.0, 0.0, 3.0, 10.0,
                               1.0, 0.0, 0.0, 0.3]])
        with torch.no_grad():
            out = self.forward_episodes(ids, clock, reasoning_mode,
                                        [claim.text], use_memory)
            preds = self.heads.predict(out["feats"])
            a, info = self.gate.decide(out["action_logits"][0])
            action = ACTIONS[a]
            conf = float(preds["value"][0])
            epi = ["known", "computable", "check_later", "unknown"][
                int(preds["epistemic"][0])]
            scope = int(preds["scope"][0])

        # A verified memory hit short-circuits to ANS: cheaper and auditable.
        # The hit must match on scope, not just surface similarity.
        scope_key = claim.family if claim.marker_present else "none"
        hit = (self.memory.verified_hit(claim.text, scope=scope_key)
               if use_memory else None)
        if hit is not None:
            return Decision("ANS", True, False, None, "known", scope, conf,
                            info["override"], "experience",
                            f"cached verified answer: {hit.answer}")

        v_scoped = adjudicate(claim.spec, claim.marker_present)

        # 1. no exact semantics -> defer, whatever the gate said
        if v_scoped.mode == OutputMode.REFUSE:
            return Decision("DEF", None, True, v_scoped, "check_later", scope,
                            conf, info["override"], v_scoped.provenance.value,
                            v_scoped.note or
                            "no exact semantics; obligation recorded")

        # 2. scope unresolved -> deliberate under the default, flag the assumption
        if not claim.marker_present:
            v = adjudicate(claim.spec, marker_present=False)
            return Decision("DELIB", v.correct, False, v, epistemic_status(v),
                            scope, conf, info["override"], v.provenance.value,
                            "scope unresolved; evaluated under default scope")

        # 3. gate
        if action == "DEF":
            return Decision("DEF", None, True, None, epi, scope, conf,
                            info["override"], None,
                            "deferred by gate; obligation recorded")

        return Decision(action, v_scoped.correct, False, v_scoped,
                        epistemic_status(v_scoped), scope, conf,
                        info["override"], v_scoped.provenance.value, v_scoped.note)

    # ------------------------------------------------------------------
    def repair(self, blocks: list[str], target_block: int, mode: str = "local",
               steps: int = 6, seed: int | None = None,
               use_reasoner: bool = False) -> list[str]:
        """Repair a corrupted document.

        Modes, with their contract implications:

          local     mask exactly one block. Used for the local-defect condition.
          editable  mask every block except block 0. Block 0 is the locked,
                    authoritative introduction, so this is the only global mode
                    that can satisfy "preserve the anchor". Exp 4 uses this.
          global    mask the whole document, including the introduction. This
                    violates the anchor contract by construction; it is kept
                    only so the violation can be measured rather than assumed
                    away.
        """
        ids, spans = blocked_spans(blocks, self.tok)
        if len(ids) > self.cfg.model.max_len:
            # A silent positional-embedding IndexError here is worse than a
            # named failure: it looks like a model bug rather than a document
            # that does not fit the context.
            raise ValueError(
                f"document has {len(ids)} tokens but model max_len is "
                f"{self.cfg.model.max_len}; filter it out at task construction")
        t = torch.tensor(ids, device=self.device).unsqueeze(0)
        order = getattr(self.cfg.model, "infill_order", "confidence")
        with torch.no_grad():
            cond = self.encode(t)[1] if use_reasoner else None
            if mode == "local":
                gen = self.backbone.infill(t, [spans[target_block]],
                                           cond=cond, steps=steps, seed=seed,
                                           order=order)
            elif mode == "editable":
                gen = self.backbone.infill(t, spans[1:], cond=cond,
                                           steps=steps, seed=seed, order=order)
            elif mode == "global":
                gen = self.backbone.infill(t, [(0, t.shape[1])],
                                           cond=cond, steps=steps, seed=seed,
                                           order=order)
            else:
                raise ValueError(mode)
        # decode back into blocks using the same span layout
        _, new_spans = blocked_spans(blocks, self.tok)
        out = []
        flat = gen[0].tolist()
        for (s, e) in new_spans:
            out.append(self.tok.decode(flat[s:e]))
        return out

    # ------------------------------------------------------------------
    def freeze_all_but_plastic(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.backbone.plastic_a.requires_grad_(True)
        self.backbone.plastic_b.requires_grad_(True)

    def plastic_params(self) -> list[torch.nn.Parameter]:
        return [self.backbone.plastic_a, self.backbone.plastic_b]

    def plastic_snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.backbone.plastic_a.detach().clone(),
                self.backbone.plastic_b.detach().clone())

    def plastic_unchanged(self, snap: tuple[torch.Tensor, torch.Tensor]) -> bool:
        return (torch.equal(snap[0], self.backbone.plastic_a.detach())
                and torch.equal(snap[1], self.backbone.plastic_b.detach()))

    @torch.no_grad()
    def restore_plastic(self, snap: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.backbone.plastic_a.copy_(snap[0])
        self.backbone.plastic_b.copy_(snap[1])

    def consolidate(self, ids: torch.Tensor, clock: torch.Tensor,
                    labels: dict[str, torch.Tensor], verified: list[bool],
                    advantage: list[float], optim: torch.optim.Optimizer,
                    reasoning_mode: str = "full",
                    reject_action: str = "VER") -> dict:
        """Offline, batch, verifier-gated plasticity (§3.7).

        Two operations, deliberately distinct. Masking both would make the
        safeguard indistinguishable from "the model learns nothing", and a test
        of that would pass while proving nothing:

          1. factual / reinforcement. Only verified-right examples with
             advantage above threshold enter this term. A verified-wrong
             example therefore contributes **zero** gradient to the plastic
             substrate: it is dropped from the denoising batch before
             backward, which is structural rather than numerical.
          2. policy. Verified-wrong examples *are* supervised, on the gate,
             with the action the verifier implies (VER by default). The
             features are detached, so this term cannot reach the substrate
             even though it updates the policy.

        Exp 6 checks both halves separately, with an unguarded control that
        must fail and a verified-right control that must pass.
        """
        keep = [i for i, (v, adv) in enumerate(zip(verified, advantage))
                if v and adv > self.advantage_threshold]
        reject = [i for i, v in enumerate(verified) if not v]
        ids = ids.to(self.device)
        clock = clock.to(self.device)
        labels = {k: v.to(self.device) for k, v in labels.items()}
        stats = {"n_total": len(verified), "n_updated": len(keep),
                 "n_masked": len(verified) - len(keep), "stepped": False,
                 "n_rejected": len(reject), "policy_supervised": False}

        loss = None
        if keep:
            idx = torch.tensor(keep, device=self.device)
            out = self.forward_episodes(ids[idx], clock[idx], reasoning_mode,
                                        hard=False)
            b = {k: v[idx] for k, v in labels.items()}
            h_loss, parts = self.heads.loss(out["preds"], b)
            # The plastic substrate sits on the output projection, so it only
            # receives gradient from the generation objective. Consolidation
            # therefore optimises both: what to predict, and what to say.
            d_loss = self.backbone.masked_denoising_loss(ids[idx], out["cond"])
            loss = h_loss + d_loss
            stats["loss"] = float(loss.detach())
            stats["loss_denoise"] = float(d_loss.detach())
            stats.update({f"loss_{k}": v for k, v in parts.items()
                          if k != "total"})

        # Policy term for verified-wrong examples. Features are computed under
        # no_grad and detached, so the gradient reaches the gate and nothing
        # else. Skipped when the gate is frozen (e.g. the plasticity-only
        # experiments), because then it would contribute no learning at all.
        gate_trainable = any(p.requires_grad for p in self.gate.parameters())
        if reject and gate_trainable:
            idx_r = torch.tensor(reject, device=self.device)
            with torch.no_grad():
                out_r = self.forward_episodes(ids[idx_r], clock[idx_r],
                                              reasoning_mode, hard=False)
            feats = out_r["feats"].detach()
            r_logits = self.gate.logits(feats)
            target = torch.full((len(reject),), A[reject_action],
                                dtype=torch.long, device=self.device)
            r_loss = F.cross_entropy(r_logits, target)
            loss = r_loss if loss is None else loss + r_loss
            stats["loss_policy"] = float(r_loss.detach())
            stats["policy_supervised"] = True

        if loss is None or not loss.requires_grad:
            return stats

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.plastic_params(),
                                       self.cfg.train.grad_clip)
        optim.step()
        stats["stepped"] = True
        return stats

    # ------------------------------------------------------------------
    def update_memory(self, text: str, scope: str = "none",
                      verified: bool = False, answer: str | None = None,
                      allow_unverified: bool = False,
                      source: str = "experience") -> int:
        """Write one entry, refusing unverified factual content.

        The guard lives here rather than in the caller because every write path
        goes through this function. A verified-wrong claim must not become a
        factual memory entry: if it did, `recall` would add it to the hidden
        state and the rejection the interpreter made would be undone by the
        store. Returns -1 when the write is refused.

        `allow_unverified` is not a policy switch. It exists so the null-property
        test can populate the store with a deliberately non-factual entry; no
        experiment path sets it.
        """
        if not verified and not allow_unverified:
            return -1
        ids = torch.tensor(self.tok.encode(text)).unsqueeze(0)
        clock = torch.zeros(1, 8)
        with torch.no_grad():
            # memory_enabled=False: the entry's value must not depend on the
            # store it is about to be written into.
            f = self.build_features(ids, clock, memory_enabled=False)
        return self.memory.add(text, f["pooled"][0], scope=scope,
                               verified=verified, answer=answer,
                               source=source)

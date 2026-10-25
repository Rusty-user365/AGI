"""Invariants of the DiffuGPT backbone, testable without the 250 MB download.

The properties here are the ones the integration could plausibly break and
that no downstream experiment would notice:

  * the FiLM side channel is an exact identity at load time, so swapping the
    backbone in cannot have changed the pre-trained model's behaviour;
  * the plastic adapter is an exact identity at load time, so C1's instrument
    starts from "nothing has been consolidated";
  * the shift convention is applied to likelihoods (token `i` is scored from
    `logits[i-1]`), which is the one thing that would silently halve the
    reported quality if it were wrong;
  * `infill` never re-generates position 0 and commits every position it was
    asked to commit.

`scripts/check_backbone.py` covers the part that needs the real checkpoint:
that the weights load and that the denoiser still works.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cgr.config import ModelConfig
from cgr.model.diffugpt import DiffuGPTBackbone
from cgr.model.diffusion import MaskedDiffusionLM

MASK_ID, PAD_ID = 1, 0


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(backbone="diffugpt", vocab_size=64, d_model=32, n_layers=2,
                n_heads=2, d_ff=64, max_len=32, dropout=0.0,
                activation="gelu_new")
    base.update(kw)
    return ModelConfig(**base)


def tiny_model(**kw) -> DiffuGPTBackbone:
    m = DiffuGPTBackbone(tiny_cfg(**kw), MASK_ID, PAD_ID, load_weights=False)
    return m.eval()


def test_both_backbones_expose_the_same_contract():
    required = ["forward", "pooled", "masked_denoising_loss", "infill",
                "span_nll", "token_nll", "plastic_a", "plastic_b",
                "mask_id", "pad_id"]
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2,
                      d_ff=64, max_len=32, dropout=0.0)
    canvas = MaskedDiffusionLM(cfg, MASK_ID, PAD_ID)
    gpt = tiny_model()
    for name in required:
        assert hasattr(canvas, name), f"canvas is missing {name}"
        assert hasattr(gpt, name), f"diffugpt is missing {name}"


def test_plastic_adapter_is_identity_at_init():
    """plastic_b is zero, so the adapter contributes exactly nothing."""
    m = tiny_model()
    ids = torch.randint(3, 60, (1, 12))
    with torch.no_grad():
        before = m(ids).clone()
        m.plastic_a.add_(torch.randn_like(m.plastic_a))
        after = m(ids)
    assert torch.equal(before, after), (
        "changing plastic_a moved the logits while plastic_b is zero; the "
        "adapter is not an identity at init and C1's baseline is invalid")


def test_film_is_identity_at_init_and_live_afterwards():
    m = tiny_model()
    ids = torch.randint(3, 60, (1, 12))
    cond = torch.randn(1, 32)
    with torch.no_grad():
        plain = m(ids)
        steered = m(ids, cond)
        assert torch.equal(plain, steered), (
            "zero-initialised FiLM still changed the logits")

        # Now make it non-zero and check the side channel actually reaches the
        # output. Without this the identity test above would pass on a backbone
        # that ignored `cond` entirely.
        for blk in m.h:
            blk.film.weight.normal_(0, 0.05)
        live = m(ids, cond)
    assert not torch.allclose(plain, live, atol=1e-6)


def test_shift_convention_in_token_nll():
    """token i must be scored from logits[i-1], not logits[i]."""
    m = tiny_model()
    ids = torch.tensor([[5, 9, 14, 20, 7]])
    with torch.no_grad():
        logits = m(ids)
        nll = m.token_nll(ids)
    logp = logits.log_softmax(-1)
    assert nll[0].item() == 0.0
    for i in range(1, ids.shape[1]):
        want = -logp[0, i - 1, ids[0, i]].item()
        assert abs(nll[i].item() - want) < 1e-5, f"position {i}"
    # And it must NOT equal the unshifted reading, or the test proves nothing.
    unshifted = -logp[0, 1, ids[0, 1]].item()
    assert abs(nll[1].item() - unshifted) > 1e-6


def test_denoising_loss_ignores_padding_and_position_zero():
    m = tiny_model()
    ids = torch.tensor([[5, 9, 14, 20, PAD_ID, PAD_ID]])
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        loss = m.masked_denoising_loss(ids, mask_ratio=1.0, generator=g)
    # mask_ratio=1.0 masks every non-pad position except 0 (rand() < 1.0 is
    # always true), so the noisy input is deterministic and the loss must be
    # the mean over exactly tokens 1..3, read from logits 0..2.
    noisy = torch.tensor([[5, MASK_ID, MASK_ID, MASK_ID, PAD_ID, PAD_ID]])
    with torch.no_grad():
        logp = m(noisy).log_softmax(-1)
        want = -torch.stack([logp[0, i - 1, ids[0, i]]
                             for i in (1, 2, 3)]).mean()
    assert torch.allclose(loss, want, atol=1e-5)


def test_block_mask_is_contiguous_and_never_masks_a_separator():
    """The block mask must match the repair operation: one contiguous block.

    If this silently masks scattered tokens, the new objective is the old one
    with a different name, and the floor will not move.
    """
    m = tiny_model()
    sep = 5
    ids = torch.tensor([[2, 10, 11, 12, 13, sep, 20, 21, 22, 23, sep,
                         30, 31, 32, 3, PAD_ID]])
    for seed in range(10):
        g = torch.Generator().manual_seed(seed)
        mask = m._block_span_mask(ids, g, 3, 5, sep)
        assert mask.any(), "block mask selected no tokens"
        pos = mask[0].nonzero(as_tuple=False).flatten().tolist()
        assert pos == list(range(pos[0], pos[-1] + 1)), (
            f"mask is not contiguous: {pos}")
        assert sep not in pos, "a block separator was masked"
        assert 0 not in pos and 14 not in pos and 15 not in pos, (
            "BOS, EOS or padding was masked")


def test_block_mask_mode_returns_a_finite_loss():
    m = tiny_model()
    ids = torch.tensor([[2, 10, 11, 12, 13, 5, 20, 21, 22, 23, 5,
                         30, 31, 32, 3, PAD_ID]])
    g = torch.Generator().manual_seed(0)
    loss = m.masked_denoising_loss(ids, mask_ratio=0.3, generator=g,
                                   mask_mode="block", block_mask_prob=1.0,
                                   sep_id=5)
    assert torch.isfinite(loss), loss
    assert loss.item() > 0.0


def test_no_separator_sequence_falls_back_to_token_mask():
    """A short claim has no block boundary; masking all of it would be
    unconditional generation, so the mixed objective keeps the token mask."""
    m = tiny_model()
    ids = torch.tensor([[2, 10, 11, 12, 13, 3, PAD_ID, PAD_ID]])
    g = torch.Generator().manual_seed(0)
    block = m._block_span_mask(ids, g, 15, 20, sep_id=5)
    assert not block.any(), "no-separator sequence received a block mask"
    loss = m.masked_denoising_loss(ids, mask_ratio=0.5, generator=g,
                                   mask_mode="block", block_mask_prob=1.0,
                                   sep_id=5)
    assert torch.isfinite(loss), loss


def test_span_nll_uses_the_shift():
    m = tiny_model()
    ids = torch.tensor([[5, 9, 14, 20, 7]])
    with torch.no_grad():
        span = m.span_nll(ids, [(2, 4)])
        nll = m.token_nll(ids)
    assert torch.allclose(span, nll[2:4].mean(), atol=1e-5)


def test_infill_never_touches_position_zero_and_commits_the_span():
    """The commit loop is tested with a stub proposal, not the weights.

    A randomly initialised model can legitimately propose the mask token, so
    asserting on its output would test the RNG rather than the plumbing. The
    stub returns a single in-vocabulary id that the input does not contain, so
    "was this position committed" is unambiguous.
    """
    m = tiny_model()
    ids = torch.tensor([[5, 9, 14, 20, 22, 3]])
    fill = 7
    calls = []

    def stub(logits, temperature, top_p, g):
        calls.append(1)
        return torch.full_like(ids, fill)

    m._propose = stub
    out = m.infill(ids, [(2, 4)], steps=3, seed=0)
    assert out[0, 0].item() == 5, "position 0 (BOS) was re-generated"
    for j in (1, 4, 5):
        assert out[0, j].item() == ids[0, j].item(), f"position {j} moved"
    for j in (2, 3):
        assert out[0, j].item() == fill, f"position {j} was never committed"
    assert len(calls) <= 3


def test_infill_one_step_commits_everything():
    """`order="random"` respects `steps` as the round count."""
    m = tiny_model()
    ids = torch.tensor([[5, 9, 14, 20, 22, 3]])
    m._propose = lambda logits, t, p, g: torch.full_like(ids, 7)
    out = m.infill(ids, [(1, 5)], steps=1, seed=0, order="random")
    for j in range(1, 5):
        assert out[0, j].item() == 7
    assert out[0, 5].item() == 3


def test_left_to_right_commits_one_position_per_round():
    """Rounds are raised to the span length, so no predecessor stays masked.

    That is the whole reason the order exists: under the shift convention a
    token is predicted from its predecessor's hidden state, and a masked
    predecessor is out of distribution.
    """
    m = tiny_model()
    ids = torch.tensor([[5, 9, 14, 20, 22, 3]])
    rounds = []

    def stub(logits, temperature, top_p, g):
        rounds.append(1)
        return torch.full_like(ids, 7)

    m._propose = stub
    m.infill(ids, [(1, 5)], steps=1, seed=0, order="left_to_right")
    assert len(rounds) == 4, f"expected 4 rounds for a 4-token span, got {len(rounds)}"


def test_infill_global_mode_preserves_bos():
    """`mode="global"` spans the whole document; position 0 must survive."""
    m = tiny_model()
    ids = torch.tensor([[5, 9, 14, 20, 22, 3]])
    m._propose = lambda logits, t, p, g: torch.full_like(ids, 7)
    out = m.infill(ids, [(0, ids.shape[1])], steps=2, seed=0)
    assert out[0, 0].item() == 5
    assert all(out[0, j].item() == 7 for j in range(1, 6))


def test_long_sequence_is_a_named_error():
    m = tiny_model(max_len=8)
    with pytest.raises(ValueError, match="positional"):
        m(torch.randint(3, 60, (1, 9)))


def test_padding_mask_keeps_real_tokens_bit_identical():
    """Padding must not leak into the real positions.

    The canvas used `nn.MultiheadAttention`'s `key_padding_mask` (True =
    ignore); SDPA's boolean mask has the opposite convention. Getting that
    backwards would let pad tokens attend, and the damage would be invisible
    until a repair scored slightly wrong.
    """
    m = tiny_model()
    ids = torch.tensor([[5, 9, 14, 20, 7]])
    padded = torch.tensor([[5, 9, 14, 20, 7, PAD_ID, PAD_ID]])
    with torch.no_grad():
        a = m(ids)
        b = m(padded)[:, :5]
    assert torch.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------------------
# PEFT: frozen base + LoRA
# ---------------------------------------------------------------------------

def test_lora_is_identity_at_init():
    """`lora_b` is zero, so the adapter contributes exactly nothing at load.

    This is what makes "pre-trained model + adapter" start from the checkpoint
    rather than from a perturbation of it, which is the only way an ablation
    against the base model means anything.
    """
    m = tiny_model()
    ids = torch.randint(3, 60, (1, 12))
    with torch.no_grad():
        before = m(ids).clone()
        for blk in m.h:
            blk.lora_attn.a.add_(torch.randn_like(blk.lora_attn.a))
            blk.lora_proj.a.add_(torch.randn_like(blk.lora_proj.a))
        after = m(ids)
    assert torch.equal(before, after), (
        "moving LoRA `a` changed the logits while `b` is zero; the adapter is "
        "not an identity at initialisation")


def test_freeze_base_leaves_only_adapters_trainable():
    m = tiny_model()
    m.freeze_base()
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable, "freeze_base froze everything"
    for n in trainable:
        assert ((".lora_" in n) or ("film" in n) or n == "const_cond"
                or n in ("plastic_a", "plastic_b")), n
    frozen = [n for n, p in m.named_parameters() if not p.requires_grad]
    for n in ("embed_tokens.weight", "wte.weight", "h.0.c_attn.weight",
              "h.0.c_fc.weight", "ln_f.weight"):
        if n in dict(m.named_parameters()):
            assert n in frozen, f"{n} should be frozen"


def test_freeze_base_keeps_the_side_channel_trainable():
    """Regression: the side channel was silently dropped from the allowlist.

    `freeze_base` froze every parameter and re-enabled only LoRA and plastic.
    The FiLM maps stayed at their zero initialisation, so `cond` had exactly no
    effect for every run, and no test noticed because "zero side channel" is a
    valid identity at load time. This asserts the contract instead.
    """
    m = tiny_model()
    m.freeze_base()
    for blk in m.h:
        assert blk.film.weight.requires_grad, "film.weight was frozen"
        assert blk.film.bias.requires_grad, "film.bias was frozen"
    assert m.const_cond.requires_grad, "const_cond was frozen"
    m.assert_side_channel_live()


def test_side_channel_receives_gradient_from_the_denoising_loss():
    """A live allowlist is not enough: the gradient has to reach FiLM."""
    m = tiny_model()
    m.freeze_base()
    m.train()
    ids = torch.randint(3, 60, (2, 12))
    cond = torch.randn(2, 32)
    g = torch.Generator().manual_seed(0)
    loss = m.masked_denoising_loss(ids, cond, mask_ratio=0.5, generator=g)
    loss.backward()
    grad = m.h[0].film.weight.grad
    assert grad is not None, "no gradient reached film.weight"
    assert float(grad.abs().max()) > 0.0, (
        "film.weight received an all-zero gradient; cond is not in the graph")


def test_const_cond_receives_gradient_in_the_matched_control():
    """The control's conditioning vector must be a learned parameter, not zeros.

    If `const_cond` were zero, `film.weight` would receive an all-zero gradient
    (grad_w is proportional to cond) and the control would collapse to a bias
    term: same parameter count on paper, far less capacity in practice.

    The vector cannot move on the *first* step, because dL/d(const_cond) is
    proportional to `film.weight`, which starts at zero. It moves from the
    second step on, once FiLM has a gradient of its own -- so this checks two
    steps, not one.
    """
    m = tiny_model()
    m.freeze_base()
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=1e-2)
    ids = torch.randint(3, 60, (2, 12))
    cond = m.const_cond.unsqueeze(0).expand(2, -1)
    before = m.const_cond.detach().clone()
    for step in range(2):
        opt.zero_grad()
        g = torch.Generator().manual_seed(step)
        loss = m.masked_denoising_loss(ids, cond, mask_ratio=0.5, generator=g)
        loss.backward()
        assert float(m.h[0].film.weight.grad.abs().max()) > 0.0, (
            "the control's FiLM weights got no gradient")
        opt.step()
    assert not torch.equal(before, m.const_cond.detach()), (
        "const_cond never moved; the control is not a learned conditioning "
        "vector")


def test_assert_side_channel_live_rejects_a_dead_channel():
    m = tiny_model()
    m.freeze_base()
    with torch.no_grad():
        for blk in m.h:
            blk.film.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="frozen"):
        m.assert_side_channel_live()

    m2 = tiny_model()
    m2.freeze_base()
    with torch.no_grad():
        m2.const_cond.zero_()
    with pytest.raises(RuntimeError, match="exactly zero"):
        m2.assert_side_channel_live()


def test_side_channel_stats_detect_a_dead_channel():
    """The diagnostic that would have caught the frozen-FiLM bug on day one."""
    from cgr.train import side_channel_stats

    m = tiny_model()
    m.freeze_base()
    ids = torch.randint(3, 60, (1, 10))
    probe = torch.randn(1, 32)
    reasoner = torch.nn.Linear(4, 4)
    ref = [p.detach().clone() for p in reasoner.parameters()]

    dead = side_channel_stats(m, reasoner, ref, ids, probe)
    assert dead["film_max_abs"] == 0.0
    assert dead["cond_probe_delta"] == 0.0
    assert dead["side_channel_live"] is False

    with torch.no_grad():
        for blk in m.h:
            blk.film.weight.normal_(0, 0.05)
        reasoner.weight.add_(1.0)
    live = side_channel_stats(m, reasoner, ref, ids, probe)
    assert live["film_max_abs"] > 0.0
    assert live["cond_probe_delta"] > 0.0
    assert live["side_channel_live"] is True
    assert live["reasoner_l2_delta"] > 0.0


def test_frozen_base_does_not_move_after_a_step():
    """The PEFT claim, checked as a bit-identity rather than a promise.

    If a single optimizer step moves a base weight, the paper cannot say the
    backbone was frozen, and every "architecture, not weights" ablation is
    confounded.
    """
    m = tiny_model()
    m.freeze_base()
    m.train()
    base_before = {n: p.detach().clone() for n, p in m.named_parameters()
                   if not p.requires_grad}
    lora_b_before = m.h[0].lora_attn.b.detach().clone()

    ids = torch.randint(3, 60, (2, 12))
    g = torch.Generator().manual_seed(0)
    loss = m.masked_denoising_loss(ids, mask_ratio=0.5, generator=g)
    loss.backward()
    trainable = [p for p in m.parameters() if p.requires_grad]
    torch.optim.AdamW(trainable, lr=1e-2).step()

    for n, before in base_before.items():
        now = dict(m.named_parameters())[n].detach()
        assert torch.equal(before, now), f"frozen parameter {n} moved"
    assert not torch.equal(lora_b_before, m.h[0].lora_attn.b.detach()), (
        "the adapter did not move; the loss is not reaching the trainable path")

"""Small, shared controls for training ablation conditions."""
from __future__ import annotations

import torch

from .config import ModelConfig
from .model.heads import HEAD_NAMES, HeadModule

ABLATION_MODES = ("none", "random_heads", "shuffled_labels", "no_loss",
                  "matched_lora")


class FrozenRandomHeadModule(HeadModule):
    """Same head architecture with fixed random readouts.

    The training entry point seeds PyTorch from ``cfg.train.seed`` before the
    model is built. Freezing parameters does not detach the readouts, so an
    active head loss can still backpropagate to the shared input features.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.requires_grad_(False)


def build_head_module(cfg: ModelConfig, mode: str) -> HeadModule:
    """Build the regular or frozen head module for an ablation mode."""
    if mode not in ABLATION_MODES:
        raise ValueError(f"ablation mode must be one of {ABLATION_MODES}")
    if mode == "random_heads":
        return FrozenRandomHeadModule(cfg)
    heads = HeadModule(cfg)
    if mode == "no_loss":
        heads.requires_grad_(False)
    return heads


def shuffle_head_labels(labels: dict[str, torch.Tensor], *, seed: int,
                        stage: str, epoch: int, batch_index: int
                        ) -> dict[str, torch.Tensor]:
    """Permute batch examples independently for every head, reproducibly."""
    missing = set(HEAD_NAMES) - set(labels)
    if missing:
        raise KeyError(f"cannot shuffle labels; missing heads: {sorted(missing)}")

    stage_offset = 0 if stage == "A" else 1_000_003
    result = dict(labels)
    for head_index, name in enumerate(HEAD_NAMES):
        target = labels[name]
        if target.ndim == 0:
            raise ValueError(f"label tensor for {name!r} has no batch dimension")
        local_seed = (int(seed) + stage_offset + int(epoch) * 2_000_033
                      + int(batch_index) * 1_000_037
                      + head_index * 97_409) % (2**63 - 1)
        # DirectML does not expose a torch.Generator device backend. Draw the
        # permutation on CPU, then move its indices to the target device.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(local_seed)
        permutation = torch.randperm(target.shape[0], generator=generator).to(
            target.device)
        result[name] = target.index_select(0, permutation)
    return result

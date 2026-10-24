"""Central configuration.

Three presets:

  smoke  -- tiny, CPU-runnable in seconds. Used by tests and the smoke run.
  dev    -- small but real. Fits a free Kaggle T4 in minutes.
  paper  -- the configuration described in the draft (§4.1).

Keeping the presets explicit matters: the draft's compute budget was not
achievable, and a config that silently differs between "test" and "paper"
is how people end up reporting numbers from the wrong model.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    # backbone
    vocab_size: int = 512
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 1024
    max_len: int = 256
    dropout: float = 0.1

    # latent reasoner (side channel)
    d_latent: int = 128
    n_blocks: int = 4
    latent_steps: int = 4
    guidance_weight: float = 0.5

    # head module
    d_head: int = 64
    d_outcome: int = 4
    n_epistemic: int = 4       # known / computable / check-later / unknown
    n_scope: int = 5           # 4 families + none
    n_source: int = 3          # weights / experience / derived

    # gate
    n_actions: int = 8
    gumbel_tau: float = 1.0

    # memory
    mem_dim: int = 64
    mem_capacity: int = 100_000
    embed_dim: int = 256

    def n_params(self) -> int:
        """Rough parameter count, for reporting the actual size (not the claimed one)."""
        emb = self.vocab_size * self.d_model
        per_layer = 4 * self.d_model * self.d_model + 2 * self.d_model * self.d_ff
        blocks = self.n_layers * per_layer
        ln = 2 * self.n_layers * 2 * self.d_model
        return emb + blocks + ln


@dataclass
class DataConfig:
    n_docs: int = 30_000
    n_val: int = 2_000
    n_test: int = 2_000
    n_reasons_min: int = 2
    n_reasons_max: int = 4
    # held-out template combinations: the anti-memorisation split.
    holdout_frac: float = 0.15
    seed: int = 0


@dataclass
class TrainConfig:
    stage_a_epochs: int = 8
    stage_b_epochs: int = 4
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 200
    grad_clip: float = 1.0
    mask_ratio_min: float = 0.15
    mask_ratio_max: float = 0.85
    # lambda weights for the seven heads (Stage A)
    head_weights: dict = field(default_factory=lambda: {
        "value": 1.0, "outcome": 1.0, "tau": 1.0,
        "epistemic": 1.0, "scope": 1.0, "source": 1.0, "disagreement": 1.0,
    })
    kl_leash: float = 0.1
    seed: int = 0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    preset_name: str = "custom"

    @staticmethod
    def preset(name: str) -> "Config":
        if name == "smoke":
            return Config(
                model=ModelConfig(
                    vocab_size=512, d_model=64, n_layers=2, n_heads=2,
                    d_ff=128, max_len=96, dropout=0.0,
                    d_latent=32, n_blocks=3, latent_steps=2,
                    d_head=32, d_outcome=4, embed_dim=64, mem_dim=32,
                ),
                data=DataConfig(n_docs=400, n_val=100, n_test=100, seed=0),
                train=TrainConfig(stage_a_epochs=2, stage_b_epochs=1,
                                  batch_size=16, lr=3e-3),
                preset_name="smoke",
            )
        if name == "dev":
            return Config(
                model=ModelConfig(
                    vocab_size=512, d_model=128, n_layers=4, n_heads=4,
                    d_ff=512, max_len=192,
                    d_latent=64, n_blocks=4, latent_steps=3,
                    d_head=48, embed_dim=128, mem_dim=64,
                ),
                data=DataConfig(n_docs=4_000, n_val=400, n_test=400),
                train=TrainConfig(stage_a_epochs=6, stage_b_epochs=3,
                                  batch_size=32, lr=3e-4),
                preset_name="dev",
            )
        if name == "paper":
            return Config(preset_name="paper")
        raise ValueError(f"unknown preset {name!r}")

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        def pick(cls, d: dict):
            # Tolerate configs written by older code: drop keys the current
            # dataclass no longer has instead of crashing on an old checkpoint.
            allowed = set(cls.__dataclass_fields__)
            return cls(**{k: v for k, v in d.items() if k in allowed})

        m = pick(ModelConfig, raw["model"])
        d = pick(DataConfig, raw["data"])
        t = pick(TrainConfig, raw["train"])
        return Config(model=m, data=d, train=t,
                      preset_name=raw.get("preset_name", "custom"))

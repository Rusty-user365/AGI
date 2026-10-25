"""Shared experiment plumbing: load/train a checkpoint, write results, report.

Every experiment writes a JSON file with the same top-level schema so the
paper's tables can be generated mechanically and so a null result is as
visible as a positive one.

    {
      "experiment": str,
      "preset": str,
      "seed": int,
      "n": int,
      "preregistered_prediction": {...},
      "conditions": {...},
      "summary": {...},
      "notes": [...],
      "timestamp": str
    }

`preregistered_prediction` is written *before* the measurement runs and is
never edited afterwards. This is the cheap defence against the draft's
original failure mode, where §5 was pre-filled with the expected answer.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cgr.config import Config
from cgr.train import load_system, pick_device, train

RESULTS = Path("results")
CHECKPOINTS = Path("checkpoints")


def ckpt_dir(preset: str, seed: int = 0) -> Path:
    """Where a given (preset, seed) checkpoint lives.

    Seed 0 keeps the plain `checkpoints/<preset>` path so existing runs and
    docs still work. Other seeds get a suffix. The suffix is not cosmetic: an
    earlier version of `kaggle_run.py` trained seeds 1 and 2 into suffixed
    directories while the experiments always loaded `checkpoints/<preset>`,
    so a three-seed run would have reported the same model three times.
    """
    return CHECKPOINTS / (preset if seed == 0 else f"{preset}_s{seed}")


def get_system(preset: str = "smoke", train_if_missing: bool = True,
               verbose: bool = True, seed: int = 0,
               checkpoint: str | None = None, device: str | None = None):
    """Load a checkpoint. `checkpoint` overrides the preset/seed path.

    The override exists so an experiment can be run against a model trained at
    a different budget (e.g. the editor-floor curve) without pretending it is
    the preset's checkpoint. The `preset` is still recorded in the result so a
    reader can see which config produced the weights.

    `device=None` follows the preset (`cfg.train.device`), so the committed
    canvas numbers stay on CPU while the 127M backbone runs where it can.
    """
    path = (Path(checkpoint) if checkpoint
            else ckpt_dir(preset, seed) / "stage_b.pt")
    if not path.exists():
        if not train_if_missing:
            raise FileNotFoundError(path)
        if verbose:
            print(f"[common] no checkpoint at {path}; training {preset} "
                  f"(seed {seed}) ...")
        cfg = Config.preset(preset)
        cfg.train.seed = seed
        train(cfg, ckpt_dir(preset, seed), verbose=verbose, device=device)
        path = ckpt_dir(preset, seed) / "stage_b.pt"
    system, cfg = load_system(path)
    dev = pick_device(device if device is not None
                      else getattr(cfg.train, "device", "cpu"))
    system = system.to(dev)
    if verbose:
        print(f"[common] loaded {path} on {dev}")
    return system, cfg


def write_result(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    seed = int(payload.get("seed", 0) or 0)
    # Seed 0 writes results/<name>.json; other seeds get a suffix so a
    # multi-seed run cannot silently overwrite itself.
    fname = f"{name}.json" if seed == 0 else f"{name}_s{seed}.json"
    payload.setdefault("env", {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "platform": platform.platform(),
    })
    p = RESULTS / fname
    p.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return p


def banner(title: str) -> None:
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def table(rows: list[dict], headers: list[str]) -> None:
    if not rows:
        return
    widths = {h: max(len(h), *(len(f"{r.get(h, ''):.4g}") if isinstance(r.get(h), float)
                               else len(str(r.get(h, ''))) for r in rows))
              for h in headers}
    print("  " + "  ".join(h.ljust(widths[h]) for h in headers))
    print("  " + "  ".join("-" * widths[h] for h in headers))
    for r in rows:
        cells = []
        for h in headers:
            v = r.get(h, "")
            cells.append(f"{v:.4g}".ljust(widths[h]) if isinstance(v, float)
                         else str(v).ljust(widths[h]))
        print("  " + "  ".join(cells))


def arg_parser(desc: str):
    import argparse
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test",
                    choices=["test", "heldout", "unseen"],
                    help="corpus split: `test` is in-distribution, `heldout` "
                         "uses template combinations never seen in training, "
                         "`unseen` uses theses never seen in training")
    ap.add_argument("--train", action="store_true",
                    help="force retraining even if a checkpoint exists")
    ap.add_argument("--checkpoint", default=None,
                    help="load this stage_b.pt instead of the preset's "
                         "checkpoint (the preset is still recorded)")
    ap.add_argument("--device", default=None,
                    help="cpu | cuda | auto. Default: follow the preset "
                         "(the canvas stays on cpu; diffugpt uses cuda if "
                         "available)")
    return ap

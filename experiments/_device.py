"""Backend-agnostic device helpers.

The harness was written against a single 4 GB GTX 1050, so the experiment
scripts call `torch.cuda.*` unconditionally. That breaks in two distinct ways
and both matter for reproducibility:

  * on a CPU-only build, `torch.cuda.reset_peak_memory_stats()` raises
    `AttributeError` even when no CUDA device is present, so a CPU run of an
    otherwise portable script dies at startup;
  * on an Intel iGPU box the same calls are simply the wrong backend.

Everything here resolves to one of `cuda`, `xpu`, `mps` or `cpu` and routes the
memory/seed/sync calls accordingly. Nothing is silently swallowed: when a
backend has no peak-memory counter the value is reported as `0.0` *and*
`peak_backend` says so, so a results file cannot be misread as a real VRAM
measurement.

Import as `from _device import ...` next to the other experiment scripts.
"""
from __future__ import annotations

import torch

_BACKENDS = ("cuda", "xpu", "mps")


def _available(name: str) -> bool:
    if name == "cuda":
        try:
            return torch.cuda.is_available()
        except Exception:  # noqa: BLE001 - a broken driver must not be fatal
            return False
    if name == "xpu":
        try:
            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except Exception:  # noqa: BLE001
            return False
    if name == "mps":
        try:
            return torch.backends.mps.is_available()
        except Exception:  # noqa: BLE001
            return False
    return False


def available() -> list[str]:
    """Which accelerators this interpreter can actually use, best first."""
    return [b for b in _BACKENDS if _available(b)]


def resolve(name: str | None = None) -> torch.device:
    """Resolve "auto"/None to the best available device.

    `"auto"` keeps the old precedence (CUDA first) but no longer assumes CUDA
    exists. An explicit name is passed through untouched, so a run that asks
    for `cuda` on a machine without one still fails loudly instead of quietly
    producing CPU numbers under a CUDA label.
    """
    if name in (None, "", "auto"):
        for backend in _BACKENDS:
            if _available(backend):
                return torch.device(backend)
        return torch.device("cpu")
    return torch.device(name)


def _module(dev: torch.device):
    if dev.type == "cuda":
        return torch.cuda
    if dev.type == "xpu":
        return torch.xpu
    return None


def empty_cache(dev: torch.device | None = None) -> None:
    mod = _module(resolve(dev))
    if mod is not None:
        try:
            mod.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def synchronize(dev: torch.device | None = None) -> None:
    mod = _module(resolve(dev))
    if mod is None:
        return
    try:
        mod.synchronize()
    except Exception:  # noqa: BLE001
        pass


def manual_seed_all(seed: int) -> None:
    """Seed every accelerator RNG that exists; CPU seeding is the caller's job."""
    for backend in _BACKENDS:
        if not _available(backend):
            continue
        try:
            if backend == "mps":
                torch.mps.manual_seed(seed)
            else:
                getattr(torch, backend).manual_seed_all(seed)
        except Exception:  # noqa: BLE001
            pass


def reset_peak(dev: torch.device | None = None) -> None:
    mod = _module(resolve(dev))
    if mod is None:
        return
    try:
        mod.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def peak_mb(dev: torch.device | None = None) -> float:
    """Peak allocated memory in MiB, or 0.0 when the backend has no counter."""
    mod = _module(resolve(dev))
    if mod is None:
        return 0.0
    try:
        return float(mod.max_memory_allocated()) / 2 ** 20
    except Exception:  # noqa: BLE001
        return 0.0


def peak_backend(dev: torch.device | None = None) -> str:
    """What `peak_mb` is measuring, so a 0.0 cannot be read as "used no VRAM"."""
    dev = resolve(dev)
    return dev.type if dev.type in ("cuda", "xpu") else "none"


def describe(dev: torch.device | None = None) -> dict:
    """The block results files should carry about the run's hardware."""
    dev = resolve(dev)
    out = {
        "device": str(dev),
        "torch": torch.__version__,
        "available": available(),
        "cuda": _available("cuda"),
        "xpu": _available("xpu"),
    }
    if dev.type == "xpu":
        try:
            out["xpu_device_name"] = torch.xpu.get_device_name(dev)
        except Exception:  # noqa: BLE001
            pass
    elif dev.type == "cuda":
        try:
            out["cuda_device_name"] = torch.cuda.get_device_name(dev)
        except Exception:  # noqa: BLE001
            pass
    return out

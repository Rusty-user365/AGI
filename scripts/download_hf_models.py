"""Download the HuggingFace checkpoints the experiments expect.

The repo keeps external models under ``checkpoints/hf/<short-name>`` so the
scripts can run offline.

Why this does not use ``snapshot_download``
-------------------------------------------
On the machine this was last run on (a 16 GB laptop on a slow consumer link),
``huggingface_hub`` 2.3.0 hung twice on the same 1.2 GB ``model.safetensors``:

* the default Xet transfer path (302 to ``xet-bridge-us``) never wrote a byte;
* with ``HF_HUB_DISABLE_XET=1`` the plain CDN wrote ~732 MB and then hung for
  the rest of the transfer.

Both times the process sat idle with an open socket and no progress, so a
"download is running" log line was not evidence that bytes were moving. This
script therefore speaks HTTP directly: short socket timeout, ``Range`` resume
from whatever is already on disk, and a bounded retry loop. A stall costs one
timeout, not the whole run.

It is idempotent: a file whose size already matches the server is skipped, and
a partial file resumes instead of restarting. ``*.sha256`` verification is left
to ``transformers``, which checks the safetensors header on load.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co"

# (repo_id, local directory, why it is here)
MODELS = [
    ("Qwen/Qwen3-0.6B-Base", "checkpoints/hf/qwen3-0.6b-base",
     "main arithmetic / localizer / CGR model"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "checkpoints/hf/qwen2.5-1.5b-instruct",
     "state-tracking control and instruct baseline"),
    ("ltg/dual-lm-470m", "checkpoints/hf/dual-lm-470m",
     "470M masked-diffusion control"),
    ("dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1",
     "checkpoints/hf/qwen2.5-coder-0.5b-diffusion-mdlm",
     "diffusion payload arm"),
    ("Qwen/Qwen2.5-Coder-0.5B-Instruct", "checkpoints/hf/qwen2.5-coder-0.5b-instruct",
     "AR twin for the diffusion arm and the coder pilot"),
]

# Large redundant formats we never load.
SKIP_SUFFIX = (".gguf", ".pth", ".onnx", ".msgpack", ".h5")
SKIP_PREFIX = ("original/",)

CHUNK = 1 << 20          # 1 MiB
TIMEOUT = 30             # seconds without a byte before we reconnect
MAX_STALLS = 8           # consecutive no-progress attempts per file


def api_files(repo: str) -> list[dict]:
    url = f"{HF}/api/models/{repo}/tree/main?recursive=1"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        entries = json.load(r)
    # The tree endpoint names the field `path`; older releases used `rfilename`.
    return [{"name": e.get("path") or e.get("rfilename"), "size": e.get("size")}
            for e in entries if e.get("type") == "file"]


def wanted(name: str) -> bool:
    return not (name.endswith(SKIP_SUFFIX) or name.startswith(SKIP_PREFIX))


def remote_size(repo: str, name: str) -> int | None:
    """Size from a HEAD on the resolve URL, so we can skip complete files."""
    req = urllib.request.Request(f"{HF}/{repo}/resolve/main/{name}", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:  # noqa: BLE001 - unknown size just means "download it"
        return None


def fetch(repo: str, name: str, dest: Path) -> str:
    """Stream one file to `dest` with Range resume. Returns a status word."""
    url = f"{HF}/{repo}/resolve/main/{name}"
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0

    stalls = 0
    while stalls < MAX_STALLS:
        headers = {"Range": f"bytes={have}-"} if have else {}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                total = r.headers.get("Content-Range", "")
                if have and r.status == 200:
                    # Server ignored the Range header; start over honestly.
                    have = 0
                with part.open("ab" if have else "wb") as fh:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            stalls += 1
            print(f"    ! {type(exc).__name__}: {str(exc)[:90]} "
                  f"(have {have / 1e6:.0f} MB, retry {stalls}/{MAX_STALLS})",
                  flush=True)
            time.sleep(min(2 ** stalls, 20))
            continue

        want = remote_size(repo, name)
        if want is not None and have < want:
            stalls += 1
            print(f"    ! short read {have}/{want} bytes, resuming "
                  f"({stalls}/{MAX_STALLS})", flush=True)
            continue
        part.replace(dest)
        return f"ok {have / 1e6:.1f} MB"

    return f"FAILED after {MAX_STALLS} stalls at {have / 1e6:.0f} MB"


def main() -> int:
    failures: list[str] = []
    for repo, rel, why in MODELS:
        dest = ROOT / rel
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {repo} -> {rel}  ({why})", flush=True)
        t0 = time.time()
        try:
            files = api_files(repo)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED listing {repo}: {type(exc).__name__}: {exc}", flush=True)
            failures.append(repo)
            continue

        for entry in files:
            name = entry["name"]
            if not wanted(name):
                continue
            out = dest / name
            out.parent.mkdir(parents=True, exist_ok=True)
            size = entry.get("size")
            if out.exists() and (size is None or out.stat().st_size == size):
                print(f"  = {name} (already present)", flush=True)
                continue
            print(f"  > {name}  {size / 1e6:.1f} MB" if size else f"  > {name}",
                  flush=True)
            status = fetch(repo, name, out)
            print(f"    {status}", flush=True)
            if status.startswith("FAILED"):
                failures.append(f"{repo}:{name}")

        got = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
        print(f"  dir {got / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)

    if failures:
        print("\nFAILURES:\n  " + "\n  ".join(failures), flush=True)
        return 1
    print("\nall models present", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

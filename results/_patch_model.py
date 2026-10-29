"""Remove the dead __main__ demo block from the diffusion checkpoint's model file.

The block imports `dllm`, which is not installed, and transformers'
`check_imports` refuses to load a dynamic module whose imports cannot be
satisfied -- even imports inside a `if __name__ == "__main__":` block that never
runs on import. Installing `dllm` would pull its own torch/transformers pins into
the venv, so the dead block is removed instead. Nothing above the marker is
touched; `diff` is printed to prove it.
"""
import difflib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = root / "checkpoints/hf/qwen2.5-coder-0.5b-diffusion-mdlm/modeling_qwen2.py"
orig_path = root / "results/_modeling_qwen2_orig.py"

src = p.read_text(encoding="utf-8")
if not orig_path.exists():
    orig_path.write_text(src, encoding="utf-8")

marker = 'if __name__ == "__main__":'
if marker in src:
    patched = src[:src.index(marker)].rstrip() + "\n"
    p.write_text(patched, encoding="utf-8")
    print(f"orig {len(src.splitlines())} lines -> patched "
          f"{len(patched.splitlines())} lines")
else:
    print("marker absent; nothing to do")

a = orig_path.read_text(encoding="utf-8").splitlines()
b = p.read_text(encoding="utf-8").splitlines()
diff = list(difflib.unified_diff(a, b, "orig", "patched", lineterm="", n=1))
print("\n".join(diff))
print(f"diff lines: {len(diff)}")

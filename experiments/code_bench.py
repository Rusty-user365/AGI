"""HumanEval harness foundation: data, prompts, and the sandboxed verifier.

This file contains no model. It exists so that the verifier is trustworthy
before any generation is measured against it. Two gates must hold:

    canonical solutions  164/164 pass
    corrupted bodies     every one fails

If either gate fails, every downstream pass rate is meaningless.

The verifier is a subprocess with a timeout and an isolated interpreter
(`-I`: no user site-packages, no PYTHONPATH, no script-dir injection). That is
enough for HumanEval, whose solutions are stdlib-only. It is not a security
sandbox: model-written code can still touch the filesystem or spin, which is why
the timeout is mandatory and the working directory is a fresh temp dir.

Prompting is raw completion, because the resident model is a base model. The
HumanEval prompt is the signature plus docstring; the model continues the body.
Generation stops at the standard HumanEval delimiters so a body cannot silently
become a second function.

Usage:
    python experiments/code_bench.py --self-check
    python experiments/code_bench.py --self-check --show-failures 5
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"

# Standard HumanEval stop sequences: a completion must not become a new
# top-level definition or start executing on import.
STOPS = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("]


def load(path: Path) -> list[dict]:
    """Read the cached corpus, downloading it once if absent."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = urllib.request.urlopen(URL, timeout=60).read()
        path.write_bytes(raw)
    rows = [json.loads(l) for l in
            gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()
            if l.strip()]
    for r in rows:
        r["full_canonical"] = r["prompt"] + r["canonical_solution"]
    return rows


def truncate(text: str) -> str:
    """Cut a completion at the first HumanEval stop sequence."""
    cut = len(text)
    for s in STOPS:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def run_tests(completion: str, test: str, entry_point: str,
              timeout: float = 5.0) -> dict:
    """Execute one candidate against its test in a fresh isolated subprocess.

    `completion` is the full program (prompt + body). Returns pass/fail plus the
    tail of stderr, which is the executable feedback the loop will feed back.
    """
    src = completion + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    with tempfile.TemporaryDirectory() as d:
        prog = Path(d) / "prog.py"
        prog.write_text(src, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(prog)], cwd=d,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "timeout", "stderr": "", "stdout": ""}
    return {
        "ok": r.returncode == 0,
        "reason": "pass" if r.returncode == 0 else "fail",
        "returncode": r.returncode,
        "stderr": (r.stderr or "")[-2000:],
        "stdout": (r.stdout or "")[-500:],
    }


def failure_kind(stderr: str) -> str:
    """Separate an exception (traceback localizes it) from a bare assertion.

    A wrong-output failure ends in AssertionError raised from the test harness,
    not from the candidate, so the traceback does *not* point at the faulty
    line. That distinction is the learned localizer's whole job.
    """
    if "AssertionError" in stderr:
        return "assert"
    if "Traceback (most recent call last)" in stderr:
        return "exception"
    return "other"


def self_check(rows: list[dict], show_failures: int, timeout: float) -> int:
    print(f"corpus: {len(rows)} problems")
    ok = 0
    bad = []
    t0 = time.perf_counter()
    for r in rows:
        res = run_tests(r["full_canonical"], r["test"], r["entry_point"], timeout)
        if res["ok"]:
            ok += 1
        else:
            bad.append((r["task_id"], res))
    dt = time.perf_counter() - t0
    print(f"canonical: {ok}/{len(rows)} pass  ({dt:.1f} s, "
          f"{dt / max(len(rows), 1) * 1e3:.0f} ms/problem)")

    corrupt_ok = 0
    for r in rows:
        body = "    raise RuntimeError('corrupt')\n"
        res = run_tests(r["prompt"] + body, r["test"], r["entry_point"], timeout)
        if res["ok"]:
            corrupt_ok += 1
            bad.append((r["task_id"], {"ok": True, "reason": "corrupt passed"}))
    print(f"corrupted: {len(rows) - corrupt_ok}/{len(rows)} fail as required")

    for tid, res in bad[:show_failures]:
        print(f"\n--- {tid} ({res['reason']}) ---")
        print(res["stderr"][-600:])

    if ok != len(rows) or corrupt_ok != 0:
        print("\nGATE FAILED: the verifier is not trustworthy")
        return 1
    print("\nGATE PASSED: canonical 164/164, corrupted 0 pass")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/humaneval.jsonl.gz")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--show-failures", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = Path(args.data)
    if not path.is_absolute():
        path = root / path
    rows = load(path)

    if args.self_check:
        return self_check(rows, args.show_failures, args.timeout)

    print(f"{len(rows)} problems; first prompt:\n")
    print(rows[0]["prompt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

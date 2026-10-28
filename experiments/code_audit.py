"""Pre-run audit: is the measurement harness correct before 4 hours of GPU?

No model is loaded. Four checks, all cheap:

    A  test sensitivity   do the official tests fail a stub body?
    B  localization math  does an exception cut land on the candidate line?
    C  truncation         does a stop sequence eat legitimate body lines?
    D  repair assembly    does prefix + continuation stay valid Python?

If any check fails, the long run is measuring the harness, not the loop.

Usage:
    python experiments/code_audit.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.code_bench import (failure_kind, load, run_tests, truncate)
from experiments.code_loop import (build_repair_prompt, feedback_of,
                                   localize_cut, top_level_start)

ROOT = Path(__file__).resolve().parents[1]
rows = load(ROOT / "data/humaneval.jsonl.gz")


def part_a() -> None:
    print("=" * 70)
    print("A. test sensitivity: a stub body must fail the official test")
    print("=" * 70)
    stubs = {
        "pass(None)": "    pass\n",
        "return 0": "    return 0\n",
        "return []": "    return []\n",
    }
    n = len(rows)
    for name, body in stubs.items():
        passed = []
        for r in rows:
            res = run_tests(r["prompt"] + body, r["test"], r["entry_point"])
            if res["ok"]:
                passed.append(r["task_id"])
        print(f"  {name:<12} fails {n - len(passed):>3}/{n}  "
              f"({len(passed)} stubs pass)")
        if passed and len(passed) <= 12:
            print(f"    passes: {', '.join(passed)}")

    # A test that never calls check() would pass everything. Verify the call
    # site exists in every test body.
    missing = [r["task_id"] for r in rows if "check(" not in r["test"]]
    print(f"  tests with a check() call: {n - len(missing)}/{n}")
    if missing:
        print(f"    MISSING: {missing}")


def part_b() -> None:
    print("\n" + "=" * 70)
    print("B. localization math: exception cut must land on the faulty line")
    print("=" * 70)
    r = rows[0]
    prompt_lines = len(r["prompt"].splitlines())

    # A body that raises on a known line, inside a loop, so the enclosing
    # statement is not the raising line itself.
    body = (
        "    total = 0\n"
        "    for x in numbers:\n"
        "        total += x\n"
        "        raise ValueError('boom')\n"
        "    return total\n"
    )
    res = run_tests(r["prompt"] + body, r["test"], r["entry_point"])
    print(f"  kind={failure_kind(res['stderr'])} failed_line={res['failed_line']!r}")
    lines = body.splitlines()
    cut, kind = localize_cut(lines, res, prompt_lines)
    print(f"  cut={cut} kind={kind} -> line {lines[cut]!r}")
    ok = lines[cut].strip().startswith("for") or lines[cut].strip().startswith("total")
    print(f"  cut points at the enclosing statement: {ok}")

    # The traceback line numbers must be offset by the prompt, not the test.
    frames = re.findall(r'prog\.py", line (\d+)', res["stderr"])
    print(f"  prog.py frames: {frames} (prompt_lines={prompt_lines}, "
          f"body_lines={len(lines)})")
    in_body = [int(f) for f in frames
               if prompt_lines < int(f) <= prompt_lines + len(lines)]
    print(f"  frames inside the candidate body: {in_body} "
          f"(expected [{prompt_lines + 4}])")


def part_c() -> None:
    print("\n" + "=" * 70)
    print("C. truncation: stop sequences must not eat legitimate body lines")
    print("=" * 70)
    samples = {
        "comment at col 0": "    x = 1\n# note\n    return x\n",
        "indented comment": "    x = 1\n    # note\n    return x\n",
        "nested def": "    def inner(y):\n        return y\n    return inner(1)\n",
        "print at col 0": "    x = 1\nprint(x)\n    return x\n",
        "class at col 0": "    x = 1\nclass A:\n    pass\n",
    }
    for name, text in samples.items():
        out = truncate(text)
        print(f"  {name:<18} kept {len(out.splitlines())}/{len(text.splitlines())} "
              f"lines, end={out.splitlines()[-1]!r}")


def part_d() -> None:
    print("\n" + "=" * 70)
    print("D. repair assembly: prefix + continuation must be valid Python")
    print("=" * 70)
    r = rows[0]
    body = r["canonical_solution"]
    lines = body.splitlines()
    prompt_lines = len(r["prompt"].splitlines())
    for cut in (0, 1, len(lines) - 1):
        cut = max(0, cut)
        rp = build_repair_prompt(r["prompt"], lines, cut, "assert f(1) == 2")
        suffix = "\n".join(lines[cut:]) + "\n"
        res = run_tests(rp + suffix, r["test"], r["entry_point"])
        print(f"  cut={cut:<2} reassembled passes: {res['ok']}  "
              f"(feedback={feedback_of({'failed_line': 'assert f(1) == 2'})!r})")
    print(f"  prompt ends with: {r['prompt'][-30:]!r}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()

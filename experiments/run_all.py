"""Run the full experiment suite and aggregate the results.

    python experiments/run_all.py --preset smoke          # seconds
    python experiments/run_all.py --preset dev --n 200    # Kaggle T4, minutes

Every experiment is a separate process, so one failure does not hide the rest
and each writes its own JSON. A non-zero exit is reported but does not abort
the suite: a failed experiment is a result too.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

EXPERIMENTS = [
    ("exp0_benchmark_validity.py", "benchmark validity (CPU gate)"),
    ("exp1_invariants.py", "design invariants"),
    ("exp2_scope_and_pressure.py", "scope, pressure, masking"),
    ("exp3_shuffle.py", "side-channel causality"),
    ("exp4_repair_taxonomy.py", "repair taxonomy"),
    ("exp5_gate_triage.py", "gate usage and triage"),
    ("exp6_structural_guard.py", "structural guard (C1)"),
]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma-separated exp prefixes")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    only = tuple(args.only.split(",")) if args.only else None
    # Warn if results from other seeds are present: aggregate will then render
    # a multi-seed table that includes runs this invocation did not make.
    stale = sorted(p.name for p in RESULTS.glob("exp*_s*.json"))
    if stale:
        print(f"[warn] results from other seeds present ({len(stale)} files). "
              f"Tables will include them; delete results/exp*_s*.json for a "
              f"single-seed table.")
    log = {}
    for script, desc in EXPERIMENTS:
        if only and not script.startswith(only):
            continue
        cmd = [sys.executable, str(HERE / script), "--preset", args.preset,
               "--seed", str(args.seed)]
        if args.n:
            cmd += ["--n", str(args.n)]
        print(f"\n>>> {script}  ({desc})")
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        dt = time.time() - t0
        status = "ok" if r.returncode == 0 else f"exit {r.returncode}"
        print(f"    {status} in {dt:.1f}s")
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
        log[script] = {"status": status, "seconds": round(dt, 1),
                       "returncode": r.returncode}
        (RESULTS / (Path(script).stem + ".log")).write_text(
            r.stdout + "\n" + r.stderr, encoding="utf-8")

    (RESULTS / "run_log.json").write_text(json.dumps(log, indent=2),
                                          encoding="utf-8")
    print("\n>>> aggregate")
    subprocess.run([sys.executable, str(HERE / "aggregate.py")],
                   cwd=str(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

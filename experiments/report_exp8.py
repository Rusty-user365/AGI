"""Print a paired exp8 artifact as a compact table.

Usage:
    python experiments/report_exp8.py results/exp8_fixup_steering_test.json

Prints one block per condition: the two arms' means, the paired difference with
its bootstrap interval, and whether that interval excludes zero. The paired
difference is the only number in the file that supports a comparative claim;
the per-arm means are descriptive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

METRICS = (
    ("c_target", "thesis restored"),
    ("q_target", "quality target"),
    ("improvement", "edit quality change"),
)


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:+.3f}"


def rate(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    d = json.loads(path.read_text(encoding="utf-8"))

    print(f"{path.name}")
    print(f"  split={d.get('split')}  n={d.get('requested_n')}  seed={d.get('seed')}")
    print(f"  cgr      {d.get('cgr_checkpoint')}  cond={d.get('cgr_cond_mode')}")
    print(f"  baseline {d.get('baseline_checkpoint')}  cond={d.get('baseline_cond_mode')}")

    conds = d.get("conditions", {})
    for key, row in conds.items():
        n = row.get("n")
        print(f"\n== {key}  (n={n})")
        for metric, label in METRICS:
            cgr = row.get(f"cgr_mean_{metric}")
            base = row.get(f"backbone_mean_{metric}")
            diff = row.get(f"{metric}_cgr_minus_backbone") or {}
            sig = diff.get("significant")
            mark = "SIG" if sig else "   "
            print(f"  {label:<20} cgr {rate(cgr):>7}  base {rate(base):>7}  "
                  f"diff {fmt(diff.get('mean_diff')):>7} "
                  f"[{fmt(diff.get('lo'))}, {fmt(diff.get('hi'))}] {mark}")
        print(f"  doc-consistent        cgr {rate(row.get('cgr_doc_consistent_rate')):>7}  "
              f"base {rate(row.get('backbone_doc_consistent_rate')):>7}  "
              f"noop {rate(row.get('noop_doc_consistent_rate')):>7}")

    # A single headline line: how many conditions favour each arm on the primary
    # metric, and whether any difference is significant. Deliberately blunt,
    # because a table of six conditions invites cherry-picking one row.
    sig_for = sig_against = 0
    for row in conds.values():
        diff = row.get("c_target_cgr_minus_backbone") or {}
        if diff.get("significant"):
            if (diff.get("mean_diff") or 0) > 0:
                sig_for += 1
            else:
                sig_against += 1
    print(f"\nheadline: c_target differs significantly in {sig_for + sig_against} "
          f"of {len(conds)} conditions ({sig_for} favouring cgr, "
          f"{sig_against} favouring the baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

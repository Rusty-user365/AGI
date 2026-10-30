"""Generate deterministic CPU-only logic, safety, and injection JSONL data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgr.data.rd_datasets import (make_injection_cases, make_logic_traps,
                                 make_safety_claims)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "generated")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logic-n", type=int, default=5_000)
    p.add_argument("--safety-n", type=int, default=5_000)
    p.add_argument("--injection-n", type=int, default=500)
    args = p.parse_args()
    outputs = {
        "logic_traps.jsonl": make_logic_traps(args.logic_n, args.seed),
        "safety_claims.jsonl": make_safety_claims(args.safety_n, args.seed + 1),
        "prompt_injections.jsonl": make_injection_cases(args.injection_n,
                                                        args.seed + 2),
    }
    for name, rows in outputs.items():
        path = args.out_dir / name
        write_jsonl(path, rows)
        print(f"{path}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One-shot entry point for Kaggle / Colab.

    !python scripts/kaggle_run.py --preset dev --n 200

Trains once, then runs every experiment at inference time. Most ablations need
no retraining, which is what makes the budget fit a free T4.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="dev",
                    choices=["smoke", "dev", "paper", "diffugpt"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    from experiments._common import ckpt_dir  # noqa: E402  (path set above)

    for seed in seeds:
        out = ROOT / ckpt_dir(args.preset, seed)
        ckpt = out / "stage_b.pt"
        if not args.skip_train and not ckpt.exists():
            print(f"\n=== training {args.preset} (seed {seed}) -> {out} ===")
            t0 = time.time()
            from cgr.config import Config
            from cgr.train import train
            cfg = Config.preset(args.preset)
            cfg.train.seed = seed
            summary = train(cfg, out, verbose=True)
            print(f"    trained in {time.time() - t0:.1f}s; "
                  f"params={summary['params']}")

        print(f"\n=== experiments (seed {seed}) ===")
        subprocess.run([sys.executable, str(ROOT / "experiments" / "run_all.py"),
                        "--preset", args.preset, "--n", str(args.n),
                        "--seed", str(seed)], cwd=str(ROOT), check=False)

    tables = ROOT / "results" / "paper_tables.md"
    if tables.exists():
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass
        print("\n" + tables.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

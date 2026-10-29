"""Compare the call-allocation curve across corpora, and price the headroom.

`interp_calls.py` writes one JSON per corpus. This reads them and answers the
two questions the paper needs:

1. Does a localized call still beat a random call at the same k, out of sample?
   (paired McNemar, per k, per corpus)
2. What fraction of the all-steps headroom does k localized calls capture?
   (loc_k - k0) / (k5 - k0), so the number is comparable across corpora with
   different solve rates.

The headroom fraction is the only cross-corpus number that makes sense. Raw
rates are not comparable because the vault's localizer is weaker (top-1 0.625
against 0.718 on the h5 holdout), so a lower vault rate is expected even if the
mechanism is identical.

Usage:
    python experiments/interp_compare.py results/interp_calls_h5.json \
        results/interp_calls_vault.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def headroom(arms: dict, k: int) -> float | None:
    base = arms["k0_rank2"]["ok"]
    top = arms["k5_all"]["ok"]
    if top == base:
        return None
    return (arms[f"k{k}_loc"]["ok"] - base) / (top - base)


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in (argv[1:] or ["results/interp_calls_h5.json"])]
    runs = [(p.stem.replace("interp_calls_", ""), load(p)) for p in paths]

    for name, d in runs:
        n = d["n"]
        a = d["arms"]
        print("=" * 78)
        print(f"{name}  N={n}  train={d.get('train_candidates', '?')}  "
              f"anchor={d.get('anchor_enforced')}")
        print("=" * 78)
        print(f"{'arm':<14} {'ok':>9} {'rate':>7} {'95% CI':>17} {'calls':>7}")
        for k in ("k0_rank2", "k1_loc", "k2_loc", "k3_loc",
                  "k1_rand", "k2_rand", "k3_rand", "k1_oracleloc", "k5_all"):
            s = a[k]
            print(f"{k:<14} {s['ok']:>4}/{n:<4} {s['rate']:>7.3f} "
                  f"[{s['ci'][0]:.3f},{s['ci'][1]:.3f}] {s['calls']:>7.2f}")
        print()
        for k in (1, 2, 3):
            p = d["paired"][f"k{k}_loc_vs_rand"]
            h = headroom(a, k)
            hs = "n/a" if h is None else f"{100 * h:5.1f}%"
            print(f"  k={k}  loc {a[f'k{k}_loc']['ok']:>3} vs rand "
                  f"{a[f'k{k}_rand']['ok']:>3}  {p['wins']:>3}W/{p['losses']:<3}L "
                  f"p={p['p']:.2e}  headroom {hs}")
        p0 = d["paired"]["k1_loc_vs_k0"]
        print(f"  k=1  loc vs k0_rank2: {p0['wins']}W/{p0['losses']}L "
              f"p={p0['p']:.2e}")
        loc = d.get("localizer", {})
        print(f"  localizer top-1 {loc.get('two_tier_top1', float('nan')):.3f} "
              f"| t_exact {loc.get('t_exact', float('nan')):.3f}")
        print()

    if len(runs) == 2:
        (n1, d1), (n2, d2) = runs
        print("=" * 78)
        print("cross-corpus headroom (fraction of the all-steps gap)")
        print("=" * 78)
        print(f"{'k':>3} {n1:>12} {n2:>12}")
        for k in (1, 2, 3):
            h1, h2 = headroom(d1["arms"], k), headroom(d2["arms"], k)
            f1 = "n/a" if h1 is None else f"{100 * h1:10.1f}%"
            f2 = "n/a" if h2 is None else f"{100 * h2:10.1f}%"
            print(f"{k:>3} {f1:>12} {f2:>12}")
        print()
        print("Read the fraction, not the rate: the vault localizer is frozen and")
        print("weaker, so its raw solve rate should be lower even if the mechanism")
        print("is identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""Exact Sudoku core: generator, checker, and the classical baseline.

This is the weapon, not a measurement. Three things live here and nothing else:

    legality      the exact verifier (row / column / box), O(1) per digit
    MRV solver    the baseline that a learned proposal has to beat
    generator     puzzles with a *proved* unique solution, built locally

Why the baseline is MRV and not random: comparing a model to random cell order
would be inventing a weak opponent. Minimum-remaining-values with constraint
propagation is what a competent classical solver does, and it is the number the
model must beat to be worth loading. Random cell order is kept only as a
sanity floor.

Why puzzles are generated rather than downloaded: uniqueness has to be a fact,
not an assumption about a corpus. `generate` removes clues one at a time and
restores any removal that destroys uniqueness, so every puzzle it returns has
exactly one solution by construction. That also means no ground-truth file can
be wrong.

Node accounting is the metric that matters downstream: `nodes` counts digit
assignments attempted, which is the search work a verifier loop is trying to
reduce. Wall-clock is reported too, but nodes are the portable number.

Usage:
    python experiments/sudoku_core.py --self-check 12 --clues 32
    python experiments/sudoku_core.py --self-check 12 --clues 32 --compare-random
"""
from __future__ import annotations

import argparse
import random
import statistics
import time

N = 9
CELLS = N * N

ROWS = [[r * N + c for c in range(N)] for r in range(N)]
COLS = [[r * N + c for r in range(N)] for c in range(N)]
BOXES = [[(br * 3 + dr) * N + (bc * 3 + dc)
          for dr in range(3) for dc in range(3)]
         for br in range(3) for bc in range(3)]

UNITS = ROWS + COLS + BOXES
PEERS = [sorted({j for u in UNITS if i in u for j in u if j != i})
         for i in range(CELLS)]


def parse(text: str) -> list[int]:
    """Read a puzzle as 81 ints, 0 = blank. Accepts dots, spaces and newlines."""
    digits = [int(ch) for ch in text if ch.isdigit()]
    if len(digits) != CELLS:
        raise ValueError(f"expected {CELLS} digits, got {len(digits)}")
    return digits


def format_grid(grid: list[int]) -> str:
    rows = []
    for r in range(N):
        cells = ["." if grid[r * N + c] == 0 else str(grid[r * N + c])
                 for c in range(N)]
        rows.append("".join(cells))
    return "\n".join(rows)


def legal(grid: list[int], idx: int, digit: int) -> bool:
    """The exact verifier. A digit is legal iff no peer holds it."""
    return all(grid[j] != digit for j in PEERS[idx])


def candidates(grid: list[int], idx: int) -> list[int]:
    used = {grid[j] for j in PEERS[idx]}
    return [d for d in range(1, N + 1) if d not in used]


def is_consistent(grid: list[int]) -> bool:
    """No unit repeats a digit. Blanks are allowed."""
    for unit in UNITS:
        seen = [grid[i] for i in unit if grid[i] != 0]
        if len(seen) != len(set(seen)):
            return False
    return True


def is_solved(grid: list[int]) -> bool:
    return all(d != 0 for d in grid) and is_consistent(grid)


def solve(grid: list[int], policy: str = "mrv", rng: random.Random | None = None,
          limit: int = 1, shuffle_values: bool = False) -> dict:
    """Backtracking search.

    policy="mrv"     fewest candidates first, lowest index breaks ties
    policy="random"  uniform random empty cell (the sanity floor, not a rival)

    Returns {"solution", "nodes", "solutions"}. `nodes` counts digit
    assignments attempted, i.e. the search work a verifier loop would try to
    reduce. `solutions` is capped at `limit`; 2 is enough to test uniqueness.
    """
    if policy not in ("mrv", "random"):
        raise ValueError(policy)
    if policy == "random" and rng is None:
        raise ValueError("policy='random' needs an rng")

    g = list(grid)
    found: list[list[int]] = []
    nodes = 0

    def rec() -> None:
        nonlocal nodes
        if len(found) >= limit:
            return

        if policy == "mrv":
            best_i, best_c = -1, None
            for i in range(CELLS):
                if g[i]:
                    continue
                c = candidates(g, i)
                if not c:
                    return                      # dead end, prune here
                if best_c is None or len(c) < len(best_c):
                    best_i, best_c = i, c
                    if len(c) == 1:
                        break
            if best_c is None:
                found.append(list(g))           # no blanks left
                return
        else:
            blanks = [i for i in range(CELLS) if g[i] == 0]
            if not blanks:
                found.append(list(g))
                return
            best_i = rng.choice(blanks)
            best_c = candidates(g, best_i)
            if not best_c:
                return

        if shuffle_values:
            rng.shuffle(best_c)
        for d in best_c:
            nodes += 1
            g[best_i] = d
            rec()
            g[best_i] = 0
            if len(found) >= limit:
                return

    rec()
    return {"solution": found[0] if found else None,
            "nodes": nodes,
            "solutions": len(found)}


def count_solutions(grid: list[int], limit: int = 2) -> int:
    """Number of solutions, capped at `limit`."""
    return solve(grid, "mrv", None, limit)["solutions"]


def generate(rng: random.Random, clues: int) -> dict:
    """A puzzle with exactly one solution and `clues` or as close as possible.

    Starts from a random complete grid, then removes clues in random order and
    keeps a removal only if uniqueness survives. Removal stops early if no
    further clue can be dropped without creating a second solution.
    """
    full = solve([0] * CELLS, "mrv", rng, 1, shuffle_values=True)["solution"]
    if full is None:
        raise RuntimeError("empty grid was unsolvable, which should be impossible")

    puzzle = list(full)
    order = list(range(CELLS))
    rng.shuffle(order)
    remaining = CELLS
    for i in order:
        if remaining <= clues:
            break
        keep = puzzle[i]
        puzzle[i] = 0
        if count_solutions(puzzle, 2) == 1:
            remaining -= 1
        else:
            puzzle[i] = keep

    return {"puzzle": puzzle, "solution": full, "clues": remaining}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", type=int, default=12,
                    help="number of puzzles to generate and verify")
    ap.add_argument("--clues", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare-random", action="store_true",
                    help="also solve each puzzle with random cell order")
    ap.add_argument("--compare-branch", action="store_true",
                    help="also solve with MRV cell choice but shuffled candidate order")
    ap.add_argument("--show", type=int, default=1,
                    help="print this many grids")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    mrv_nodes, rnd_nodes = [], []
    mrv_ms, rnd_ms = [], []
    br_nodes, br_ms = [], []
    t_gen = 0.0

    print(f"generating {args.self_check} puzzles at ~{args.clues} clues ...")
    for k in range(args.self_check):
        t0 = time.perf_counter()
        item = generate(rng, args.clues)
        t_gen += time.perf_counter() - t0
        p, sol = item["puzzle"], item["solution"]

        # correctness gates: the puzzle is legal, unique, and the solution solves it
        assert is_consistent(p), f"puzzle {k} is not a legal grid"
        assert is_solved(sol), f"solution {k} is not a solved grid"
        assert all(p[i] == 0 or p[i] == sol[i] for i in range(CELLS)), \
            f"puzzle {k} disagrees with its solution"
        assert count_solutions(p, 2) == 1, f"puzzle {k} does not have a unique solution"

        t0 = time.perf_counter()
        r = solve(p, "mrv")
        mrv_ms.append((time.perf_counter() - t0) * 1e3)
        assert r["solution"] == sol, f"MRV did not recover the unique solution {k}"
        mrv_nodes.append(r["nodes"])

        if args.compare_random:
            t0 = time.perf_counter()
            r2 = solve(p, "random", random.Random(1000 + k))
            rnd_ms.append((time.perf_counter() - t0) * 1e3)
            assert r2["solution"] is not None, f"random search failed on puzzle {k}"
            rnd_nodes.append(r2["nodes"])

        if args.compare_branch:
            t0 = time.perf_counter()
            r3 = solve(p, "mrv", random.Random(2000 + k), shuffle_values=True)
            br_ms.append((time.perf_counter() - t0) * 1e3)
            assert r3["solution"] is not None, f"shuffled-branch search failed on {k}"
            br_nodes.append(r3["nodes"])

        if k < args.show:
            print(f"\n--- puzzle {k}: {item['clues']} clues ---")
            print(format_grid(p))

    print(f"\n{'=' * 66}")
    print(f"SUDOKU CORE SELF-CHECK  ({args.self_check} puzzles, {args.clues} clues)")
    print(f"{'=' * 66}")
    print(f"generation:            {t_gen:.1f} s total, "
          f"{t_gen / max(args.self_check, 1):.2f} s/puzzle")
    print(f"uniqueness:            {args.self_check}/{args.self_check} proved")
    print(f"MRV nodes:             median {statistics.median(mrv_nodes):.0f}, "
          f"mean {statistics.mean(mrv_nodes):.1f}, max {max(mrv_nodes)}")
    print(f"MRV time:              median {statistics.median(mrv_ms):.1f} ms, "
          f"max {max(mrv_ms):.1f} ms")
    if args.compare_random:
        print(f"random-order nodes:    median {statistics.median(rnd_nodes):.0f}, "
              f"mean {statistics.mean(rnd_nodes):.1f}, max {max(rnd_nodes)}")
        print(f"random-order time:     median {statistics.median(rnd_ms):.1f} ms, "
              f"max {max(rnd_ms):.1f} ms")
        ratio = statistics.median(rnd_nodes) / max(statistics.median(mrv_nodes), 1)
        print(f"MRV advantage:         {ratio:.1f}x fewer nodes than random order")
    if args.compare_branch:
        print(f"shuffled-branch nodes: median {statistics.median(br_nodes):.0f}, "
              f"mean {statistics.mean(br_nodes):.1f}, max {max(br_nodes)}")
        print(f"shuffled-branch time:  median {statistics.median(br_ms):.1f} ms, "
              f"max {max(br_ms):.1f} ms")
        ratio = statistics.median(br_nodes) / max(statistics.median(mrv_nodes), 1)
        print(f"branch-order cost:     {ratio:.2f}x nodes vs ascending order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Sudoku is a control, not a target (2026-10-29)

**Status:** complete. Verdict: Sudoku is closed as a model target in the
difficulty range this generator can produce.

**Artifacts:** `experiments/sudoku_core.py`

## What was built

The exact weapon, no model and no GPU:

- `legal` / `candidates` — the verifier, O(1) per digit via precomputed peers.
- `generate` — puzzles with a **proved** unique solution. Starts from a random
  complete grid and removes clues in random order, restoring any removal that
  creates a second solution. Uniqueness is a fact about each puzzle, not an
  assumption about a downloaded corpus.
- `solve(policy="mrv")` — minimum-remaining-values backtracking with forward
  checking. This is the baseline a learned proposal has to beat.
- `solve(policy="random")` — random cell order. Kept only as a sanity floor.

Gates, all asserted in the self-check: every puzzle is a legal grid, has exactly
one solution, and MRV recovers that solution. 40/40 and 25/25 passed.

## Measurements

Same seed, generated locally, `nodes` = digit assignments attempted.

| Clues | Puzzles | Uniqueness proved | MRV nodes (median) | MRV time (median) | Random cell order | Shuffled branch order |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 32 | 40 | 40/40 | **50** | 1.6 ms | 2,481,272 nodes / 23.6 s | 55 nodes (**1.10x**) |
| ~24 | 25 | 25/25 | **285** | 11.9 ms | not run | 186 nodes (**0.65x**) |

Three numbers decide this:

1. **Cell selection is the whole game, and it is solved exactly.** Random cell
   order needs 2.48M nodes and 23.6 s; MRV needs 50 nodes and 1.6 ms. That is a
   50,638x gap in nodes. A learned cell selector would have to beat a 1.6 ms
   exact procedure to be worth loading.

2. **Branch order has almost no headroom.** With cell choice held fixed at MRV,
   shuffling the candidate order costs **1.10x** nodes at 32 clues and is
   actually **0.65x** at 24 clues. Ascending order is arbitrary, so a perfect
   value-ranker could at best save a fraction of a solve that already takes
   12 ms. Even a perfect ranker is worth milliseconds.

3. **The hard tail is out of reach of this generator.** Greedy clue removal
   bottoms out at **23-26 clues** (median 24) when asked for 17. The 17-clue
   puzzles where MRV genuinely explodes are therefore **untested**, and the
   "18 clues" row above is really a 24-clue row. That limitation is stated
   rather than hidden.

## Verdict

**Sudoku cannot support the CGR claim, and a 0.6B model cannot add value to it.**

The claim was "localized verifier calls reach the same result with fewer calls".
In Sudoku the verifier localizes the violation for free, and MRV orders the
search near-optimally. There is no allocation problem left to solve, and the
absolute headroom is milliseconds. Any model placed in this loop would be
decorative at best and slower at worst.

This is a **negative result obtained in five minutes with no GPU**, and it is
the reason not to spend two days here. Sudoku stays in the repo as a control:
it is the domain where an exact constraint exists, so it is the clean
demonstration that *having* a verifier is not the same as *needing* a model.

The remaining candidate is **code**, because no MRV exists for "write a function
that passes these tests": the verifier says pass/fail plus a traceback, and
ordering the search over programs is not a solved problem.

## What would reopen Sudoku

Only 17-clue puzzles, where MRV's node count is orders of magnitude higher and
branch order might matter. The generator cannot produce them; a real corpus
would be needed. Even then the honest claim would be about nodes, not about a
0.6B model, which has no mechanism for exact constraint propagation.

#!/usr/bin/env python3
"""Neurosymbolic Diagonal Sudoku solver (text input only).

  python main.py                           # solve examples/inputs/standard_classic.txt (rules from its "# variant:" line)
  python main.py solve-board FILE          # FILE in the format  board = [[6,_,_,...], ...]  (_ = blank)
  python main.py random [--clues 30]       # random puzzle (printed in the input format) and its solution
  python main.py train                     # (re)train the neural module (a few minutes)
  python main.py --variant standard ...    # classic rules instead of diagonal
Options for solve-board / random:  --mode {plain,guided,hints,both}  (default: both)
                                   --explain   show an unsatisfied condition (conflict) and why the search cannot go on
"""
import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MODEL = ROOT / "models" / "guide.pkl"
DEFAULT_INPUT = str(ROOT / "examples" / "inputs" / "standard_classic.txt")


def load_guide():
    from sudoku_ns.neural import NeuralGuide
    if not MODEL.exists():
        print("no trained neural model found; training (a few minutes)...", flush=True)
        MODEL.parent.mkdir(exist_ok=True)
        NeuralGuide.train().save(MODEL)
    return NeuralGuide.load(MODEL)


def solve_and_report(grid, mode, variant, truth=None, explain=False):
    from sudoku_ns import generator as G
    from sudoku_ns.neurosym import NeuroSymbolicSolver
    guide = load_guide() if mode != "plain" else None
    ns = NeuroSymbolicSolver(guide)
    res = ns.solve(grid, mode, check_unique=True)
    if res.solution is None:
        print(f"\nUNSAT: the givens contradict the {variant} Sudoku rules"
              + (" (try --variant standard)" if variant == "diagonal" else ""))
        return
    print("\nSolution:"); print(G.pretty(res.solution))
    base = ns.solve(grid, "plain")
    print(f"\nunique solution: {res.unique}" + (f" | matches generator: {res.solution == truth}" if truth else ""))
    print(f"mode={mode}: decisions={res.decisions} conflicts={res.conflicts} solver-calls={res.rounds} "
          f"hints used={res.hints_used} dropped (wrong)={res.hints_dropped} time={res.time:.3f}s")
    print(f"plain CDCL : decisions={base.decisions} conflicts={base.conflicts} time={base.time:.3f}s")
    if explain:
        from sudoku_ns.explain import explain_run
        print(explain_run(grid, res.solution))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--variant", choices=["diagonal", "standard"], default=None,
                    help="rules; default: the file's `# variant:` line, else diagonal")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("solve-board"); b.add_argument("file", nargs="?", default=DEFAULT_INPUT)
    b.add_argument("--mode", choices=["plain", "guided", "hints", "both"], default="both")
    b.add_argument("--explain", action="store_true", help="show unsatisfied conditions (conflicts) and why")
    r = sub.add_parser("random"); r.add_argument("--clues", type=int, default=0, help="0 = minimal puzzle")
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--mode", choices=["plain", "guided", "hints", "both"], default="both")
    r.add_argument("--explain", action="store_true")
    sub.add_parser("train")
    a = ap.parse_args()
    if a.cmd is None:
        a.cmd, a.file, a.mode, a.explain = "solve-board", DEFAULT_INPUT, "both", False
    if a.variant is None:                              # `# variant: standard` line in the input file
        a.variant = "diagonal"
        if a.cmd == "solve-board" and a.file != "-" and Path(a.file).exists():
            for line in Path(a.file).read_text().splitlines():
                if line.strip().lower().replace(" ", "").startswith("#variant:"):
                    a.variant = line.split(":", 1)[1].split()[0].lower()
    if a.variant not in ("diagonal", "standard"):
        sys.exit("variant must be 'diagonal' or 'standard'")
    os.environ["SUDOKU_VARIANT"] = a.variant          # must precede importing sudoku_ns

    if a.cmd == "train":
        from sudoku_ns.neural import NeuralGuide
        MODEL.parent.mkdir(exist_ok=True)
        NeuralGuide.train().save(MODEL); print("saved", MODEL)
    elif a.cmd == "random":
        from sudoku_ns import generator as G
        puz, truth = G.random_puzzle(random.Random(a.seed), a.clues)
        print(f"Random {a.variant} puzzle ({sum(1 for x in puz if x)} clues):"); print(G.pretty(puz))
        print("\nAs an input file:"); print(G.grid_to_list_text(puz))
        solve_and_report(puz, a.mode, a.variant, truth, a.explain)
    else:
        from sudoku_ns import generator as G
        text = sys.stdin.read() if a.file == "-" else Path(a.file).read_text()
        try:
            grid = G.parse_board_list(text)
        except ValueError as e:
            sys.exit(f"Invalid input file: {e}\nExpected format (see examples/inputs/standard_classic.txt):\n"
                     "board = [\n    [6, _, _, _, 9, 8, _, _, 4],\n    ... 9 rows ...\n]")
        print(f"Input ({a.variant} rules):"); print(G.pretty(grid))
        solve_and_report(grid, a.mode, a.variant, None, a.explain)


if __name__ == "__main__":
    main()

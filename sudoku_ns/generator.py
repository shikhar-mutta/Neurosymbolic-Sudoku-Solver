"""Random Sudoku generation with guaranteed unique solutions."""
from __future__ import annotations

import random
import re

from .csp import CSPSolver


def random_solution(rng: random.Random) -> list[int]:
    return CSPSolver(rng).solve([0] * 81, 1, randomize=True)[0]


def make_puzzle(solution, rng: random.Random, target_clues: int = 0) -> list[int]:
    """Remove clues in random order while keeping the solution unique.
    ``target_clues=0`` -> remove until minimal (no clue can be dropped)."""
    puz = list(solution)
    solver = CSPSolver()
    order = list(range(81))
    rng.shuffle(order)
    clues = 81
    for i in order:
        if target_clues and clues <= target_clues:
            break
        keep = puz[i]
        puz[i] = 0
        if solver.count(puz, 2) != 1:
            puz[i] = keep
        else:
            clues -= 1
    return puz


def random_puzzle(rng: random.Random, clues: int = 0):
    sol = random_solution(rng)
    return make_puzzle(sol, rng, clues), sol


def symmetry_transform(grid, rng: random.Random) -> list[int]:
    """Validity-preserving transformation for BOTH variants: digit relabelling and the
    8 symmetries of the square (rotations/reflections), which map the two diagonals onto
    themselves.  Zero stays zero."""
    perm = list(range(1, 10))
    rng.shuffle(perm)
    mp = [0] + perm
    g = [mp[x] for x in grid]
    for _ in range(rng.randrange(4)):                     # rotate 90 degrees
        g = [g[9 * (8 - c) + r] for r in range(9) for c in range(9)]
    if rng.random() < 0.5:                                # mirror
        g = [g[9 * r + 8 - c] for r in range(9) for c in range(9)]
    return g


def grid_to_str(grid) -> str:
    return "".join(str(d) if d else "_" for d in grid)


def str_to_grid(s: str) -> list[int]:
    s = "".join(ch for ch in s if ch in "0123456789._")      # blank = _  (also . or 0)
    if len(s) != 81:
        raise ValueError("a puzzle string needs exactly 81 cells")
    return [0 if ch in "._0" else int(ch) for ch in s]


def pretty(grid) -> str:
    lines = []
    for r in range(9):
        if r % 3 == 0 and r:
            lines.append("------+-------+------")
        row = []
        for c in range(9):
            if c % 3 == 0 and c:
                row.append("|")
            row.append(str(grid[9 * r + c]) if grid[9 * r + c] else "_")
        lines.append(" ".join(row))
    return "\n".join(lines)


_ROW = re.compile(r"\[([^\[\]]*)\]")


def parse_board_list(text: str) -> list[int]:
    """Strict parser for the ONLY accepted input format::

        board = [
            [6, _, _, _, 9, 8, _, _, 4],
            ... (9 rows of 9 entries; digits 1-9, `_` = blank)
        ]
    Lines starting with ``#`` are comments.  Anything else raises ValueError."""
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    m = re.match(r"\s*board\s*=\s*\[(.*)\]\s*$", text, re.S)
    if not m:
        raise ValueError("input must look like:  board = [ [6, _, _, ...], ... ]")
    rows = _ROW.findall(m.group(1))
    if len(rows) != 9:
        raise ValueError(f"expected 9 rows, found {len(rows)}")
    grid = []
    for k, row in enumerate(rows, 1):
        toks = [t.strip() for t in row.split(",")]
        if toks and toks[-1] == "":
            toks.pop()                           # allow trailing comma
        if len(toks) != 9:
            raise ValueError(f"row {k}: expected 9 entries, found {len(toks)}")
        for t in toks:
            if t == "_":
                grid.append(0)
            elif t in tuple("123456789"):
                grid.append(int(t))
            else:
                raise ValueError(f"row {k}: invalid entry {t!r} (use digits 1-9 or _ for blank)")
    return grid


def grid_to_list_text(grid) -> str:
    rows = ["    [" + ", ".join(str(d) if d else "_" for d in grid[9 * r:9 * r + 9]) + "],"
            for r in range(9)]
    return "board = [\n" + "\n".join(rows) + "\n]"


def sym_params(rng: random.Random):
    perm = list(range(1, 10))
    rng.shuffle(perm)
    return perm, rng.randrange(4), rng.random() < 0.5


def apply_sym(grid, params) -> list[int]:
    """Apply the same diagonal-preserving symmetry (digit relabel, rotation, mirror)."""
    perm, rot, mirror = params
    mp = [0] + perm
    g = [mp[x] for x in grid]
    for _ in range(rot):
        g = [g[9 * (8 - c) + r] for r in range(9) for c in range(9)]
    if mirror:
        g = [g[9 * r + 8 - c] for r in range(9) for c in range(9)]
    return g

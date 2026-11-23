"""Propositional encoding of Sudoku (variable x[r,c,d] <=> cell (r,c) holds digit d+1)."""
from __future__ import annotations

from .sat import CDCLSolver


def var(r: int, c: int, d: int) -> int:
    """DIMACS variable for 'cell (r,c) has digit d' with d in 1..9."""
    return 81 * r + 9 * c + (d - 1) + 1


def decode(v: int) -> tuple[int, int, int]:
    v -= 1
    return v // 81, (v // 9) % 9, v % 9 + 1


from .config import units as _units_fn


def _units():
    return _units_fn()


def base_clauses(extended: bool = True) -> list[list[int]]:
    """CNF of the Sudoku rules. ``extended`` adds the redundant 'every digit appears in
    every unit' clauses, which strengthen unit propagation (hidden singles)."""
    cl: list[list[int]] = []
    for r in range(9):
        for c in range(9):
            cl.append([var(r, c, d) for d in range(1, 10)])            # >=1 digit
            for d1 in range(1, 10):
                for d2 in range(d1 + 1, 10):
                    cl.append([-var(r, c, d1), -var(r, c, d2)])        # <=1 digit
    for unit in _units():
        for d in range(1, 10):
            if extended:
                cl.append([var(r, c, d) for r, c in unit])             # digit appears
            for i in range(9):
                for j in range(i + 1, 9):
                    (r1, c1), (r2, c2) = unit[i], unit[j]
                    cl.append([-var(r1, c1, d), -var(r2, c2, d)])      # at most once
    return cl


def build_solver(extended: bool = True, **kw) -> CDCLSolver:
    s = CDCLSolver(729, **kw)
    for c in base_clauses(extended):
        s.add_clause(c)
    return s


def unit_masks(grid) -> list[int] | None:
    """One int per unit (9 rows, 9 cols, 9 boxes, +2 diagonals); bit d-1 set <=> digit d present.
    Start at 0, traverse the grid once, OR in each given's bit.  None if a unit has a duplicate."""
    from .csp import CELL_UNITS, UNITS
    um = [0] * len(UNITS)
    for i, d in enumerate(grid):
        if d:
            bit = 1 << (d - 1)
            for k in CELL_UNITS[i]:
                if um[k] & bit:
                    return None
                um[k] |= bit
    return um


def puzzle_units(grid) -> list[int] | None:
    """Puzzle-specific unit clauses from the bit-masks: for every blank cell, each digit already
    present in one of its units is ruled out (-x).  Returns None if the givens already clash."""
    from .csp import CELL_UNITS
    um = unit_masks(grid)
    if um is None:
        return None
    out = []
    for i in range(81):
        if grid[i]:
            continue
        seen = 0
        for k in CELL_UNITS[i]:
            seen |= um[k]
        out += [-var(i // 9, i % 9, d + 1) for d in range(9) if seen >> d & 1]
    return out


def givens_to_assumptions(grid) -> list[int]:
    return [var(i // 9, i % 9, d) for i, d in enumerate(grid) if d]


def model_to_grid(model) -> list[int]:
    g = [0] * 81
    for r in range(9):
        for c in range(9):
            for d in range(1, 10):
                if model[var(r, c, d)] == 1:
                    g[9 * r + c] = d
    return g


def solve_grid(grid, solver: CDCLSolver | None = None, check_unique: bool = False,
               extended: bool = True, **kw):
    """Solve a puzzle (81 ints, 0 = blank).  Returns (solution|None, unique|None, solver)."""
    s = solver or build_solver(extended, **kw)
    ass = givens_to_assumptions(grid)
    if not s.solve(ass):
        return None, None, s
    sol = model_to_grid(s.model)
    uniq = None
    if check_unique:
        tv = [v for v in range(1, 730) if s.model[v] == 1]
        uniq = s.is_unique_given(ass, tv)
    return sol, uniq, s


def is_valid_solution(grid) -> bool:
    if len(grid) != 81 or any(d < 1 or d > 9 for d in grid):
        return False
    for unit in _units():
        if sorted(grid[9 * r + c] for r, c in unit) != list(range(1, 10)):
            return False
    return True

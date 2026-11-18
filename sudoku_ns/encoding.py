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

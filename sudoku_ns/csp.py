"""Classical CSP baseline: bit-mask backtracking with MRV (minimum remaining values)
and naked-single propagation.  Also used for fast puzzle generation / solution counting.
Honors the rule variant (standard / diagonal) from ``config``."""
from __future__ import annotations

import random

from .config import units

UNITS = [[9 * r + c for r, c in u] for u in units()]
CELL_UNITS = [[k for k, u in enumerate(UNITS) if i in u] for i in range(81)]
FULL = 0x1FF


class CSPSolver:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng
        self.nodes = 0

    def _setup(self, grid):
        self.g = list(grid)
        self.um = [0] * len(UNITS)
        for i, d in enumerate(grid):
            if d:
                b = 1 << (d - 1)
                if any(self.um[k] & b for k in CELL_UNITS[i]):
                    return False
                for k in CELL_UNITS[i]:
                    self.um[k] |= b
        return True

    def solve(self, grid, limit: int = 1, randomize: bool = False):
        """Return list of up to ``limit`` solutions."""
        self.nodes = 0
        self.sols: list[list[int]] = []
        self.limit = limit
        self.randomize = randomize
        if self._setup(grid):
            self._search()
        return self.sols

    def count(self, grid, limit: int = 2) -> int:
        return len(self.solve(grid, limit))

    def _cand(self, i):
        m = 0
        for k in CELL_UNITS[i]:
            m |= self.um[k]
        return FULL & ~m

    def _search(self):
        if len(self.sols) >= self.limit:
            return
        best, best_mask, best_n = -1, 0, 10
        g = self.g
        for i in range(81):
            if g[i] == 0:
                m = self._cand(i)
                n = bin(m).count("1")
                if n < best_n:
                    best, best_mask, best_n = i, m, n
                    if n <= 1:
                        break
        if best < 0:
            self.sols.append(list(g))
            return
        if best_n == 0:
            return
        digits = [d for d in range(9) if best_mask >> d & 1]
        if self.randomize and self.rng is not None:
            self.rng.shuffle(digits)
        for d in digits:
            self.nodes += 1
            bit = 1 << d
            g[best] = d + 1
            for k in CELL_UNITS[best]:
                self.um[k] |= bit
            self._search()
            for k in CELL_UNITS[best]:
                self.um[k] &= ~bit
            g[best] = 0
            if len(self.sols) >= self.limit:
                return

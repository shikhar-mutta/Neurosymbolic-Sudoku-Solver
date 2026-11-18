"""Neurosymbolic solving: neural proposals + CDCL verification with core-guided repair.

Modes
-----
plain   : CDCL only (baseline)
guided  : the network's probabilities set CDCL's initial branching order and polarity
hints   : the most confident predictions are *assumed* as hints; if they contradict the rules
          the solver returns an UNSAT core, the wrong hints in the core are dropped and the
          solver retries (core-guided repair of the network's output)
both    : hints + guided

The final answer is always a model of the Sudoku CNF under the givens, so it is correct
regardless of how bad the network is; the network only changes *how fast* it is found.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .encoding import build_solver, decode, model_to_grid, var


@dataclass
class Result:
    solution: list[int] | None
    unique: bool | None = None
    mode: str = "plain"
    decisions: int = 0
    conflicts: int = 0
    rounds: int = 0                 # solver calls in the repair loop
    hints_used: int = 0
    hints_dropped: int = 0
    time: float = 0.0
    hint_cells: list[int] = field(default_factory=list)


def _solve_core(grid, P, mode, hint_conf, check_unique):
    t0 = time.time()
    s = build_solver(True)
    givens = [var(i // 9, i % 9, d) for i, d in enumerate(grid) if d]
    blanks = [i for i in range(81) if not grid[i]]
    if mode in ("guided", "both") and P is not None:
        ph, act = {}, {}
        for i in blanks:
            top = int(P[i].argmax())
            for d in range(9):
                v = var(i // 9, i % 9, d + 1)
                ph[v] = 1 if d == top else -1
                act[v] = float(P[i, d])
        s.set_guidance(ph, act)
    hints: list[tuple[int, int]] = []          # (cell, digit) sorted by confidence
    if mode in ("hints", "both") and P is not None:
        order = sorted(blanks, key=lambda i: -P[i].max())
        hints = [(i, int(P[i].argmax()) + 1) for i in order if P[i].max() >= hint_conf]
    n_hints = len(hints)
    rounds = 0
    while True:
        rounds += 1
        ass = givens + [var(i // 9, i % 9, d) for i, d in hints]
        ok = s.solve(ass)
        if ok:
            sol = model_to_grid(s.model)
            uniq = None
            if check_unique:
                tv = [v for v in range(1, 730) if s.model[v] == 1]
                uniq = s.is_unique_given(givens, tv)
            return Result(sol, uniq, mode, s.stats.decisions, s.stats.conflicts, rounds,
                          n_hints, n_hints - len(hints), time.time() - t0, [i for i, _ in hints])
        bad = {decode(l)[0] * 9 + decode(l)[1] for l in s.core}
        kept = [(i, d) for i, d in hints if i not in bad]
        if len(kept) == len(hints):            # no hint in the core: givens themselves contradict
            return Result(None, None, mode, s.stats.decisions, s.stats.conflicts, rounds,
                          n_hints, n_hints - len(hints), time.time() - t0)
        hints = kept


class NeuroSymbolicSolver:
    def __init__(self, guide=None, hint_conf: float = 0.9):
        self.guide = guide
        self.hint_conf = hint_conf

    def solve(self, grid, mode: str = "both", check_unique: bool = False) -> Result:
        P = self.guide.predict(grid) if (self.guide is not None and mode != "plain") else None
        return _solve_core(list(grid), P, mode if P is not None else "plain", self.hint_conf, check_unique)

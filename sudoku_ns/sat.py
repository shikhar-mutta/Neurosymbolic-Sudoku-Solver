"""A from-scratch CDCL SAT solver (MiniSat style) with assumptions and unsat cores.

Features
--------
* two-watched-literal unit propagation
* first-UIP conflict analysis with clause learning and non-chronological backjumping
* VSIDS branching (lazy heap), phase saving, Luby restarts
* solving under assumptions + extraction of a failed-assumption core
  (``analyze_final``); the core is what drives the neurosymbolic repair loop.

Literals are non-zero integers (DIMACS convention): variable ``v`` is ``v`` and
its negation is ``-v``.
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field


def luby(i: int) -> int:
    """i-th element (1-based) of the Luby restart sequence."""
    k = 1
    while (1 << k) - 1 < i:
        k += 1
    while True:
        if i == (1 << k) - 1:
            return 1 << (k - 1)
        i -= (1 << (k - 1)) - 1
        k = 1
        while (1 << k) - 1 < i:
            k += 1


@dataclass
class Stats:
    decisions: int = 0
    conflicts: int = 0
    propagations: int = 0
    restarts: int = 0
    learned: int = 0

    def as_dict(self):
        return dict(self.__dict__)


class CDCLSolver:
    def __init__(self, n_vars: int = 0, *, use_vsids: bool = True,
                 use_restarts: bool = True, restart_base: int = 64,
                 var_decay: float = 0.95, seed: int | None = None):
        self.n = 0
        self.clauses: list[list[int]] = []
        self.watches: list[list[int]] = [[], []]
        self.assigns = [0]          # 0 unassigned, 1 true, -1 false
        self.level = [0]
        self.reason: list[int | None] = [None]
        self.activity = [0.0]
        self.phase = [-1]
        self.trail: list[int] = []
        self.trail_lim: list[int] = []
        self.qhead = 0
        self.var_inc = 1.0
        self.var_decay = var_decay
        self.heap: list[tuple[float, int]] = []
        self.ok = True
        self.use_vsids = use_vsids
        self.use_restarts = use_restarts
        self.restart_base = restart_base
        self.rng = random.Random(seed)
        self.stats = Stats()
        self.model: list[int] | None = None
        self.core: list[int] = []
        for _ in range(n_vars):
            self.new_var()

    # ------------------------------------------------------------------ vars
    def new_var(self) -> int:
        self.n += 1
        self.assigns.append(0)
        self.level.append(0)
        self.reason.append(None)
        self.activity.append(0.0 if self.use_vsids else self.rng.random() * 1e-3)
        self.phase.append(-1)
        self.watches.append([])
        self.watches.append([])
        heapq.heappush(self.heap, (-self.activity[self.n], self.n))
        return self.n

    @staticmethod
    def _idx(lit: int) -> int:
        return (lit << 1) if lit > 0 else ((-lit) << 1) | 1

    def value(self, lit: int) -> int:
        a = self.assigns[lit if lit > 0 else -lit]
        return a if lit > 0 else -a

    def decision_level(self) -> int:
        return len(self.trail_lim)

    # --------------------------------------------------------------- clauses
    def add_clause(self, lits) -> bool:
        """Add a clause at decision level 0. Returns False if formula became UNSAT."""
        if not self.ok:
            return False
        self._cancel_until(0)
        seen = set()
        out = []
        for l in lits:
            if -l in seen:
                return True            # tautology
            if l in seen:
                continue
            v = self.value(l)
            if v == 1:
                return True            # already satisfied at level 0
            if v == -1:
                continue               # false at level 0: drop
            seen.add(l)
            out.append(l)
        if not out:
            self.ok = False
            return False
        if len(out) == 1:
            self._enqueue(out[0], None)
            if self._propagate() is not None:
                self.ok = False
                return False
            return True
        self._attach(out)
        return True

    def _attach(self, lits: list[int]) -> int:
        ci = len(self.clauses)
        self.clauses.append(lits)
        self.watches[self._idx(lits[0])].append(ci)
        self.watches[self._idx(lits[1])].append(ci)
        return ci

    # ----------------------------------------------------------- assignment
    def _enqueue(self, lit: int, reason: int | None) -> None:
        v = lit if lit > 0 else -lit
        self.assigns[v] = 1 if lit > 0 else -1
        self.level[v] = len(self.trail_lim)
        self.reason[v] = reason
        self.trail.append(lit)

    def _cancel_until(self, lvl: int) -> None:
        if len(self.trail_lim) <= lvl:
            return
        start = self.trail_lim[lvl]
        for i in range(len(self.trail) - 1, start - 1, -1):
            lit = self.trail[i]
            v = lit if lit > 0 else -lit
            self.phase[v] = 1 if lit > 0 else -1
            self.assigns[v] = 0
            self.reason[v] = None
            heapq.heappush(self.heap, (-self.activity[v], v))
        del self.trail[start:]
        del self.trail_lim[lvl:]
        self.qhead = len(self.trail)

    # ------------------------------------------------------------ propagate
    def _propagate(self) -> int | None:
        clauses, watches, assigns = self.clauses, self.watches, self.assigns
        while self.qhead < len(self.trail):
            p = self.trail[self.qhead]
            self.qhead += 1
            self.stats.propagations += 1
            fl = -p
            ws = watches[(fl << 1) if fl > 0 else ((-fl) << 1) | 1]
            i = j = 0
            n = len(ws)
            while i < n:
                ci = ws[i]
                i += 1
                c = clauses[ci]
                if c[0] == fl:
                    c[0], c[1] = c[1], c[0]
                first = c[0]
                a = assigns[first if first > 0 else -first]
                if (a if first > 0 else -a) == 1:
                    ws[j] = ci
                    j += 1
                    continue
                found = False
                for k in range(2, len(c)):
                    lk = c[k]
                    a = assigns[lk if lk > 0 else -lk]
                    if (a if lk > 0 else -a) != -1:
                        c[1], c[k] = lk, c[1]
                        watches[(lk << 1) if lk > 0 else ((-lk) << 1) | 1].append(ci)
                        found = True
                        break
                if found:
                    continue
                ws[j] = ci
                j += 1
                a = assigns[first if first > 0 else -first]
                if (a if first > 0 else -a) == -1:      # conflict
                    while i < n:
                        ws[j] = ws[i]
                        j += 1
                        i += 1
                    del ws[j:]
                    self.qhead = len(self.trail)
                    return ci
                self._enqueue(first, ci)
            del ws[j:]
        return None

    # -------------------------------------------------------------- analyze
    def _bump(self, v: int) -> None:
        if not self.use_vsids:
            return
        self.activity[v] += self.var_inc
        if self.activity[v] > 1e100:
            for u in range(1, self.n + 1):
                self.activity[u] *= 1e-100
            self.var_inc *= 1e-100
            self.heap = [(-self.activity[u], u) for u in range(1, self.n + 1)
                         if self.assigns[u] == 0]
            heapq.heapify(self.heap)
            return
        if self.assigns[v] == 0:
            heapq.heappush(self.heap, (-self.activity[v], v))

    def _analyze(self, confl: int) -> tuple[list[int], int]:
        seen = [False] * (self.n + 1)
        learnt = [0]
        counter = 0
        p = 0
        idx = len(self.trail) - 1
        dl = self.decision_level()
        while True:
            c = self.clauses[confl]
            for q in (c if p == 0 else c[1:]):
                v = q if q > 0 else -q
                if not seen[v] and self.level[v] > 0:
                    self._bump(v)
                    seen[v] = True
                    if self.level[v] >= dl:
                        counter += 1
                    else:
                        learnt.append(q)
            while not seen[abs(self.trail[idx])]:
                idx -= 1
            p = self.trail[idx]
            idx -= 1
            confl = self.reason[abs(p)]
            seen[abs(p)] = False
            counter -= 1
            if counter == 0:
                break
        learnt[0] = -p
        if len(learnt) == 1:
            bt = 0
        else:
            mi = max(range(1, len(learnt)), key=lambda k: self.level[abs(learnt[k])])
            learnt[1], learnt[mi] = learnt[mi], learnt[1]
            bt = self.level[abs(learnt[1])]
        self.var_inc /= self.var_decay
        return learnt, bt

    def _analyze_final(self, p: int) -> list[int]:
        """Assumption literals responsible for assumption ``p`` being falsified."""
        core = [p]
        if self.decision_level() == 0:
            return core
        seen = [False] * (self.n + 1)
        seen[abs(p)] = True
        for i in range(len(self.trail) - 1, self.trail_lim[0] - 1, -1):
            lit = self.trail[i]
            x = abs(lit)
            if seen[x]:
                if self.reason[x] is None:
                    core.append(lit)           # an assumption (decision)
                else:
                    for q in self.clauses[self.reason[x]][1:]:
                        if self.level[abs(q)] > 0:
                            seen[abs(q)] = True
                seen[x] = False
        return core

    # --------------------------------------------------------------- search
    def _pick_branch(self) -> int:
        while self.heap:
            neg_act, v = heapq.heappop(self.heap)
            if self.assigns[v] == 0 and -neg_act == self.activity[v]:
                return v if self.phase[v] > 0 else -v
        for v in range(1, self.n + 1):          # fallback safety net
            if self.assigns[v] == 0:
                return v if self.phase[v] > 0 else -v
        return 0

    def solve(self, assumptions=(), max_conflicts: int | None = None) -> bool | None:
        """Return True (SAT, see ``self.model``), False (UNSAT, see ``self.core``)
        or None when ``max_conflicts`` was exhausted."""
        self.model = None
        self.core = []
        if not self.ok:
            return False
        assumptions = list(assumptions)
        self._cancel_until(0)
        restart_no = 1
        budget = luby(restart_no) * self.restart_base
        conflicts_here = 0
        start_conflicts = self.stats.conflicts
        while True:
            confl = self._propagate()
            if confl is not None:
                self.stats.conflicts += 1
                conflicts_here += 1
                if self.decision_level() == 0:
                    self.ok = False
                    return False
                learnt, bt = self._analyze(confl)
                self._cancel_until(bt)
                if len(learnt) == 1:
                    self._enqueue(learnt[0], None)
                else:
                    ci = self._attach(learnt)
                    self.stats.learned += 1
                    self._enqueue(learnt[0], ci)
                if (max_conflicts is not None
                        and self.stats.conflicts - start_conflicts >= max_conflicts):
                    self._cancel_until(0)
                    return None
                continue
            if self.use_restarts and conflicts_here >= budget:
                self.stats.restarts += 1
                restart_no += 1
                budget = luby(restart_no) * self.restart_base
                conflicts_here = 0
                self._cancel_until(0)
                continue
            nxt = 0
            while self.decision_level() < len(assumptions):
                a = assumptions[self.decision_level()]
                val = self.value(a)
                if val == 1:
                    self.trail_lim.append(len(self.trail))   # dummy level
                elif val == -1:
                    self.core = self._analyze_final(a)
                    self._cancel_until(0)
                    return False
                else:
                    nxt = a
                    break
            if nxt == 0:
                nxt = self._pick_branch()
                if nxt == 0:
                    self.model = [0] + [1 if self.assigns[v] == 1 else -1
                                        for v in range(1, self.n + 1)]
                    self._cancel_until(0)
                    return True
                self.stats.decisions += 1
            self.trail_lim.append(len(self.trail))
            self._enqueue(nxt, None)

    def set_guidance(self, phases: dict[int, int] | None = None,
                     activities: dict[int, float] | None = None) -> None:
        """Heuristic hints from an external (neural) model: initial polarity per variable
        (+1/-1) and initial VSIDS activity.  Affects search order only, never soundness."""
        self._cancel_until(0)
        for v, sgn in (phases or {}).items():
            self.phase[v] = 1 if sgn > 0 else -1
        for v, a in (activities or {}).items():
            self.activity[v] = float(a)
        self.heap = [(-self.activity[v], v) for v in range(1, self.n + 1) if self.assigns[v] == 0]
        heapq.heapify(self.heap)

    # ------------------------------------------------------------- helpers
    def true_vars(self) -> list[int]:
        assert self.model is not None
        return [v for v in range(1, self.n + 1) if self.model[v] == 1]

    def is_unique_given(self, assumptions, model_true_vars) -> bool:
        """After a SAT answer: is the solution (restricted to ``model_true_vars``)
        the only one under ``assumptions``?  Uses a selector literal so the
        blocking clause stays inert for later calls."""
        sel = self.new_var()
        self.add_clause([-sel] + [-v for v in model_true_vars])
        saved = self.model
        res = self.solve(list(assumptions) + [sel])
        self.model = saved
        self.add_clause([-sel])         # permanently disable the blocking clause
        return res is False

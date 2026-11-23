"""Human-readable explanations of conflicts met by the SAT solver.

A *conflict* is a clause (a rule) that is violated under the current partial assignment: the
search cannot move further on that branch.  We translate the violated clause, the chain of
implications that led to it, and the learned clause into plain Sudoku language.
"""
from __future__ import annotations

from . import config
from .csp import UNITS
from .encoding import build_solver, decode, var

_NAMES: list[str] = []
for _i in range(9):
    _NAMES += [f"row {_i + 1}", f"column {_i + 1}"]
_NAMES += [f"box {_k + 1}" for _k in range(9)]
if config.DIAGONAL:
    _NAMES += ["the main diagonal", "the anti-diagonal"]


def cell(r, c):
    return f"r{r + 1}c{c + 1}"


def lit_text(l: int) -> str:
    r, c, d = decode(abs(l))
    return f"{cell(r, c)} {'=' if l > 0 else '≠'} {d}"


def _cells(clause):
    return {decode(abs(l))[:2] for l in clause}


def _common_units(cells) -> list[str]:
    idx = {9 * r + c for r, c in cells}
    return [_NAMES[k] for k, u in enumerate(UNITS) if idx <= set(u)]


def describe_violation(clause) -> str:
    cs = _cells(clause)
    if all(l > 0 for l in clause):
        if len(cs) == 1:
            (r, c), = cs
            return f"cell {cell(r, c)} has no candidate digit left (all digits 1-9 are excluded)"
        d = decode(clause[0])[2]
        u = _common_units(cs)
        return f"digit {d} has no place left in {u[0] if u else 'a unit'}"
    if all(l < 0 for l in clause) and len(clause) == 2:
        (r1, c1, d1), (r2, c2, d2) = decode(abs(clause[0])), decode(abs(clause[1]))
        if (r1, c1) == (r2, c2):
            return f"cell {cell(r1, c1)} would hold two digits ({d1} and {d2})"
        u = _common_units(cs)
        return f"digit {d1} would appear twice in {u[0] if u else 'a unit'} (at {cell(r1, c1)} and {cell(r2, c2)})"
    return "a learned constraint is violated"


def describe_reason(t: int, clause) -> str:
    facts = [-o for o in clause if o != t]
    if t < 0 and len(clause) == 2 and facts and facts[0] > 0:
        r1, c1, d1 = decode(abs(t)); r2, c2, d2 = decode(facts[0])
        if (r1, c1) == (r2, c2):
            return f"{lit_text(t)}   because {cell(r2, c2)} = {d2} (a cell holds one digit)"
        u = _common_units({(r1, c1), (r2, c2)})
        return f"{lit_text(t)}   because {cell(r2, c2)} = {d2} is in {u[0]}" if u else f"{lit_text(t)}   because of a learned rule"
    if t > 0 and all(f < 0 for f in facts):
        cs = _cells(clause)
        r, c, d = decode(t)
        if len(cs) == 1:
            return f"{lit_text(t)}   because every other digit is excluded in {cell(r, c)}"
        u = _common_units(cs)
        if not u:
            return f"{lit_text(t)}   because of a learned rule"
        return f"{lit_text(t)}   because it is the only cell in {u[0]} that can still hold {d}"
    return f"{lit_text(t)}   because of a learned rule"


class Explainer:
    """Solver hook: records a readable description of the first ``limit`` conflicts."""

    def __init__(self, limit: int = 2):
        self.limit = limit
        self.events: list[dict] = []

    def on_conflict(self, s, confl, learnt, bt):
        if len(self.events) >= self.limit:
            return
        clause = list(s.clauses[confl])
        decisions = [l for l in s.trail if s.level[abs(l)] > 0 and s.reason[abs(l)] is None]
        pos = {l: i for i, l in enumerate(s.trail)}
        seen, steps, known = set(), [], []

        def visit(t):
            if t in seen:
                return
            seen.add(t)
            v = abs(t)
            if s.level[v] == 0:
                known.append(t)
            elif s.reason[v] is None:
                steps.append((pos[t], f"{lit_text(t)}   (guess / decision)"))
            else:
                rc = s.clauses[s.reason[v]]
                for o in rc:
                    if o != t:
                        visit(-o)
                steps.append((pos[t], describe_reason(t, rc)))

        for l in clause:
            visit(-l)
        steps.sort()
        self.events.append(dict(
            violation=describe_violation(clause),
            decisions=[lit_text(d) for d in decisions],
            chain=[text for _, text in steps],
            known=[lit_text(k) for k in sorted(known, key=lambda x: pos[x])],
            learned=[lit_text(-l) for l in learnt],
            level=s.decision_level(), backjump=bt))


def _solver_with_givens(grid):
    s = build_solver(True)
    for i, d in enumerate(grid):
        if d:
            s.add_clause([var(i // 9, i % 9, d)])
    return s


def _chain_lines(chain, keep: int = 12) -> list[str]:
    """Show only the last ``keep`` steps (those closest to the conflict)."""
    first = max(0, len(chain) - keep)
    out = [f"    ... ({first} earlier steps omitted)"] if first else []
    return out + [f"    {n}. {t}" for n, t in enumerate(chain[first:], first + 1)]


def _format_event(k, ev) -> str:
    out = [f"Conflict #{k}: the search cannot continue on this branch"]
    if ev["decisions"]:
        out.append("  Guesses made on this branch: " + ", ".join(ev["decisions"][-6:])
                   + (f"  (+{len(ev['decisions']) - 6} earlier)" if len(ev["decisions"]) > 6 else ""))
    out.append(f"  UNSATISFIED CONDITION: {ev['violation']}.")
    out.append("  Why (chain of forced consequences, in order):")
    out += _chain_lines(ev["chain"])
    if ev["known"]:
        out.append(f"    (facts already known from the puzzle: {', '.join(ev['known'][:8])}"
                   + (", ..." if len(ev["known"]) > 8 else "") + ")")
    out.append("  Learned: these cannot all be true together -> " + "; ".join(ev["learned"]))
    out.append(f"  Backjump from level {ev['level']} to level {ev['backjump']} and try another option.")
    return "\n".join(out)


def explain_run(grid, solution=None, limit: int = 2) -> str:
    """Explain conflicts met while solving ``grid``.  If the search never conflicts,
    demonstrate a dead end with a 'what if' wrong guess."""
    s = _solver_with_givens(grid)
    ex = Explainer(limit)
    s.hook = ex
    s.solve()
    lines = ["", "=" * 70, "WHY THE SOLVER CANNOT ALWAYS MOVE FORWARD (unsatisfied conditions)", "=" * 70]
    if ex.events:
        lines.append(f"The search hit {s.stats.conflicts} conflict(s); the first {len(ex.events)} are explained.\n")
        lines += [_format_event(k, ev) + "\n" for k, ev in enumerate(ex.events, 1)]
        return "\n".join(lines)
    lines.append("This puzzle was solved by pure deduction (0 conflicts), so no real dead end occurred.\n"
                 "Here is a 'what if' wrong guess and the rule it violates:\n")
    if solution is None:
        s2 = _solver_with_givens(grid); s2.solve(); solution = [0] * 81
        if s2.model:
            from .encoding import model_to_grid
            solution = model_to_grid(s2.model)
    import random
    relaxed, hidden = list(grid), 0
    order = [i for i in range(81) if grid[i]]
    random.Random(0).shuffle(order)

    def find_demo(g):
        """Wrong guess refuted purely by propagation (conflict at decision level 1)."""
        base = _solver_with_givens(g)
        cands = [(i, d) for i in range(81) if not g[i] for d in range(1, 10)
                 if d != solution[i] and base.value(var(i // 9, i % 9, d)) != -1]
        best = None
        for i, d in cands[:40]:
            s3 = _solver_with_givens(g); h = Explainer(1); s3.hook = h
            if s3.solve([var(i // 9, i % 9, d)]) is False and h.events and h.events[0]["level"] == 1:
                n = len(h.events[0]["chain"])
                score = (0 if n >= 3 else 1, n if n >= 3 else -n)
                if best is None or score < best[0]:
                    best = (score, i, d, h.events[0])
        return best

    best = find_demo(relaxed)
    while best is None and order:               # hide clues until a wrong guess is still possible
        for _ in range(2):
            if order:
                relaxed[order.pop()] = 0; hidden += 1
        best = find_demo(relaxed)
    if hidden:
        lines.append(f"(Deduction alone is so strong here that no wrong guess survives, so {hidden} clue(s) are\n"
                     "hidden temporarily for this illustration.)\n")
    grid = relaxed
    if best is None:
        lines.append("(no wrong guess could be illustrated for this puzzle.)")
        return "\n".join(lines)
    _, i, d, ev = best
    lines.append(f"What if {cell(i // 9, i % 9)} = {d}?   (the true value is {solution[i]})")
    lines.append(f"  UNSATISFIED CONDITION: {ev['violation']}.")
    lines.append("  Why (chain of forced consequences, in order):")
    lines += _chain_lines(ev["chain"])
    if ev["known"]:
        lines.append(f"    (also already known from the puzzle: {', '.join(ev['known'][:8])}"
                     + (", ..." if len(ev["known"]) > 8 else "") + ")")
    lines.append(f"  Therefore {cell(i // 9, i % 9)} ≠ {d}: that branch is a dead end and is discarded.")
    return "\n".join(lines)

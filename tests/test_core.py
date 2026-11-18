import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sudoku_ns import encoding as E, generator as G
from sudoku_ns.csp import CSPSolver
from sudoku_ns.neurosym import NeuroSymbolicSolver
from sudoku_ns.sat import CDCLSolver



def test_cdcl_basic():
    s = CDCLSolver(3)
    s.add_clause([1, 2]); s.add_clause([-1, 3]); s.add_clause([-2, 3]); s.add_clause([-3])
    assert s.solve() is False
    s = CDCLSolver(3)
    s.add_clause([1, 2]); s.add_clause([-1, 3])
    assert s.solve() is True


def test_assumption_core():
    s = CDCLSolver(3)
    s.add_clause([-1, -2])
    assert s.solve([1, 2, 3]) is False
    assert set(abs(l) for l in s.core) == {1, 2}
    assert s.solve([1, 3]) is True


def test_generated_puzzles_unique_and_solved():
    rng = random.Random(5)
    for _ in range(5):
        p, sol = G.random_puzzle(rng)
        assert CSPSolver().count(p, 2) == 1
        got, uniq, _ = E.solve_grid(p, check_unique=True)
        assert got == sol and uniq and E.is_valid_solution(got)


def test_solution_respects_diagonals():
    rng = random.Random(11)
    for _ in range(5):
        p, sol = G.random_puzzle(rng)
        assert sorted(sol[9 * i + i] for i in range(9)) == list(range(1, 10))
        assert sorted(sol[9 * i + 8 - i] for i in range(9)) == list(range(1, 10))


def test_standard_grid_with_repeated_diagonal_is_rejected():
    g = [((3 * (r % 3) + r // 3 + c) % 9) + 1 for r in range(9) for c in range(9)]
    diag = [g[10 * i] for i in range(9)]
    assert len(set(diag)) < 9          # valid standard Sudoku, bad diagonal
    assert not E.is_valid_solution(g)


def test_diagonal_conflict_is_unsat():
    p = [0] * 81
    p[0] = p[80] = 5                    # same digit twice on the main diagonal
    assert E.solve_grid(p)[0] is None


def test_unsat_puzzle():
    p = [0] * 81
    p[0] = p[1] = 5
    assert E.solve_grid(p)[0] is None


def test_hints_with_wrong_prediction_are_repaired():
    from sudoku_ns.neurosym import _solve_core
    rng = random.Random(9)
    p, sol = G.random_puzzle(rng, 28)
    P = np.full((81, 9), 0.01)
    blanks = [i for i in range(81) if not p[i]]
    for i in blanks:
        P[i, sol[i] - 1] = 0.95
    wrong = blanks[0]                          # confidently wrong prediction for one cell
    P[wrong, :] = 0.01
    P[wrong, sol[wrong] % 9] = 0.99
    r = _solve_core(p, P, "hints", 0.9, True)
    assert r.solution == sol and r.unique
    assert r.hints_dropped >= 1 or r.rounds >= 1


def test_guided_and_plain_agree():
    from sudoku_ns.neurosym import _solve_core
    p, sol = G.random_puzzle(random.Random(4), 26)
    P = np.random.default_rng(0).random((81, 9)) ** 3          # garbage "network"
    for mode in ("plain", "guided", "hints", "both"):
        assert _solve_core(p, P, mode, 0.9, False).solution == sol


def test_parser_strict():
    good = "board = [\n" + "".join("    [" + ", ".join(["_"] * 9) + "],\n" for _ in range(9)) + "]"
    assert G.parse_board_list(good) == [0] * 81
    for bad in (good.replace("_", "0", 1), good.replace("_", ".", 1), "5" * 81, good.replace("[_", "[10", 1)):
        try:
            G.parse_board_list(bad)
        except ValueError:
            continue
        raise AssertionError("bad input accepted")


def test_symmetry_preserves_validity():
    rng = random.Random(2)
    p, sol = G.random_puzzle(rng)
    t = G.symmetry_transform(sol, rng)
    assert E.is_valid_solution(t)

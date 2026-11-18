"""Neural module: an MLP that predicts, for every empty cell, a distribution over digits.

Input  : the puzzle one-hot (81x10) + the *symbolic* candidate mask (81x9) obtained from
         row/column/box/diagonal elimination (so symbolic knowledge is a neural input too).
Output : P[i, d]  = p(cell i holds digit d+1 in the solution), masked by the candidates.
The prediction is only a *heuristic proposal*; the SAT solver decides what is true.
"""
from __future__ import annotations

import pickle
import random

import numpy as np
from sklearn.neural_network import MLPRegressor

from . import generator as G
from .csp import CELL_UNITS, FULL, UNITS


def candidates(grid) -> np.ndarray:
    """(81, 9) 0/1 mask: digits not yet used by any peer (given cells: only their digit)."""
    used = [0] * len(UNITS)
    for i, d in enumerate(grid):
        if d:
            for k in CELL_UNITS[i]:
                used[k] |= 1 << (d - 1)
    out = np.zeros((81, 9), dtype=np.float32)
    for i in range(81):
        if grid[i]:
            out[i, grid[i] - 1] = 1
        else:
            m = FULL
            for k in CELL_UNITS[i]:
                m &= ~used[k]
            for d in range(9):
                out[i, d] = (m >> d) & 1
    return out


def features(grid) -> np.ndarray:
    oh = np.zeros((81, 10), dtype=np.float32)
    oh[np.arange(81), grid] = 1
    return np.concatenate([oh.ravel(), candidates(grid).ravel()])


def make_dataset(n_base: int, aug: int, seed: int):
    rng = random.Random(seed)
    X, Y = [], []
    for _ in range(n_base):
        puz, sol = G.random_puzzle(rng, clues=rng.randint(22, 34))
        for a in range(aug):
            prm = G.sym_params(rng) if a else ([*range(1, 10)], 0, False)
            p2, s2 = G.apply_sym(puz, prm), G.apply_sym(sol, prm)
            X.append(features(p2))
            y = np.zeros((81, 9), dtype=np.float32)
            y[np.arange(81), np.array(s2) - 1] = 1
            Y.append(y.ravel())
    return np.array(X), np.array(Y)


class NeuralGuide:
    def __init__(self, model: MLPRegressor):
        self.model = model

    @classmethod
    def train(cls, n_base=1200, aug=16, seed=0, verbose=True):
        X, Y = make_dataset(n_base, aug, seed)
        if verbose:
            print(f"training samples: {len(X)} ({n_base} base puzzles x {aug} symmetries)", flush=True)
        m = MLPRegressor(hidden_layer_sizes=(768, 768), batch_size=256, max_iter=40,
                         early_stopping=True, n_iter_no_change=4, random_state=seed)
        m.fit(X, Y)
        if verbose:
            print(f"epochs: {m.n_iter_}, best val R2: {m.best_validation_score_:.4f}", flush=True)
        return cls(m)

    def predict(self, grid) -> np.ndarray:
        """(81, 9) probabilities, masked by candidates and normalised per cell."""
        cand = candidates(grid)
        raw = self.model.predict(features(grid)[None])[0].reshape(81, 9)
        p = np.clip(raw, 0, None) * cand + 1e-6 * cand
        return p / np.maximum(p.sum(1, keepdims=True), 1e-12)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls(pickle.load(f))

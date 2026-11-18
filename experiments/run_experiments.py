"""Reproduces all numbers/figures of the report.  Usage: python experiments/run_experiments.py [--quick]"""
import json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sudoku_ns import encoding as E, generator as G
from sudoku_ns.csp import CSPSolver
from sudoku_ns.neural import NeuralGuide
from sudoku_ns.neurosym import NeuroSymbolicSolver, _solve_core

QUICK = "--quick" in sys.argv
N = 20 if QUICK else 60
FIG = ROOT / "report" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
OUT = {}
guide = NeuralGuide.load(ROOT / "models" / "guide.pkl")

def pool(n, clues, seed):
    rng = random.Random(seed); return [G.random_puzzle(rng, clues) for _ in range(n)]
sets = {"medium (30 clues)": pool(N, 30, 1001), "hard (minimal)": pool(N, 0, 1002)}
print("puzzles ready", flush=True)

# ---------------- 1. quality of the neural proposals -------------------
tau_list = [0.0, 0.5, 0.7, 0.9, 0.95]
q = {}
for name, ps in sets.items():
    corr = {t: 0 for t in tau_list}; cnt = {t: 0 for t in tau_list}; valid = 0; cell = []
    for p, s in ps:
        P = guide.predict(p); pred = P.argmax(1) + 1
        full = [p[i] if p[i] else int(pred[i]) for i in range(81)]
        valid += E.is_valid_solution(full)
        for i in range(81):
            if p[i]: continue
            c = P[i].max(); ok = pred[i] == s[i]; cell.append(ok)
            for t in tau_list:
                if c >= t: cnt[t] += 1; corr[t] += ok
    q[name] = dict(blank_cell_acc=float(np.mean(cell)), valid_grid_rate=valid / len(ps),
                   precision={str(t): (corr[t] / cnt[t] if cnt[t] else None) for t in tau_list},
                   coverage={str(t): cnt[t] / (len(cell)) for t in tau_list})
OUT["neural"] = q; print(json.dumps(q, indent=1), flush=True)

# ---------------- 2. mode comparison --------------------------------------
modes = ["plain", "guided", "hints", "both"]
cmp_ = {}
for name, ps in sets.items():
    cmp_[name] = {}
    Ps = [guide.predict(p) for p, _ in ps]
    for m in modes:
        dec, cf, tm, drop, used, rounds, ok = [], [], [], [], [], [], 0
        for (p, s), P in zip(ps, Ps):
            r = _solve_core(p, P, m, 0.9, False)
            ok += r.solution == s; dec.append(r.decisions); cf.append(r.conflicts); tm.append(r.time)
            drop.append(r.hints_dropped); used.append(r.hints_used); rounds.append(r.rounds)
        cmp_[name][m] = dict(correct=ok / len(ps), decisions=float(np.mean(dec)), conflicts=float(np.mean(cf)),
                             conflicts_max=int(max(cf)), time_ms=1000 * float(np.mean(tm)),
                             hints=float(np.mean(used)), dropped=float(np.mean(drop)), rounds=float(np.mean(rounds)))
OUT["modes"] = cmp_; print(json.dumps(cmp_, indent=1), flush=True)

# hint-threshold sweep on hard puzzles
sw = {}
ps = sets["hard (minimal)"]; Ps = [guide.predict(p) for p, _ in ps]
for tau in [0.5, 0.7, 0.9, 0.95, 0.99]:
    cf, dr, hu = [], [], []
    for (p, s), P in zip(ps, Ps):
        r = _solve_core(p, P, "hints", tau, False); cf.append(r.conflicts); dr.append(r.hints_dropped); hu.append(r.hints_used)
    sw[str(tau)] = dict(conflicts=float(np.mean(cf)), dropped=float(np.mean(dr)), hints=float(np.mean(hu)))
OUT["hint_sweep"] = sw; print(sw, flush=True)

# ---------------- 3. CDCL vs CSP ------------------------------------------
def bench(puzzles_):
    res = {"CDCL": [], "CDCL no-VSIDS": [], "CDCL no-restart": [], "CSP-MRV": []}
    cfl = {k: [] for k in res}
    for p in puzzles_:
        for name, kw in [("CDCL", {}), ("CDCL no-VSIDS", dict(use_vsids=False)), ("CDCL no-restart", dict(use_restarts=False))]:
            t = time.time(); s = E.build_solver(True, **kw); sol, _, sv = E.solve_grid(p, solver=s)
            res[name].append(time.time() - t); cfl[name].append(sv.stats.conflicts)
        t = time.time(); c = CSPSolver(); c.solve(p); res["CSP-MRV"].append(time.time() - t); cfl["CSP-MRV"].append(c.nodes)
    return {k: (float(np.mean(res[k])), float(np.mean(cfl[k]))) for k in res}
rng = random.Random(77); nb = 15 if QUICK else 40
bm = {"easy (36)": bench([G.random_puzzle(rng, 36)[0] for _ in range(nb)]),
      "medium (30)": bench([G.random_puzzle(rng, 30)[0] for _ in range(nb)]),
      "minimal": bench([G.random_puzzle(rng, 0)[0] for _ in range(nb)])}
cands = [G.random_puzzle(rng, 0)[0] for _ in range(20 if QUICK else 60)]
def _cf(p):
    sv = E.build_solver(True); E.solve_grid(p, solver=sv); return sv.stats.conflicts
hardest = max(cands, key=_cf); OUT["hardest_puzzle"] = G.grid_to_list_text(hardest)
bm["hardest of 60"] = bench([hardest])
OUT["bench"] = bm; print(json.dumps(bm, indent=1), flush=True)

enc = {}
for ext in (False, True):
    cf = []
    for p in [G.random_puzzle(random.Random(i), 0)[0] for i in range(nb)]:
        s = E.build_solver(ext); sol, _, sv = E.solve_grid(p, solver=s); cf.append(sv.stats.conflicts)
    enc["extended" if ext else "minimal"] = float(np.mean(cf))
OUT["encoding"] = enc

json.dump(OUT, open(ROOT / "results" / "results.json", "w"), indent=1)

# ---- figures
fig, ax = plt.subplots(figsize=(6, 3.2))
labels = list(bm); names = list(next(iter(bm.values()))); w = 0.2
for j, nme in enumerate(names):
    ax.bar(np.arange(len(labels)) + j * w, [1000 * bm[l][nme][0] for l in labels], w, label=nme)
ax.set_xticks(np.arange(len(labels)) + 1.5 * w); ax.set_xticklabels(labels, fontsize=7)
ax.set_yscale("log"); ax.set_ylabel("mean time (ms)"); ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(FIG / "solver_bench.pdf"); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
for name in sets:
    xs = [float(t) for t in tau_list[1:]]
    axs[0].plot(xs, [100 * q[name]["precision"][str(t)] if q[name]["precision"][str(t)] is not None else np.nan for t in tau_list[1:]], "o-", label=name)
    axs[1].plot(xs, [100 * q[name]["coverage"][str(t)] for t in tau_list[1:]], "s-", label=name)
axs[0].set_ylabel("precision of top-1 prediction (%)"); axs[1].set_ylabel("blank cells covered (%)")
for a in axs: a.set_xlabel("confidence threshold"); a.grid(alpha=.3); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "neural_precision.pdf"); plt.close()
print("done", flush=True)

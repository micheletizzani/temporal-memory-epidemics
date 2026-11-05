# %%
"""Phase space of the shuffle ratio for exponential Hawkes node memory.

Time rescaling t -> t*mu maps (mu, n, gamma) to (1, n, gamma/mu), so the whole family of
exponential-Hawkes processes is a TWO-parameter family: the branching ratio n and the memory-to-
waiting ratio g = gamma/mu (memory rate over baseline rate). R = n_hat(shuffled)/n_hat(real)
therefore depends only on (n, g) and this sweep maps it exactly (no other nuisance parameter).
For each cell: S seeds of a process with ~EV events, single-exponential estimator as in the paper.
Also records the model-free lag-one memory coefficient M (within-node rank version of memory_stats.py; the Pearson value is kept as M_pearson), so each cell can be placed on the
(B, M) plane. Writes results/hawkes_phase_space.json.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

import kernel_phase as kp  # noqa: E402
from memory_stats import rank_memory_train  # noqa: E402
from sum_exp_kernel import multi_replay, nll_multi  # noqa: E402

OUT = os.path.join(ROOT, "results", "hawkes_phase_space.json")
NS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
GS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]          # gamma / mu
EV, SEEDS = 3000, 4
GRID = np.geomspace(1e-3, 1e3, 14)                    # estimation grid for the single rate


def n_exp(ts, T):
    ev = [(float(t), 0, 1) for t in ts]; init = [0] * len(ev)
    S, C, counts = multi_replay(ev, init, 2, GRID, T)
    best = (-1e18, 0.0)
    for r in range(len(GRID)):
        w = np.zeros(len(GRID)); w[r] = 1.0
        o = minimize(lambda x: nll_multi(1 / (1 + math.exp(-x[0])), w, S, C, counts, init, 2, T)[0],
                     [0.0], method="Nelder-Mead", options=dict(maxiter=200, xatol=1e-3, fatol=1e-3))
        if -o.fun > best[0]:
            best = (-o.fun, 1 / (1 + math.exp(-o.x[0])))
    return best[1]


def shuffle(ts, rng):
    d = np.diff(np.concatenate([[0.0], np.sort(ts)])); rng.shuffle(d)
    return np.cumsum(d)


def B_M(ts):
    d = np.diff(np.sort(ts)); d = d[d > 0]
    B = (d.std() - d.mean()) / (d.std() + d.mean())
    return float(B), float(np.corrcoef(d[:-1], d[1:])[0, 1]), rank_memory_train(ts)


if __name__ == "__main__":
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for n in NS:
        for g in GS:
            key = f"{n}|{g}"
            if key in out:
                continue
            T = EV * (1 - n)                          # mu = 1: rate 1/(1-n) -> ~EV events
            Rs, Bs, Ms, Mps = [], [], [], []
            for s in range(SEEDS):
                rng = np.random.default_rng(100 * s + int(1000 * n) + int(10 * g))
                ts = kp.hawkes_mix_sim(1.0, n, [1.0], [g], T, rng)
                if len(ts) < 500:
                    continue
                T_eff = float(ts[-1]) + 1e-9
                a = n_exp(ts, T_eff); b = n_exp(shuffle(ts, rng), T_eff)
                if a > 1e-3:
                    Rs.append(b / a)
                    B, Mp, M = B_M(ts); Bs.append(B); Ms.append(M); Mps.append(Mp)
            out[key] = dict(n=n, g=g, R=float(np.mean(Rs)), Rsd=float(np.std(Rs)), B=float(np.mean(Bs)),
                            M=float(np.mean(Ms)), M_pearson=float(np.mean(Mps)), k=len(Rs))
            print(f"n={n} g={g:>5}: R={out[key]['R']:.3f} +/- {out[key]['Rsd']:.3f}  B={out[key]['B']:+.2f} "
                  f"M={out[key]['M']:+.3f}", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)

# %%
"""Shuffle-ratio intervals for the neural records, with the protocol of scalefree_screen_hardened.

R_exp and R_mix as mean and 5-95% over B_SUB contiguous subsamples x S_SHUF shuffle seeds, for
the simulated spiking network (bursting and refractory regimes) and the cortical spike trains.
The helper functions are copied from scalefree_screen_hardened.py, whose import would rerun that
module's whole pipeline. Each record is read from its first MAX_EV events, as in Table S1.
Writes results/neural_screen.json.
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

from sum_exp_kernel import multi_replay, nll_multi, fit_mixture, load_directed  # noqa: E402

GAMMAS = np.geomspace(1e-3, 1e1, 8)
B_SUB, S_SHUF, SUB = 4, 3, 4000
MAX_EV = 60000
DATA = os.path.abspath(os.path.join(ROOT, "data"))
RESULTS = os.path.join(ROOT, "results", "neural_screen.json")
DATASETS = [  # (name, file, tcol, icol, jcol); all three are (t, i, j)
    ("SNN (bursting)",   "snn_bursty_network.txt",     0, 1, 2),
    ("SNN (refractory)", "snn_refractory_network.txt", 0, 1, 2),
    ("Cortex (Allen)",   "allen_brain_network.txt",    0, 1, 2),
]


# %%
def fit_nR(ev, N, T, gammas=GAMMAS):
    """(n_exp, ll_exp, n_mix, ll_mix). Mixture seeded at the single-exp MLE (as in the screen)."""
    init = [i for _t, i, _j in ev]; R = len(gammas)
    S, C, counts = multi_replay(ev, init, N, gammas, T)
    best = (-1e18, 0.0, 0)
    for r in range(R):
        w = np.zeros(R); w[r] = 1.0
        o = minimize(lambda x: nll_multi(1 / (1 + math.exp(-x[0])), w, S, C, counts, init, N, T)[0],
                     [0.0], method="Nelder-Mead", options=dict(maxiter=250, xatol=1e-3, fatol=1e-3))
        n = 1 / (1 + math.exp(-o.x[0]))
        if -o.fun > best[0]:
            best = (-o.fun, n, r)
    ll_exp, n_exp, r_star = best
    z0 = np.full(R, -3.0); z0[r_star] = 3.0
    x0 = np.concatenate([[math.log(n_exp / (1 - n_exp + 1e-9))], z0])
    n_mix, _w, ll_mix = fit_mixture(S, C, counts, init, N, T, R, x0=x0)
    return n_exp, ll_exp, n_mix, ll_mix


def per_node_shuffle(evs, rng):
    """Destroy per-node timing order, keep the marginal (and the (i,j) pairing)."""
    tnew = [t for t, _i, _j in evs]; idxby = {}
    for k, (t, i, j) in enumerate(evs):
        idxby.setdefault(i, []).append(k)
    for i, idx in idxby.items():
        ts = sorted(tnew[k] for k in idx)
        d = np.diff([0.0] + ts); rng.shuffle(d); cs = np.cumsum(d)
        for k, tt in zip(idx, cs):
            tnew[k] = float(tt)
    return sorted((tnew[k], evs[k][1], evs[k][2]) for k in range(len(evs)))


def screen(ev, N, tag, seed=0):
    evs = sorted(ev); E = len(evs); rng = np.random.default_rng(seed)
    Re, Rm, nes_real = [], [], []
    for b in range(B_SUB):
        s = int(rng.integers(0, E - SUB)) if E > SUB else 0
        window = evs[s:s + SUB]
        Nw = 1 + max(max(i, j) for _t, i, j in window)
        Tw = window[-1][0] - window[0][0] + 1e-9
        w0 = [(t - window[0][0], i, j) for t, i, j in window]
        ne, _lle, nm, _llm = fit_nR(w0, Nw, Tw); nes_real.append(ne)
        for s2 in range(S_SHUF):
            wsh = per_node_shuffle(list(w0), np.random.default_rng(1000 * b + s2))
            nes, _, nms, _ = fit_nR(wsh, Nw, Tw)
            Re.append(nes / max(ne, 1e-3)); Rm.append(nms / max(nm, 1e-3))
        print(f"  {tag} subsample {b}: n_exp={ne:.3f} R_exp={np.mean(Re[-S_SHUF:]):.3f} "
              f"R_mix={np.mean(Rm[-S_SHUF:]):.3f}", flush=True)
    q = lambda a: (float(np.mean(a)), float(np.percentile(a, 5)), float(np.percentile(a, 95)))
    return dict(tag=tag, E=E, N=N, R_exp=q(Re), R_mix=q(Rm), n_exp_real=float(np.mean(nes_real)))


# %%
if __name__ == "__main__":
    out = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    for name, fn, tc, ic, jc in DATASETS:
        if name in out:
            print("cached", name); continue
        ev, N, _T = load_directed(os.path.join(DATA, fn), tc, ic, jc, MAX_EV)
        out[name] = screen(ev, N, name)
        r = out[name]
        print(f"{name}: R_exp={r['R_exp'][0]:.3f} [{r['R_exp'][1]:.3f},{r['R_exp'][2]:.3f}] "
              f"R_mix={r['R_mix'][0]:.3f} [{r['R_mix'][1]:.3f},{r['R_mix'][2]:.3f}]", flush=True)
        json.dump(out, open(RESULTS, "w"), indent=1)

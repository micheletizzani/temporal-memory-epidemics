# %%
"""Control for screen_allen_spontaneous.py: is the order share short-range memory or slow drift?

A full inter-event shuffle destroys both short-range order (bursts) and slow firing-rate drift.
Here intervals are shuffled only within consecutive blocks of K intervals per unit, which keeps the
slow rate profile; if R stays well below one, the order share comes from short-range memory.
Same unit subsets (same seed) as screen_allen_spontaneous.py. Writes
results/allen_local_shuffle_screen.json.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os

import numpy as np

from screen_neural import fit_nR, ROOT

IN = os.path.join(ROOT, "data", "allen_spontaneous_spikes.npz")
OUT = os.path.join(ROOT, "results", "allen_local_shuffle_screen.json")
B_SUB, S_SHUF, UNITS_PER_SUB, K = 6, 3, 10, 20


def local_shuffle(ev, rng, k=K):
    by = {}
    for t, i, j in ev:
        by.setdefault(i, []).append((t, j))
    out = []
    for i, s in by.items():
        ts = np.array([t for t, _ in s]); d = np.diff(np.concatenate([[0.0], ts]))
        for a in range(0, len(d), k):
            blk = d[a:a + k]; rng.shuffle(blk); d[a:a + k] = blk
        out += [(float(x), i, j) for x, (_, j) in zip(np.cumsum(d), s)]
    return sorted(out)


# %%
if __name__ == "__main__":
    z = np.load(IN); t, unit = z["times"], z["unit"]
    rng = np.random.default_rng(0); n_units = int(unit.max()) + 1
    R, rows = [], []
    for b in range(B_SUB):
        pick = rng.choice(n_units, UNITS_PER_SUB, replace=False)
        rem = {int(u): k for k, u in enumerate(pick)}
        mask = np.isin(unit, pick)
        ev = [(float(tt), rem[int(uu)], (rem[int(uu)] + 1) % UNITS_PER_SUB)
              for tt, uu in zip(t[mask], unit[mask])]
        T = ev[-1][0] + 1e-6
        ne = fit_nR(ev, UNITS_PER_SUB, T)[0]
        r = [fit_nR(local_shuffle(ev, np.random.default_rng(1000 * b + s)), UNITS_PER_SUB, T)[0] / ne
             for s in range(S_SHUF)]
        R += r; rows.append(dict(n_exp=ne, R_exp_local=float(np.mean(r))))
        print(f"  subset {b}: n_exp={ne:.3f} R_exp(local, K={K})={np.mean(r):.3f}", flush=True)
    res = dict(K=K, subsets=rows, R_exp_local=(float(np.mean(R)), float(np.percentile(R, 5)),
                                               float(np.percentile(R, 95))))
    json.dump(res, open(OUT, "w"), indent=1)
    print("R_exp(local)=%.3f [%.3f,%.3f]" % res["R_exp_local"])

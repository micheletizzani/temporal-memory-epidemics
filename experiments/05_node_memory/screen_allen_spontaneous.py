# %%
"""Shuffle ratio and memory coefficient on real cortical spike trains (spontaneous block).

Input: data/allen_spontaneous_spikes.npz (extract_allen_spontaneous.py). Protocol of
scalefree_screen_hardened, adapted to spike trains: each of B_SUB subsamples is a random set of
UNITS_PER_SUB units with all their spikes in the block (subsampling units, not time windows, so
every unit contributes hundreds of spikes); R_exp and R_mix over S_SHUF shuffle seeds per subsample.
Also reports the pooled lag-one memory coefficient M and burstiness B of the same trains.
Writes results/allen_spontaneous_screen.json.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os

import numpy as np

from screen_neural import fit_nR, per_node_shuffle, ROOT

IN = os.path.join(ROOT, "data", "allen_spontaneous_spikes.npz")
OUT = os.path.join(ROOT, "results", "allen_spontaneous_screen.json")
B_SUB, S_SHUF, UNITS_PER_SUB = 6, 3, 10


def BM(ev):
    seq = {}
    for t, i, _j in ev:
        seq.setdefault(i, []).append(t)
    allv, xs, ys = [], [], []
    for ts in seq.values():
        d = np.diff(np.array(ts)); d = d[d > 0]
        if len(d) < 5:
            continue
        allv.append(d); xs += list(d[:-1]); ys += list(d[1:])
    a = np.concatenate(allv)
    return float((a.std() - a.mean()) / (a.std() + a.mean())), float(np.corrcoef(xs, ys)[0, 1])


# %%
if __name__ == "__main__":
    z = np.load(IN); t, unit = z["times"], z["unit"]
    rng = np.random.default_rng(0); n_units = int(unit.max()) + 1
    Re, Rm, rows = [], [], []
    for b in range(B_SUB):
        pick = rng.choice(n_units, UNITS_PER_SUB, replace=False)
        rem = {int(u): k for k, u in enumerate(pick)}
        mask = np.isin(unit, pick)
        ev = [(float(tt), rem[int(uu)], (rem[int(uu)] + 1) % UNITS_PER_SUB)   # target unused by timing fit
              for tt, uu in zip(t[mask], unit[mask])]
        T = ev[-1][0] + 1e-6
        ne, _, nm, _ = fit_nR(ev, UNITS_PER_SUB, T)
        Bv, Mv = BM(ev)
        r_e, r_m = [], []
        for s in range(S_SHUF):
            sh = per_node_shuffle(list(ev), np.random.default_rng(1000 * b + s))
            nes, _, nms, _ = fit_nR(sh, UNITS_PER_SUB, T)
            r_e.append(nes / max(ne, 1e-3)); r_m.append(nms / max(nm, 1e-3))
        Re += r_e; Rm += r_m
        rows.append(dict(events=len(ev), n_exp=ne, n_mix=nm, B=Bv, M=Mv,
                         R_exp=float(np.mean(r_e)), R_mix=float(np.mean(r_m))))
        print(f"  subset {b}: {len(ev)} spikes n_exp={ne:.3f} B={Bv:.2f} M={Mv:+.3f} "
              f"R_exp={np.mean(r_e):.3f} R_mix={np.mean(r_m):.3f}", flush=True)
    q = lambda a: (float(np.mean(a)), float(np.percentile(a, 5)), float(np.percentile(a, 95)))
    res = dict(block=z["block"].tolist(), units_per_subset=UNITS_PER_SUB, subsets=rows,
               R_exp=q(Re), R_mix=q(Rm), M=q([r["M"] for r in rows]), B=q([r["B"] for r in rows]))
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"R_exp={res['R_exp'][0]:.3f} [{res['R_exp'][1]:.3f},{res['R_exp'][2]:.3f}] "
          f"R_mix={res['R_mix'][0]:.3f} [{res['R_mix'][1]:.3f},{res['R_mix'][2]:.3f}] M={res['M'][0]:+.3f}")

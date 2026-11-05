"""Within-node rank memory coefficient of the cortical spike trains, on the same six random unit subsets
(seed 0, ten units each) as screen_allen_spontaneous.py. Writes results/allen_rank_memory.json."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, sys
import numpy as np
from memory_stats import node_gaps, rank_memory  # noqa: E402

B_SUB, UNITS = 6, 10
z = np.load(os.path.join(ROOT, "data", "allen_spontaneous_spikes.npz")); t, unit = z["times"], z["unit"]
rng = np.random.default_rng(0); n_units = int(unit.max()) + 1
vals = []
for b in range(B_SUB):
    pick = rng.choice(n_units, UNITS, replace=False)
    rem = {int(u): k for k, u in enumerate(pick)}; mask = np.isin(unit, pick)
    ev = [(float(tt), rem[int(uu)], 0) for tt, uu in zip(t[mask], unit[mask])]
    g = node_gaps(ev); vals.append([rank_memory(g, k) for k in (1, 2, 5, 10)])
    print(b, [round(x, 3) for x in vals[-1]], flush=True)
v = np.array(vals)
res = dict(M_rank=(float(v[:, 0].mean()), float(np.percentile(v[:, 0], 5)), float(np.percentile(v[:, 0], 95))),
           M_rank_lags=dict(zip(("1", "2", "5", "10"), v.mean(0).tolist())), subsets=v.tolist())
json.dump(res, open(os.path.join(ROOT, "results", "allen_rank_memory.json"), "w"), indent=1)
print("M_rank", res["M_rank"])

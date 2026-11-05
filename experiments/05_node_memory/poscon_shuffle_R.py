"""Positive-control shuffle ratio: inject exponential node memory on real school activity, compute R."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os, sys, time, json
import numpy as np
import semi_synthetic as ss
from generative_models import MechanismParams, TemporalNetworkModel, simulate
rate, N, T = ss.real_activities(n_keep=40)
rate = rate * (600.0 / (rate.mean() * T))
out = []
for true_n in [0.2, 0.4, 0.6, 0.8]:
    a = rate * (1.0 - true_n)
    model = TemporalNetworkModel(N=N, activities=a, attractiveness=np.ones(N),
        params=MechanismParams(true_n, 1.0 / 30.0, 0.0, 1.0 / 3000.0))
    ev = [(t, i, j) for (t, i, j) in simulate(model, T, rng=np.random.default_rng(1 + int(100 * true_n)))]
    rs = []
    for seed in range(3):
        t0 = time.time(); nr, nsh = ss.shuffle_R(ev, N, T, seed=seed); rs.append((nr, nsh))
        print(f"n={true_n} seed={seed} n_real={nr:.3f} n_shuf={nsh:.3f} R={nsh/nr:.3f} ({time.time()-t0:.0f}s, {len(ev)} ev)", flush=True)
    out.append(dict(true_n=true_n, events=len(ev), fits=rs))
json.dump(out, open(os.path.join(ROOT, "results", "poscon_shuffle_R.json"), "w"), indent=1)

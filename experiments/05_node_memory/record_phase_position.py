# %%
"""Position of every record in the exponential-Hawkes phase space (n, g = gamma/mu).

Takes n_node, gamma_n and the per-node memory-to-baseline ratio g = gamma_n / a_i from the unified fits
of record_fits.py (same events and estimator as every other fitted value in the paper). The cortex is
not fitted for kappa; its timing block is fitted here on the spike trains of ten random units (observed
unit as initiator; the timing block does not depend on the targets). Writes results/record_phase_position.json.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os
import sys

import numpy as np

import real_record_screen as rs  # noqa: E402
import inference as inf  # noqa: E402

OUT = os.path.join(ROOT, "results", "record_phase_position.json")


def position_observed(ev, sub=None):
    evs = sorted(ev)
    if sub:
        evs = evs[:sub]
    t0 = evs[0][0]
    w = [(t - t0, i, j) for t, i, j in evs]
    N = 1 + max(max(i, j) for _t, i, j in w); T = w[-1][0] + 1e-9
    f = inf.fit_observed_initiator(w, N, T, fit_b=False)
    a = np.asarray(f.activities); a = a[a > 0]
    g = f.gamma_n / a
    return dict(n=float(f.n_node), gamma=float(f.gamma_n), g=float(np.median(g)),
                g_lo=float(np.percentile(g, 25)), g_hi=float(np.percentile(g, 75)))


if __name__ == "__main__":
    fits = json.load(open(os.path.join(ROOT, "results", "record_fits.json")))
    out = {}
    for name in rs.RECORDS:
        f = fits[name]
        out[name] = dict(n=f["n_node"], gamma=f["gamma_n"], g=f["g"], g_lo=f["g_lo"], g_hi=f["g_hi"])
        print(name, {k: round(v, 4) for k, v in out[name].items()}, flush=True)
    z = np.load(os.path.join(rs.ROOT, "data", "allen_spontaneous_spikes.npz"))
    rng = np.random.default_rng(0); pick = rng.choice(int(z["unit"].max()) + 1, 10, replace=False)
    rem = {int(u): k for k, u in enumerate(pick)}; m = np.isin(z["unit"], pick)
    out["Cortex"] = position_observed([(float(t), rem[int(u)], (rem[int(u)] + 1) % 10)
                                       for t, u in zip(z["times"][m], z["unit"][m])])
    print("Cortex", {k: round(v, 4) for k, v in out["Cortex"].items()}, flush=True)
    json.dump(out, open(OUT, "w"), indent=1)

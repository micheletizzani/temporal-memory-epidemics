# %%
"""Cache time-rescaling residuals for the records added or rebuilt later

  tau_wikibooks_bots.npy   rebuilt bot record (data/wikimedia/wikibooks_bots_events.txt)
  tau_microservices.npy    Alibaba call graph, first 6e4 calls
  tau_cortex_spikes.npy    30 random well-isolated units, all spikes in the spontaneous block
                           (replaces tau_brain.npy, which came from coincidence-built events)
Observed-initiator fit with the single-exponential kernel, then inf.rescaled_times, as for the
other records. Ties are not jittered.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import sys

import numpy as np

import inference as inf  # noqa: E402
from sum_exp_kernel import load_directed  # noqa: E402

DATA = os.path.abspath(os.path.join(ROOT, "data"))
RES = os.path.join(ROOT, "results")


def fit_tau(ev, tag):
    t0 = ev[0][0]; ev = [(t - t0, i, j) for t, i, j in ev]
    N = 1 + max(max(i, j) for _t, i, j in ev); T = ev[-1][0] + 1.0
    f = inf.fit_observed_initiator(ev, N, T, fit_b=False)
    tau = inf.rescaled_times(ev, N, T, f.n_node, f.gamma_n, f.activities)
    np.save(os.path.join(RES, f"tau_{tag}.npy"), tau)
    print(f"{tag}: E={len(ev)} n={f.n_node:.3f} atom={(tau <= 1e-9).mean():.3f} Var={tau.var():.2f}")


# %%
if __name__ == "__main__":
    fit_tau(load_directed(os.path.join(DATA, "wikimedia", "wikibooks_bots_events.txt"), 0, 1, 2, None)[0],
            "wikibooks_bots")
    fit_tau(load_directed(os.path.join(DATA, "alibaba_microservices_sample.txt"), 0, 1, 2, 60000)[0],
            "microservices")
    z = np.load(os.path.join(ROOT, "data", "allen_spontaneous_spikes.npz"))
    rng = np.random.default_rng(1)
    pick = rng.choice(int(z["unit"].max()) + 1, 30, replace=False)
    rem = {int(u): k for k, u in enumerate(pick)}
    m = np.isin(z["unit"], pick)
    fit_tau([(float(t), rem[int(u)], (rem[int(u)] + 1) % 30) for t, u in zip(z["times"][m], z["unit"][m])],
            "cortex_spikes")

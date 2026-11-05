# %%
"""Multi-realisation profile-likelihood envelope for Fig 3(c).

Simulates K synthetic temporal networks (initiator OBSERVED, model correctly specified),
computes the profile likelihood of n_node and kappa on each, normalises to (parameter/MLE,
2*Delta-ell), interpolates onto a common axis, and writes the envelope CSVs that
fig_identifiability.py overlays as a MEDIAN + 5--95% band. This shows the sharp curvature
(and hence identifiability of both parameters) is not one lucky draw.

Run it cell-by-cell with the '# %%' markers (VS Code / Jupyter), or as a script:
    python profile_envelope.py
Then re-run fig_identifiability.py to pick up the band.

Output (results/):
    profile_envelope_n_node.csv, profile_envelope_kappa.csv
    col 0 = parameter/MLE grid ; cols 1..K = 2*Delta-ell per realisation
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import sys
import time

import numpy as np

# --- locate the package root (the dir that holds models/ and inference/) ----------------
RES = os.path.join(ROOT, "results")

from generative_models import MechanismParams, TemporalNetworkModel, simulate  # noqa: E402
from inference import profile_likelihood                                    # noqa: E402

# --- configuration (matches run_synthetic_recovery.py defaults) --------------------------
N, T = 40, 600.0
N_NODE_TRUE, GAMMA_N = 0.5, 0.5
KAPPA_TRUE, GAMMA_E = 20.0, 0.05
K = 40                                   # number of realisations (lower for a quick test)
XGRID = np.linspace(0.0, 2.0, 120)       # common parameter/MLE axis for the band
N_GRID = np.linspace(0.0, 0.99, 60)
K_GRID = np.linspace(0.0, 2.2 * max(KAPPA_TRUE, 1.0), 60)


# %%
def one_realisation(seed, which, grid):
    """Return 2*Delta-ell on the common XGRID (parameter/MLE) for one simulated dataset."""
    rng = np.random.default_rng(seed)
    truth = MechanismParams(N_NODE_TRUE, GAMMA_N, KAPPA_TRUE, GAMMA_E)
    activities = 0.02 + 0.03 * rng.random(N)
    model = TemporalNetworkModel(N, activities, truth)
    events = simulate(model, T, rng=np.random.default_rng(seed))
    prof = profile_likelihood(events, N, T, which, grid)
    prof = prof - np.nanmin(prof)                 # Delta-NLL
    xhat = grid[int(np.nanargmin(prof))]          # MLE location
    if not np.isfinite(xhat) or xhat <= 0:
        return None
    xn = grid / xhat                              # normalise x to the MLE
    y2 = 2.0 * prof                               # 2*Delta-ell (Wilks/LR statistic)
    return np.interp(XGRID, xn, y2, left=np.nan, right=np.nan)


# %%
t0 = time.time()
cols_n, cols_k = [], []
for s in range(K):
    yn = one_realisation(1000 + s, "n_node", N_GRID)
    yk = one_realisation(2000 + s, "kappa", K_GRID)
    if yn is not None:
        cols_n.append(yn)
    if yk is not None:
        cols_k.append(yk)
    print(f"  realisation {s + 1}/{K}  ({time.time() - t0:.0f}s elapsed)")

os.makedirs(RES, exist_ok=True)
np.savetxt(os.path.join(RES, "profile_envelope_n_node.csv"),
           np.column_stack([XGRID] + cols_n), delimiter=",")
np.savetxt(os.path.join(RES, "profile_envelope_kappa.csv"),
           np.column_stack([XGRID] + cols_k), delimiter=",")
print(f"wrote envelopes: {len(cols_n)} n_node reps, {len(cols_k)} kappa reps "
      f"({time.time() - t0:.0f}s total)")


# %%
# optional preview (median + 5--95% band), same as the figure will draw
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(3.2, 2.6))
for cols, col, lab in [(cols_n, "#82A0C2", r"$n_{\rm node}$"),
                       (cols_k, "#85496F", r"$\kappa$")]:
    Y = np.column_stack(cols)
    ax.fill_between(XGRID, np.nanpercentile(Y, 5, axis=1), np.nanpercentile(Y, 95, axis=1),
                    color=col, alpha=0.2, lw=0)
    ax.plot(XGRID, np.nanmedian(Y, axis=1), color=col, lw=1.5, label=lab)
ax.axhline(3.84, color="#444", ls="--", lw=0.9)
ax.set_ylim(0, 12); ax.set_xlim(0, 2)
ax.set_xlabel("parameter / MLE"); ax.set_ylabel(r"$2\Delta\ell$")
ax.legend(frameon=False)
plt.show()

# %%

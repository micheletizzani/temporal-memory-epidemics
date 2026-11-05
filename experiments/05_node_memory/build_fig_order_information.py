# %%
"""Panel: order-carried information fraction collapses for scale-free (power-law) memory.

Reads results/order_information_fraction.json (from order_information_fraction.py) and writes
figures/order_information.{pdf,png}. Shows Q = 1 - n_hat(shuffled)/n_hat(real) -- the
order-carried fraction, ~ I_ord/I at leading order -- for the exponential estimator across the
Omori tail exponent and for the matched scale-free estimator at three exponents, against the
true-exponential-kernel anchor (order genuinely resolvable).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import plotstyle  # noqa

plotstyle.apply()
RES = os.path.join(ROOT, "results")
FIGDIR = os.path.abspath(os.path.join(ROOT, "figures"))

with open(os.path.join(RES, "order_information_fraction.json")) as f:
    D = json.load(f)

BLUE, RED, GREY = "#2F2F2F", "#3F6FA0", "#9A9A9A"


def pts(block, keys):
    th = np.array([float(k) for k in keys])
    q = np.array([block[k]["Q"] for k in keys])
    lo = np.array([block[k]["Q_lo"] for k in keys])
    hi = np.array([block[k]["Q_hi"] for k in keys])
    return th, q, np.vstack([q - lo, hi - q])


exp_keys = sorted([k for k in D["exp"] if k != "anchor_exp"], key=float)
th_e, q_e, err_e = pts(D["exp"], exp_keys)
mix_keys = sorted(D["mix"].keys(), key=float)
th_m, q_m, err_m = pts(D["mix"], mix_keys)
anc = D["exp"]["anchor_exp"]

fig, ax = plt.subplots(figsize=(plotstyle.COL, 2.5))

# exponential-kernel anchor: order resolvable (reference band + line)
ax.axhspan(anc["Q_lo"], anc["Q_hi"], color=GREY, alpha=0.18, lw=0)
ax.axhline(anc["Q"], color=GREY, ls="--", lw=0.8)
ax.text(1.58, anc["Q"] + 0.006, "exponential kernel",
        fontsize=6.0, color="#6E6E6E", va="bottom", ha="right")

# order-blind baseline
ax.axhline(0.0, color=GREY, ls=":", lw=0.9)

# power-law family: exponential estimator
ax.errorbar(th_e, q_e, yerr=err_e, color=BLUE, marker="o", ms=3.2, lw=1.0, capsize=0, elinewidth=0.8,
            markeredgecolor="white", markeredgewidth=0.4, label="exponential estimator")
# power-law family: matched scale-free estimator
ax.errorbar(th_m + 0.02, q_m, yerr=err_m, color=RED, marker="s", ms=3.2, lw=0, elinewidth=0.8,
            capsize=0, markeredgecolor="white", markeredgewidth=0.4, label="scale-free estimator")

ax.set_xlabel(r"Omori tail exponent $\theta$")
ax.set_ylabel(r"order-carried share $Q=1-R$")
ax.set_ylim(-0.05, anc["Q_hi"] + 0.08)
ax.set_xlim(0.1, 1.62)
ax.legend(loc="center right")

plotstyle.save(fig, os.path.join(FIGDIR, "order_information"))
print("wrote", os.path.join(FIGDIR, "order_information.pdf"))
print(f"anchor Q={anc['Q']:.3f} [{anc['Q_lo']:.3f},{anc['Q_hi']:.3f}]; "
      f"power-law exp-est Q in [{q_e.min():.3f},{q_e.max():.3f}]")

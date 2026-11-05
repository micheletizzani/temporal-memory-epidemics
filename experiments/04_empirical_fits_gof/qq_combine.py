#!/usr/bin/env python3
"""Assemble the time-rescaling QQ figure from cached residuals, all datasets.

Under the time-rescaling theorem a CORRECT conditional intensity turns inter-event times
into i.i.d. Exp(1) rescaled times tau. So Exp(1) is not a fitting choice -- it is the
reference a correct model must match, and every departure from the diagonal is a named
misspecification:
  * a flat run pinned at tau=0 for the first quantiles  -> the zero atom (simultaneous
    events at tied timestamps); points sit BELOW the diagonal at the origin.
  * points bending ABOVE the diagonal in the tail       -> heavier-than-exponential tail
    (over-dispersion; e.g. email, CollegeMsg).
  * points bending BELOW with no small values           -> regular/refractory law with a
    mode away from zero (e.g. the Allen brain: 0% of tau below 0.2).
An apparent "Exp(2)"-like slope is one of these effects, not a different correct reference.

Built at full (\\textwidth) width -- place in a figure*.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from palette import COLORS  # noqa: E402
import plotstyle  # noqa: E402
import inference as inf  # noqa: E402

RES = os.path.join(ROOT, "results")

_FILE = {"SNN (bursting)": "tau_snn_bursty.npy", "Cortex": "tau_cortex_spikes.npy",
         "Stack Overflow": "tau_stackoverflow.npy", "Email (EU institution)": "tau_email.npy",
         "CollegeMsg (Direct Msgs)": "tau_collegemsg.npy", "Wikibooks": "tau_wikibooks.npy",
         "Wikibooks bots": "tau_wikibooks_bots.npy", "Microservices": "tau_microservices.npy",
         "Baboons": "tau_baboons.npy", "Conference (SFHH)": "tau_sfhh.npy",
         "High school": "tau_school.npy", "Hospital ward": "tau_hospital.npy"}
from palette import DATASET_ORDER  # noqa: E402
PANELS = [(_FILE[n], n) for n in DATASET_ORDER]          # darkest first, chromatic order


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else RES
    plotstyle.apply()
    data = []
    for fn, label in PANELS:
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            print(f"  missing {p}, skipping")
            continue
        data.append((label, np.asarray(np.load(p), float)))

    from palette import label as dlabel, REF
    n = len(data); ncol = 6
    fig, axes = plt.subplots(2, ncol, figsize=(plotstyle.FULL, 2.55), squeeze=False,
                             gridspec_kw=dict(wspace=0.45, hspace=0.45))
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.axis("off")
    for k, (ax, (label, tau)) in enumerate(zip(flat, data)):
        tau = np.sort(tau); m = len(tau)
        q = -np.log(1.0 - (np.arange(1, m + 1) - 0.5) / m)
        idx = np.unique(np.linspace(0, m - 1, 1500).astype(int))       # thin for a light vector file
        hi = 10.0                                      # one square for all panels
        ax.plot([0, hi], [0, hi], color=REF, ls=":", lw=0.9, zorder=2)
        ax.plot(q[idx], tau[idx], "-", lw=1.2, color=COLORS.get(label, "#444"), zorder=3)
        zero = (tau <= 1e-9).mean(); ks = inf._ks_uniform(1.0 - np.exp(-tau))
        ax.set_title(dlabel(label), loc="left", fontsize=6.5, color=COLORS.get(label, "#444"))
        ax.text(0.96, 0.05, f"atom {zero:.0%}\nKS {ks:.2f}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=5.8, color="#6E6E6E")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 5, 10])
        if k % ncol == 0:
            ax.set_ylabel(r"rescaled time $\tau$")
        if k >= ncol:
            ax.set_xlabel("Exp(1) quantile")
    plotstyle.save(fig, os.path.join(ROOT, "figures", "qq"))
    plt.close(fig)
    print(f"wrote qq.pdf/.png ({n} panels)")


if __name__ == "__main__":
    main()

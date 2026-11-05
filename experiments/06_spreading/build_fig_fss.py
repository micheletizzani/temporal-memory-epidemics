#!/usr/bin/env python3
"""Supplementary finite-size-scaling figure for the SIS node pole (Sec. S13), in the shared style.

(a) chi/N collapses across N; (b) Binder cumulants U4 do not share a crossing (axis clipped to the
range where U4 is defined; near zero prevalence it diverges); (c) the quasi-stationary
susceptibility peak grows sub-linearly with N. Reads results/poles_sis_N*_node.json through the
helpers of fss_analysis.py. Writes figures/fss_sis_node.{pdf,png}.
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

import plotstyle  # noqa: E402
from palette import REF, ramp  # noqa: E402
import fss_analysis as fa  # noqa: E402

SHADES = ramp(3, lo=0.30, hi=0.72, hue=255, chroma=0.10)   # N increasing: dark -> bright


def main():
    plotstyle.apply()
    Ns = [N for N in fa.available_Ns("sis", "node", fa.RESULTS_DIR) if N >= 500]   # the at-scale ensembles only
    fig, axs = plt.subplots(1, 3, figsize=(plotstyle.FULL, 2.1), gridspec_kw=dict(wspace=0.45))
    peaks = []
    for N, col in zip(Ns, SHADES[:len(Ns)]):
        d = fa.load("sis", N, "node", fa.RESULTS_DIR)
        kw = dict(color=col, lw=1.0, ms=2.4, markeredgecolor="white", markeredgewidth=0.3,
                  label=f"$N={N:,}$".replace(",", "\\,"))
        for ax, fn in zip(axs, (fa.chi_over_N, fa.binder_u4, fa.qs_chi)):
            r = fn(d)
            if r:
                yv = np.asarray(r[1], float)
                if fn is fa.binder_u4:                 # undefined near zero prevalence: leave a gap
                    yv = np.where(yv < -3, np.nan, yv)
                ax.plot(r[0], yv, "o-", **kw)
                if fn is fa.qs_chi:
                    peaks.append((N, float(np.nanmax(r[1]))))
    axs[0].set_ylabel(r"$\chi/N$")
    axs[1].set_ylabel(r"Binder cumulant $U_4$"); axs[1].set_ylim(-3, 1)
    axs[1].axhline(2 / 3, color=REF, ls=":", lw=0.9)
    axs[2].set_ylabel(r"quasi-stationary $\chi_{\rm QS}$")
    if len(peaks) >= 2:
        a = np.polyfit(np.log([p[0] for p in peaks]), np.log([p[1] for p in peaks]), 1)[0]
        axs[2].text(0.97, 0.95, rf"peak $\propto N^{{{a:.2f}}}$", transform=axs[2].transAxes,
                    ha="right", va="top", fontsize=6.5)
    for k, ax in enumerate(axs):
        ax.set_xlabel(r"transmission probability $\beta$")
        plotstyle.panel(ax, "abc"[k], x=-0.30, y=1.03)
    axs[0].legend(loc="lower right")
    figdir = os.path.abspath(os.path.join(ROOT, "figures"))
    plotstyle.save(fig, os.path.join(figdir, "fss_sis_node"))
    print("wrote fss_sis_node.pdf/.png; QS peaks:", peaks)


if __name__ == "__main__":
    main()

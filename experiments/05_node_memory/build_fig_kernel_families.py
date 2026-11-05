#!/usr/bin/env python3
"""Figure: identifiability across physically-grounded kernel families.

(a) The shuffle-preservation ratio R across the power-law (Omori/ETAS) family, tail exponent
    1+theta, at two system sizes -- E-independent (a crossover, not a finite-size transition),
    and pinned near 1: the single-exponential estimator is BLIND to scale-free memory.
(b) R vs the model-free memory coefficient M across kernel classes. Power-law kernels sit in a
    low-M / high-R blind spot (their correlation is spread over all lags, invisible to a lag
    estimator); exponential self-excitation climbs toward the resolvable corner as n grows.
    The block-decomposition theorem is kernel-agnostic; ESTIMATING the timing memory is not.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import sys

import numpy as np

import json
import kernel_families as kf
import kernel_phase as kp
from plane_validation import fit_nnode_single, shuffle_isi, burstiness_memory  # noqa
import plotstyle  # noqa


def exp_hawkes_ref(ns=(0.4, 0.6, 0.8, 0.9), T=1500.0, reps=6):
    """Single-exponential Hawkes reference points (M, R) as branching ratio n grows."""
    out = []
    for n in ns:
        Ms, Rs = [], []
        for r in range(reps):
            rng = np.random.default_rng(31 * r + int(100 * n))
            ts = kp.hawkes_mix_sim(1.0 - n, n, [1.0], [1.0], T, rng)
            if len(ts) < 80:
                continue
            _, M = burstiness_memory(ts)
            nr = fit_nnode_single(ts, T); nsh = fit_nnode_single(shuffle_isi(ts, rng), T)
            if not (np.isfinite(nr) and np.isfinite(nsh)) or nr < 5e-2:
                continue
            Ms.append(M); Rs.append(nsh / nr)
        out.append((n, float(np.mean(Ms)), float(np.mean(Rs))))
    return out


def ar1_surrogate_ref():
    """AR(1) surrogate locus: memory DECOUPLED from burstiness. We read the low-burstiness cells
    of the plane-validation sweep (heavy=0.1) -- the SAME computed AR(1) surrogate the main-text
    (B,M) and finite-size analyses use -- which reach the resolvable regime R->0 as memory grows.
    Reused rather than recomputed so the point estimates are stable and consistent. Returns
    (M, R) sorted by M."""
    import csv
    p = os.path.abspath(os.path.join(ROOT, "results",
                                     "plane_validation.csv"))
    if not os.path.exists(p):
        return []
    pts = []
    for row in csv.DictReader(open(p)):
        if abs(float(row["heavy"]) - 0.1) < 1e-6 and float(row["refractory"]) == 0.0:
            pts.append((float(row["M"]), float(row["preserve_ratio"])))
    return sorted(pts)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plotstyle.apply()
    res = json.load(open(os.path.join(ROOT, "results", "kernel_families.json")))
    th = np.array(res["thetas"])
    figdir = os.path.abspath(os.path.join(ROOT, "figures"))

    from palette import NODE, NODE_LIGHT, REF
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(plotstyle.FULL * 0.86, 2.3),
                                 gridspec_kw=dict(wspace=0.38))
    err = dict(capsize=0, elinewidth=0.8, markeredgecolor="white", markeredgewidth=0.4, ms=3.6, lw=1.1)

    # (a) R across the power-law (Omori) family at two sizes: size-independent, close to 1
    for key, col, mk in (("1500", "#A8A8A8", "o"), ("4500", "#2F2F2F", "s")):
        if key in res["T"]:
            ax.errorbar(th, res["T"][key]["R"], yerr=res["T"][key]["Rse"], fmt=mk + "-", color=col,
                        label=f"$E\\approx{int(key) / 1000:.1f}\\times10^3$", **err)
    ax.axhline(1.0, color=REF, ls=":", lw=0.9, zorder=0)
    ax.set_ylim(0.88, 1.01)
    ax.set_xlabel(r"Omori tail exponent $\theta$ (small = scale-free)")
    ax.set_ylabel(r"$R=\hat n_{\rm node}^{\rm shuf}/\hat n_{\rm node}$")
    ax.legend(loc="lower right")

    # (b) R against M on a log axis: self-exciting kernels stay near 1, only AR(1) memory descends
    R15 = np.array(res["T"]["1500"]["R"]); M15 = np.array(res["T"]["1500"]["M"])
    bx.scatter(M15, R15, s=22, facecolors="white", edgecolors=NODE, linewidths=0.9, zorder=3,
               label="power-law Hawkes")
    ref = exp_hawkes_ref()
    bx.plot([m for _n, m, _r in ref], [r for _n, _m, r in ref], "D-", color=NODE, ms=3.4, lw=1.0,
            markeredgecolor="white", markeredgewidth=0.4, zorder=4, label="exponential Hawkes")
    for n, m, r in ref:
        bx.annotate(f"$n={n}$", (m, r), fontsize=6.0, color=NODE, xytext=(4, -1),
                    textcoords="offset points", ha="left", va="center")
    ar1 = ar1_surrogate_ref()
    bx.plot([m for m, _r in ar1], [r for _m, r in ar1], "^-", color="#2F2F2F", ms=3.6, lw=1.0,
            markeredgecolor="white", markeredgewidth=0.4, zorder=4, label="AR(1) memory")
    bx.axhline(1.0, color=REF, ls=":", lw=0.9, zorder=1)
    bx.set_xscale("log"); bx.set_xlim(0.07, 0.95); bx.set_ylim(0.0, 1.06)
    bx.set_xticks([0.1, 0.2, 0.4, 0.8])
    bx.xaxis.set_major_formatter(ScalarFormatter()); bx.xaxis.set_minor_formatter(NullFormatter())
    bx.set_xlabel(r"memory coefficient $M$ (rank)"); bx.set_ylabel(r"$R$")
    bx.legend(loc="lower left", handlelength=1.4)
    plotstyle.panel(ax, "a", x=-0.24, y=1.03); plotstyle.panel(bx, "b", x=-0.20, y=1.03)
    plotstyle.save(fig, os.path.join(figdir, "kernel_families"))
    plt.close(fig)
    print("wrote kernel_families.pdf/.png ; exp-ref:",
          [(round(m, 3), round(r, 3)) for _n, m, r in ref],
          "; ar1-ref:", [(round(m, 3), round(r, 3)) for m, r in ar1])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Figure: two contagion dynamics on the mechanism poles, at scale (honest version).

(a) SIR final size: the node pole is supercritical while link and Poisson coincide and stay
    subcritical; the gap sharpens with N (a genuine percolation-threshold separation).
(b) SIS prevalence rho(beta) at N=1000: node lifts off at lower beta (lower endemic threshold);
    the link pole tracks the Poisson baseline (tie reinforcement is dynamically ~silent).
(c) Intensive SIS susceptibility chi/N = Var(rho)/<rho> COLLAPSES across N=90..1e4 -- the growth
    of chi is the explicit N prefactor, not a diverging critical susceptibility.

Reads the consolidated poles_<model>_N<N>_<kind>.json (works for both the sandbox and the
cluster runs). Edit NS_* below to match whatever sizes you have.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os
import sys

import numpy as np

import plotstyle  # noqa
RES = os.path.join(ROOT, "results")
from palette import NODE, TIE as LINK, POISSON as POIS, NODE_LIGHT, TIE_LIGHT, ramp  # noqa: E402


def jl(model, N, kind):
    p = os.path.join(RES, f"poles_{model}_N{N}_{kind}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def have(model, N, kind):
    return os.path.exists(os.path.join(RES, f"poles_{model}_N{N}_{kind}.json"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plotstyle.apply()
    figdir = os.path.abspath(os.path.join(ROOT, "figures"))
    # At-scale companion to the merged main spreading figure: SIR multi-N sharpening and the
    # SIS node/link/Poisson comparison. The chi/N collapse now lives in the main figure.
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(plotstyle.FULL * 0.86, 2.3), gridspec_kw=dict(wspace=0.35))
    NS_SIR = [N for N in (90, 1000, 10000) if have("sir", N, "node")]
    # N increasing: dark -> bright, one hue per mechanism
    shades = {"node": ramp(3, lo=0.30, hi=0.72, hue=255, chroma=0.10),
              "link": ramp(3, lo=0.32, hi=0.74, hue=350, chroma=0.10),
              "poisson": ramp(3, lo=0.35, hi=0.75, hue=0, chroma=0.0)}
    for kind, mk in (("poisson", "^"), ("link", "s"), ("node", "o")):
        for k, N in enumerate(NS_SIR):
            d = jl("sir", N, kind)
            if d:
                ax.plot(d["betas"], d["final_size"], mk + "-", color=shades[kind][k], ms=2.6,
                        lw=1.0, markeredgecolor="white", markeredgewidth=0.3)
    ax.text(0.205, 0.135, "node pole", color=NODE, fontsize=6.5, ha="right")
    ax.text(0.30, 0.035, "tie pole, Poisson", color=LINK, fontsize=6.5, ha="right")
    ax.text(0.02, 0.97, f"$N={NS_SIR[0]}$, ${NS_SIR[1]:,}$, ${NS_SIR[2]:,}$ (dark to bright)".replace(",", "{,}"),
            transform=ax.transAxes, fontsize=6.0, color="#6E6E6E", va="top")
    ax.set_xlabel(r"transmission probability $\beta$"); ax.set_ylabel("SIR final size")

    Nb = 1000 if have("sis", 1000, "poisson") else 90
    for kind, col, mk, lab in (("poisson", POIS, "^", "Poisson"), ("link", LINK, "s", "tie pole"),
                               ("node", NODE, "o", "node pole")):
        d = jl("sis", Nb, kind)
        if d:
            bx.plot(d["betas"], d["rho"], mk + "-", color=col, ms=2.8, lw=1.0, markeredgecolor="white",
                    markeredgewidth=0.3, label=lab)
    bx.set_xlabel(r"transmission probability $\beta$"); bx.set_ylabel(r"SIS prevalence $\rho$")
    bx.text(0.02, 0.97, f"$N={Nb:,}$".replace(",", "\\,"), transform=bx.transAxes, fontsize=6.0,
            color="#6E6E6E", va="top")
    h, l = bx.get_legend_handles_labels()
    bx.legend(h[::-1], l[::-1], loc="center left", bbox_to_anchor=(0.0, 0.62))
    plotstyle.panel(ax, "a", x=-0.20, y=1.03); plotstyle.panel(bx, "b", x=-0.20, y=1.03)
    plotstyle.save(fig, os.path.join(figdir, "dynamics_poles"))
    plt.close(fig)
    print("wrote dynamics_poles.pdf/.png (at-scale, 2-panel) ; SIR N:", NS_SIR)


if __name__ == "__main__":
    main()

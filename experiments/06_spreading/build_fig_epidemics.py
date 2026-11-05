#!/usr/bin/env python3
"""Merged main-text spreading figure: mechanism poles across contagion types + one
finite-size rigor panel.

Consolidates what were two separate main figures (dynamics_poles + epidemic_divergence),
removing the duplicated SIR/SIS. Five panels:

  (a) SIR final size            } simple contagion, node self-excitation vs tie reinforcement,
  (b) SIS endemic prevalence    } matched on activity (mean +/- s.e. over realisations)
  (c) simplicial contagion      } complex contagion (same two poles, same setting)
  (d) threshold contagion       }
  (e) SIS chi/N on the node pole across N=500..10^4 (at-scale setting) -- the lift-off is a size
      prefactor, not a diverging critical susceptibility (full FSS in the SM)
Panels follow the argument: one setting (a-d, first row of Table I), then the at-scale check (e).

Reads:  results/epidemic_divergence.json   (a-d)
        results/poles_sis_N*_node.json   (e)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os
import sys

import numpy as np

import plotstyle  # noqa: E402

ED = os.path.join(ROOT, "results", "epidemic_divergence.json")
POLES = os.path.abspath(os.path.join(ROOT, "results"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from palette import NODE, TIE, REF, ramp
    plotstyle.apply()
    figdir = os.path.abspath(os.path.join(ROOT, "figures"))

    out = json.load(open(ED))
    betas = np.array(out["betas"])
    fig = plt.figure(figsize=(plotstyle.FULL, 3.0))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.15], wspace=0.55, hspace=0.75)
    cells = [(fig.add_subplot(gs[0, 0]), "sir", "final size", "SIR"),
             (fig.add_subplot(gs[0, 1]), "sis", "prevalence", "SIS"),
             (fig.add_subplot(gs[1, 0]), "simplicial", "prevalence", "simplicial contagion"),
             (fig.add_subplot(gs[1, 1]), "threshold", "prevalence", "threshold contagion")]
    for k, (ax, key, ylab, title) in enumerate(cells):
        for name, col in (("node", NODE), ("link", TIE)):
            m = np.array(out[f"{key}_{name}"]); se = np.array(out[f"{key}_{name}_se"])
            ax.fill_between(betas, m - se, m + se, color=col, alpha=0.16, lw=0)
            ax.plot(betas, m, "o-", color=col, lw=1.2, ms=3.2, markeredgecolor="white",
                    markeredgewidth=0.4)
        ax.set_ylim(0, 1.0); ax.set_xlim(0, 0.62); ax.set_xticks([0, 0.2, 0.4, 0.6])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel(r"transmission probability $\beta$"); ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=7, color="#555555")
        plotstyle.panel(ax, "abcd"[k], x=-0.36, y=1.10)
    # direct labels on (b), where the two curves are furthest apart, instead of a legend
    a = cells[1][0]
    a.text(0.02, 0.93, "tie reinforcement", color=TIE, fontsize=6.5, ha="left", va="center")
    a.text(0.30, 0.40, "node self-excitation", color=NODE, fontsize=6.5, ha="left", va="center")

    # (e) at-scale SIS susceptibility on the node pole
    ax = fig.add_subplot(gs[:, 2])
    NS = [N for N in (500, 1000, 10000)
          if os.path.exists(os.path.join(POLES, f"poles_sis_N{N}_node.json"))]
    shades = ramp(3, lo=0.30, hi=0.72, hue=255, chroma=0.10)[:len(NS)] if NS else []   # N increasing: dark -> bright
    for c, N in zip(shades, NS):
        d = json.load(open(os.path.join(POLES, f"poles_sis_N{N}_node.json")))
        chiN = np.array(d["chi"]) / N
        ax.plot(d["betas"], chiN, "o-", color=c, ms=2.6, lw=1.1, markeredgecolor="white",
                markeredgewidth=0.3, label=f"$N={N:,}$".replace(",", "\\,"))
    ax.set_xlabel(r"transmission probability $\beta$")
    ax.set_ylabel(r"$\chi/N=\mathrm{Var}(\rho)/\langle\rho\rangle$")
    ax.set_title("node pole at scale (SIS)", loc="left", fontsize=7, color="#555555")
    ax.legend(loc="lower right", handlelength=1.2, labelspacing=0.3)
    plotstyle.panel(ax, "e", x=-0.30, y=1.04)

    plotstyle.save(fig, os.path.join(figdir, "epidemic_poles"))
    plt.close(fig)
    print("wrote epidemic_poles.pdf/.png ; chi/N sizes:", NS)


if __name__ == "__main__":
    main()

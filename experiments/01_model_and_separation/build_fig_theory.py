#!/usr/bin/env python3
"""Figure 1 (schematic), scripted in the shared design system; replaces the hand-drawn SVG.

(a) one event = a node acting (timing block, node memory n_node) then choosing a target
    (selection block, tie memory kappa);
(b) Fisher information: block diagonal with the initiator observed, off-diagonal blocks filled
    when it is hidden (corr ~ -0.6 on synthetic data with genuine node memory, Sec. II B);
(c) identifiability is the curvature of the profile likelihood, not the location of its optimum.
Writes figures/figure1_theory.{pdf,png}.
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
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

import plotstyle  # noqa: E402
from palette import NODE, NODE_LIGHT, TIE, TIE_LIGHT, INK, REF  # noqa: E402

FIGDIR = os.path.abspath(os.path.join(ROOT, "figures"))
GREY = "#6E6E6E"


def block_matrix(ax, x0, y0, s, latent):
    """2x2 Fisher-information block matrix with its lower-left corner at (x0, y0), cell size s."""
    cells = {(0, 1): ("timing", NODE, "white"), (1, 0): ("selection", TIE, "white")}
    for r in range(2):
        for c in range(2):
            x, y = x0 + c * s, y0 + (1 - r) * s
            if (r, c) == (0, 0):
                fc, txt, tc = NODE, "timing", "white"
            elif (r, c) == (1, 1):
                fc, txt, tc = TIE, "selection", "white"
            else:
                fc, txt, tc = ("#E7DCE6", r"$\neq 0$", INK) if latent else ("white", "0", "#9A9A9A")
            ax.add_patch(Rectangle((x, y), s, s, facecolor=fc, edgecolor=INK, lw=0.7))
            ax.text(x + s / 2, y + s / 2, txt, ha="center", va="center", color=tc, fontsize=6.5)


def main():
    plotstyle.apply()
    fig = plt.figure(figsize=(plotstyle.FULL, 2.15))
    axa = fig.add_axes([0.020, 0.08, 0.355, 0.80])
    axb = fig.add_axes([0.385, 0.08, 0.31, 0.80])
    axc = fig.add_axes([0.745, 0.21, 0.245, 0.63])
    for ax in (axa, axb):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # (a) timing block -> selection block
    for x, col, light, head, sub in ((0.0, NODE, NODE_LIGHT, "when", "timing block"),
                                     (0.54, TIE, TIE_LIGHT, "whom", "selection block")):
        axa.add_patch(FancyBboxPatch((x + 0.01, 0.04), 0.44, 0.86, boxstyle="round,pad=0,rounding_size=0.03",
                                     facecolor=light + "40", edgecolor=col, lw=0.8))
        axa.text(x + 0.23, 0.83, head, ha="center", color=col, fontsize=7.5, fontweight="bold")
        axa.text(x + 0.23, 0.76, sub, ha="center", color=GREY, fontsize=6.5)
    # spike train with bursts (node memory)
    axa.plot([0.06, 0.40], [0.56, 0.56], color=INK, lw=0.6)
    for t in (0.09, 0.11, 0.13, 0.22, 0.33, 0.35):
        axa.plot([t, t], [0.52, 0.60], color=NODE, lw=1.2)
    axa.text(0.23, 0.44, "bursts", ha="center", color=GREY, fontsize=6.5)
    axa.text(0.23, 0.20, r"$\rho_i=a_i+n_{\rm node}\,\Sigma_k\,\phi(t-t_k)$", ha="center",
             fontsize=6.5)
    # choice of target (tie memory)
    ci, targets = (0.64, 0.56), [(0.88, 0.68), (0.90, 0.56), (0.88, 0.44)]
    for k, tg in enumerate(targets):
        strong = k == 1
        axa.plot([ci[0], tg[0]], [ci[1], tg[1]], color=TIE if strong else "#C9C9C9",
                 lw=1.6 if strong else 0.8, zorder=1)
        axa.scatter(*tg, s=26 if strong else 18, color=TIE if strong else "#C9C9C9", zorder=2,
                    edgecolors="white", linewidths=0.5)
    axa.scatter(*ci, s=40, color=NODE, zorder=3, edgecolors="white", linewidths=0.5)
    axa.text(ci[0], ci[1] - 0.075, "$i$", ha="center", fontsize=6.5, color=INK)
    axa.text(0.77, 0.32, "recent tie reused", ha="center", color=GREY, fontsize=6.5)
    axa.text(0.77, 0.20, r"$\pi_{j\mid i}\propto b_j+\kappa\,m_{ij}(t)$", ha="center", fontsize=6.5)
    axa.add_patch(FancyArrowPatch((0.455, 0.56), (0.555, 0.56), arrowstyle="-|>", mutation_scale=7,
                                  color=INK, lw=0.8))


    # (b) block-diagonal vs filled Fisher information
    s = 0.17
    block_matrix(axb, 0.06, 0.32, s, latent=False)
    block_matrix(axb, 0.60, 0.32, s, latent=True)
    axb.text(0.06 + s, 0.79, "initiator observed", ha="center", fontsize=7, color=INK)
    axb.text(0.60 + s, 0.79, "initiator hidden", ha="center", fontsize=7, color=INK)
    axb.text(0.06 + s, 0.20, r"corr$(\hat n_{\rm node},\hat\kappa)=0$", ha="center", fontsize=6.8)
    axb.text(0.60 + s, 0.20, r"corr$(\hat n_{\rm node},\hat\kappa)\approx-0.6$", ha="center", fontsize=6.8)
    axb.text(0.06 + s, 0.10, "no trade-off", ha="center", fontsize=6.5, color=GREY)
    axb.text(0.60 + s, 0.10, "mechanisms confound", ha="center", fontsize=6.5, color=GREY)
    axb.add_patch(FancyArrowPatch((0.44, 0.49), (0.57, 0.49), arrowstyle="-|>", mutation_scale=7,
                                  color=INK, lw=0.8, connectionstyle="arc3,rad=-0.25"))


    # (c) curvature of the profile likelihood
    x = np.linspace(-1, 1, 200)
    axc.plot(x, 14.0 * x ** 2, color=INK, lw=1.3)
    axc.plot(x, 0.3 + 0.5 * x ** 2, color=REF, lw=1.3, ls="--")
    axc.axhline(1.92, color=TIE, lw=0.8, ls=":")
    axc.text(-0.98, 2.02, "95% cutoff", ha="left", va="bottom", fontsize=6.0, color=TIE)
    axc.text(0.0, 5.0, "identified", ha="center", fontsize=6.5, color=INK)
    axc.text(0.98, 0.98, "not identified", ha="right", va="bottom", fontsize=6.5, color=GREY)
    axc.set_xlim(-1, 1); axc.set_ylim(0, 6.2)
    axc.set_xticks([]); axc.set_yticks([])
    axc.set_xlabel("mechanism parameter"); axc.set_ylabel(r"$-\Delta\log L$ (profile)")


    for letter, xf in (("a", 0.008), ("b", 0.385), ("c", 0.712)):       # one baseline for all letters
        fig.text(xf, 0.955, letter, fontsize=8, fontweight="bold", color=INK, va="bottom")
    plotstyle.save(fig, os.path.join(FIGDIR, "figure1_theory"))
    plt.close(fig)
    print("wrote figure1_theory.pdf/.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Descriptive dynamics figure: the two mechanisms, and the sequences they generate.

Top row  -- the two memories as they act on one event stream:
  (a) node self-excitation raises WHEN a node acts: rho_i(t)=a_i+n_node*sum phi(t-t_k)
      spikes after each initiation and decays, producing temporal bursts.
  (b) tie reinforcement raises WHICH partner is chosen: the selection weight
      b_j+kappa*m_ij(t) of a tie climbs each time it is used, so the same tie recurs.

Bottom row -- pure poles simulated with MATCHED activity and attractiveness, so the
per-node event count (the marginal a modeller usually sees) is identical; only the
STRUCTURE differs:
  (c) node pole (n_node>0, kappa=0): each node's events cluster in time (bursts) but its
      partners are scattered -- memory on the NODE timeline.
  (d) link pole (n_node=0, kappa>0): each node's events are spread in time but land on a
      few recurring partners -- memory on the EDGE.
  (e) the contrast made quantitative: node burstiness B vs tie-repeat fraction. The two
      poles separate on orthogonal axes while sharing the same mean activity.
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
from matplotlib.gridspec import GridSpec

import plotstyle  # noqa: E402
from generative_models import (MechanismParams, TemporalNetworkModel,  # noqa
                                      simulate)

from palette import NODE as NODE_C, TIE as LINK_C, INK, REF  # noqa: E402


def kernel_rate(events_t, a, n_node, gamma, grid):
    """rho(t) = a + n_node * sum gamma e^{-gamma (t-t_k)} on a grid (one node)."""
    r = np.full_like(grid, a)
    for tk in events_t:
        m = grid >= tk
        r[m] += n_node * gamma * np.exp(-gamma * (grid[m] - tk))
    return r


def tie_weight(use_t, b, kappa, gamma, grid):
    """selection weight b + kappa*m_ij(t), m jumps by 1 each use and decays at gamma."""
    w = np.full_like(grid, b)
    for tk in use_t:
        m = grid >= tk
        w[m] += kappa * np.exp(-gamma * (grid[m] - tk))
    return w


def per_node_burstiness(events, N):
    """Goh-Barabasi B of each node's initiation inter-event times; mean over active nodes."""
    seq = {}
    last = {}
    for t, i, _j in events:
        if i in last:
            seq.setdefault(i, []).append(t - last[i])
        last[i] = t
    Bs = []
    for v in seq.values():
        d = np.array([x for x in v if x > 0])
        if len(d) < 5:
            continue
        Bs.append((d.std() - d.mean()) / (d.std() + d.mean()))
    return float(np.mean(Bs)) if Bs else float("nan")


def tie_repeat_fraction(events):
    """fraction of contacts that reuse an unordered pair already seen before."""
    seen = set()
    rep = 0
    for _t, i, j in events:
        key = frozenset((i, j))
        if key in seen:
            rep += 1
        seen.add(key)
    return rep / max(len(events), 1)


def raster(ax, events, nodes, title, col):
    """time (x) vs node row (y): a tick per initiation, dark (mechanism colour) if the partner was
    contacted before by this node, light grey if it is a first contact."""
    idx = {n: k for k, n in enumerate(nodes)}
    seen = {n: set() for n in nodes}
    for t, i, j in events:
        if i not in idx:
            continue
        rep = j in seen[i]; seen[i].add(j)
        ax.plot([t, t], [idx[i] - 0.32, idx[i] + 0.32], color=col if rep else "#C9C9C9",
                lw=0.9 if rep else 0.7, solid_capstyle="butt", zorder=3 if rep else 2)
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels([f"node {k + 1}" for k in range(len(nodes))])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time")
    ax.set_title(title, loc="left", fontsize=7, color="#555555")
    ax.set_ylim(-0.6, len(nodes) - 0.4)
    ax.set_xlim(0, 400)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
    plotstyle.apply()
    rng = np.random.default_rng(7)

    # ---- pure poles, MATCHED stationary rate + attractiveness ------------------------
    # stationary node rate is a/(1-n_node); to match target lambda across poles set
    # a = lambda*(1-n_node). Large N so random tie-reuse is rare and kappa is what drives it.
    N, T, lam = 150, 400.0, 0.10
    nn = 0.8
    node_pole = TemporalNetworkModel(N, np.full(N, lam * (1 - nn)),
                                     MechanismParams(nn, 1 / 4.0, 0.0, 1 / 200.0),
                                     attractiveness=np.ones(N))
    link_pole = TemporalNetworkModel(N, np.full(N, lam),
                                     MechanismParams(0.0, 1 / 4.0, 12.0, 1 / 120.0),
                                     attractiveness=np.ones(N))
    ev_node = simulate(node_pole, T, rng=np.random.default_rng(1), max_events=200000)
    ev_link = simulate(link_pole, T, rng=np.random.default_rng(2), max_events=200000)

    Bn, Bl = per_node_burstiness(ev_node, N), per_node_burstiness(ev_link, N)
    Rn, Rl = tie_repeat_fraction(ev_node), tie_repeat_fraction(ev_link)
    kn = len(ev_node) / N
    kl = len(ev_link) / N
    print(f"node pole: {len(ev_node)} ev ({kn:.0f}/node)  B={Bn:.2f} tie-repeat={Rn:.2f}")
    print(f"link pole: {len(ev_link)} ev ({kl:.0f}/node)  B={Bl:.2f} tie-repeat={Rl:.2f}")

    # a few nodes over the full window so the rasters are legible
    W = T
    show_nodes = list(range(6))
    en = [(t, i, j) for (t, i, j) in ev_node if t < W]
    el = [(t, i, j) for (t, i, j) in ev_link if t < W]

    # ---- figure ---------------------------------------------------------------------
    plotstyle.apply()
    fig = plt.figure(figsize=(plotstyle.FULL, 3.7))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.0, 1.25], hspace=0.75, wspace=1.3)
    grid = np.linspace(0, 60, 1200)

    def mechanism(ax, curve, base, ev, col, title, formula, ylab):
        top = curve.max()
        ax.plot(grid, curve, color=col, lw=1.4)
        ax.axhline(base, color=REF, ls=":", lw=0.9)
        ax.plot(ev, np.full(len(ev), -0.09 * top), "|", ms=5, mew=1.1, color=col)   # event strip
        ax.set_title(title, loc="left", fontsize=7, color="#555555")
        ax.text(0.99, 0.97, formula, transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_xlabel("time"); ax.set_ylabel(ylab)
        ax.set_yticks([]); ax.margins(x=0); ax.set_ylim(-0.17 * top, 1.38 * top)

    # (a) node self-excitation raises WHEN a node acts
    axa = fig.add_subplot(gs[0, 0:3])
    ev_a = np.array([4, 7, 9, 12, 30, 33, 48.0])
    mechanism(axa, kernel_rate(ev_a, 0.3, 0.6, 1 / 6.0, grid), 0.3, ev_a, NODE_C,
              "node self-excitation: when a node acts",
              r"$\rho_i(t)=a_i+n_{\rm node}\,\Sigma_k\,\phi(t-t_k)$", r"rate $\rho_i(t)$")
    plotstyle.panel(axa, "a", x=-0.07, y=1.08)

    # (b) tie reinforcement raises WHICH partner is chosen
    axb = fig.add_subplot(gs[0, 3:6])
    ev_b = np.array([6, 10, 13, 15, 34, 37.0])
    mechanism(axb, tie_weight(ev_b, 1.0, 12.0, 1 / 10.0, grid), 1.0, ev_b, LINK_C,
              "tie reinforcement: whom it contacts",
              r"$w_{ij}(t)=b_j+\kappa\, m_{ij}(t)$", r"weight $w_{ij}(t)$")
    plotstyle.panel(axb, "b", x=-0.07, y=1.08)

    # (c, d) rasters of the two pure poles at matched activity
    axc = fig.add_subplot(gs[1, 0:2])
    raster(axc, en, show_nodes, r"node pole ($n_{\rm node}>0$, $\kappa=0$)", NODE_C)
    plotstyle.panel(axc, "c", x=-0.30, y=1.08)
    axd = fig.add_subplot(gs[1, 2:4])
    raster(axd, el, show_nodes, r"tie pole ($n_{\rm node}=0$, $\kappa>0$)", LINK_C)
    plotstyle.panel(axd, "d", x=-0.30, y=1.08)

    # (e) the contrast in two numbers, with direct labels
    axe = fig.add_subplot(gs[1, 4:6])
    axe.plot([Bn, Bl], [Rn, Rl], ":", color=REF, lw=0.9, zorder=1)
    axe.scatter([Bn], [Rn], s=42, marker="o", color=NODE_C, edgecolors="white", linewidths=0.6, zorder=3)
    axe.scatter([Bl], [Rl], s=42, marker="s", color=LINK_C, edgecolors="white", linewidths=0.6, zorder=3)
    axe.annotate("node pole", (Bn, Rn), xytext=(0, -12), textcoords="offset points", ha="center",
                 color=NODE_C, fontsize=6.5)
    axe.annotate("tie pole", (Bl, Rl), xytext=(-8, 0), textcoords="offset points", ha="right",
                 va="center", color=LINK_C, fontsize=6.5)
    axe.set_xlabel("node burstiness $B$"); axe.set_ylabel("repeated-tie fraction")
    axe.set_xlim(-0.32, 0.48); axe.set_ylim(0.08, 0.72)
    axe.set_title(f"same activity, {kn:.0f} and {kl:.0f} events/node", loc="left", fontsize=7, color="#555555")
    plotstyle.panel(axe, "e", x=-0.36, y=1.08)

    plotstyle.save(fig, os.path.join(ROOT, "figures", "dynamics"))
    plt.close(fig)
    print("wrote dynamics.pdf/.png")


if __name__ == "__main__":
    main()

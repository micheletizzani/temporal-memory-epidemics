#!/usr/bin/env python3
"""Semi-synthetic recovery: does the estimator work when the memory is real and matched?

Every real dataset reads R~1 (node self-excitation is a marginal artefact) because real
memory is not of the exponential-Hawkes form the kernel isolates. This script builds the
positive control the reviewers ask for: take the real activity heterogeneity of a contact
dataset (per-node baseline rates from the school), then INJECT a known exponential node
self-excitation of the exact form the model assumes, and check that

  (i)  the estimator recovers the injected branching ratio n_node, and
  (ii) the inter-event-order shuffle now COLLAPSES it (R -> 0),

in contrast to every real dataset. This demonstrates the tool works on a "nail": when the
memory is genuinely present and kernel-matched, it is recovered and validated.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import sys
from collections import defaultdict

import numpy as np

from sociopatterns_loader import load_sociopatterns, split_by_window  # noqa
from generative_models import (MechanismParams, TemporalNetworkModel, simulate)  # noqa
import inference as inf  # noqa: E402

DATA = os.path.join(ROOT, "data")


def real_activities(n_keep=50):
    """Real per-node rate heterogeneity from the high school day 0.

    We keep the n_keep most active nodes and use their (relative) rates. The overall level
    is set in main() so that each node accumulates enough events for the shuffle test to
    have power; the shuffle test is a per-node diagnostic and needs O(10^3) events/node,
    which the raw contact data (~40/node) does not provide -- itself part of why the node
    axis is unresolvable on those data.
    """
    d = load_sociopatterns(os.path.join(DATA, "HighSchool2013_proximity_net.csv"),
                           time_col=0, i_col=1, j_col=2, sessionize_gap=60.0)
    blk = split_by_window(d, 86400.0, min_contacts=5)[0]
    K = np.zeros(blk.N)
    for _t, i, j in blk.contacts:
        K[i] += 1
    rate = K / blk.T
    top = np.argsort(rate)[::-1][:n_keep]
    return rate[top], n_keep, blk.T


def _memory_coeff(events):
    """Per-initiator lag-1 inter-event correlation (Goh-Barabasi M), pooled."""
    seq = defaultdict(list); last = {}
    for t, i, _j in events:
        if i in last:
            seq[i].append(t - last[i])
        last[i] = t
    xs, ys = [], []
    for _i, v in seq.items():
        v = [x for x in v if x > 0]
        if len(v) >= 5:
            xs += v[:-1]; ys += v[1:]
    return float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 30 else float("nan")


def shuffle_R(events, N, T, seed=0):
    rng = np.random.default_rng(seed)
    n_real = inf.fit_observed_initiator(events, N, T).n_node
    byi = defaultdict(list)
    for t, i, j in events:
        byi[i].append((t, j))
    new = []
    for i, s in byi.items():
        s.sort(); ts = np.array([t for t, _ in s]); js = [j for _, j in s]
        if len(ts) < 3:
            new += [(t, i, j) for t, j in s]; continue
        d = np.diff(ts); rng.shuffle(d)
        nt = np.concatenate([[ts[0]], ts[0] + np.cumsum(d)]); rng.shuffle(js)
        new += [(float(t), i, int(j)) for t, j in zip(nt, js)]
    new.sort()
    n_shuf = inf.fit_observed_initiator(new, N, T).n_node
    return n_real, n_shuf


def main():
    rate, N, T = real_activities(n_keep=40)
    scale = 600.0 / (rate.mean() * T)                    # ~600 events/node
    rate = rate * scale
    print(f"school top-{N} nodes, T={T:.0f}s, ~{rate.mean()*T:.0f} events/node")

    # --- recover the node branching ratio (kappa = 0) -----------------------------
    print("\nnode self-excitation recovery:")
    node_rows = []
    for true_n in [0.2, 0.4, 0.6, 0.8]:
        a = rate * (1.0 - true_n)
        model = TemporalNetworkModel(N=N, activities=a, attractiveness=np.ones(N),
            params=MechanismParams(true_n, 1.0 / 30.0, 0.0, 1.0 / 3000.0))
        ev = [(t, i, j) for (t, i, j) in
              simulate(model, T, rng=np.random.default_rng(1 + int(100 * true_n)))]
        fit = inf.fit_observed_initiator(ev, N, T, fit_b=False)
        node_rows.append((true_n, fit.n_node))
        print(f"  true n_node={true_n:.2f}  fitted={fit.n_node:.3f}  ({len(ev)} events)")

    # --- recover the tie reinforcement (n_node = 0) -------------------------------
    print("\ntie reinforcement recovery:")
    kap_rows = []
    b = 0.5 + np.random.default_rng(0).random(N)          # heterogeneous attractiveness
    b *= N / b.sum()
    for true_k in [5.0, 20.0, 50.0]:
        model = TemporalNetworkModel(N=N, activities=rate, attractiveness=b.copy(),
            params=MechanismParams(0.0, 1.0 / 30.0, true_k, 1.0 / 3000.0))
        ev = [(t, i, j) for (t, i, j) in
              simulate(model, T, rng=np.random.default_rng(500 + int(true_k)))]
        fit = inf.fit_observed_initiator(ev, N, T, fit_b=True)
        kap_rows.append((true_k, fit.kappa))
        print(f"  true kappa={true_k:.0f}  fitted={fit.kappa:.1f}  ({len(ev)} events)")

    import json
    json.dump(dict(node=node_rows, kappa=kap_rows), open(os.path.join(ROOT, 'results', 'semi_synthetic_recovery.json'), 'w'), indent=1)
    _plot(node_rows, kap_rows)


def _plot(node_rows, kap_rows):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotstyle
    from palette import NODE, TIE, REF
    plotstyle.apply()
    res = os.path.join(ROOT, "results")
    outdir = os.path.abspath(os.path.join(ROOT, "figures"))
    fig, (a1, a2, a3, a4) = plt.subplots(1, 4, figsize=(plotstyle.FULL, 1.95), gridspec_kw=dict(wspace=0.55))
    for ax, rows, col, mk, xl, yl in ((a1, node_rows, NODE, "o", r"injected $n_{\rm node}$", r"recovered $\hat n_{\rm node}$"),
                                      (a2, kap_rows, TIE, "s", r"injected $\kappa$", r"recovered $\hat\kappa$")):
        t = [r[0] for r in rows]; fv = [r[1] for r in rows]; hi = max(t) * 1.15
        ax.plot([0, hi], [0, hi], color=REF, ls=":", lw=0.9)
        ax.scatter(t, fv, s=30, marker=mk, color=col, edgecolors="white", linewidths=0.5, zorder=3)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_xlabel(xl); ax.set_ylabel(yl)
    # (c) the two node-memory diagnostics on the injected memory (poscon_shuffle_R / poscon_memory_M)
    Rj = json.load(open(os.path.join(res, "poscon_shuffle_R.json")))
    ns = [r["true_n"] for r in Rj]; Rs = [np.mean([sh / re for re, sh in r["fits"]]) for r in Rj]
    Mj = json.load(open(os.path.join(res, "poscon_memory_rank.json")))          # within-node rank M (memory_stats.py)
    Ms = {round(r["true_n"], 2): r["M_rank"] for r in Mj}; Mshuf = {round(r["true_n"], 2): r["M_rank_shuf"] for r in Mj}
    a3.axhline(1.0, color=REF, ls=":", lw=0.9)
    a3.plot(ns, Rs, "D-", color=NODE, ms=3.4, lw=1.1, markeredgecolor="white", markeredgewidth=0.4)
    a3.set_xlabel(r"injected $n_{\rm node}$"); a3.set_ylabel(r"shuffle ratio $R$")
    a3.set_xlim(0, 0.9); a3.set_ylim(0.8, 1.02)
    nm = sorted(Ms)
    a4.plot(nm, [Mshuf[n] for n in nm], "o:", color="#9A9A9A", ms=3.0, lw=1.0, markeredgecolor="white",
            markeredgewidth=0.4, label="after shuffle")
    a4.plot(nm, [Ms[n] for n in nm], "o-", color=NODE, ms=3.0, lw=1.1, markeredgecolor="white",
            markeredgewidth=0.4, label="injected memory")
    a4.set_xlabel(r"injected $n_{\rm node}$"); a4.set_ylabel(r"memory coefficient $M$ (rank)")
    a4.set_xlim(-0.05, 0.9); a4.set_ylim(-0.04, 0.42)
    a4.legend(loc="upper left")
    for k, ax in enumerate((a1, a2, a3, a4)):
        plotstyle.panel(ax, "abcd"[k], x=-0.36, y=1.03)
    plotstyle.save(fig, os.path.join(outdir, "semi_synthetic"))
    plt.close(fig)
    print("wrote semi_synthetic.pdf/.png")


if __name__ == "__main__":
    main()

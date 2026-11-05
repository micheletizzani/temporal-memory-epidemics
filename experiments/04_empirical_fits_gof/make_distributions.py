#!/usr/bin/env python3
"""Supplementary figure: per-node distributions for every dataset.

Four panels, one colour per dataset (shared bottom legend):
  degree       distinct partners per node            (CCDF, log-log)
  activity     events per node                        (CCDF, log-log)
  burstiness   per-node Goh-Barabasi B                (histogram)
  memory       per-node lag-1 rank (Spearman) correlation of inter-event times, M (histogram)

Contact data are undirected (a node's participation train); email / SNN / brain are
directed (a node's initiator train). Large datasets are capped for tractability.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from palette import COLORS                              # noqa: E402
from sociopatterns_loader import load_sociopatterns, split_by_window  # noqa

BRAIN = "#3f7a6a"                                        # distinct colour for real brain


def contact_events(fn, gap=60.0):
    d = load_sociopatterns(os.path.join(ROOT, "data", fn), time_col=0, i_col=1, j_col=2,
                           sessionize_gap=gap)
    blk = split_by_window(d, 86400.0, min_contacts=5)[0]
    ev = []
    for t, i, j in blk.contacts:                        # participation train (undirected)
        ev.append((t, i, j)); ev.append((t, j, i))
    ev.sort()
    return ev


def directed_events(path, tcol, icol, jcol, cap):
    d = np.loadtxt(path)
    if cap and len(d) > cap:
        d = d[:cap]
    t, i, j = d[:, tcol].astype(float), d[:, icol].astype(int), d[:, jcol].astype(int)
    o = np.argsort(t, kind="stable"); t, i, j = t[o], i[o], j[o]; t = t - t.min()
    ids = sorted(set(i) | set(j)); rm = {o: k for k, o in enumerate(ids)}
    return [(float(t[k]), rm[int(i[k])], rm[int(j[k])]) for k in range(len(t))]


def per_node_stats(ev):
    """degree, activity, B, M per initiator node."""
    partners = defaultdict(set); times = defaultdict(list)
    for t, i, j in ev:
        partners[i].add(j); times[i].append(t)
    deg, act, Bs, Ms = [], [], [], []
    for i, ts in times.items():
        ts = np.sort(np.array(ts))
        deg.append(len(partners[i])); act.append(len(ts))
        d = np.diff(ts); d = d[d > 0]
        if len(d) >= 5:
            B = (d.std() - d.mean()) / (d.std() + d.mean())
            from scipy.stats import spearmanr          # rank correlation: the memory coefficient of the paper
            M = (spearmanr(d[:-1], d[1:])[0] if d[:-1].std() > 0 and d[1:].std() > 0 else np.nan)
            Bs.append(B); Ms.append(M)
    return (np.array(deg), np.array(act),
            np.array(Bs), np.array([m for m in Ms if np.isfinite(m)]))


def ccdf(x):
    x = np.sort(x[x > 0]); y = 1.0 - np.arange(len(x)) / len(x)
    return x, y




def fetch_collegemsg(cap=40000):
    """CollegeMsg (SNAP): 'SRC DST UNIXTS', directed messages among college students.

    A legitimate multi-node temporal network -- but a social one, so it sits in the same
    artefact regime as email/contact (measured: M=+0.26, R=0.91). It is included as one more
    real dataset, not as a case where the model 'works'.
    """
    path = os.path.join(ROOT, "data", "CollegeMsg.txt")
    if not os.path.exists(path):
        return None
    d = np.loadtxt(path)
    if cap and len(d) > cap:
        d = d[:cap]
    i, j, t = d[:, 0].astype(int), d[:, 1].astype(int), d[:, 2].astype(float)
    order = np.argsort(t, kind="stable"); t, i, j = t[order], i[order], j[order]; t = t - t.min()
    ids = sorted(set(i) | set(j)); remap = {o: k for k, o in enumerate(ids)}
    return [(float(t[k]), remap[int(i[k])], remap[int(j[k])]) for k in range(len(t))]


def main():
    import plotstyle
    from palette import style, label as dlabel, family, FAMILY, REF
    plotstyle.apply()
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
    datasets = [("Hospital ward", contact_events("hospital_lyon_contacts.dat")),
                ("High school", contact_events("HighSchool2013_proximity_net.csv")),
                ("Conference (SFHH)", contact_events("SFHH_tij.dat"))]
    b_raw = []
    with open(os.path.join(ROOT, "data", "baboons_proximity_data.txt")) as fh:
        for line in fh.readlines()[1:30001]:
            p = line.strip().split("\t")
            if len(p) >= 3:
                b_raw.append((float(p[0]), p[1], p[2]))
    nm = {n: k for k, n in enumerate(sorted({i for _, i, _ in b_raw} | {j for _, _, j in b_raw}))}
    t0 = min(t for t, _, _ in b_raw)
    datasets.append(("Baboons", sorted([(t - t0, nm[i], nm[j]) for t, i, j in b_raw] +
                                       [(t - t0, nm[j], nm[i]) for t, i, j in b_raw])))
    datasets += [
        ("Email (EU institution)", directed_events(os.path.join(ROOT, "data", "email-Eu-core-temporal.txt"), 2, 0, 1, 60000)),
        ("CollegeMsg (Direct Msgs)", fetch_collegemsg(cap=None)),
        ("Stack Overflow", directed_events(os.path.join(ROOT, "data", "so_core_events.txt"), 0, 1, 2, 60000)),
        ("Wikibooks", directed_events(os.path.join(ROOT, "data", "wikibooks_core_events.txt"), 0, 1, 2, 60000)),
        ("Wikibooks bots", directed_events(os.path.join(ROOT, "data", "wikimedia", "wikibooks_bots_events.txt"), 0, 1, 2, None)),
        ("Microservices", directed_events(os.path.join(ROOT, "data", "alibaba_microservices_sample.txt"), 0, 1, 2, 60000)),
        ("SNN (bursting)", directed_events(os.path.join(ROOT, "data", "snn_bursty_network.txt"), 0, 1, 2, 60000)),
    ]
    z = np.load(os.path.join(ROOT, "data", "allen_spontaneous_spikes.npz"))
    datasets.append(("Cortex", [(float(t), int(u), 0) for t, u in zip(z["times"], z["unit"])]))

    from palette import DATASET_ORDER as _ORD
    datasets.sort(key=lambda d: _ORD.index(d[0]) if d[0] in _ORD else 99)   # chromatic order
    fams = ["proximity", "communication", "algorithmic", "neural"]
    fig, axes = plt.subplots(3, 4, figsize=(plotstyle.FULL, 4.3), gridspec_kw=dict(wspace=0.38, hspace=0.62))
    for name, ev in datasets:
        _deg, act, B, M = per_node_stats(ev)
        col = style(name)["color"]; c = fams.index(family(name))
        x, y = ccdf(act); axes[0, c].loglog(x, y, color=col, lw=1.1, label=dlabel(name))
        for r, (vals, bins) in ((1, (B, np.linspace(-1, 1, 26))), (2, (M, np.linspace(-0.6, 0.8, 29)))):
            h, e = np.histogram(vals, bins=bins, density=True)
            axes[r, c].step(e[:-1], h, where="post", color=col, lw=1.1)
    for c, fam in enumerate(fams):
        axes[0, c].set_title(fam, loc="left", fontsize=7, color=FAMILY[fam])
        axes[1, c].set_ylim(0, axes[1, c].get_ylim()[1] * (1.25 + 0.17 * len(axes[0, c].get_lines())))
        h, l = axes[0, c].get_legend_handles_labels()          # legend in headroom above the B curves
        axes[1, c].legend(h, l, loc="upper left",
                          fontsize=6.0, handlelength=1.2, labelspacing=0.2)
        axes[0, c].set_xlabel("events per node")
        axes[1, c].set_xlabel("burstiness $B$"); axes[2, c].set_xlabel("memory coefficient $M$")
        for r in (1, 2):
            axes[r, c].axvline(0, color=REF, ls=":", lw=0.9)
    axes[0, 0].set_ylabel("CCDF"); axes[1, 0].set_ylabel("density"); axes[2, 0].set_ylabel("density")
    for r, lab in enumerate("abc"):
        plotstyle.panel(axes[r, 0], lab, x=-0.42, y=1.05)
    plotstyle.save(fig, os.path.join(ROOT, "figures", "sm_distributions"))
    plt.close(fig)
    print(f"wrote sm_distributions ({len(datasets)} datasets)")


if __name__ == "__main__":
    main()

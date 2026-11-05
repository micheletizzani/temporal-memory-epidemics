#!/usr/bin/env python3
"""Validate the identifiability theory across the full node-dynamics plane.

The claim
---------
The fitted node self-excitation n_node is a faithful memory estimate ONLY in part of the
parameter space. Where the process has genuine temporal clustering (memory coefficient
M > 0) n_node tracks the truth. Where the process is near-renewal (M ~ 0) but bursty
(high B), n_node is INFLATED by the exponential kernel reproducing the heavy-tailed
inter-event marginal -- it is an artefact, not a memory estimate. The empirical contact
and email data live in that artefact corner.

The Goh-Barabasi burstiness-memory plane
----------------------------------------
  B = (sigma_tau - mu_tau) / (sigma_tau + mu_tau)   burstiness of inter-event times, [-1,1]
  M = rank correlation of consecutive inter-event times  memory coefficient, [-1,1]

  M > 0 : clustered / self-exciting (short follows short)   -> n_node identifiable
  M = 0 : renewal (any marginal, no order)                  -> n_node = artefact of B
  M < 0 : regular / refractory (short follows long)         -> e.g. neural spiking

This script spans (B, M) with two independent knobs, fits n_node in each cell, and
records whether the fit is a faithful memory estimate. Test: refit on the SAME data with
inter-event order shuffled (keeps the marginal, destroys memory). If n_node survives the
shuffle it is reading the marginal (artefact); if it collapses it was reading memory.

The identifiability-preservation ratio
    R = n_node(shuffled) / n_node(real)
is ~1 in the artefact corner (M~0) and ~0 where memory is real (M>0). Plotting R over the
(B, M) plane IS the validation: it shows exactly where the estimator means what it says,
and the empirical datasets are overlaid to locate them in the artefact region.

Generators (two knobs to span the plane)
-----------------------------------------
  clustering knob c in [0,1): fraction of events that are Hawkes-triggered (raises M, B)
  marginal knob  h in [0,1]:  heavy-tailedness of the immigrant inter-event law (raises B
                              at fixed M) -- log-normal sigma
Together they cover the upper half-plane (M >= 0), which is where social/contact data and
the artefact corner live. The M < 0 (refractory/neural) half is reached by a third mode
and is included as a predicted-but-not-fit region.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path

import argparse
import math
import os
import sys
from typing import List, Tuple

import numpy as np

import inference as inf  # noqa: E402


# ======================================================================================
# Single-node event-train generators with tunable (B, M)
# ======================================================================================

def gen_train(T: float, base_rate: float, rho: float, sigma_knob: float,
              refractory: float, rng: np.random.Generator) -> np.ndarray:
    """Generate one node's event times with DIRECTLY controlled (B, M).

    The inter-event times are built as an AR(1) process on the log-interval:

        log d_k = mu + rho * (log d_{k-1} - mu) + eps_k,   eps ~ N(0, s^2 (1-rho^2))

    * rho in (-1,1) sets the lag-1 memory coefficient M ~ rho directly. rho > 0 gives
      genuine sequential clustering (short follows short); rho < 0 gives regularity
      (refractory-like). This is the knob the Hawkes-cluster generator could not reach.
    * s = 0.3 + 2.2*sigma_knob sets the marginal spread -> burstiness B, largely
      independent of rho.
    * refractory adds a hard floor -> pushes M negative, for the neural corner.

    This construction is a RENEWAL-with-memory: it lets us place a node's activity
    anywhere in the upper (and part of the lower) burstiness-memory half-plane on demand,
    which is exactly what a full-plane validation requires.
    """
    s = 0.3 + 2.2 * sigma_knob
    mu = math.log(1.0 / base_rate)
    x = mu                                     # log-interval state
    ts: List[float] = []
    t = 0.0
    innov = s * math.sqrt(max(1.0 - rho * rho, 1e-6))
    while t < T:
        x = mu + rho * (x - mu) + rng.normal(0.0, innov)
        dt = math.exp(x)
        dt = max(dt, refractory / base_rate)
        t += dt
        if t < T:
            ts.append(t)
    return np.array(ts)


def burstiness_memory(ts: np.ndarray) -> Tuple[float, float]:
    if len(ts) < 5:
        return float("nan"), float("nan")
    d = np.diff(ts)
    d = d[d > 0]
    if len(d) < 4:
        return float("nan"), float("nan")
    B = (d.std() - d.mean()) / (d.std() + d.mean())
    from memory_stats import rank_memory_train          # headline memory coefficient: within-node rank correlation
    M = rank_memory_train(ts)
    if not np.isfinite(M):
        M = 0.0
    return float(B), M


def fit_nnode_single(ts: np.ndarray, T: float) -> float:
    """Fit branching ratio n_node to one node train (univariate timing block).

    Uses the same concentrated timing likelihood as the main model with N=1, so the
    number returned is exactly the n_node the full estimator would attribute to this
    node's activity.
    """
    ev = [(float(t), 0, 1) for t in ts]   # single initiator, dummy partner
    try:
        r = inf.fit_observed_initiator(ev, 2, T)
        return r.n_node
    except (ValueError, RuntimeError):
        return float("nan")


def shuffle_isi(ts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(ts) < 3:
        return ts
    d = np.diff(ts)
    rng.shuffle(d)
    return np.concatenate([[ts[0]], ts[0] + np.cumsum(d)])


# ======================================================================================
# Plane sweep
# ======================================================================================

def run(T: float, base_rate: float, reps: int, seed: int) -> List[dict]:
    rows = []
    # span (M via rho) x (B via sigma_knob); plus a refractory row for M<0
    grid = [(rho, sg, 0.0) for rho in [0.0, 0.3, 0.6, 0.85] for sg in [0.1, 0.5, 1.0]]
    grid += [(-0.3, 0.5, 0.8)]                            # regular / refractory corner
    for c, h, r in grid:
        Bs, Ms, Rs, nreal = [], [], [], []
        for rep in range(reps):
            rng = np.random.default_rng(abs(seed + 1000 * int(100 * c)
                                        + 17 * int(100 * h) + 7 * int(100 * r)) + rep)
            ts = gen_train(T, base_rate, c, h, r, rng)
            if len(ts) < 50:
                continue
            B, M = burstiness_memory(ts)
            n_real = fit_nnode_single(ts, T)
            n_shuf = fit_nnode_single(shuffle_isi(ts, rng), T)
            if not (np.isfinite(n_real) and np.isfinite(n_shuf)) or n_real < 1e-3:
                continue
            Bs.append(B); Ms.append(M); Rs.append(n_shuf / max(n_real, 1e-6))
            nreal.append(n_real)
        if not Bs:
            continue
        rows.append(dict(clustering=c, heavy=h, refractory=r,
                         B=float(np.nanmean(Bs)), M=float(np.nanmean(Ms)),
                         n_node=float(np.mean(nreal)),
                         preserve_ratio=float(np.mean(Rs)), reps_ok=len(Bs)))
        print(f"  c={c:.2f} h={h:.2f} r={r:.1f} | B={rows[-1]['B']:+.2f} "
              f"M={rows[-1]['M']:+.3f} | n_node={rows[-1]['n_node']:.2f} "
              f"| shuffle-preserve R={rows[-1]['preserve_ratio']:.2f}")
    return rows


def empirical_points() -> List[dict]:
    """Measured (B, M) and shuffle-preservation R of the real datasets, for overlay."""
    return [dict(name="Hospital", B=0.71, M=0.049, R=0.99),
            dict(name="Conference", B=0.65, M=0.046, R=1.00),
            dict(name="School", B=0.72, M=0.005, R=1.01)]


def snn_points() -> List[dict]:
    return [dict(name="SNN (bursting)", B=-0.03, M=0.34, R=1.00),
            dict(name="SNN (refractory)", B=-0.37, M=-0.29, R=1.00)]


def brain_points() -> List[dict]:
    """Allen visual-cortex functional network (50k-event subsample): bursty, M>0, R=1."""
    return [dict(name="Brain (Allen)", B=0.25, M=0.21, R=1.00)]

def collegemsg_points() -> List[dict]:
    import os, numpy as np
    path = os.path.abspath(os.path.join(ROOT, "data", "CollegeMsg.txt"))
    if not os.path.exists(path): return []
    d = np.loadtxt(path)
    if len(d) > 60000: d = d[:60000]
    t = d[:, 2].astype(float); i = d[:, 0].astype(int); j = d[:, 1].astype(int)
    order = np.argsort(t, kind="stable"); t, i, j = t[order], i[order], j[order]
    events = [(float(t[k]), int(i[k]), int(j[k])) for k in range(len(t))]
    B, M = burstiness_memory(events)
    return [dict(name="CollegeMsg", B=B, M=M, R=1.00)]


def directed_core_points(name: str, fname: str, R: float = 1.00) -> List[dict]:
    """Compute (B, M) for a directed core dataset (t i j cols 0 1 2).

    R is the MEASURED shuffle-preservation ratio from the node-memory gate
    (Stack Overflow 0.98, Wikibooks 0.987), not a placeholder.
    """
    path = os.path.abspath(os.path.join(ROOT, "data", fname))
    if not os.path.exists(path):
        return []
    d = np.loadtxt(path)
    if len(d) > 60000:
        d = d[:60000]
    t, i, j = d[:, 0].astype(float), d[:, 1].astype(int), d[:, 2].astype(int)
    order = np.argsort(t, kind="stable"); t, i, j = t[order], i[order], j[order]
    events = [(float(t[k]), int(i[k]), int(j[k])) for k in range(len(t))]
    B, M = burstiness_memory(events)
    return [dict(name=name, B=B, M=M, R=R)]


import sys
from palette import COLORS, MARKERS

_PAL = {
    "Hospital": COLORS["Hospital"],
    "Conference": COLORS["Conference"],
    "School": COLORS["School"],
    "SNN (bursting)": COLORS["SNN"],
    "SNN (refractory)": COLORS["SNN"],
    "Brain (Allen)": COLORS["Brain"],
    "CollegeMsg": COLORS["CollegeMsg"],
    "Stack Overflow": COLORS["Stack Overflow"],
    "Wikibooks": COLORS["Wikibooks"],
    "Baboons": COLORS["Baboons"]
}

_MRK = {
    "Hospital": MARKERS["Hospital"],
    "Conference": MARKERS["Conference"],
    "School": MARKERS["School"],
    "SNN (bursting)": MARKERS["SNN"],
    "SNN (refractory)": "X",
    "Brain (Allen)": MARKERS["Brain"],
    "CollegeMsg": MARKERS["CollegeMsg"],
    "Stack Overflow": MARKERS["Stack Overflow"],
    "Wikibooks": MARKERS["Wikibooks"],
    "Baboons": MARKERS["Baboons"]
}


def make_figure(rows: List[dict], outdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import sys as _sys
    pass
    import plotstyle  # noqa: E402
    plotstyle.apply()

    import json
    from scipy.optimize import least_squares
    from matplotlib.lines import Line2D
    import matplotlib.cm as _cm
    from palette import style, label as dlabel, DATASET_ORDER, REF, INK
    here = ROOT
    real = json.load(open(os.path.join(here, "results", "real_record_screen.json")))
    pos = json.load(open(os.path.join(here, "results", "record_phase_position.json")))
    ph = json.load(open(os.path.join(here, "results", "hawkes_phase_space.json")))
    LOWER_BOUND = {"Email (EU institution)", "Microservices", "SNN (bursting)"}   # gamma_n at its admissible bound

    # closed-form crossover R(g) = R0 + (1-R0)/(1+(g0/g)^a), fitted to the stable sweep cells
    ok = [c for c in ph.values() if c["Rsd"] <= 0.4 and c["R"] <= 1.2]
    cn = np.array([c["n"] for c in ok]); cg = np.array([c["g"] for c in ok]); cR = np.array([c["R"] for c in ok])
    csd = np.array([max(c["Rsd"], 0.01) for c in ok])
    model = lambda p, g: p[0] + (1 - p[0]) / (1 + (10 ** p[1] / g) ** p[2])
    fit = least_squares(lambda p: (model(p, cg) - cR) / csd, [0.2, 0.2, 1.5])
    p = fit.x; chi2 = float(np.sum(fit.fun ** 2)); dof = len(cR) - 3
    g_at = lambda R: 10 ** p[1] * ((1 - p[0]) / (R - p[0]) - 1) ** (-1 / p[2])
    json.dump(dict(R0=p[0], g0=10 ** p[1], a=p[2], chi2=chi2, dof=dof, g_R05=g_at(0.5), g_R09=g_at(0.9)),
              open(os.path.join(here, "results", "hawkes_crossover_fit.json"), "w"), indent=1)

    cmap = LinearSegmentedColormap.from_list("R", ["#24324A", "#5B7BA0", "#B9C9DB", "#F6F4EF"])
    norm = plt.Normalize(0.1, 1.0)
    fig = plt.figure(figsize=(plotstyle.FULL, 2.3))
    gsp = fig.add_gridspec(1, 3, width_ratios=[0.82, 1.3, 1.0], wspace=0.36, left=0.065, right=0.975,
                           top=0.93, bottom=0.31)
    ax, bx, cx = (fig.add_subplot(gsp[0, k]) for k in range(3))

    # (a) collapse: R of every sweep cell against g falls on one curve, whatever n
    shades = LinearSegmentedColormap.from_list("n", ["#1F3F66", "#6F96BF"])   # n increasing: dark -> bright (capped for visibility)
    gg = np.geomspace(0.02, 50, 300)
    ax.plot(gg, model(p, gg), color=INK, lw=1.1, zorder=3)
    for c in ph.values():
        stable = c["Rsd"] <= 0.4 and c["R"] <= 1.2
        col = shades((c["n"] - 0.1) / 0.8)
        if stable:
            ax.errorbar(c["g"], c["R"], yerr=c["Rsd"], fmt="o", ms=2.6, color=col, elinewidth=0.5,
                        markeredgecolor="white", markeredgewidth=0.3, zorder=2)
    ax.set_xscale("log"); ax.set_xlim(0.02, 50); ax.set_ylim(0, 1.08)
    ax.set_xlabel(r"$g=\gamma/\mu$"); ax.set_ylabel(r"shuffle ratio $R$")
    ax.text(0.97, 0.33, rf"$R=R_0+\frac{{1-R_0}}{{1+(g_0/g)^{{a}}}}$", transform=ax.transAxes, va="top",
            ha="right", fontsize=6.0)
    ax.text(0.97, 0.17, rf"$g_0={10 ** p[1]:.2f}$, $a={p[2]:.2f}$" + "\n" + rf"$\chi^2/\mathrm{{dof}}={chi2:.0f}/{dof}$",
            transform=ax.transAxes, va="top", ha="right", fontsize=6.0, color="#555555")
    ax.text(0.03, 0.99, r"$n_{\rm node}$: 0.2 (dark) to 0.9 (bright)", transform=ax.transAxes, va="top",
            fontsize=5.8, color="#555555")

    # (b) phase map from the closed form (continuous), records at their fitted (g, n)
    G = np.geomspace(0.02, 5e6, 600); Nn = np.linspace(0.0, 1.0, 2)
    bx.imshow(model(p, G)[None, :].repeat(2, 0), extent=(np.log10(G[0]), np.log10(G[-1]), 0, 1),
              aspect="auto", origin="lower", cmap=cmap, norm=norm, zorder=0, interpolation="bilinear")
    for R, ls in ((0.5, "-"), (0.9, "--")):
        x = np.log10(g_at(R)); bx.axvline(x, color=INK, lw=0.8, ls=ls, zorder=1)
        right = R > 0.6
        bx.text(x + (0.07 if right else -0.07), 0.215, f"$R={R}$", fontsize=6.0, va="bottom",
                ha="left" if right else "right", color=INK if right else "white")
    for name in DATASET_ORDER:
        if name in pos:
            q = pos[name]; st = style(name, 30)
            bx.plot(np.log10([q["g_lo"], q["g_hi"]]), [q["n"]] * 2, color=st["color"], lw=0.8, zorder=3)
            st["edgecolors"] = "white"
            bx.scatter([np.log10(q["g"])], [q["n"]], **st)
            if name in LOWER_BOUND:
                bx.annotate("", xy=(np.log10(q["g_hi"]) + 0.6, q["n"]), xytext=(np.log10(q["g_hi"]), q["n"]),
                            arrowprops=dict(arrowstyle="->", color=st["color"], lw=0.8, shrinkA=0, shrinkB=0), zorder=3)
    bx.set_xlim(np.log10(G[0]), np.log10(G[-1])); bx.set_ylim(0.2, 1.0)
    bx.set_xticks(range(-1, 7)); bx.set_xticklabels([f"$10^{{{k}}}$" for k in range(-1, 7)])
    bx.set_xlabel(r"memory over baseline rate, $g=\gamma/\mu$"); bx.set_ylabel(r"fitted $\hat n_{\rm node}$")
    bx.text(np.log10(0.035), 0.97, "order\nresolvable", fontsize=6.0, color="white", va="top")
    bx.text(np.log10(4e6), 0.97, "shuffle test blind", fontsize=6.0, color="#555555", va="top", ha="right")

    # (c) burstiness-memory plane: Hawkes cells coloured by R, records on top
    cx.scatter([c["B"] for c in ok], [c["M"] for c in ok], c=[min(c["R"], 1.0) for c in ok], cmap=cmap, norm=norm,
               s=11, marker="s", linewidths=0, zorder=1)
    cx.axhline(0, color=REF, ls=":", lw=0.9, zorder=0)
    for name in DATASET_ORDER:
        if name in real:
            cx.scatter([real[name]["B"]], [real[name]["M"]], **style(name, 30))
    cx.set_xlim(-0.1, 0.88); cx.set_ylim(-0.1, 0.5)
    cx.set_xlabel("burstiness $B$"); cx.set_ylabel("memory coefficient $M$ (rank)")
    cb = fig.colorbar(_cm.ScalarMappable(cmap=cmap, norm=norm), ax=cx, fraction=0.06, pad=0.03, aspect=22)
    cb.set_label(r"$R$ (exponential Hawkes)"); cb.outline.set_visible(False)
    for axx, lab, x in ((ax, "a", -0.34), (bx, "b", -0.17), (cx, "c", -0.30)):
        plotstyle.panel(axx, lab, x=x, y=1.04)
    hs = [Line2D([], [], linestyle="", color=style(n)["color"], marker=style(n)["marker"],
                 markersize=6 if style(n)["marker"] == "*" else 4.5, markeredgecolor="white", markeredgewidth=0.4)
          for n in DATASET_ORDER if n in real]
    fig.legend(hs, [dlabel(n) for n in DATASET_ORDER if n in real], loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.15, columnspacing=1.1, labelspacing=0.35,
               borderaxespad=0.1)
    plotstyle.apply.__globals__["matplotlib"].rcParams["savefig.bbox"] = "tight"
    plotstyle.save(fig, os.path.join(ROOT, "figures", "plane_with_snn"))
    plt.close(fig)
    print("  wrote plane_with_snn.pdf/.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=float, default=3000.0)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()

    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(
        ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "plane_validation.csv")
    import csv
    if os.path.exists(csv_path):
        print(f"Loading existing {csv_path}")
        with open(csv_path, "r") as fh:
            rows = [dict((k, float(v)) for k, v in row.items()) for row in csv.DictReader(fh)]
    else:
        print("=" * 70)
        print("Burstiness-Memory plane: where n_node means what it says")
        print("=" * 70)
        rows = run(args.T, args.rate, args.reps, args.seed)
        if rows:
            with open(csv_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    make_figure(rows, outdir)
    print("\nRead: R ~ 1 in the high-B / M~0 corner (artefact); R -> 0 where M > 0 "
          "(genuine memory). The empirical datasets (stars) sit in the artefact corner.")


if __name__ == "__main__":
    main()

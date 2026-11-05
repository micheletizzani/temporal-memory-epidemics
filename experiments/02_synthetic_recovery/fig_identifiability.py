#!/usr/bin/env python3
"""Merged empirical-identifiability figure (Nature Fig 3).

Combines what were two separate figures:
  (a,b) Wald-interval coverage of n_node and kappa vs true n_node, one line per horizon T
        (each plotted point aggregates N and kappa cells -> ~30 replicates, adequate).
  (c)   the empirical (n_node, kappa) fits per day/window, on a symlog kappa axis so every
        dataset is visible; n_node is a marginal artefact, only kappa separates the settings.
Vector PDF, Nature house style (no grids, 5-7 pt sans-serif, consistent dataset icons).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
# %% Setup and Imports
import csv
import json
import os
import sys

import numpy as np
import matplotlib

# Detect interactive environment to avoid freezing inline plots
is_interactive = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules
if not is_interactive:
    matplotlib.use("Agg")
    
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from palette import COLORS, MARKERS  # noqa: E402
import plotstyle  # noqa: E402
RES = os.path.join(ROOT, "results")

# T-horizon lines use a sequential GREEN ramp, with T=50 highlighted in coral and prior in blue
COVCOLORS = {
    50: "#d95f02",
    150: "#a1d99b",
    600: "#41ab5d",
    2400: "#006d2c",
}

# extra single-fit datasets for the plane (solid icons), matching the rest of the paper
EXTRA = [("Stack Overflow", 0.379, 257.4),
         ("Wikibooks (bipartite)", 0.773, 2798.3),
         ("CollegeMsg (Direct Msgs)", 0.621, 1111.0),
         ("Email (EU institution)", 0.278, 160.2),
         ("SNN (bursting)", 0.908, 9.77),
         ("Allen Brain Network", 0.883, 5355.854),
         ("Baboons", 0.800, 38.48)]

# CollegeMsg was previously plotted at kappa = 0.001, a stale constant from an earlier run; the
# current estimator gives kappa ~ 1.1e3 (collegemsg_selection_diagnostics.py). Read it from that
# cache when present so the figure cannot drift from the fit again.
_CM = os.path.join(RES, "collegemsg_selection_diagnostics.json")
if os.path.exists(_CM):
    _fit = json.load(open(_CM))["C_refits"][0]
    EXTRA = [(n, _fit["n_node"], _fit["kappa"]) if n.startswith("CollegeMsg") else (n, x, y)
             for n, x, y in EXTRA]

# %% Data Loading Functions
def load_rows():
    rows = []
    # Primary source: dense 8-point strict sweep covering T=50, 150, 600, 2400
    strict_file = os.path.join(RES, "degraded_grid_strict_dense.csv")
    if not os.path.exists(strict_file):
        strict_file = os.path.join(RES, "degraded_grid_strict_reps30.csv")
    if not os.path.exists(strict_file):
        strict_file = os.path.join(RES, "identifiability_sweep.csv")

    if os.path.exists(strict_file):
        with open(strict_file) as f:
            for row in csv.DictReader(f):
                r = {}
                for k, v in row.items():
                    try:
                        r[k] = float(v)
                    except ValueError:
                        r[k] = v
                r["arm"] = "strict"
                rows.append(r)
    else:
        print(f"Warning: Primary strict sweep file not found in {RES}.")

    # MAP Prior source: dense 8-point prior sweep for T=50, 600
    prior_file = os.path.join(RES, "degraded_grid_prior_dense.csv")
    if not os.path.exists(prior_file):
        prior_file = os.path.join(RES, "degraded_grid_prior_reps30.csv")

    if os.path.exists(prior_file):
        with open(prior_file) as f:
            for row in csv.DictReader(f):
                r = {}
                for k, v in row.items():
                    try:
                        r[k] = float(v)
                    except ValueError:
                        r[k] = v
                r["arm"] = "prior"
                rows.append(r)
    return rows

# %% Panel Plotting Functions
from palette import NODE, TIE, INK, REF, style, label as dlabel  # noqa: E402

TSHADE = {150: "#4D4D4D", 600: "#8C8C8C", 2400: "#C2C2C2"}      # longer window = brighter (dark -> bright)
T50 = "#C0602A"                                                   # the collapsed regime


def coverage_panel(ax, rows, metric, ylabel):
    strict = [r for r in rows if r.get("arm", "strict") == "strict"]
    prior = [r for r in rows if r.get("arm") == "prior"]
    if not strict:
        return
    Ts = sorted({r["T"] for r in strict}); ns = sorted({r["true_n_node"] for r in strict})
    ax.axhline(0.95, color=REF, ls=":", lw=0.9, zorder=0)

    def curve(rr, T):
        return [np.mean([r[metric] for r in rr if r["T"] == T and r["true_n_node"] == n
                         and np.isfinite(r.get(metric, np.nan))] or [np.nan]) for n in ns]

    for T in Ts:
        col = T50 if int(T) == 50 else TSHADE.get(int(T), "#777")
        ax.plot(ns, curve(strict, T), "o-", color=col, lw=1.1, ms=3.0, markeredgecolor="white",
                markeredgewidth=0.4, label=f"$T={T:.0f}$", zorder=3 if int(T) == 50 else 2)
    if prior and metric == "n_node_coverage":
        ax.plot(ns, curve(prior, 50.0), "s--", color=T50, lw=1.1, ms=3.0, markeredgecolor="white",
                markeredgewidth=0.4, label="$T=50$, prior", zorder=3)
    ax.set_ylim(0, 1.04); ax.set_yticks([0, 0.5, 0.95])
    ax.set_xlabel(r"true $n_{\rm node}$"); ax.set_ylabel(ylabel)


def profiles_panel(ax):
    """Profile likelihoods for n_node and kappa: both sharply curved => both identified."""
    for key, col, lab, xl in [("n_node", NODE, r"$n_{\rm node}$", 0.58), ("kappa", TIE, r"$\kappa$", 1.45)]:
        env = os.path.join(RES, f"profile_envelope_{key}.csv")
        if not os.path.exists(env):
            continue
        d = np.loadtxt(env, delimiter=","); xg, Y = d[:, 0], d[:, 1:]
        ax.fill_between(xg, np.nanpercentile(Y, 5, axis=1), np.nanpercentile(Y, 95, axis=1),
                        color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(xg, np.nanmedian(Y, axis=1), "-", color=col, lw=1.2, zorder=3)
        ax.text(xl, 9.2, lab, color=col, ha="center", fontsize=7)
    ax.axhline(3.84, color=REF, ls=":", lw=0.9, zorder=1)
    ax.text(1.98, 3.95, "95% cutoff", fontsize=6.0, color="#6E6E6E", ha="right", va="bottom")
    ax.set_ylim(0, 11); ax.set_xlim(0.4, 2.0); ax.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("parameter / MLE"); ax.set_ylabel(r"$2\Delta\ell$")


def plane_panel(ax):
    """kappa_hat per record on a log axis, one pipeline for all records (04_identifiability_phase):
    the point is the day-0 (or whole-record) fit, the bar is the range over observation days for the
    three SocioPatterns records. The fitted n_node is printed at the row end, not plotted (Sec. II G)."""
    from palette import DATASET_ORDER
    root = os.path.abspath(os.path.join(ROOT, "results"))
    fits = json.load(open(os.path.join(root, "record_fits.json")))
    perday = json.load(open(os.path.join(root, "record_perday.json")))
    vals = {}
    for nm in DATASET_ORDER:
        if nm not in fits:                                    # cortex: spike trains, no targets, no kappa
            continue
        f = fits[nm]
        rng = None
        if nm in perday and len(perday[nm]) > 1:
            k = [d["kappa"] for d in perday[nm] if d["events"] >= 1000]   # days with < 1000 events excluded
            rng = (min(k), max(k))
        vals[nm] = (f["n_node"], f["kappa"], rng)
    rows = [nm for nm in DATASET_ORDER if nm in vals]          # darkest record at the top
    y = np.arange(len(rows), dtype=float)[::-1]
    for yy, nm in zip(y, rows):
        x, kap, rng = vals[nm]; st = style(nm, 30)
        if rng:
            ax.plot(rng, [yy, yy], color=st["color"], lw=1.0, alpha=0.7, zorder=2)
        ax.scatter([kap], [yy], **st)
        ax.text(1.03, yy, f"{x:.2f}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.0, color="#8A8A8A")
    ax.set_yticks(y); ax.set_yticklabels([dlabel(nm) + (r"$^\dagger$" if fits[nm]["at_bounds"] else "") for nm in rows])
    for t, nm in zip(ax.get_yticklabels(), rows):
        t.set_color(style(nm)["color"])
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xscale("log"); ax.set_xlim(1, 1e4); ax.set_ylim(-0.7, y[0] + 1.3)
    ax.set_xlabel(r"fitted tie reinforcement $\hat\kappa$")
    ax.text(1.03, y[0] + 0.95, r"$\hat n_{\rm node}$", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=6.5, color="#8A8A8A")


def key_of(name):
    from palette import key
    return key(name)


# %% Execution
def main(outdir=None):
    # Ignore sys.argv if running in Jupyter/VS Code to avoid passing kernel args like '-f'
    if outdir is None:
        if not is_interactive and len(sys.argv) > 1:
            outdir = sys.argv[1]
        else:
            outdir = RES
            
    # Ensure the target directory exists before saving
    os.makedirs(outdir, exist_ok=True)
    
    plotstyle.apply()
    rows = load_rows()
    fig = plt.figure(figsize=(plotstyle.FULL, 4.1))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.45], hspace=0.55, wspace=0.42, top=0.95)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[0, 2])
    axd = fig.add_subplot(gs[1, 0:2])
    coverage_panel(axa, rows, "n_node_coverage", r"coverage of $n_{\rm node}$")
    coverage_panel(axb, rows, "kappa_coverage", r"coverage of $\kappa$")
    profiles_panel(axc)
    plane_panel(axd)
    h, l = axa.get_legend_handles_labels()
    axl = fig.add_subplot(gs[1, 2]); axl.axis("off")          # legend for (a, b) in the free cell
    axl.legend(h, l, loc="center left", bbox_to_anchor=(0.18, 0.62), title="observation window (a, b)",
               title_fontsize=6.5, handlelength=1.8, labelspacing=0.5, alignment="left")
    for ax, lab, x in ((axa, "a", -0.24), (axb, "b", -0.24), (axc, "c", -0.24), (axd, "d", -0.20)):
        plotstyle.panel(ax, lab, x=x, y=1.05)
    try:
        plotstyle.save(fig, os.path.join(ROOT, "figures", "fig_identifiability"))
        print(f"wrote fig_identifiability.pdf/.png to {outdir}")
    except Exception as e:
        print(f"Could not save using plotstyle.save (Error: {e}). Trying standard savefig...")
        fig.savefig(os.path.join(ROOT, "figures", "fig_identifiability.png"), dpi=300)
        fig.savefig(os.path.join(ROOT, "figures", "fig_identifiability.pdf"))
        
    if is_interactive:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

# %%
#!/usr/bin/env python3
"""Merged finite-size-scaling figure (Nature Fig 5).

(a) Shuffle ratio R(M) for the AR(1) generator at several events per node E (grey, light->dark),
    the semi-synthetic positive control (genuine exponential node memory on the high-school
    activity pattern, results/poscon_*), and every real record at its measured (M, R) with the
    5-95% interval (results/real_record_screen.json; one protocol for all records).
(b) The same R values per record as a dot plot (no overlapping markers); M printed at the row end.
    Cortex: full shuffle (filled) and local shuffle within 20 intervals (open).
(c) Width of the R(M) decline (M interval for R: 0.9->0.7) against E on log-log axes: a positive
    slope, i.e. a smooth crossover rather than a sharpening threshold.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
# %% Setup and Imports
import glob
import os
import sys

import numpy as np
import matplotlib

# Detect interactive environment to avoid freezing inline plots
is_interactive = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules
if not is_interactive:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(ROOT, "results")

from palette import COLORS, MARKERS  # noqa: E402
import plotstyle  # noqa: E402
import phase_combine as pc           # reuse REAL list (incl. CollegeMsg), ramp, MMIN  # noqa
from phase_width import width_at     # noqa: E402

# %% Panel Functions
import json  # noqa: E402
from palette import style, label as dlabel, family, DATASET_ORDER, NODE, REF  # noqa: E402

from palette import ramp  # noqa: E402
GREYS = ramp(5, lo=0.30, hi=0.80, hue=0, chroma=0.0)   # E increasing: dark -> bright
RESULTS = os.path.abspath(os.path.join(ROOT, "results"))


def load_real():
    d = json.load(open(os.path.join(RES, "real_record_screen.json")))
    return {k: v for k, v in d.items() if k in DATASET_ORDER}


def positive_control():
    """(M, R) of exponential node memory injected into the high-school activity pattern (rank M)."""
    Rj = json.load(open(os.path.join(RESULTS, "poscon_shuffle_R.json")))
    Mj = {round(r["true_n"], 2): r["M_rank"] for r in json.load(open(os.path.join(RESULTS, "poscon_memory_rank.json")))}
    pts = [(Mj[round(r["true_n"], 2)], np.mean([sh / re for re, sh in r["fits"]])) for r in Rj
           if round(r["true_n"], 2) in Mj]
    return np.array(sorted(pts))


def panel_scaling(ax, real):
    files = sorted(glob.glob(os.path.join(RES, "phase2_T*.npz")), key=lambda p: float(np.load(p)["Emean"]))
    for i, f in enumerate(files):
        d = np.load(f)
        M, R = d["M"], d["R"]; Rsem = d["Rsem"] if "Rsem" in d else np.zeros_like(R)
        ok = np.isfinite(M) & np.isfinite(R) & (M >= pc.MMIN)
        o = np.argsort(M[ok]); M, R, Rsem = M[ok][o], R[ok][o], Rsem[ok][o]
        col = GREYS[len(GREYS) - len(files) + i]
        ax.plot(M, R, "-", color=col, lw=1.0, zorder=2, label=f"{float(d['Emean']):.0f}")
    pcx = positive_control()
    (pch,) = ax.plot(pcx[:, 0], pcx[:, 1], "D-", color=NODE, ms=3.0, lw=1.0, markeredgecolor="white",
                     markeredgewidth=0.4, zorder=5)
    for name, v in real.items():
        ax.scatter([v["M"]], [v["R_exp"][0]], **style(name, 16))
    ax.axhline(1.0, color=REF, ls=":", lw=0.9, zorder=0)
    ax.set_xlim(-0.08, 0.86); ax.set_ylim(0.45, 1.05)
    ax.set_xlabel("memory coefficient $M$ (within-node rank)")
    ax.set_ylabel(r"$R=\hat n_{\rm node}^{\rm shuf}/\hat n_{\rm node}$")
    h, l = ax.get_legend_handles_labels()
    labels = ["positive control"] + [f"AR(1), $E={l[0]}$"] + [f"$E={x}$" for x in l[1:]]
    leg = ax.legend([pch] + h, labels, loc="lower left", fontsize=6.0, handlelength=1.4,
                    labelspacing=0.2)
    leg.get_texts()[0].set_color(NODE)


def panel_records(ax, real):
    rows = [n for n in DATASET_ORDER if n in real]           # darkest record at the top
    y = np.arange(len(rows), dtype=float)[::-1]
    ax.axvline(1.0, color=REF, ls=":", lw=0.9, zorder=0)
    for yy, n in zip(y, rows):
        v = real[n]; st = style(n, 22); col = st["color"]
        ax.plot([v["R_exp"][1], v["R_exp"][2]], [yy, yy], color=col, lw=1.0, alpha=0.8, zorder=2)
        ax.scatter([v["R_exp"][0]], [yy], **st)
        if "R_exp_local" in v:                                   # cortex: local shuffle, open marker
            lo, mid, hi = v["R_exp_local"][1], v["R_exp_local"][0], v["R_exp_local"][2]
            ax.plot([lo, hi], [yy - 0.42, yy - 0.42], color=col, lw=1.0, alpha=0.8, zorder=2)
            ax.scatter([mid], [yy - 0.42], s=22, marker="o", facecolors="white", edgecolors=col,
                       linewidths=0.9, zorder=4)
            ax.annotate("local shuffle", (lo, yy - 0.42), xytext=(-5, -1), textcoords="offset points",
                        ha="right", va="center", fontsize=6.0, color=col)
        ax.text(1.03, yy, f"{v['M']:+.2f}", transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=6.0, color="#8A8A8A")
    ax.text(1.03, y[0] + 0.95, "$M$", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=6.5, color="#8A8A8A")
    ax.set_yticks(y); ax.set_yticklabels([dlabel(n) for n in rows])
    for t, n in zip(ax.get_yticklabels(), rows):
        t.set_color(style(n)["color"])
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0.45, 1.05); ax.set_ylim(-0.9, y[0] + 1.3)
    ax.set_xlabel(r"shuffle ratio $R$ (5–95%)")


def panel_width(ax, nboot=400, seed=0):
    """Width per E, with a parametric bootstrap error bar propagated from the per-point SEM."""
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(os.path.join(RES, "phase2_T*.npz")), key=lambda p: float(np.load(p)["Emean"]))
    Es, Ws, Werr = [], [], []
    for f in files:
        d = np.load(f)
        M, R = d["M"], d["R"]; Rsem = d["Rsem"] if "Rsem" in d else np.zeros_like(R)
        Es.append(float(d["Emean"])); Ws.append(width_at(M, R))
        boot = [w for w in (width_at(M, R + rng.normal(0, np.nan_to_num(Rsem), size=R.shape))
                            for _ in range(nboot)) if np.isfinite(w)]
        Werr.append(float(np.std(boot)) if boot else 0.0)
    Es, Ws, Werr = np.array(Es), np.array(Ws), np.array(Werr)
    a, b = np.polyfit(np.log10(Es), np.log10(Ws), 1)
    xx = np.array([Es.min() * 0.8, Es.max() * 1.25])
    ax.plot(xx, 10 ** b * xx ** a, "-", color=REF, lw=1.0, zorder=1)
    ax.errorbar(Es, Ws, yerr=Werr, fmt="o", ms=3.5, color="#3A3A3A", elinewidth=0.8,
                markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([0.3, 0.4, 0.6, 0.8]))
    ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.text(0.96, 0.05, f"slope ${a:+.2f}$", transform=ax.transAxes, fontsize=6.5, va="bottom", ha="right")
    ax.set_xlabel("events per node $E$"); ax.set_ylabel(r"decline width in $M$")
    return a


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
    real = load_real()
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(plotstyle.FULL, 2.75),
                                        gridspec_kw=dict(width_ratios=[1.2, 1.15, 0.72], wspace=0.75))
    panel_scaling(axa, real)
    panel_records(axb, real)
    slope = panel_width(axc)
    print(f"width log-log slope = {slope:+.2f} (>=0 => crossover)")
    for ax, lab, x in ((axa, "a", -0.22), (axb, "b", -0.52), (axc, "c", -0.34)):
        plotstyle.panel(ax, lab, x=x, y=1.03)
    try:
        plotstyle.save(fig, os.path.join(ROOT, "figures", "fig_finitesize"))
        print(f"wrote fig_finitesize.pdf/.png to {outdir}")
    except Exception as e:
        print(f"Could not save using plotstyle.save (Error: {e}). Trying standard savefig...")
        fig.savefig(os.path.join(ROOT, "figures", "fig_finitesize.png"), dpi=300)
        fig.savefig(os.path.join(ROOT, "figures", "fig_finitesize.pdf"))
        
    if is_interactive:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

# %%
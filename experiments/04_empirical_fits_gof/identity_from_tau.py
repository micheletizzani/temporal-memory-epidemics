#!/usr/bin/env python3
"""Rebuild the KS = zero-atom identity figure from cached residuals, all datasets.

Two panels sharing one bottom legend:
  left  : KS statistic vs zero-atom fraction. Datasets WITH simultaneous events fall on the
          diagonal -- their entire marginal misfit IS the atom. Directed datasets with no
          atom (brain, CollegeMsg) sit ABOVE it: their misfit is a residual SHAPE defect
          (neural regularity / over-dispersion), not simultaneity.
  right : Var(tau) vs zero-atom fraction. A correct model has Var(tau)=1; points off that
          line carry a second defect (email and the SNN are over-dispersed).

Built at full (\\textwidth) physical width -- this figure must be placed in a figure*.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
# %%
import os
import sys

import numpy as np
import matplotlib

# Detect interactive environment to avoid freezing inline plots
is_interactive = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules
if not is_interactive:
    matplotlib.use("Agg")
    
import matplotlib.pyplot as plt

from palette import COLORS, MARKERS  # noqa: E402
import plotstyle  # noqa: E402
import inference as inf  # noqa: E402
RES = os.path.join(ROOT, "results")

SPEC = [
    ("tau_hospital.npy", "Hospital ward", False),
    ("tau_school.npy",   "High school", False),
    ("tau_sfhh.npy",     "Conference (SFHH)", False),
    ("tau_baboons.npy", "Baboons", False),
    ("tau_email.npy",    "Email (EU institution)", False),
    ("tau_collegemsg.npy", "CollegeMsg (Direct Msgs)", False),
    ("tau_stackoverflow.npy", "Stack Overflow", False),
    ("tau_wikibooks.npy", "Wikibooks", False),
    ("tau_wikibooks_bots.npy", "Wikibooks bots", False),       # rebuilt record (R1)
    ("tau_microservices.npy", "Microservices", False),
    ("tau_snn_bursty.npy", "SNN (bursting)", False),
    ("tau_cortex_spikes.npy", "Cortex", False),                 # spike trains; replaces tau_brain
]
# direct-label offsets (points): panel a, panel b
OFF = {
    "Hospital ward": ((7, -4, "left"), (6, -6, "left")),
    "High school": ((7, 2, "left"), (6, 5, "left")),
    "Conference (SFHH)": ((-6, 6, "right"), (6, -4, "left")),
    "Baboons": ((-6, 0, "right"), (6, 3, "left")),
    "Email (EU institution)": ((7, 0, "left"), (6, 0, "left")),
    "CollegeMsg (Direct Msgs)": ((7, 0, "left"), (7, 0, "left")),
    "Stack Overflow": ((7, 0, "left"), (7, 0, "left")),
    "Wikibooks": ((7, 0, "left"), (7, 0, "left")),
    "Wikibooks bots": ((7, 0, "left"), (7, 0, "left")),
    "Microservices": ((-7, 0, "right"), (6, 0, "left")),
    "SNN (bursting)": ((-7, 0, "right"), (-7, 0, "right")),
    "Cortex": ((7, 0, "left"), (7, -1, "left")),
}
ATOM_TOL = 0.02   # datasets with atom below this are "no-atom" (directed) -> off-diagonal


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
    pts = []
    for fn, lab, bound in SPEC:
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            print(f"  missing {p}"); continue
        tau = np.asarray(np.load(p), float)
        atom = float((tau <= 1e-9).mean())
        ks = float(inf._ks_uniform(1.0 - np.exp(-tau)))
        var = float(tau.var())
        pts.append((lab, bound, atom, ks, var))
        print(f"  {lab:16s} atom={atom:.3f} KS={ks:.3f} Var={var:.2f} bound={bound}")

    from palette import style, label as dlabel, family, REF, FAMILY
    # dot plot: one row per record, grouped by family (no overlapping markers)
    from palette import DATASET_ORDER
    order = {n: k for k, n in enumerate(DATASET_ORDER)}
    rows = sorted(pts, key=lambda r: order.get(r[0], 99))        # darkest record at the top
    y = np.arange(len(rows), dtype=float)[::-1]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(plotstyle.FULL, 2.9), sharey=True,
                                   gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.08))
    for yy, (lab, bound, atom, ks, var) in zip(y, rows):
        st = style(lab, 26); col = st["color"]
        axL.plot([atom, ks], [yy, yy], color=col, lw=1.0, alpha=0.6, zorder=2)
        axL.scatter([atom], [yy], s=26, marker="o", facecolors="white", edgecolors=col,
                    linewidths=0.9, zorder=3)
        axL.scatter([ks], [yy], s=26, marker="o", color=col, edgecolors="white", linewidths=0.4, zorder=4)
        axR.scatter([var], [yy], s=26, marker="o", color=col, edgecolors="white", linewidths=0.4, zorder=4)
        axR.plot([1.0, var], [yy, yy], color=col, lw=1.0, alpha=0.6, zorder=2)
    axL.set_yticks(y); axL.set_yticklabels([dlabel(r[0]) for r in rows])
    for t, r in zip(axL.get_yticklabels(), rows):
        t.set_color(style(r[0])["color"])
    axL.tick_params(axis="y", length=0); axL.spines["left"].set_visible(False)
    axR.tick_params(axis="y", length=0, labelleft=False); axR.spines["left"].set_visible(False)
    axL.set_xlim(-0.02, 1.0); axL.set_xlabel("zero-atom fraction (open), KS statistic (filled)")
    axR.set_xscale("log"); axR.set_xlim(0.5, 400); axR.axvline(1.0, color=REF, ls=":", lw=0.9, zorder=1)
    axR.set_xlabel(r"Var$(\tau)$ of rescaled times (correct model: 1)")
    top = y[0] + 0.9
    axL.set_ylim(-0.7, top + 0.6)
    axL.scatter([0.55], [top], s=26, facecolors="white", edgecolors="#555", linewidths=0.9)
    axL.text(0.575, top, "zero atom", va="center", fontsize=6.5, color="#555")
    axL.scatter([0.78], [top], s=26, color="#555", edgecolors="white", linewidths=0.4)
    axL.text(0.805, top, "KS statistic", va="center", fontsize=6.5, color="#555")
    plotstyle.panel(axL, "a", x=-0.30, y=1.0); plotstyle.panel(axR, "b", x=-0.04, y=1.0)
    plotstyle.save(fig, os.path.join(ROOT, "figures", "fig1_identity"))
    print(f"wrote fig1_identity.pdf/.png to {outdir} ({len(pts)} datasets)")
    
    if is_interactive:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()

# %%
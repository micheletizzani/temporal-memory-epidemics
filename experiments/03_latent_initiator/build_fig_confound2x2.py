#!/usr/bin/env python3
"""Real-data 2x2 confound on directed communication data (Stack Overflow core).

Honest framing: the missing initiator confounds the SELECTION channel. As the label-reveal
fraction rho -> 1, (a) the across-imputation spread sigma(kappa) collapses and (b) the point
estimate kappa de-biases toward its observed-initiator value. The NODE channel is pinned
(sigma(n) ~ 0), so this is a selection-channel missing-information effect, not a node-tie
confound; the correlation corr(n,kappa) is NOT reported because it is ill-conditioned when the
node spread vanishes.

Usage:  python build_fig_confound2x2.py stackoverflow
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
KCOL = "#1F3B63"


def main():
    tags = sys.argv[1:] or ["stackoverflow", "wikibooks"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plotstyle.apply()
    figdir = os.path.abspath(os.path.join(ROOT, "figures"))
    from palette import style, REF
    fig, (ax1, bx1) = plt.subplots(1, 2, figsize=(plotstyle.FULL * 0.86, 2.2), gridspec_kw=dict(wspace=0.35))
    for tag in tags:
        p = os.path.join(RES, f"confound2x2_{tag}.json")
        if not os.path.exists(p):
            print("missing", p); continue
        d = json.load(open(p)); rows = sorted(d["rows"], key=lambda r: r["rho"])
        rho = np.array([r["rho"] for r in rows]); sk = np.array([r["sk"] for r in rows])
        kh = np.array([r["khat"] for r in rows])
        name = "Stack Overflow" if "stack" in tag.lower() else "Wikibooks"
        st = style(name); col, mk = st["color"], st["marker"]
        kw = dict(color=col, marker=mk, ms=4.5 if mk == "*" else 3.6, lw=1.1, markeredgecolor="white",
                  markeredgewidth=0.4)
        ax1.plot(rho, sk / sk[0], label=name, **kw)
        bx1.plot(rho, 100 * (kh / kh[-1] - 1), label=name, **kw)
    ax1.set_ylabel(r"spread of $\hat\kappa$ (relative to $\rho=0$)"); ax1.set_ylim(0, 1.05)
    bx1.axhline(0, color=REF, ls=":", lw=0.9)
    bx1.set_ylabel(r"inflation of $\hat\kappa$ (%)")
    for ax in (ax1, bx1):
        ax.set_xlabel(r"initiator labels revealed, $\rho$")
    ax1.legend(loc="upper right")
    plotstyle.panel(ax1, "a", x=-0.20, y=1.03); plotstyle.panel(bx1, "b", x=-0.20, y=1.03)
    plotstyle.save(fig, os.path.join(figdir, "confound2x2_real"))
    plt.close(fig)
    print("wrote confound2x2_real.pdf/.png")


if __name__ == "__main__":
    main()

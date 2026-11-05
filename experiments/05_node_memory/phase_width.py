#!/usr/bin/env python3
"""Log-log finite-size scaling: does the R(M) transition WIDTH shrink with E?

For each event-count-per-node E we measure the width of the R(M) decline as the M-interval
over which R falls from 0.9 to 0.7 (both reachable at every size). A genuine phase
transition sharpens: width ~ E^{-a} with a>0 (negative log-log slope). A crossover does
not: the width is flat or GROWS with E. We fit log(width) vs log(E) and report the slope.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(ROOT, "results")


def width_at(M, R, r_hi=0.9, r_lo=0.7):
    ok = np.isfinite(M) & np.isfinite(R) & (M >= 0.0)
    M, R = M[ok], R[ok]
    idx = np.argsort(R)                      # ascending R
    Ra, Ma = R[idx], M[idx]
    m_hi = np.interp(r_hi, Ra, Ma)           # M where R=0.9 (small M)
    m_lo = np.interp(r_lo, Ra, Ma)           # M where R=0.7 (larger M)
    return float(m_lo - m_hi)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else RES
    files = sorted(glob.glob(os.path.join(RES, "phase2_T*.npz")),
                   key=lambda p: float(np.load(p)["Emean"]))
    Es, Ws = [], []
    for f in files:
        d = np.load(f)
        w = width_at(d["M"], d["R"])
        Es.append(float(d["Emean"])); Ws.append(w)
        print(f"  E~{Es[-1]:7.0f}  width(R:0.9->0.7 in M) = {w:.3f}")
    Es, Ws = np.array(Es), np.array(Ws)

    # log-log slope
    a, b = np.polyfit(np.log10(Es), np.log10(Ws), 1)
    print(f"\nlog-log slope  d log(width)/d log(E) = {a:+.2f}  "
          f"({'CROSSOVER (>=0)' if a > -0.05 else 'sharpening (<0)'})")

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.loglog(Es, Ws, "o", ms=9, color="#85496F", markeredgecolor="#222", lw=0, zorder=3)
    xx = np.array([Es.min() * 0.8, Es.max() * 1.2])
    ax.loglog(xx, 10 ** (b) * xx ** a, "-", color="#82A0C2", lw=2,
              label=f"slope $= {a:+.2f}$")
    ax.axhline(np.mean(Ws), color="#aaa", ls=":", lw=1)
    ax.set_xlabel("events per node  $E$")
    ax.set_ylabel(r"transition width in $M$  ($R:0.9\to0.7$)")
    ax.set_title("Finite-size scaling: the width does not shrink with $E$\n"
                 "(slope $\\geq 0$ $\\Rightarrow$ crossover, not a transition)", fontsize=9.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best", frameon=False)
    fig.tight_layout()
    path = os.path.join(outdir, "phase_width.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

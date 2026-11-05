#!/usr/bin/env python3
"""Figure: Proposition 2 (weighted) demonstrated. An initiator-attributed mark joins the
node/tie confound under a latent initiator (its parameter spread collapses as labels are
revealed); a pair-attributed mark stays decoupled.

Incremental compute (background-friendly):
  python build_fig_mark.py compute 1 0.0,0.25,0.5,0.75,1.0
  python build_fig_mark.py plot
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os
import sys

import numpy as np

import mark_confound as mc  # noqa: E402
RES = os.path.join(ROOT, "results", "mark_sim.json")


def compute(seed, rhos):
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    for mode in ("init", "pair"):
        ev, w, N, T = mc.make_marked(mode, seed=seed)
        for rho in rhos:
            D = mc.confound_3x3(ev, w, N, T, rho, np.random.default_rng(7 + seed),
                                mark_model=mode, seed=seed)
            res[f"{seed}|{mode}|{rho:.2f}"] = dict(
                sig=float(D[:, 2].std()), sig_n=float(D[:, 0].std()),
                cov_ncw=float(np.cov(D[:, 0], D[:, 2])[0, 1]),   # missing-information off-diagonals
                cov_kcw=float(np.cov(D[:, 1], D[:, 2])[0, 1]),   # (covariance, not corr: vanishes when spread->0)
                cnk=mc._corr(D, 0, 1), cncw=mc._corr(D, 0, 2), ckcw=mc._corr(D, 1, 2))
            print(f"seed={seed} {mode} rho={rho:.2f} sig(cW)={res[f'{seed}|{mode}|{rho:.2f}']['sig']:.3f}", flush=True)
    os.makedirs(os.path.dirname(RES), exist_ok=True)
    json.dump(res, open(RES, "w"), indent=2)


def _agg(res, mode, rhos, key):
    seeds = sorted({k.split("|")[0] for k in res})
    M = np.full((len(seeds), len(rhos)), np.nan)
    for a, s in enumerate(seeds):
        for b, rho in enumerate(rhos):
            k = f"{s}|{mode}|{rho:.2f}"
            if k in res:
                M[a, b] = res[k][key]
    return np.nanmean(M, 0), np.nanstd(M, 0) / np.sqrt(np.maximum((~np.isnan(M)).sum(0), 1))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotstyle
    plotstyle.apply()
    res = json.load(open(RES))
    rhos = sorted({float(k.split("|")[2]) for k in res})
    figdir = os.path.abspath(os.path.join(ROOT, "figures"))
    from palette import NODE as INIT, TIE as PAIR
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(plotstyle.FULL * 0.86, 2.2), gridspec_kw=dict(wspace=0.38))

    # (a) spread of the mark parameter vs label reveal fraction
    for mode, col, mk, lab in [("init", INIT, "o", r"initiator-attributed ($H_i$)"),
                               ("pair", PAIR, "s", r"pair-attributed ($H_{ij}$)")]:
        m, e = _agg(res, mode, rhos, "sig")
        ax.errorbar(rhos, m, yerr=e, fmt=mk + "-", color=col, ms=3.6, lw=1.1, capsize=0, elinewidth=0.8,
                    markeredgecolor="white", markeredgewidth=0.4, label=lab)
    ax.set_xlabel(r"initiator labels revealed, $\rho$")
    ax.set_ylabel(r"spread of the mark parameter $\hat c_W$")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")

    # (b) missing-information off-diagonals |cov(.,c_W)| at rho=0: nonzero only for H_i marks
    r0 = f"{rhos[0]:.2f}"
    labels = [r"$|\mathrm{cov}(\hat n,\hat c_W)|$", r"$|\mathrm{cov}(\hat\kappa,\hat c_W)|$"]
    keys = ["cov_ncw", "cov_kcw"]
    xi = np.arange(2)
    for off, mode, col, lab in [(-0.18, "init", INIT, r"initiator-attr. ($H_i$)"),
                                (0.18, "pair", PAIR, r"pair-attr. ($H_{ij}$)")]:
        vals = [np.nanmean([abs(res[k][key]) for k in res
                            if k.split("|")[1] == mode and k.endswith(r0) and key in res[k]]) for key in keys]
        bx.bar(xi + off, vals, width=0.34, color=col, edgecolor="white", linewidth=0.5, label=lab)
    bx.set_xticks(xi); bx.set_xticklabels(labels, fontsize=6.5); bx.tick_params(axis="x", length=0)
    bx.set_ylabel(r"off-diagonal $|\mathrm{cov}|$ at $\rho=0$")
    plotstyle.panel(ax, "a", x=-0.22, y=1.03); plotstyle.panel(bx, "b", x=-0.22, y=1.03)
    plotstyle.save(fig, os.path.join(figdir, "mark_confound"))
    plt.close(fig)
    print("wrote mark_confound.pdf/.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compute":
        compute(int(sys.argv[2]), [float(x) for x in sys.argv[3].split(",")])
    else:
        plot()

#!/usr/bin/env python3
"""Figure: the latent-initiator confound is the missing-information effect, and it is an
information-theoretic order parameter -- revealing a fraction rho of true initiator labels
tunes it continuously to zero.

(a) The across-imputation spread of kappa_hat (the missing-information magnitude) collapses as
    labels are revealed, tracking the label-entropy order parameter (1-rho) H_bar.
(b) Both point estimates are de-confounded: the latent label biases n_hat DOWN and kappa_hat UP;
    revealing labels removes the bias monotonically. Averaged over 3 realizations (mean +/- s.e.).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os
import sys

import numpy as np

import plotstyle  # noqa


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plotstyle.apply()
    res = json.load(open(os.path.join(ROOT, "results", "confound_rho.json")))
    seeds = sorted(res.keys())
    rhos = sorted({float(r) for s in seeds for r in res[s]})
    def stack(key):
        M = np.full((len(seeds), len(rhos)), np.nan)
        for a, s in enumerate(seeds):
            for b, rho in enumerate(rhos):
                k = f"{rho:.2f}"
                if k in res[s]:
                    M[a, b] = res[s][k][key]
        return M
    sk = stack("sk"); H = stack("entropy"); nh = stack("nhat"); kh = stack("khat")
    rho = np.array(rhos)
    def ms(M):
        return np.nanmean(M, 0), np.nanstd(M, 0) / np.sqrt(np.sum(~np.isnan(M), 0))

    figdir = os.path.abspath(os.path.join(ROOT, "figures"))
    from palette import NODE, TIE, INK, REF
    fig, axs = plt.subplots(1, 3, figsize=(plotstyle.FULL * 0.86, 1.75))
    err = dict(capsize=0, elinewidth=0.8, markeredgecolor="white", markeredgewidth=0.5, ms=4.2, lw=1.2)
    xl = r"labels revealed, $\rho$"

    # (a) missing-information spread and the (1-rho) H_bar prediction (scaled to rho = 0)
    ax = axs[0]
    m, e = ms(sk)
    op = (1 - rho) * np.nanmean(H, 0); op = op / op[0] * m[0]
    ax.plot(rho, op, "--", color=REF, lw=1.0, zorder=1)
    ax.errorbar(rho, m, yerr=e, fmt="o-", color=TIE, zorder=3, **err)
    ax.text(0.97, 0.93, r"$(1-\rho)\,\bar H$", transform=ax.transAxes, ha="right", va="top",
            color="#6E6E6E", fontsize=6.5)
    ax.set_ylabel(r"spread of $\hat\kappa$ across imputations")
    ax.set_ylim(0, 0.65)

    # (b) n_hat, (c) kappa_hat: each against its truth, no twin axis
    for ax, M, truth, col, mk, lab in ((axs[1], nh, 0.5, NODE, "s", r"$\hat n_{\rm node}$"),
                                       (axs[2], kh, 10.0, TIE, "^", r"$\hat\kappa$")):
        mm, ee = ms(M)
        ax.axhline(truth, color=REF, ls=":", lw=0.9, zorder=1)
        ax.errorbar(rho, mm, yerr=ee, fmt=mk + "-", color=col, zorder=3, **err)
        ax.text(0.02, truth, " truth", transform=ax.get_yaxis_transform(), color="#6E6E6E",
                fontsize=6.5, va="bottom")
        ax.set_ylabel(lab)
    axs[1].set_ylim(0.34, 0.52); axs[2].set_ylim(9.5, 14.0)

    for k, ax in enumerate(axs):
        ax.set_xlabel(xl); ax.set_xticks([0, 0.25, 0.5, 0.75]); ax.set_xlim(-0.05, 0.8)
        plotstyle.panel(ax, "abc"[k], x=-0.30)
    fig.tight_layout(w_pad=1.6)
    plotstyle.save(fig, os.path.join(figdir, "confound_orderparam"))
    plt.close(fig)
    print("wrote confound_orderparam.pdf/.png ; seeds:", seeds, "rhos:", rhos)


if __name__ == "__main__":
    main()

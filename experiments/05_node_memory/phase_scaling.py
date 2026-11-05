#!/usr/bin/env python3
"""Finite-size scaling test: is the shuffle-preservation R a transition or a crossover?

We sweep the injected-memory knob rho (AR(1) coefficient on the log-inter-event time,
which sets the memory coefficient M; M is the rank coefficient of memory_stats.py) and, crucially, repeat the sweep at several
events-per-node counts E (by varying the observation window T). For each (rho, E) cell we
fit n_node on the real train and on the inter-event-order-shuffled train and record

    R(rho, E) = mean over reps of n_node_shuf / n_node_real.

Interpretation (the only rigorous way to call it a transition):
  * TRANSITION  -> R(rho) steepens as E grows; the curves cross at a single rho* and the
                   maximum slope dR/drho diverges with E (finite-size sharpening).
  * CROSSOVER   -> the R(rho) curves collapse onto one E-independent smooth curve; the
                   width of the change does not shrink with E.

Outputs results/phase_scaling.npz and results/phase_scaling.png.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import sys

import numpy as np

import inference as inf  # noqa: E402
from plane_validation import gen_train, burstiness_memory, fit_nnode_single, shuffle_isi  # noqa: E402
from memory_stats import rank_memory_train  # noqa: E402


def cell(T, rho, sigma, reps, base_rate, seed):
    """Monte-Carlo over `reps` independent realisations; return means AND standard errors."""
    Rs, Ms, Es = [], [], []
    for rep in range(reps):
        rng = np.random.default_rng(abs(seed + 7919 * rep + int(1000 * rho)))
        ts = gen_train(T, base_rate, rho, sigma, 0.0, rng)
        if len(ts) < 40:
            continue
        M = rank_memory_train(ts)                 # headline memory coefficient (within-node rank)
        n_real = fit_nnode_single(ts, T)
        n_shuf = fit_nnode_single(shuffle_isi(ts, rng), T)
        if not (np.isfinite(n_real) and np.isfinite(n_shuf)) or n_real < 5e-2:
            continue
        Rs.append(n_shuf / n_real); Ms.append(M); Es.append(len(ts))
    if not Rs:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0
    Rs, Ms = np.array(Rs), np.array(Ms)
    rsem = float(Rs.std(ddof=1) / np.sqrt(len(Rs))) if len(Rs) > 1 else np.nan
    msem = float(Ms.std(ddof=1) / np.sqrt(len(Ms))) if len(Ms) > 1 else np.nan
    return float(Rs.mean()), rsem, float(Ms.mean()), msem, float(np.mean(Es)), len(Rs)


def main():
    # phase_scaling.py <T> <reps>  -> compute ONE size, cache results/phase_T<T>.npz.
    # Resumable: re-running continues the rho values not yet filled in the cache.
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    base_rate = 1.0
    sigma = 0.5
    rhos = np.linspace(-0.85, 0.92, 18)        # finer memory grid (was 10 points)
    outdir = os.path.join(ROOT, "results")
    os.makedirs(outdir, exist_ok=True)
    cache = os.path.join(outdir, f"phase2_T{int(T)}.npz")

    R = np.full(len(rhos), np.nan); Rsem = np.full(len(rhos), np.nan)
    M = np.full(len(rhos), np.nan); Msem = np.full(len(rhos), np.nan)
    if os.path.exists(cache):                  # resume if grid matches
        d = np.load(cache)
        if d["rhos"].shape == rhos.shape and np.allclose(d["rhos"], rhos):
            R, M = d["R"].copy(), d["M"].copy()
            Rsem = d["Rsem"].copy() if "Rsem" in d else Rsem
            Msem = d["Msem"].copy() if "Msem" in d else Msem
    es = []
    for ir, rho in enumerate(rhos):
        if np.isfinite(R[ir]):
            print(f"  T={T:7.0f} rho={rho:+.2f} | cached R={R[ir]:.3f}")
            continue
        r, rsem, m, msem, e, ok = cell(T, rho, sigma, reps, base_rate, seed=100 + int(T))
        R[ir], Rsem[ir], M[ir], Msem[ir] = r, rsem, m, msem
        if np.isfinite(e):
            es.append(e)
        print(f"  T={T:7.0f} rho={rho:+.2f} | E~{e:6.0f} M={m:+.3f} "
              f"R={r:.3f}+/-{rsem:.3f} (ok={ok})")
        emean = float(np.nanmean(es)) if es else float(np.load(cache)["Emean"]) if os.path.exists(cache) else np.nan
        np.savez(cache, R=R, Rsem=Rsem, M=M, Msem=Msem, rhos=rhos, T=T, Emean=emean)
    print(f"  saved {os.path.basename(cache)}")


def _plot(R, M, rhos, Emean, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    SEQ = ["#82A0C2", "#A8C4CC", "#F1A890", "#85496F"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for it in range(R.shape[0]):
        ax.plot(M[it], R[it], "-o", color=SEQ[it % len(SEQ)], lw=2,
                markeredgecolor="#333", markeredgewidth=0.4,
                label=f"E $\\approx$ {Emean[it]:.0f} events/node")
    ax.axhline(0.5, color="#777", ls=":", lw=1)
    ax.set_xlabel("memory coefficient  $M$")
    ax.set_ylabel(r"shuffle-preservation  $R$")
    ax.set_title("Finite-size scaling of $R$: do the curves sharpen (transition)\n"
                 "or collapse (crossover) as events/node grow?", fontsize=10)
    ax.grid(alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    path = os.path.join(outdir, "phase_scaling.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Single-realisation synthetic check: can the A and B mechanisms be told apart?

Runs four things on one simulated temporal network:

  1. MLE with the initiator observed          -> point estimates + asymptotic SEs
  2. MLE with the initiator latent            -> the empirical-data situation
  3. Likelihood-ratio tests of the two nulls  -> kappa = 0 (pure A), n_node = 0 (no A-memory)
  4. Profile negative log-likelihoods         -> the curvature that decides identifiability

The contrast between (1) and (2) is the point of the experiment. With initiator labels
the timing and selection blocks of the likelihood share no parameters, so node memory
and tie memory are estimated from disjoint statistics and cannot trade off. Without
labels they mix, and the across-imputation correlation reported by (2) measures how
badly.

Caveat that must be carried into any writeup: part of the gap between (1) and (2) is
the sequential-imputation approximation in the Monte Carlo EM, not pure information
loss. Do not attribute the whole difference to the missing labels.

Usage
-----
    python scripts/run_synthetic_recovery.py
    python scripts/run_synthetic_recovery.py --N 60 --T 900 --n-node 0.6 --kappa 10
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path

import argparse
import os
import sys
import time

import numpy as np


from generative_models import (  # noqa: E402
    MechanismParams, TemporalNetworkModel, simulate, strip_initiators, summarise)
from inference import (  # noqa: E402
    _split_events, fit_latent_initiator, fit_observed_initiator,
    likelihood_ratio_test, profile_likelihood, selection_nll, timing_nll)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=40, help="number of nodes")
    p.add_argument("--T", type=float, default=600.0, help="observation window")
    p.add_argument("--n-node", type=float, default=0.5,
                   help="true node branching ratio in [0, 1)")
    p.add_argument("--gamma-n", type=float, default=0.5, help="true node memory decay")
    p.add_argument("--kappa", type=float, default=20.0, help="true tie reinforcement")
    p.add_argument("--gamma-e", type=float, default=0.05, help="true tie memory decay")
    p.add_argument("--seed", type=int, default=7, help="simulation seed")
    p.add_argument("--skip-latent", action="store_true",
                   help="skip the (slower) latent-initiator fit")
    p.add_argument("--outdir", type=str, default=None,
                   help="directory for the profile-likelihood figure and CSVs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    truth = MechanismParams(args.n_node, args.gamma_n, args.kappa, args.gamma_e)
    activities = 0.02 + 0.03 * rng.random(args.N)
    model = TemporalNetworkModel(args.N, activities, truth)

    print("=" * 78)
    print("A-vs-B identifiability: single-realisation recovery")
    print("=" * 78)
    print(f"truth: n_node={truth.n_node}  gamma_n={truth.gamma_n}  "
          f"kappa={truth.kappa}  gamma_e={truth.gamma_e}")
    print(f"       N={args.N}  T={args.T}  seed={args.seed}")

    t0 = time.time()
    events = simulate(model, args.T, rng=np.random.default_rng(args.seed))
    stats = summarise(events, args.N, args.T)
    print(f"\nsimulated {int(stats['n_events'])} events in {time.time() - t0:.1f}s")
    print(f"  event rate      = {stats['rate']:.3f} / time unit")
    print(f"  distinct dyads  = {int(stats['n_distinct_dyads'])} "
          f"({stats['dyad_saturation']:.1%} of all pairs)")
    print(f"  repeat fraction = {stats['repeat_fraction']:.3f}")
    if stats["n_events"] < 200:
        print("  WARNING: fewer than 200 events; estimates will be very noisy. "
              "Increase --T or the baseline activities.")

    # ---------------------------------------------------------------- 1. observed
    print("\n" + "-" * 78)
    print("1. Initiator OBSERVED (exact MLE, activities profiled out)")
    print("-" * 78)
    t0 = time.time()
    res_obs = fit_observed_initiator(events, args.N, args.T)
    print(res_obs.summary())
    print(f"({time.time() - t0:.1f}s)")

    for name, truth_val, est in [("n_node", truth.n_node, res_obs.n_node),
                                 ("kappa", truth.kappa, res_obs.kappa)]:
        se = res_obs.se[name]
        if np.isfinite(se) and se > 0:
            print(f"  {name}: (est - truth)/se = {(est - truth_val) / se:+.2f}")

    # ---------------------------------------------------------------- 3. LR tests
    print("\n" + "-" * 78)
    print("3. Likelihood-ratio tests against the two boundary nulls")
    print("-" * 78)
    contacts, initiator, initiator_side = _split_events(events)

    nll_k0 = selection_nll(0.0, res_obs.gamma_e, contacts, initiator_side, args.N)
    print(f"  H0: kappa = 0  (pure node-driven partner choice)  "
          f"p = {likelihood_ratio_test(res_obs.nll_selection, nll_k0, 1):.3g}")
    nll_n0 = timing_nll(0.0, res_obs.gamma_n, contacts, initiator, args.N, args.T)
    print(f"  H0: n_node = 0 (memoryless node activity)         "
          f"p = {likelihood_ratio_test(res_obs.nll_timing, nll_n0, 1):.3g}")
    print("  (both nulls sit on the parameter-space boundary, so these p-values are")
    print("   conservative by roughly a factor of two; see likelihood_ratio_test)")

    # ---------------------------------------------------------------- 4. profiles
    print("\n" + "-" * 78)
    print("4. Profile negative log-likelihoods (curvature = identifiability)")
    print("-" * 78)
    # linear, dense grids centred so the MLE sits mid-range -> smooth, evenly spaced curves
    n_grid = np.linspace(0.0, 0.99, 60)
    k_grid = np.linspace(0.0, 2.2 * max(truth.kappa, 1.0), 60)
    prof_n = profile_likelihood(events, args.N, args.T, "n_node", n_grid)
    prof_k = profile_likelihood(events, args.N, args.T, "kappa", k_grid)

    def report(label: str, grid: np.ndarray, prof: np.ndarray, truth_val: float) -> None:
        prof = prof - np.nanmin(prof)
        inside = grid[prof < 0.5 * 3.841]   # 95% profile-likelihood interval
        lo, hi = (inside.min(), inside.max()) if len(inside) else (np.nan, np.nan)
        note = ""
        if len(inside) <= 1:
            note = "  (CI narrower than the grid spacing; refine the grid to resolve it)"
        print(f"  {label}: argmin = {grid[np.nanargmin(prof)]:.3f}, "
              f"truth = {truth_val:.3f}, 95% profile CI = [{lo:.3f}, {hi:.3f}]{note}")
        print(f"      dNLL at grid ends: {prof[0]:.1f}, {prof[-1]:.1f} "
              "(small = flat profile = not identified)")

    report("n_node", n_grid, prof_n, truth.n_node)
    report("kappa ", k_grid, prof_k, truth.kappa)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        np.savetxt(os.path.join(args.outdir, "profile_n_node.csv"),
                   np.column_stack([n_grid, prof_n]), delimiter=",",
                   header="n_node,nll", comments="")
        np.savetxt(os.path.join(args.outdir, "profile_kappa.csv"),
                   np.column_stack([k_grid, prof_k]), delimiter=",",
                   header="kappa,nll", comments="")
        _plot_profiles(args.outdir, n_grid, prof_n, k_grid, prof_k, truth)
        print(f"\n  profiles written to {args.outdir}")

    # ---------------------------------------------------------------- 2. latent
    if not args.skip_latent:
        print("\n" + "-" * 78)
        print("2. Initiator LATENT (Monte Carlo EM) -- the empirical-data situation")
        print("-" * 78)
        t0 = time.time()
        res_lat = fit_latent_initiator(strip_initiators(events), args.N, args.T,
                                       n_imputations=4, n_em=6, seed=args.seed + 1)
        print(res_lat.summary())
        print(f"({time.time() - t0:.1f}s)")

        print("\n  cost of the missing initiator label:")
        for name, obs, lat in [("n_node", res_obs.n_node, res_lat.n_node),
                               ("kappa", res_obs.kappa, res_lat.kappa)]:
            se_o, se_l = res_obs.se[name], res_lat.se[name]
            ratio = (se_l / se_o if np.isfinite(se_o) and se_o > 0 and np.isfinite(se_l)
                     else float("nan"))
            print(f"    {name:7s} observed {obs:8.3f}   latent {lat:8.3f}   "
                  f"se inflation x{ratio:.2f}")
        print(f"    corr(n_node, kappa): {res_obs.corr_n_kappa:+.3f} "
              f"(exactly 0 by construction) -> {res_lat.corr_n_kappa:+.3f}")
        print("\n  CAVEAT: this gap mixes genuine information loss with the sequential-")
        print("  imputation approximation in the EM. Separating the two would need an")
        print("  exact sampler over labels, which is not implemented here.")

    print("\n" + "=" * 78)


def _plot_profiles(outdir: str, n_grid, prof_n, k_grid, prof_k, truth) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available; skipping figure)")
        return
    import sys as _sys
    pass
    from palette import COLORS  # noqa: E402
    cutoff = "#F1A890"
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ax2 = ax1.twiny()
    
    # n_node on bottom axis (ax1)
    prof1 = prof_n - np.nanmin(prof_n)
    ax1.plot(n_grid, prof1, "-o", ms=4, color=COLORS["School"], lw=2, label=r"$n_{\rm node}$ profile")
    ax1.axvline(truth.n_node, color=COLORS["School"], ls="--", lw=1.5, label=r"truth ($n_{\rm node}$)")
    
    # kappa on top axis (ax2)
    prof2 = prof_k - np.nanmin(prof_k)
    ax2.plot(k_grid, prof2, "-s", ms=4, color=COLORS["SNN"], lw=2, label=r"$\kappa$ profile")
    ax2.axvline(truth.kappa, color=COLORS["SNN"], ls="--", lw=1.5, label=r"truth ($\kappa$)")
    
    # 95% cutoff line
    ax1.axhline(0.5 * 3.841, color=cutoff, ls=":", lw=1.6, label="95% cutoff")
    
    ax1.set_xlabel(r"Node branching ratio $n_{\rm node}$", color=COLORS["School"])
    ax1.tick_params(axis='x', colors=COLORS["School"])
    ax2.set_xlabel(r"Tie reinforcement $\kappa$", color=COLORS["SNN"])
    ax2.tick_params(axis='x', colors=COLORS["SNN"])
    
    ax1.set_ylabel(r"$\Delta$ profile NLL")
    ax1.grid(alpha=0.25)
    
    # combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.1), fontsize=8)
    
    fig.tight_layout(rect=(0, 0, 1, 1))
    fig.savefig(os.path.join(outdir, "profiles.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

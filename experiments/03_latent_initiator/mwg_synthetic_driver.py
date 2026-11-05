#!/usr/bin/env python3
"""Large-scale synthetic confound baseline: Python prep/fit + Julia label sampling.

Pipeline:
  1. Simulate a large (N=100) synthetic temporal network with genuine node and edge memory.
  2. Fit theta_hat = (n_node, gamma_n, kappa, gamma_e) by Monte-Carlo EM (observed-data MLE).
  3. Hand events + theta_hat to mwg_full.jl, which samples initiator labels.
  4. Fit (n_node, kappa) on each returned label draw with the tested estimator and report
     the across-draw confound corr(n_node, kappa) and CV(n_node).

This confirms whether the structural label confound survives at an empirical scale
when node memory is genuine, isolating the difference between memory and the marginal artefact.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

import inference as inf  # noqa: E402
from generative_models import MechanismParams, TemporalNetworkModel, simulate  # noqa: E402


def fit_draw(ev, sides, N, T):
    dir_ev = [(t, i, j) if s == 0 else (t, j, i) for (t, i, j), s in zip(ev, sides)]
    fit = inf.fit_observed_initiator(dir_ev, N, T, fit_b=True)
    return fit.n_node, fit.kappa


def corr_ci(ns, ks, nboot=2000, seed=0):
    """5-95% bootstrap interval of corr(n_node, kappa) over label draws."""
    ns, ks = np.asarray(ns), np.asarray(ks); r = np.random.default_rng(seed); out = []
    for _ in range(nboot):
        i = r.integers(0, len(ns), len(ns))
        if np.std(ns[i]) > 0 and np.std(ks[i]) > 0:
            out.append(np.corrcoef(ns[i], ks[i])[0, 1])
    return [float(np.percentile(out, 5)), float(np.percentile(out, 95))] if out else [float("nan")] * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--T", type=float, default=600.0)  # scale T to get ~5000 events
    ap.add_argument("--M", type=int, default=60)
    ap.add_argument("--burnin", type=int, default=20)
    ap.add_argument("--thin", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--julia", default="julia")
    args = ap.parse_args()

    np.random.seed(args.seed)
    # 1. Simulate dataset with GENUINE memory
    a0 = 0.05 + 0.05 * np.random.default_rng(args.seed).random(args.N)
    # MechanismParams: n_node=0.5, gamma_n=0.5, kappa=20.0, gamma_e=0.05
    mdl = TemporalNetworkModel(args.N, a0, MechanismParams(0.5, 0.5, 20.0, 0.05))
    sim_ev = simulate(mdl, args.T, rng=np.random.default_rng(args.seed+1))
    
    # Strip labels for the observed sequence
    ev = [(float(t), int(min(i, j)), int(max(i, j))) for (t, i, j) in sim_ev]
    N, T = args.N, float(args.T)
    print(f"SYNTHETIC N={N}: {len(ev)} contacts over T={T:.0f}")

    # 2. Fit observed-data MLE (latent initiator EM)
    fitL = inf.fit_latent_initiator(ev, N, T, n_imputations=6, n_em=8, fit_b=True, seed=args.seed)
    theta = (fitL.n_node, fitL.gamma_n, fitL.kappa, fitL.gamma_e)
    a, b = fitL.activities, fitL.attractiveness
    print(f"observed-data MLE: n_node={theta[0]:.3f} gamma_n={theta[1]:.4g} "
          f"kappa={theta[2]:.2f} gamma_e={theta[3]:.4g}")

    # 3. Sample labels with Julia
    with tempfile.TemporaryDirectory() as d:
        np.savetxt(os.path.join(d, "events.tsv"),
                   np.array([[t, i, j] for (t, i, j) in ev]), fmt="%.6f\t%d\t%d")
        np.savetxt(os.path.join(d, "scalars.tsv"),
                   np.array([theta[0], theta[1], theta[2], theta[3], N, T,
                             args.M, args.burnin, args.seed, args.thin]))
        np.savetxt(os.path.join(d, "a.tsv"), np.asarray(a))
        np.savetxt(os.path.join(d, "b.tsv"), np.asarray(b))
        try:
            subprocess.run([args.julia, os.path.join(ROOT, "julia", "mwg_full.jl"), d, d], check=True)
        except FileNotFoundError:
            sys.exit("Julia not found. Install Julia and/or pass --julia /path/to/julia.")
        fil = np.atleast_2d(np.loadtxt(os.path.join(d, "filter.tsv"))).astype(int)
        smo = np.atleast_2d(np.loadtxt(os.path.join(d, "smoother.tsv"))).astype(int)
        try:
            with open(os.path.join(d, "smoother_acceptance.txt"), "r") as f:
                acc = float(f.read().strip())
        except FileNotFoundError:
            acc = float("nan")

    # 4. Evaluate confound
    def confound(draws, tag):
        ns, ks = [], []
        for row in draws:
            n, k = fit_draw(ev, row, N, T)
            ns.append(n); ks.append(k)
        c = float(np.corrcoef(ns, ks)[0, 1]) if len(ns) > 2 else float("nan")
        cv = float(np.std(ns) / np.mean(ns)) if len(ns) > 2 and np.mean(ns) != 0 else float("nan")
        print(f"  {tag:9s}: draws={len(ns)} corr(n_node,kappa)={c:+.2f} [{corr_ci(ns, ks)[0]:+.2f},{corr_ci(ns, ks)[1]:+.2f}]  mean n_node={np.mean(ns):.3f}  CV(n_node)={cv:.3f}")
        return c, float(np.mean(ns)), cv, ns, ks

    print("\nlatent-initiator confound on the LARGE SYNTHETIC record:")
    cf, nf, cvf, nsf, ksf = confound(fil, "filter")
    cs, nsm, cvs, nss, kss = confound(smo, "smoother")
    if not np.isnan(acc):
        print(f"  smoother acceptance rate: {acc:.3f}")

    if cvs < 0.02:
        verdict = "collapses (n_node is essentially pinned, CV < 2%)"
    elif abs(cs) > 0.2:
        verdict = "survives smoothing (genuine information loss)"
    else:
        verdict = "weakens significantly under smoothing"
    print(f"  => confound {verdict}")

    np.savez(os.path.join(ROOT, "results", "mwg_synthetic_full.npz"),
             corr_filter=cf, corr_smoother=cs, n_filter=nf, n_smoother=nsm, 
             cv_filter=cvf, cv_smoother=cvs, ci_filter=corr_ci(nsf, ksf), ci_smoother=corr_ci(nss, kss), n_draws=len(nss),
             ns_filter=nsf, ks_filter=ksf, ns_smoother=nss, ks_smoother=kss, acceptance=acc, N=N, E=len(ev))


if __name__ == "__main__":
    main()

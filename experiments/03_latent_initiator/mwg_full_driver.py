#!/usr/bin/env python3
"""Full-record latent-initiator confound test: Python prep/fit + Julia label sampling.

Pipeline
  1. load a proximity record, fit theta_hat = (n_node, gamma_n, kappa, gamma_e, a, b) by
     Monte-Carlo EM (observed-data MLE);
  2. hand events + theta_hat to mwg_full.jl, which samples initiator labels under both the
     sequential-imputation filter and the Metropolis-within-Gibbs smoother at fixed theta_hat;
  3. fit (n_node, kappa) on each returned label draw with the tested observed-initiator
     estimator and report the across-draw confound corr(n_node, kappa) for each method.

Only step 2 (the O(E^2)/sweep sampler) is in Julia; all fitting stays in the validated
Python estimator. This is the scalable version of mwg_real.py.

Usage
  # validation: reproduce the mwg_real.py subsample result (~ -0.6 both methods)
  python3 scripts/mwg_full_driver.py --n_keep 20 --max_events 250
  # full high-school record
  python3 scripts/mwg_full_driver.py --n_keep 0 --max_events 0
  # another SocioPatterns record (supply its CSV)
  python3 scripts/mwg_full_driver.py --csv /path/hospital.csv --n_keep 0 --max_events 0
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
from sociopatterns_loader import load_sociopatterns, split_by_window  # noqa: E402
DATA = os.path.join(ROOT, "data")


def load(csv, n_keep, max_events):
    d = load_sociopatterns(csv, time_col=0, i_col=1, j_col=2, sessionize_gap=60.0)
    blk = split_by_window(d, 86400.0, min_contacts=5)[0]
    ev = list(blk.contacts)
    if n_keep and n_keep > 0:
        deg = {}
        for _t, i, j in ev:
            deg[i] = deg.get(i, 0) + 1; deg[j] = deg.get(j, 0) + 1
        keep = set(sorted(deg, key=deg.get, reverse=True)[:n_keep])
        ev = [(t, i, j) for (t, i, j) in ev if i in keep and j in keep]
    if max_events and max_events > 0:
        ev = ev[:max_events]
    nodes = sorted({i for _t, i, _j in ev} | {j for _t, _i, j in ev})
    rm = {n: k for k, n in enumerate(nodes)}
    t0 = ev[0][0]
    ev = [(float(t) - t0, rm[i], rm[j]) for (t, i, j) in ev]
    T = ev[-1][0] + 20.0
    return ev, len(nodes), float(T)


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
    ap.add_argument("--csv", default=os.path.join(DATA, "HighSchool2013_proximity_net.csv"))
    ap.add_argument("--n_keep", type=int, default=0, help="restrict to top-k active nodes (0=all)")
    ap.add_argument("--max_events", type=int, default=0, help="cap events (0=all)")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--burnin", type=int, default=6)
    ap.add_argument("--thin", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--julia", default="julia")
    args = ap.parse_args()

    ev, N, T = load(args.csv, args.n_keep, args.max_events)
    print(f"{os.path.basename(args.csv)}: {len(ev)} contacts, N={N}, T={T:.0f}")
    fitL = inf.fit_latent_initiator(ev, N, T, n_imputations=6, n_em=8, fit_b=True, seed=0)
    theta = (fitL.n_node, fitL.gamma_n, fitL.kappa, fitL.gamma_e)
    a, b = fitL.activities, fitL.attractiveness
    print(f"observed-data MLE: n_node={theta[0]:.3f} gamma_n={theta[1]:.4g} "
          f"kappa={theta[2]:.2f} gamma_e={theta[3]:.4g}")

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

    def confound(draws, tag):
        ns, ks = [], []
        for row in draws:
            n, k = fit_draw(ev, row, N, T); ns.append(n); ks.append(k)
        c = float(np.corrcoef(ns, ks)[0, 1]) if len(ns) > 2 else float("nan")
        cv = float(np.std(ns) / np.mean(ns)) if len(ns) > 2 and np.mean(ns) != 0 else float("nan")
        print(f"  {tag:9s}: draws={len(ns)} corr(n_node,kappa)={c:+.2f} [{corr_ci(ns, ks)[0]:+.2f},{corr_ci(ns, ks)[1]:+.2f}]  mean n_node={np.mean(ns):.3f}  CV(n_node)={cv:.3f}")
        return c, float(np.mean(ns)), cv, ns, ks

    print("\nlatent-initiator confound on the FULL record:")
    cf, nf, cvf, nsf, ksf = confound(fil, "filter")
    cs, nsm, cvs, nss, kss = confound(smo, "smoother")
    print("  => confound " + ("survives smoothing (genuine information loss)"
                              if abs(cs) > 0.2 else "collapses (filtering artefact)"))
    base_name = os.path.basename(args.csv).split('.')[0]
    np.savez(os.path.join(ROOT, "results", f"mwg_full_{base_name}.npz"),
             corr_filter=cf, corr_smoother=cs, n_filter=nf, n_smoother=nsm, 
             cv_filter=cvf, cv_smoother=cvs, ci_filter=corr_ci(nsf, ksf), ci_smoother=corr_ci(nss, kss), n_draws=len(nss),
             ns_filter=nsf, ks_filter=ksf, ns_smoother=nss, ks_smoother=kss, N=N, E=len(ev))


if __name__ == "__main__":
    main()

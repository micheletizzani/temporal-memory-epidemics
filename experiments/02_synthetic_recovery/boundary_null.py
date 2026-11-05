#!/usr/bin/env python3
"""Parametric-bootstrap null distribution for the boundary test H0: n_node = 0.

The likelihood-ratio test of no node self-excitation puts the null on the boundary of the
parameter space (n_node >= 0), and the memory timescale gamma_n is unidentified there. The
asymptotic null is therefore not chi^2_1 but the mixture chi-bar^2 = 1/2 delta_0 + 1/2 chi^2_1
(Self & Liang 1987), whose 95th percentile is 2.71, below the chi^2_1 value 3.84. Using the
uncorrected chi^2_1 threshold is thus CONSERVATIVE. This script confirms that directly:
generate well-specified data under H0 (Poisson node timing, n_node = 0), refit the full
exponential-Hawkes timing model, and build the bootstrap distribution of
    LR = 2 [ nll(n_node = 0) - min_{n_node, gamma_n} nll ] .
If the empirical 95th percentile falls at ~2.7 (< 3.84), the chi^2_1 rule cannot mask a
failure to reject: any rejection made against 3.84 would also be a rejection against the
correct boundary null.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import sys

import numpy as np

import inference as inf  # noqa: E402


def gen_poisson(activities, T, rng):
    """Node timing is a homogeneous Poisson process per node (n_node = 0); random partner."""
    N = len(activities)
    ev = []
    for i in range(N):
        t = 0.0
        while True:
            t += rng.exponential(1.0 / activities[i])
            if t >= T:
                break
            j = int(rng.integers(N - 1)); j += (j >= i)
            ev.append((t, i, j))
    ev.sort()
    return ev


def fit_full_nll(contacts, initiator, N, T):
    """min timing NLL over (n_node, gamma_n); grid then local refine so interior optima
    (and hence small chance LR values) are not missed. Return (nll_full, nll_0)."""
    from scipy.optimize import minimize_scalar
    n_grid = np.linspace(0.0, 0.95, 16)
    g_grid = np.geomspace(1.0 / (5 * T), 5.0 / _min_gap(contacts), 8)
    best, best_g = np.inf, g_grid[0]
    for g in g_grid:
        for n in n_grid:
            v = inf.timing_nll(float(n), float(g), contacts, initiator, N, T)
            if v < best:
                best, best_g = v, g
    r = minimize_scalar(lambda n: inf.timing_nll(float(n), float(best_g), contacts,
                        initiator, N, T), bounds=(0.0, 0.98), method="bounded")
    if r.fun < best:
        best = float(r.fun)
    nll_0 = inf.timing_nll(0.0, float(g_grid[0]), contacts, initiator, N, T)
    return best, nll_0


def _min_gap(contacts):
    t = np.array([c[0] for c in contacts])
    d = np.diff(np.sort(t)); d = d[d > 0]
    return float(d.min()) if len(d) else 1.0


def main():
    outdir = os.path.join(ROOT, "results")
    os.makedirs(outdir, exist_ok=True)
    N, T, B = 20, 300.0, 80
    rng = np.random.default_rng(0)
    activities = np.exp(rng.normal(np.log(0.1), 0.5, size=N))   # heavy-ish rate heterogeneity
    LR = np.zeros(B)
    for b in range(B):
        ev = gen_poisson(activities, T, rng)
        initiator = [i for (_t, i, _j) in ev]
        nll_full, nll_0 = fit_full_nll(ev, initiator, N, T)
        LR[b] = max(0.0, 2.0 * (nll_0 - nll_full))
    frac0 = float(np.mean(LR < 1e-6))
    p95 = float(np.percentile(LR, 95))
    p99 = float(np.percentile(LR, 99))
    print(f"boundary-null bootstrap (B={B}, well-specified H0: n_node=0)")
    print(f"  P(LR = 0)            = {frac0:.2f}   (chi-bar^2 predicts 0.50)")
    print(f"  bootstrap 95th pct   = {p95:.2f}   (chi-bar^2: 2.71 ; chi^2_1: 3.84)")
    print(f"  bootstrap 99th pct   = {p99:.2f}   (chi-bar^2: 5.41 ; chi^2_1: 6.63)")
    print(f"  => chi^2_1 threshold 3.84 is {'CONSERVATIVE' if p95 < 3.84 else 'NOT conservative'}")
    np.savez(os.path.join(outdir, "boundary_null.npz"), LR=LR, frac0=frac0, p95=p95, p99=p99)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Proof of concept: a sum-of-exponentials node kernel, vs the single exponential.

The node memory kernel is generalised from a single exponential to
    phi(tau) = sum_r w_r * gamma_r e^{-gamma_r tau},   sum_r w_r = 1,  w_r >= 0,
on a FIXED logarithmic grid of rates {gamma_r}. The unit-integral constraint (sum w = 1)
keeps n_node the branching ratio, so the node intensity is
    rho_i(t) = a_i + n_node * sum_r w_r S_i^{(r)}(t),
with S_i^{(r)} the r-th exponential excitation state. Each S^{(r)} obeys the same O(1)
recursion as the single-exponential model, so the whole fit is R passes of replay_timing.

We fit the TIMING block on directed data (observed initiator) two ways -- single
exponential and R-rate mixture -- and report:
  * n_node (single) vs n_node = sum_r w_r (mixture): is the branching ratio robust?
  * the timing log-likelihood gain: does a heavier-tailed kernel fit better?
  * the fitted weights and their identifiability (how flat the likelihood is in w).

Reuses inference.replay_timing / profile_activities, so it inherits the exact profiling
of the baseline activities.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

import inference as inf  # noqa: E402


def load_directed(path, tcol, icol, jcol, max_events, sessionize=None):
    d = np.loadtxt(path)
    if max_events and len(d) > max_events:
        d = d[:max_events]
    t, i, j = d[:, tcol].astype(float), d[:, icol].astype(int), d[:, jcol].astype(int)
    o = np.argsort(t, kind="stable"); t, i, j = t[o], i[o], j[o]
    ids = sorted(set(i) | set(j)); rm = {o: k for k, o in enumerate(ids)}
    ev = [(float(t[k]), rm[int(i[k])], rm[int(j[k])]) for k in range(len(t))]
    return ev, len(ids), float(t.max())


def multi_replay(ev, initiator, N, gammas, T):
    stats = [inf.replay_timing(ev, initiator, N, g, T) for g in gammas]
    S = np.stack([s.S_at_event for s in stats], axis=1)   # (E, R)
    C = np.stack([s.compensator for s in stats], axis=1)  # (N, R)
    return S, C, stats[0].counts


def nll_multi(n_node, w, S, C, counts, initiator, N, T):
    Scomb = S @ w
    Ccomb = C @ w
    st = inf.TimingStats(S_at_event=Scomb, initiator=np.asarray(initiator, int),
                         compensator=Ccomb, counts=counts, T=T)
    a = inf.profile_activities(st, n_node, N)
    lam = a[np.asarray(initiator, int)] + n_node * Scomb
    if np.any(lam <= 0):
        return np.inf, a
    ll = float(np.sum(np.log(lam)) - np.sum(a * T + n_node * Ccomb))
    return -ll, a


def fit_mixture(S, C, counts, initiator, N, T, R, x0=None):
    """Optimise (n_node, w on simplex via softmax) for an R-rate mixture."""
    def unpack(x):
        n = 1.0 / (1.0 + math.exp(-x[0]))            # n_node in (0,1)
        z = x[1:1 + R]
        w = np.exp(z - z.max()); w /= w.sum()
        return n, w

    def obj(x):
        n, w = unpack(x)
        return nll_multi(n, w, S, C, counts, initiator, N, T)[0]

    if x0 is None:
        x0 = np.concatenate([[0.0], np.zeros(R)])
    best = minimize(obj, x0, method="Nelder-Mead",
                    options=dict(maxiter=2500, xatol=1e-3, fatol=1e-3))
    n, w = unpack(best.x)
    return n, w, -best.fun


def run(name, ev, N, T, gammas):
    init = [i for _t, i, _j in ev]
    S, C, counts = multi_replay(ev, init, N, gammas, T)
    R = len(gammas)
    # single exponential = best single rate (fit n_node at each rate, take best)
    best1 = (-np.inf, None, None)
    for r in range(R):
        w = np.zeros(R); w[r] = 1.0
        # 1-D optimise n_node at this rate
        def o1(xn):
            n = 1.0 / (1.0 + math.exp(-xn[0]))
            return nll_multi(n, w, S, C, counts, init, N, T)[0]
        m = minimize(o1, [0.0], method="Nelder-Mead",
                     options=dict(maxiter=300, xatol=1e-3, fatol=1e-3))
        n = 1.0 / (1.0 + math.exp(-m.x[0])); ll = -m.fun
        if ll > best1[0]:
            best1 = (ll, n, 1.0 / gammas[r])
    ll1, n1, tau1 = best1
    r_star = int(np.argmin(np.abs(1.0 / gammas - tau1)))
    z0 = np.full(R, -3.0); z0[r_star] = 3.0                 # seed w at the single-exp rate
    x0 = np.concatenate([[math.log(n1 / (1 - n1))], z0])    # seed n_node at single-exp MLE
    nR, wR, llR = fit_mixture(S, C, counts, init, N, T, R, x0=x0)
    print(f"\n=== {name}: {len(ev)} events, N={N} ===")
    print(f"  single exp : n_node={n1:.3f}  best 1/gamma={tau1:8.1f}s  logL={ll1:.1f}")
    print(f"  mixture(R={R}) : n_node=sum(w)={nR:.3f}  logL={llR:.1f}  "
          f"gain={llR - ll1:+.1f}")
    print("  fitted weights on 1/gamma grid:")
    for g, wv in zip(gammas, wR):
        bar = "#" * int(round(wv * 40))
        print(f"    1/gamma={1.0 / g:8.1f}s  w={wv:5.3f} {bar}")
    eff = np.exp(np.sum(wR * np.log(1.0 / gammas)))          # geometric-mean timescale
    print(f"  effective timescale (geom. mean) ~ {eff:.1f}s;  weights spread over "
          f"{np.sum(wR > 0.05)} of {R} rates")


def main():
    here = ROOT
    data = os.path.join(os.path.dirname(here), "data")                  # experiment data
    wsdata = os.path.abspath(os.path.join(here, "data"))  # workspace data

    which = sys.argv[1] if len(sys.argv) > 1 else "snn"
    if which == "snn":
        snn = os.path.join(wsdata, "snn_bursty_network.txt")
        ev, N, T = load_directed(snn, 0, 1, 2, 15000)
        gammas = 1.0 / np.array([1.0, 8.0, 60.0, 500.0, 2500.0])
        run("SNN bursting", ev, N, T, gammas)
    if which == "email":
        em = os.path.join(data, "email-Eu-core-temporal.txt")
        ev, N, T = load_directed(em, 2, 0, 1, 40000)
        gammas = 1.0 / np.array([300.0, 3600.0, 86400.0, 6 * 86400.0, 30 * 86400.0])
        run("Email (EU)", ev, N, T, gammas)
    if which in ("hospital", "school"):
        from sociopatterns_loader import load_sociopatterns, split_by_window
        fn = ("hospital_lyon_contacts.dat" if which == "hospital"
              else "HighSchool2013_proximity_net.csv")
        d = load_sociopatterns(os.path.join(data, fn), time_col=0, i_col=1, j_col=2,
                               sessionize_gap=60.0)
        blk = split_by_window(d, 86400.0, min_contacts=5)[0]
        ev = []                                    # participation trains: each endpoint fires
        for t, i, j in blk.contacts:
            ev.append((t, i, j)); ev.append((t, j, i))
        ev.sort()
        gammas = 1.0 / np.array([20.0, 60.0, 300.0, 1200.0, 7200.0])
        run(f"{which} (participation trains)", ev, blk.N, blk.T, gammas)


if __name__ == "__main__":
    main()

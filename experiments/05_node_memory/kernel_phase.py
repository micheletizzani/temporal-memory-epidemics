#!/usr/bin/env python3
"""Exploration: is the identifiability boundary a crossover or a transition in KERNEL space?

The main text finds, along the memory-strength axis M (an AR(1) surrogate), that the
shuffle-preservation ratio R is a smooth crossover, not a phase transition. Here we open a
second axis -- the SHAPE of the memory kernel -- and ask the same question with a TRUE
self-exciting (Hawkes) generator.

Generator. A single node's intensity is lambda(t) = mu + n * sum_k phi(t - t_k), with a
mixture-of-exponentials kernel
    phi(tau) = sum_m w_m gamma_m e^{-gamma_m tau},   sum_m w_m = 1  (unit integral),
so n is exactly the branching ratio. A mixture of exponentials is simulated EXACTLY in O(E)
by carrying one decaying state per component (no O(E^2) kernel resummation).

Mismatch axis. We fix a two-component kernel with a fast timescale (1) and a slow timescale
(rho >= 1), equal weight. rho = 1 is a single exponential -- matched to the single-exponential
estimator the field uses. As rho grows the memory becomes multi-scale and the single-exp
estimator can no longer represent it. We fit the single-exp estimator, destroy the order by
shuffling inter-event times, and record
    R = n_node(shuffled) / n_node(real)
(R -> 0 : the estimate reads genuine, representable memory ; R -> 1 : it reads the marginal).
Sweeping rho at several event-counts E asks whether the resolvable -> artefact boundary
sharpens with system size (a transition) or not (a crossover) -- now in kernel space.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import sys

import numpy as np

from plane_validation import fit_nnode_single, shuffle_isi, burstiness_memory  # noqa: E402


def hawkes_mix_sim(mu, n, ws, gammas, T, rng, max_events=300000):
    """Exact Ogata thinning for a mixture-of-exponentials Hawkes process (O(E)).

    Between events the intensity only decreases (kernel is a sum of decaying exponentials),
    so the intensity just after the last event is a valid thinning upper bound.
    """
    ws = np.asarray(ws, float); gammas = np.asarray(gammas, float)
    S = np.zeros(len(ws)); ev = []; t = 0.0
    while len(ev) < max_events:
        lam_bar = mu + n * float(ws @ S)                 # upper bound over the next interval
        if lam_bar <= 0:
            break
        dt = rng.exponential(1.0 / lam_bar)
        t += dt
        if t >= T:
            break
        S = S * np.exp(-gammas * dt)                     # decay to candidate time
        lam = mu + n * float(ws @ S)
        if rng.random() <= lam / lam_bar:
            ev.append(t)
            S = S + gammas                               # each component jumps by gamma_m
    return np.array(ev)


def cell(rho, T, n, mu, reps, seed):
    """Mean R, M over reps for a fast/slow two-exponential kernel with separation rho."""
    ws = np.array([0.5, 0.5]); gammas = np.array([1.0, 1.0 / rho])
    Rs, Ms, Es = [], [], []
    for r in range(reps):
        rng = np.random.default_rng(seed + 9973 * r + int(100 * rho))
        ts = hawkes_mix_sim(mu, n, ws, gammas, T, rng)
        if len(ts) < 60:
            continue
        _, M = burstiness_memory(ts)
        n_real = fit_nnode_single(ts, T)
        n_shuf = fit_nnode_single(shuffle_isi(ts, rng), T)
        if not (np.isfinite(n_real) and np.isfinite(n_shuf)) or n_real < 5e-2:
            continue
        Rs.append(n_shuf / n_real); Ms.append(M); Es.append(len(ts))
    if not Rs:
        return np.nan, np.nan, np.nan, np.nan, 0
    Rs = np.array(Rs)
    return (float(Rs.mean()), float(Rs.std(ddof=1) / np.sqrt(len(Rs))) if len(Rs) > 1 else np.nan,
            float(np.mean(Ms)), float(np.mean(Es)), len(Rs))


def main():
    outdir = os.path.join(ROOT, "results"); os.makedirs(outdir, exist_ok=True)
    n, mu = 0.5, 0.5
    rhos = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0])
    Ts = [1000.0, 3000.0, 9000.0]                        # three system sizes (events/node)
    reps = 12
    res = {"rhos": rhos.tolist(), "Ts": Ts, "R": {}, "Rse": {}, "M": {}, "E": {}}
    for T in Ts:
        R, Rse, Mv, Ev = [], [], [], []
        for rho in rhos:
            r, rse, m, e, ok = cell(rho, T, n, mu, reps, seed=100 + int(T))
            R.append(r); Rse.append(rse); Mv.append(m); Ev.append(e)
            print(f"  T={T:6.0f} rho={rho:5.1f} | E~{e:6.0f} M={m:+.3f} "
                  f"R={r:.3f}+/-{(rse if np.isfinite(rse) else 0):.3f} (ok={ok})")
        res["R"][str(T)] = R; res["Rse"][str(T)] = Rse; res["M"][str(T)] = Mv; res["E"][str(T)] = Ev
    import json
    json.dump(res, open(os.path.join(outdir, "kernel_phase.json"), "w"), indent=2)
    print("wrote kernel_phase.json")


if __name__ == "__main__":
    main()

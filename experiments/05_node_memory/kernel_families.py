#!/usr/bin/env python3
"""Physically-grounded kernel families: does identifiability change across real systems?

Different systems carry qualitatively different memory kernels:
  * short-memory / single-scale     -- exponential                     (baseline)
  * scale-free / long-memory        -- power-law (Omori/ETAS seismic,   BOUCHAUD-style
                                       economic order-flow, animal bouts)
  * non-monotone / refractory       -- excitation after a dead time     (neural, some
                                       animal behaviour: a lull then a burst)

We keep ONE simulator: every kernel is written as a mixture of exponentials
    phi(tau) = sum_m w_m gamma_m e^{-gamma_m tau},   sum_m w_m = 1  (unit integral),
so n is exactly the branching ratio and the process simulates EXACTLY in O(E) via one
decaying state per component (hawkes_mix_sim in kernel_phase.py). A power-law is recovered
by log-spacing the rates gamma_m over many decades with weights w_m ~ gamma_m^theta:

    int rho(gamma) gamma e^{-gamma tau} dgamma  ~  tau^{-(1+theta)}   for rho(gamma) ~ gamma^{theta-1},
    and with log-spaced gamma_m (d gamma ~ gamma) the discrete weight is w_m ~ gamma_m^theta.

theta -> 0 : weight uniform per decade  -> maximally scale-free (the scale-invariant / critical
             kernel: no characteristic timescale).
theta large: weight collapses onto gamma_max -> effectively a single fast exponential (short
             memory). One knob interpolates seismic <-> social <-> exponential.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import sys

import numpy as np

from kernel_phase import hawkes_mix_sim  # noqa: E402


def powerlaw_mixture(theta, gmin=1e-3, gmax=1e2, K=28):
    """Mixture-of-exponentials approximation to phi(tau) ~ tau^{-(1+theta)}.

    Returns (ws, gammas) with sum(ws)=1. The power law holds for 1/gmax << tau << 1/gmin.
    """
    gammas = np.geomspace(gmin, gmax, K)
    ws = gammas ** theta
    ws = ws / ws.sum()
    return ws, gammas


def refractory_mixture(g_slow=1.0, g_fast=8.0, w_neg=0.6):
    """Non-monotone kernel: a fast NEGATIVE (inhibitory) lobe minus... implemented as a
    difference of exponentials phi(tau) = A[e^{-g_slow tau} - e^{-g_fast tau}], which is 0 at
    tau=0, rises to a delayed peak, then decays -- a refractory 'dead time then excitation'.
    Expressed in the unit-integral mixture form; the fast term carries a negative weight, so we
    return (ws, gammas) with a sign on ws (hawkes_mix_sim tolerates negative component states as
    long as the total intensity is clamped >= 0 by the caller regime).
    """
    # difference-of-exponentials, each written as a unit-integral component
    gammas = np.array([g_slow, g_fast])
    ws = np.array([1.0, -1.0]) * 1.0
    # renormalise so the POSITIVE integral is 1 (net integral of a diff-of-exp is 1-1=0, which
    # would give zero branching; instead we normalise by the positive part only)
    ws = ws / ws[ws > 0].sum()
    return ws, gammas


def _slope_check(theta):
    ws, gammas = powerlaw_mixture(theta)
    tau = np.geomspace(1.0 / gammas.max() * 3, 1.0 / gammas.min() / 3, 40)
    phi = (ws[:, None] * gammas[:, None] * np.exp(-gammas[:, None] * tau[None, :])).sum(0)
    lo, hi = 8, 30                                  # fit the clean interior decades
    s = np.polyfit(np.log(tau[lo:hi]), np.log(phi[lo:hi]), 1)[0]
    return s


def cell_powerlaw(theta, T, n, mu, reps, seed):
    """Mean R, M, B over reps for a power-law (Omori) kernel of tail exponent 1+theta."""
    import kernel_phase as kp
    from plane_validation import fit_nnode_single, shuffle_isi, burstiness_memory
    ws, gm = powerlaw_mixture(theta)
    Rs, Ms, Bs, Es = [], [], [], []
    for r in range(reps):
        rng = np.random.default_rng(seed + 9973 * r + int(1000 * theta))
        ts = kp.hawkes_mix_sim(mu, n, ws, gm, T, rng)
        if len(ts) < 80:
            continue
        B, M = burstiness_memory(ts)
        nr = fit_nnode_single(ts, T)
        ns = fit_nnode_single(shuffle_isi(ts, rng), T)
        if not (np.isfinite(nr) and np.isfinite(ns)) or nr < 5e-2:
            continue
        Rs.append(ns / nr); Ms.append(M); Bs.append(B); Es.append(len(ts))
    if not Rs:
        return dict(R=np.nan, Rse=np.nan, M=np.nan, B=np.nan, E=np.nan, ok=0)
    Rs = np.array(Rs)
    return dict(R=float(Rs.mean()), Rse=float(Rs.std(ddof=1) / np.sqrt(len(Rs))),
                M=float(np.mean(Ms)), B=float(np.mean(Bs)), E=float(np.mean(Es)), ok=len(Rs))


def sweep(Ts, thetas, n=0.6, mu=0.4, reps=8):
    import json
    outdir = os.path.join(ROOT, "results"); os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, "kernel_families.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {"n": n, "mu": mu, "thetas": list(thetas), "T": {}}
    for T in Ts:
        row = {k: [] for k in ("R", "Rse", "M", "B", "E")}
        for th in thetas:
            c = cell_powerlaw(th, float(T), n, mu, reps, seed=100 + int(T))
            for k in row:
                row[k].append(c[k])
            print(f"  T={T:6.0f} theta={th:.2f} | E~{c['E']:6.0f} B={c['B']:+.3f} "
                  f"M={c['M']:+.3f} R={c['R']:.3f}+/-{c['Rse']:.3f} (ok={c['ok']})")
        res["T"][str(int(T))] = row
        json.dump(res, open(fn, "w"), indent=2)
    print("wrote", fn)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        Ts = [float(x) for x in sys.argv[2:]] or [1500.0]
        sweep(Ts, thetas=[0.2, 0.35, 0.5, 0.7, 1.0, 1.5])
    else:
        print("power-law kernel construction check (target slope = -(1+theta)):")
        for th in [0.2, 0.5, 1.0, 1.5]:
            print(f"  theta={th:.2f}  measured log-log slope={_slope_check(th):+.3f}  target={-(1+th):+.3f}")

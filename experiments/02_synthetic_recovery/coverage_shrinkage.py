#!/usr/bin/env python3
"""Test the incidental-parameter (Neyman-Scott) hypothesis for the n_node coverage
collapse, ON THE REAL GENERATOR AND ESTIMATOR.

Hypothesis: the coverage collapse at intermediate n_node is driven by the per-node
nuisance activities a_i (Lemma 1 profiling) draining/biasing the shared branching ratio.
Remedy under test: replace strict per-node profiling of a_i with a hierarchical Gamma
shrinkage prior (empirical-Bayes mean), MAP-estimating a_i.

The 'profile' arm is the paper's own estimator (fit_observed_initiator, exactly as the
sweep uses it). The 'map' arm reuses the paper's replay_timing / stats / Hessian and
differs ONLY in how the a_i of initiating nodes are estimated. Cells are constructed
exactly as run_identifiability_sweep.run_cell (activities = 0.02 + 0.03*U, same simulate)
and chosen to match the degraded cells of Supplementary Table S4.

Every number is measured here; nothing is imputed.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import sys, os, math
import numpy as np
from scipy.optimize import minimize

from generative_models import (MechanismParams, TemporalNetworkModel, simulate)  # noqa
import inference as inf  # noqa
from inference import (replay_timing, timing_nll, _split_events,
                                 _numerical_hessian, fit_observed_initiator)  # noqa

GAMMA_N, GAMMA_E = 0.5, 0.05
KAPPA_SHAPE = 4.0   # Gamma prior shape; larger = stronger shrinkage toward the mean


# ---- MAP profiling of a_i under a Gamma(shape, shape/abar) prior (mean abar) --------
def profile_activities_map(stats, n_node, N, abar, shape=KAPPA_SHAPE):
    T = stats.T
    a = np.zeros(N)
    E = len(stats.initiator)
    if E == 0:
        return a
    order = np.argsort(stats.initiator, kind="stable")
    init_sorted = stats.initiator[order]
    S_sorted = stats.S_at_event[order]
    bounds = np.flatnonzero(np.diff(init_sorted)) + 1
    starts = np.concatenate(([0], bounds)); ends = np.concatenate((bounds, [E]))
    rate = shape / abar
    for s, e in zip(starts, ends):
        i = int(init_sorted[s]); K = stats.counts[i]
        if K == 0.0:
            continue
        base = n_node * S_sorted[s:e]
        def score(x):  # penalised stationarity: data score + prior score
            return float(np.sum(1.0 / (x + base))) - T + (shape - 1.0) / x - rate
        lo, hi = 1e-9, max(K / T, abar)
        while score(hi) > 0 and hi < 1e6:
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if score(mid) > 0:
                lo = mid
            else:
                hi = mid
        a[i] = 0.5 * (lo + hi)
    return a


def timing_nll_map(n_node, gamma_n, contacts, initiator, N, T, abar, shape=KAPPA_SHAPE):
    if not (0.0 <= n_node < 1.0) or gamma_n <= 0.0:
        return np.inf
    stats = replay_timing(contacts, initiator, N, gamma_n, T)
    a = profile_activities_map(stats, n_node, N, abar, shape)
    rate = a[stats.initiator] + n_node * stats.S_at_event
    if np.any(rate <= 0.0):
        return np.inf
    nll = -float(np.sum(np.log(rate))) + float(np.sum(a) * T + n_node * np.sum(stats.compensator))
    pos = a > 0
    logprior = float(np.sum((shape - 1.0) * np.log(a[pos]) - (shape / abar) * a[pos]))
    return nll - logprior          # negative log-posterior (up to const)


def fit_map(events, N, T):
    contacts, initiator, side = _split_events(events)
    # empirical-Bayes prior mean = population mean of strict-profile activities at a pilot fit
    pilot = fit_observed_initiator(events, N, T)
    abar = float(np.mean(pilot.activities[pilot.activities > 0])) if np.any(pilot.activities > 0) else 0.03
    obj = lambda z: timing_nll_map(z[0], z[1], contacts, initiator, N, T, abar)
    res = minimize(obj, x0=[max(0.1, pilot.n_node), GAMMA_N],
                   method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-3, maxiter=400))
    n_hat, g_hat = res.x
    H = _numerical_hessian(lambda z: timing_nll_map(z[0], z[1], contacts, initiator, N, T, abar),
                           np.array([n_hat, g_hat]),
                           np.array([max(1e-3, 0.02 * max(n_hat, 1e-2)), 0.02 * g_hat]))
    try:
        se = math.sqrt(np.linalg.inv(H)[0, 0])
    except Exception:
        se = float("nan")
    return n_hat, se


# ---- cell runner (mirrors run_identifiability_sweep.run_cell) ----------------------
def run_cell(N, T, n_true, kappa, reps, seed0):
    truth = MechanismParams(n_true, GAMMA_N, kappa, GAMMA_E)
    P = dict(n=[], se=[]); M = dict(n=[], se=[]); nev = []
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        activities = 0.02 + 0.03 * rng.random(N)
        model = TemporalNetworkModel(N, activities, truth)
        try:
            events = simulate(model, T, rng=np.random.default_rng(seed0 + 100000 + r))
        except RuntimeError:
            continue
        nev.append(len(events))
        if len(events) < 30:
            continue
        try:
            res = fit_observed_initiator(events, N, T)
            P["n"].append(res.n_node); P["se"].append(res.se["n_node"])
        except Exception:
            pass
        try:
            nm, sm = fit_map(events, N, T)
            M["n"].append(nm); M["se"].append(sm)
        except Exception:
            pass
    def summ(D):
        n = np.array(D["n"], float); s = np.array(D["se"], float)
        ok = np.isfinite(n) & np.isfinite(s) & (s > 0)
        cov = float(np.mean(np.abs(n[ok] - n_true) < 1.96 * s[ok])) if ok.any() else float("nan")
        return (float(np.mean(n[ok])) if ok.any() else float("nan"),
                float(np.mean(n[ok]) - n_true) if ok.any() else float("nan"),
                float(np.mean(s[ok])) if ok.any() else float("nan"),
                float(np.std(n[ok])) if ok.any() else float("nan"), cov, int(ok.sum()))
    return summ(P), summ(M), (np.mean(nev) if nev else 0)


CELLS = [   # (N, T, n_true, kappa)  chosen to match Table S4 degraded cells + a control
    (25, 50.0, 0.30, 10.0),
    (50, 50.0, 0.30, 10.0),
    (50, 50.0, 0.60, 10.0),
    (50, 600.0, 0.30, 10.0),   # control: more events, expect coverage nearer nominal
]

def _print_row(N, T, n, name, a, ev):
    print(f"{N:3d} {T:6.0f} {n:5.2f} {name:>7} {a[0]:7.3f} {a[1]:+7.3f} "
          f"{a[2]:7.3f} {a[3]:6.3f} {a[4]:5.2f} {a[5]:4d} {ev:6.0f}", flush=True)


HEADER = (f"{'N':>3} {'T':>6} {'n_tru':>5} {'arm':>7} {'mean_n':>7} {'bias':>7} "
          f"{'mean_se':>7} {'sd_n':>6} {'cov':>5} {'nok':>4} {'ev':>6}")

if __name__ == "__main__":
    import json
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cell_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None   # run one cell if given
    out = sys.argv[3] if len(sys.argv) > 3 else None
    print(f"gamma_n={GAMMA_N} gamma_e={GAMMA_E} prior_shape={KAPPA_SHAPE} reps={reps}")
    print(HEADER)
    cells = [CELLS[cell_idx]] if cell_idx is not None else CELLS
    rows = []
    for (N, T, n, k) in cells:
        Pp, Mm, ev = run_cell(N, T, n, k, reps, seed0=1000)
        _print_row(N, T, n, "profile", Pp, ev); _print_row(N, T, n, "map", Mm, ev)
        rows.append(dict(N=N, T=T, n=n, kappa=k, ev=ev, reps=reps, profile=Pp, map=Mm))
    if out:
        json.dump(rows, open(out, "w"))

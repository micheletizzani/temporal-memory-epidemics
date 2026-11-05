#!/usr/bin/env python3
"""Do the node channel and the tie channel resolve independently under kernel generality?

The block factorisation (Prop 2) makes the TIMING (node self-excitation) and SELECTION (tie
reinforcement) blocks share no parameters GIVEN the observed initiator, so they are provably
independent there for any kernel. This script probes the two places that independence can still
break: (i) burstiness is a SHARED observable -- tie reuse also clusters a node's contacts in
time, so a model-free timing diagnostic can read tie memory as node memory, and vice versa;
(ii) kernel MISSPECIFICATION -- fitting an exponential to a scale-free channel.

We simulate the full two-channel process with configurable kernel SHAPE on BOTH channels
(node kernel unit-integral so n_node is the branching ratio; tie kernel unit-peak so kappa is
"how many times more attractive a just-contacted partner is"), keep the observed initiator, and
measure each channel's shuffle-preservation ratio:

    R_node = n_hat(time-shuffled) / n_hat(real)     -- node timing order destroyed
    R_tie  = kappa_hat(tie-shuffled) / kappa_hat(real) -- tie-reuse order destroyed

R -> 0 resolvable, R -> 1 unresolvable. Cross-leakage is read from the OFF cells: with the tie
channel off, does R_node still resolve node memory (no false positive from selection)? with the
node channel off, does R_tie still resolve tie memory? Sweeping (theta_node, theta_tie) asks
whether the two channels stay simultaneously resolvable across kernel-configuration space.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import sys

import numpy as np

from kernel_families import powerlaw_mixture           # node kernel, unit integral  # noqa
import inference as inf                       # noqa: E402


def powerlaw_tie(theta, dmin=1e-3, dmax=1e2, K=24):
    """Unit-PEAK mixture approx to a power-law tie kernel: sum_m v_m e^{-delta_m tau},
    sum_m v_m = 1 (so a just-contacted partner contributes exactly 1). theta small = scale-free.
    """
    deltas = np.geomspace(dmin, dmax, K)
    vs = deltas ** theta
    vs = vs / vs.sum()
    return vs, deltas


def simulate_two_channel(N, a, b, n_node, node_ws, node_gm, kappa, tie_vs, tie_dl,
                         T, rng, max_events=200000, return_marks=False):
    """Exact O(E) Ogata thinning for node self-excitation (mixture kernel, unit integral) +
    tie reinforcement (mixture kernel, unit peak). Returns events (t, initiator, partner).

    If return_marks, also returns two per-event covariates used to build attributed marks:
      xS_e = initiator's self-excitation state at the event  (INITIATOR-attributed, label-defined),
      xP_e = the (i,j) tie-memory state at the event          (PAIR-attributed, label-free).
    """
    node_ws = np.asarray(node_ws, float); node_gm = np.asarray(node_gm, float)
    tie_vs = np.asarray(tie_vs, float); tie_dl = np.asarray(tie_dl, float)
    Kn, Ke = len(node_gm), len(tie_dl)
    S = np.zeros((N, Kn))                    # per-node self-excitation component states
    tie = {}                                 # (i,j) -> [component vector, last_update_t]
    b = np.asarray(b, float); a = np.asarray(a, float)
    ev = []; xS = []; xP = []; t = 0.0
    # S[i,m] jumps by gamma_m per event and decays at gamma_m, so S[i,m]=sum_k gamma_m e^{-gamma_m tau}
    # and the excitation sum_k phi(tau)=sum_m w_m S[i,m] (kernel unit-integral => n_node is branching).
    wg = node_ws

    def node_intensity():
        return a + n_node * (S @ wg)          # rho_i = a_i + n_node sum_m w_m S_im

    while len(ev) < max_events:
        lam = node_intensity()
        lam_star = float(lam.sum())           # valid upper bound: intensity only decays until next event
        if lam_star <= 0:
            break
        dt = rng.exponential(1.0 / lam_star)
        t += dt
        if t >= T:
            break
        S *= np.exp(-node_gm * dt)            # incremental decay by the step (tie states decayed on access)
        lam = node_intensity()
        lam_tot = float(lam.sum())
        if rng.random() * lam_star >= lam_tot:
            continue                          # thinned
        i = int(rng.choice(N, p=lam / lam_tot))
        # partner weights for initiator i: b_l + kappa * m_il(t)
        w = b.copy(); w[i] = 0.0
        if kappa > 0:
            for (p, q), (vec, tp) in tie.items():
                if p == i:
                    l = q
                elif q == i:
                    l = p
                else:
                    continue
                m = float(tie_vs @ (vec * np.exp(-tie_dl * (t - tp))))
                w[l] += kappa * m
        wt = float(w.sum())
        if wt <= 0:
            continue
        j = int(rng.choice(N, p=w / wt))
        key = (i, j) if i < j else (j, i)
        if return_marks:
            xS.append(float(node_ws @ S[i]))                 # initiator excitation (pre-bump, label-defined)
            if key in tie:
                vec, tp = tie[key]
                xP.append(float(tie_vs @ (vec * np.exp(-tie_dl * (t - tp)))))  # pair memory (label-free)
            else:
                xP.append(0.0)
        # register: bump node-i excitation, bump tie-(i,j) memory
        S[i] += node_gm
        if key in tie:
            vec, tp = tie[key]
            vec = vec * np.exp(-tie_dl * (t - tp)) + 1.0     # unit-peak: each comp +1
            tie[key] = [vec, t]
        else:
            tie[key] = [np.ones(Ke), t]
        ev.append((t, i, j))
    if return_marks:
        return ev, np.array(xS), np.array(xP)
    return ev


# ---- diagnostics -------------------------------------------------------------------------
def _fit(events, N, T):
    r = inf.fit_observed_initiator(events, N, float(T), fit_b=False)
    return float(r.n_node), float(r.kappa)


def shuffle_node_time(events, rng):
    """Per-initiator ISI shuffle of event TIMES; keep (initiator, partner) pairing and the
    global time set per node. Destroys node timing order, preserves the timing marginal."""
    ev = sorted(events)
    by = {}
    for k, (t, i, j) in enumerate(ev):
        by.setdefault(i, []).append(k)
    tnew = [t for (t, _i, _j) in ev]
    for i, idx in by.items():
        ts = sorted(tnew[k] for k in idx)
        d = np.diff([0.0] + ts)
        rng.shuffle(d)
        cs = np.cumsum(d)
        for k, tt in zip(idx, cs):
            tnew[k] = float(tt)
    out = [(tnew[k], ev[k][1], ev[k][2]) for k in range(len(ev))]
    return sorted(out)


def shuffle_tie_partner(events, rng):
    """Per-initiator permutation of PARTNER labels; keep times and initiators. Destroys
    tie-reuse order, preserves each initiator's partner frequency."""
    ev = sorted(events)
    by = {}
    for k, (t, i, j) in enumerate(ev):
        by.setdefault(i, []).append(k)
    out = [list(e) for e in ev]
    for i, idx in by.items():
        parts = [ev[k][2] for k in idx]
        rng.shuffle(parts)
        for k, p in zip(idx, parts):
            out[k][2] = p
    return sorted(tuple(e) for e in out)


def cell(theta_n, theta_e, N, T, n_node, kappa, reps, seed, a_het=0.6, e_target=3000):
    """Mean (R_node, R_tie, n_hat, kappa_hat, E) over reps at kernel shapes (theta_n, theta_e).

    Activities are normalised so the stationary rate Sum(a)/(1-n_node) yields ~e_target events.
    """
    nws, ngm = powerlaw_mixture(theta_n)
    tvs, tdl = powerlaw_tie(theta_e)
    Rn, Rt, NH, KH, ES = [], [], [], [], []
    for r in range(reps):
        rng = np.random.default_rng(seed + 101 * r)
        a = np.exp(a_het * rng.standard_normal(N))           # lognormal activity heterogeneity
        a *= (e_target / T) * (1.0 - n_node) / a.sum()       # hit the target event count
        b = np.ones(N)
        ev = simulate_two_channel(N, a, b, n_node, nws, ngm, kappa, tvs, tdl, T, rng)
        if len(ev) < 200:
            continue
        n_real, k_real = _fit(ev, N, T)
        n_sh, _ = _fit(shuffle_node_time(ev, rng), N, T)
        _, k_sh = _fit(shuffle_tie_partner(ev, rng), N, T)
        if n_real > 5e-2:
            Rn.append(n_sh / n_real)
        if k_real > 5e-2:
            Rt.append(k_sh / k_real)
        NH.append(n_real); KH.append(k_real); ES.append(len(ev))
    m = lambda x: float(np.mean(x)) if x else np.nan
    return dict(Rnode=m(Rn), Rtie=m(Rt), nhat=m(NH), khat=m(KH), E=m(ES),
                okn=len(Rn), okt=len(Rt))


if __name__ == "__main__":
    import json
    N, T = 30, 4000.0
    # quick validation + leakage corners at matched exponential kernels (theta=1)
    print("validation / leakage corners (theta_n=theta_e=1):")
    for tag, nn, kk in [("both", 0.5, 12.0), ("node-only", 0.5, 0.0), ("tie-only", 0.0, 12.0),
                        ("null", 0.0, 0.0)]:
        c = cell(1.0, 1.0, N, T, nn, kk, reps=4, seed=7)
        print(f"  {tag:10s} n_node={nn} kappa={kk} -> nhat={c['nhat']:.3f} khat={c['khat']:.2f} "
              f"Rnode={c['Rnode']:.3f} Rtie={c['Rtie']:.3f} E~{c['E']:.0f}")

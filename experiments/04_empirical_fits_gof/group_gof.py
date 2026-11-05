#!/usr/bin/env python3
"""Does modelling groups natively remove the timing misspecification, or only part of it?

The main text argues that proximity data break the point-process model in two ways: (i) the
zero atom, from a group conversation being logged as several simultaneous dyads, and (ii) a
residual serial structure that the exponential kernel misses. A reviewer noted that we only
*remove* tied residuals rather than *fit* a group model, so we cannot separate the two.

Here we do the separation. We fit the timing block under two encodings of the SAME record:
  (dyadic)  every logged contact is an initiation event -- coincident dyads included;
  (group)   each node's coincident contacts are collapsed to ONE group activation, i.e. a
            node that reaches several partners at one instant fires once (the native
            higher-order timing the group model implies).
For each we compute the time-rescaling residuals tau, the KS statistic against Exp(1), the
zero-atom fraction, and the lag-1 serial-correlation z-score of tau. If the group encoding
removes the KS/atom failure but the serial-correlation z-score stays large, defect (ii)
survives native group modelling -- the point process is still misspecified for these data.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import sys

import numpy as np

import inference as inf  # noqa: E402
from sociopatterns_loader import load_sociopatterns, split_by_window  # noqa: E402
DATA = os.path.join(ROOT, "data")


def load_hs(n_keep=40, max_events=6000):
    d = load_sociopatterns(os.path.join(DATA, "HighSchool2013_proximity_net.csv"),
                           time_col=0, i_col=1, j_col=2, sessionize_gap=60.0)
    blk = split_by_window(d, 86400.0, min_contacts=5)[0]
    deg = {}
    for _t, i, j in blk.contacts:
        deg[i] = deg.get(i, 0) + 1; deg[j] = deg.get(j, 0) + 1
    keep = set(sorted(deg, key=deg.get, reverse=True)[:n_keep])
    ev = [(float(t), i, j) for (t, i, j) in blk.contacts if i in keep and j in keep][:max_events]
    nodes = sorted({i for _t, i, _j in ev} | {j for _t, _i, j in ev})
    rm = {n: k for k, n in enumerate(nodes)}
    t0 = ev[0][0]
    ev = [(t - t0, rm[i], rm[j]) for (t, i, j) in ev]
    T = ev[-1][0] + 20.0
    return ev, len(nodes), float(T)


def collapse_groups(events):
    """Collapse each node's coincident contacts into one activation (initiator = lower id)."""
    seen = set()
    out = []
    for (t, i, j) in events:
        a, b = (i, j) if i <= j else (j, i)      # deterministic initiator for undirected data
        key = (a, round(t, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((t, a, b))
    out.sort()
    return out


def gof(events, N, T, label):
    fit = inf.fit_observed_initiator(events, N, T, fit_b=False)
    tau = np.asarray(inf.rescaled_times(events, N, T, fit.n_node, fit.gamma_n, fit.activities), float)
    ks = float(inf._ks_uniform(1.0 - np.exp(-tau)))
    atom = float((tau <= 1e-9).mean())
    nz = tau[tau > 1e-9]                          # lag-1 serial correlation of non-atom residuals
    if len(nz) > 20:
        r1 = float(np.corrcoef(nz[:-1], nz[1:])[0, 1])
        z = r1 * np.sqrt(len(nz))
    else:
        r1, z = float("nan"), float("nan")
    print(f"  {label:8s}: E={len(events):5d} n_node={fit.n_node:.3f} | atom={atom:.3f} "
          f"KS={ks:.3f} | lag-1 r={r1:+.3f} z={z:+.1f}")
    return dict(E=len(events), n_node=fit.n_node, atom=atom, ks=ks, r1=r1, z=z)


def main():
    ev, N, T = load_hs()
    print(f"high-school active subnetwork: {len(ev)} contacts, N={N}")
    dy = gof(ev, N, T, "dyadic")
    gp = gof(collapse_groups(ev), N, T, "group")
    print("\nreading:")
    print(f"  KS: {dy['ks']:.3f} (dyadic) -> {gp['ks']:.3f} (group);  "
          f"atom: {dy['atom']:.3f} -> {gp['atom']:.3f}")
    print(f"  lag-1 z: {dy['z']:+.1f} (dyadic) -> {gp['z']:+.1f} (group)")
    verdict = ("group encoding removes the atom but the serial-correlation defect survives"
               if abs(gp['z']) > 3 and gp['atom'] < 0.5 * dy['atom'] + 1e-9
               else "see numbers")
    print(f"  => {verdict}")
    np.savez(os.path.join(ROOT, "results", "group_gof.npz"), **{f"dy_{k}": v for k, v in dy.items()},
             **{f"gp_{k}": v for k, v in gp.items()})


if __name__ == "__main__":
    main()

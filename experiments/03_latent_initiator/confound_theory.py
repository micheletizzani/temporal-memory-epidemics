#!/usr/bin/env python3
"""Theory test: the latent-initiator confound is the missing-information off-diagonal.

Given the observed initiator the Fisher information is block-diagonal (Prop 2), so
corr(n_node, kappa) = 0. Marginalising the latent initiator subtracts the missing
information (Louis 1982): I_obs = I_complete - I_missing, and I_missing is the covariance of
the complete-data score over the label posterior. Its timing-selection off-diagonal IS the
confound; its magnitude is controlled by the ENTROPY of the initiator posterior.

Prediction (falsifiable). Reveal a fraction rho of the TRUE initiator labels and impute the
rest. As rho: 0 -> 1 the label posterior collapses to a point mass, I_missing -> 0, and the
across-imputation corr(n_node, kappa) climbs from its saturating negative value to 0, tracking
the mean binary label entropy of the UNREVEALED events. A monotone corr(rho) that scales with
that entropy nails the confound to an information-theoretic order parameter -- it is not an
artefact of any one imputation algorithm.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

import inference as inf                       # noqa: E402
from joint_kernel_map import simulate_two_channel        # noqa: E402
_TINY = 1e-300


def _impute_clamped(ordered, N, T, theta, rng, revealed, true_side):
    """Forward-pass imputation that FIXES revealed[e]=True events to their true label and
    draws the rest. Also returns the per-event binary entropy of the draw distribution."""
    n_node, gamma_n, kappa, gamma_e = (float(theta[0]), float(theta[1]),
                                       float(theta[2]), float(theta[3]))
    S = np.zeros(N); M = np.zeros(N)
    nbr: List[Dict[int, Tuple[float, float]]] = [dict() for _ in range(N)]
    counts = np.zeros(N)
    for _t, i, j in ordered:
        counts[i] += 0.5; counts[j] += 0.5
    a = np.maximum(counts / max(T, 1e-12), 1e-9)
    b_total = float(N)
    side = np.zeros(len(ordered), dtype=int)
    ent = np.zeros(len(ordered))
    t_prev = 0.0
    for e, (t, i, j) in enumerate(ordered):
        dt = t - t_prev
        S *= math.exp(-gamma_n * dt); M *= math.exp(-gamma_e * dt); t_prev = t
        entry = nbr[i].get(j)
        m_now = 0.0 if entry is None else entry[0] * math.exp(-gamma_e * (t - entry[1]))
        num = 1.0 + kappa * m_now
        w_i = (a[i] + n_node * S[i]) * num / max((b_total - 1.0) + kappa * M[i], _TINY)
        w_j = (a[j] + n_node * S[j]) * num / max((b_total - 1.0) + kappa * M[j], _TINY)
        tot = w_i + w_j
        p = 0.5 if tot <= 0 else w_i / tot
        ent[e] = 0.0 if p <= 0 or p >= 1 else -(p * math.log(p) + (1 - p) * math.log(1 - p))
        if revealed[e]:
            s = int(true_side[e])
        else:
            s = 0 if rng.random() < p else 1
        side[e] = s
        init = i if s == 0 else j
        S[init] += gamma_n
        m_new = m_now + 1.0
        nbr[i][j] = (m_new, t); nbr[j][i] = (m_new, t)
        M[i] += 1.0; M[j] += 1.0
    return side, ent


def fit_partial(events_true, N, T, rho, rng, n_em=4, n_imp=4, seed=0):
    """Clamped-imputation EM revealing fraction rho of true labels. Returns
    (corr_n_kappa across pooled draws, mean unrevealed-event entropy, n_hat, k_hat)."""
    ordered = [(t, i, j) if i < j else (t, j, i) for (t, i, j) in events_true]
    true_side = np.array([0 if (i < j) else 1 for (t, i, j) in events_true])  # 0 => min-index initiated
    E = len(ordered)
    rr = np.random.default_rng(seed)
    revealed = rr.random(E) < rho
    # seed theta from a coin-flip fit
    flip = rr.integers(0, 2, E)
    seed_ev = [(t, (i if s == 0 else j), (j if s == 0 else i)) for (t, i, j), s in zip(ordered, flip)]
    r0 = inf.fit_observed_initiator(seed_ev, N, T)
    theta = [r0.n_node, r0.gamma_n, r0.kappa, r0.gamma_e]
    draws, ents = [], []
    for sweep in range(n_em):
        sweep_theta = []
        for _ in range(n_imp):
            side, ent = _impute_clamped(ordered, N, T, theta, rng, revealed, true_side)
            ev = [(t, (i if s == 0 else j), (j if s == 0 else i)) for (t, i, j), s in zip(ordered, side)]
            res = inf.fit_observed_initiator(ev, N, T, x0=theta)
            sweep_theta.append([res.n_node, res.gamma_n, res.kappa, res.gamma_e])
            if sweep >= n_em - 2:
                draws.append([res.n_node, res.kappa])
                ents.append(float(ent[~revealed].mean()) if (~revealed).any() else 0.0)
        theta = list(np.array(sweep_theta).mean(axis=0))
    D = np.array(draws)
    sn, sk = float(D[:, 0].std(ddof=1)), float(D[:, 1].std(ddof=1))
    cov = float(np.cov(D[:, 0], D[:, 1])[0, 1])
    corr = float(np.corrcoef(D[:, 0], D[:, 1])[0, 1]) if sn > 0 and sk > 0 else float("nan")
    return dict(corr=corr, cov=cov, sn=sn, sk=sk, entropy=float(np.mean(ents)),
                nhat=float(D[:, 0].mean()), khat=float(D[:, 1].mean()))


def make_data(N=20, T=2500.0, n_node=0.5, kappa=10.0, seed=1):
    rng = np.random.default_rng(seed)
    a = np.exp(0.6 * rng.standard_normal(N)); a *= (1800.0 / T) * (1 - n_node) / a.sum()
    b = np.ones(N)
    nws, ngm = np.array([1.0]), np.array([1.0])       # true single-exp node kernel
    tvs, tdl = np.array([1.0]), np.array([1.0])       # true single-exp tie kernel
    ev = simulate_two_channel(N, a, b, n_node, nws, ngm, kappa, tvs, tdl, T, rng)
    return ev, N, T


if __name__ == "__main__":
    import json
    # argv: <data_seed> <rho1> <rho2> ...   -> append res[seed][rho], averaged later
    data_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rhos = [float(x) for x in (sys.argv[2:] or ["0.0", "1.0"])]
    ev, N, T = make_data(seed=data_seed)
    print(f"realization seed={data_seed}: E={len(ev)} N={N} true n_node=0.5 kappa=10", flush=True)
    fn = os.path.join(ROOT, "results", "confound_rho.json")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    row = res.get(str(data_seed), {})
    for rho in rhos:
        d = fit_partial(ev, N, T, rho, np.random.default_rng(100 + data_seed), n_em=3, n_imp=6, seed=data_seed)
        row[f"{rho:.2f}"] = d
        print(f"  rho={rho:.2f}  H={d['entropy']:.3f} | sn={d['sn']:.3f} sk={d['sk']:.2f} "
              f"cov={d['cov']:+.3f} corr={d['corr']:+.3f} | nhat={d['nhat']:.3f} khat={d['khat']:.2f}",
              flush=True)
    res[str(data_seed)] = row
    json.dump(res, open(fn, "w"), indent=2)

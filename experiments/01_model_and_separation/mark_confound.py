#!/usr/bin/env python3
"""Empirical test of Proposition 2 (weighted): does an initiator-attributed mark join the
node/tie confound under a latent initiator, while a pair-attributed mark stays clean?

Three mechanisms: node self-excitation (n_node), tie reinforcement (kappa), and a MARK whose
log-scale couples to a covariate with strength c_W (the mark parameter we estimate). The mark is
built two ways on the SAME simulated network:
  * INITIATOR-attributed: covariate = the initiator's self-excitation state (label-defined, H_i);
  * PAIR-attributed:       covariate = the (i,j) tie-memory state (label-free, H_ij).
Under a latent initiator we impute the labels; for each imputation we refit (n_hat, kappa_hat)
and re-estimate c_W by OLS of log-mark on the imputed covariate, then read the 3x3 across-
imputation correlation. Prediction: init-attributed marks show non-zero corr(n,c_W) and
corr(kappa,c_W) (a full 3x3 confound); pair-attributed marks leave c_W orthogonal (corr ~ 0).
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


def make_marked(mode, N=25, T=2500.0, n_node=0.5, kappa=10.0, c_W=1.5, sig_W=0.5, seed=1):
    rng = np.random.default_rng(seed)
    a = np.exp(0.6 * rng.standard_normal(N)); a *= (1800.0 / T) * (1 - n_node) / a.sum()
    b = np.ones(N)
    e1 = (np.array([1.0]), np.array([1.0]))
    ev, xS, xP = simulate_two_channel(N, a, b, n_node, *e1, kappa, *e1, T, rng, return_marks=True)
    cov = xS if mode == "init" else xP                   # true attribution of the mark
    z = (cov - cov.mean()) / (cov.std() + 1e-9)
    logw = c_W * z + sig_W * rng.standard_normal(len(ev))
    return ev, np.exp(logw), N, T


def _ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    vx = x.var()
    return float(((x - x.mean()) * (y - y.mean())).mean() / vx) if vx > 1e-12 else 0.0


def _impute_pass(ordered, N, T, theta, rng, revealed, true_side):
    """Forward-pass initiator imputation (clamping revealed events), returning the labels and,
    per event, the imputed initiator's self-excitation state (the init-attributed covariate)."""
    n_node, gamma_n, kappa, gamma_e = (float(theta[0]), float(theta[1]),
                                       float(theta[2]), float(theta[3]))
    S = np.zeros(N); M = np.zeros(N)
    nbr: List[Dict[int, Tuple[float, float]]] = [dict() for _ in range(N)]
    counts = np.zeros(N)
    for _t, i, j in ordered:
        counts[i] += 0.5; counts[j] += 0.5
    a = np.maximum(counts / max(T, 1e-12), 1e-9); b_total = float(N)
    side = np.zeros(len(ordered), dtype=int)
    xhat_init = np.zeros(len(ordered)); xhat_pair = np.zeros(len(ordered))
    ts_n = np.zeros(N); ts_e = np.zeros(N)          # per-node last-touch times for lazy decay
    for e, (t, i, j) in enumerate(ordered):
        # decay ONLY the two nodes accessed this event, exactly (O(1)/event -> O(E) total,
        # not O(E*N)); S[k] read only when k is i or j, so lazy decay is equivalent.
        S[i] *= math.exp(-gamma_n * (t - ts_n[i])); ts_n[i] = t
        S[j] *= math.exp(-gamma_n * (t - ts_n[j])); ts_n[j] = t
        M[i] *= math.exp(-gamma_e * (t - ts_e[i])); ts_e[i] = t
        M[j] *= math.exp(-gamma_e * (t - ts_e[j])); ts_e[j] = t
        entry = nbr[i].get(j)
        m_now = 0.0 if entry is None else entry[0] * math.exp(-gamma_e * (t - entry[1]))
        num = 1.0 + kappa * m_now
        w_i = (a[i] + n_node * S[i]) * num / max((b_total - 1.0) + kappa * M[i], _TINY)
        w_j = (a[j] + n_node * S[j]) * num / max((b_total - 1.0) + kappa * M[j], _TINY)
        tot = w_i + w_j
        p = 0.5 if tot <= 0 else w_i / tot
        s = int(true_side[e]) if revealed[e] else (0 if rng.random() < p else 1)
        side[e] = s
        init = i if s == 0 else j
        xhat_init[e] = S[init]         # initiator's excitation (label-DEFINED covariate)
        xhat_pair[e] = m_now           # (i,j) tie memory (label-FREE covariate)
        S[init] += gamma_n
        m_new = m_now + 1.0
        nbr[i][j] = (m_new, t); nbr[j][i] = (m_new, t)
        M[i] += 1.0; M[j] += 1.0
    return side, xhat_init, xhat_pair


def confound_3x3(ev, w, N, T, rho, rng, mark_model, n_em=3, n_imp=6, seed=0):
    """Across-imputation draws of (n_hat, kappa_hat, c_W_hat) at reveal fraction rho, where the
    mark parameter c_W is fit with the covariate of the POSITED `mark_model` ('init' or 'pair')."""
    ordered = [(t, i, j) if i < j else (t, j, i) for (t, i, j) in ev]
    true_side = np.array([0 if i < j else 1 for (t, i, j) in ev])
    logw = np.log(w); E = len(ordered)
    rr = np.random.default_rng(1000 + seed); revealed = rr.random(E) < rho
    flip = rr.integers(0, 2, E)
    seed_ev = [(t, (i if s == 0 else j), (j if s == 0 else i)) for (t, i, j), s in zip(ordered, flip)]
    r0 = inf.fit_observed_initiator(seed_ev, N, T)
    theta = [r0.n_node, r0.gamma_n, r0.kappa, r0.gamma_e]
    draws = []
    for sweep in range(n_em):
        st = []
        for _ in range(n_imp):
            side, xhat_init, xhat_pair = _impute_pass(ordered, N, T, theta, rng, revealed, true_side)
            evd = [(t, (i if s == 0 else j), (j if s == 0 else i)) for (t, i, j), s in zip(ordered, side)]
            res = inf.fit_observed_initiator(evd, N, T, x0=theta)
            st.append([res.n_node, res.gamma_n, res.kappa, res.gamma_e])
            if sweep >= n_em - 2:
                cov = xhat_init if mark_model == "init" else xhat_pair
                draws.append([res.n_node, res.kappa, _ols_slope(cov, logw)])
        theta = list(np.array(st).mean(axis=0))
    return np.array(draws)


def _corr(D, a, b):
    if D[:, a].std() < 1e-9 or D[:, b].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(D[:, a], D[:, b])[0, 1])


if __name__ == "__main__":
    rho = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    # data generated with a given attribution; analyst fits the MATCHING mark model
    for mode in ("init", "pair"):
        ev, w, N, T = make_marked(mode, seed=1)
        D = confound_3x3(ev, w, N, T, rho, np.random.default_rng(7), mark_model=mode, seed=3)
        lab = {"init": "initiator-attributed (H_i)", "pair": "pair-attributed (H_ij)"}[mode]
        print(f"{lab:28s} E={len(ev)} rho={rho}: "
              f"corr(n,kappa)={_corr(D,0,1):+.2f}  corr(n,cW)={_corr(D,0,2):+.2f}  "
              f"corr(kappa,cW)={_corr(D,1,2):+.2f}  |  cW={D[:,2].mean():+.2f}  "
              f"sigma(cW)={D[:,2].std():.3f}", flush=True)

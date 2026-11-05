#!/usr/bin/env python3
"""Is the latent-initiator confound genuine information loss, or an artefact of the filter?

Reviewer 1's worry: the sequential-imputation E-step is a forward FILTER, and the confound
corr(n_node, kappa) it reports (~ -0.6 on synthetic data) might be an algorithm artefact
rather than the true information loss. We settle it by replacing the filter with a proper
SMOOTHER: a Metropolis-within-Gibbs sampler over the initiator labels, which targets the
exact label posterior P(sigma | data, theta). If the confound survives under smoothing it is
real; if it collapses toward 0 it was a filtering artefact.

We fix theta at the observed-data MLE (the E-step is what is under test) and, for each label
draw from (a) sequential imputation and (b) MH smoothing, compute the complete-data estimator
(n_node, kappa) and its across-draw correlation.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import math
import os
import sys

import numpy as np

from generative_models import MechanismParams, TemporalNetworkModel, simulate  # noqa
import inference as inf  # noqa: E402


def _events(sigma, ordered):
    return [(t, (u if s == 0 else v), (v if s == 0 else u))
            for (t, u, v), s in zip(ordered, sigma)]


def total_nll(sigma, ordered, N, T, theta):
    n, gn, k, ge = theta
    contacts, initiator, side = inf._split_events(_events(sigma, ordered))
    return (inf.timing_nll(n, gn, contacts, initiator, N, T)
            + inf.selection_nll(k, ge, contacts, side, N, fit_b=True))


def estimator(sigma, ordered, N, T):
    fit = inf.fit_observed_initiator(_events(sigma, ordered), N, T, fit_b=True)
    return fit.n_node, fit.kappa


def main():
    N, T = 12, 75.0
    rng = np.random.default_rng(0)
    a = 0.05 + 0.05 * rng.random(N)
    model = TemporalNetworkModel(N, a, MechanismParams(0.5, 0.5, 20.0, 0.05))
    ev = [(t, i, j) for (t, i, j) in simulate(model, T, rng=np.random.default_rng(1))]
    fit_obs = inf.fit_observed_initiator(ev, N, T, fit_b=True)
    theta = (fit_obs.n_node, fit_obs.gamma_n, fit_obs.kappa, fit_obs.gamma_e)
    print(f"{len(ev)} events, N={N}. observed-initiator fit: "
          f"n_node={theta[0]:.3f} kappa={theta[2]:.1f}  corr(n,kappa)=0 (exact, Prop.2)")

    ordered = sorted((t, min(i, j), max(i, j)) for (t, i, j) in ev)
    E = len(ordered)
    M = 20

    # (a) sequential imputation (forward filter) -----------------------------------
    sn, sk = [], []
    for m in range(M):
        side = inf._impute_initiators(ordered, N, T, list(theta),
                                      np.random.default_rng(100 + m))
        n, k = estimator(side, ordered, N, T); sn.append(n); sk.append(k)
    corr_seq = float(np.corrcoef(sn, sk)[0, 1])

    # (b) Metropolis-within-Gibbs smoother -----------------------------------------
    sigma = np.array(inf._impute_initiators(ordered, N, T, list(theta),
                                            np.random.default_rng(7)), int)
    cur = total_nll(sigma, ordered, N, T, theta)
    mr = np.random.default_rng(3)
    mn, mk = [], []
    burnin, thin = 6, 1
    for sweep in range(burnin + M * thin):
        for e in range(E):                       # sweep: propose flipping each label
            sigma[e] ^= 1
            new = total_nll(sigma, ordered, N, T, theta)
            if math.log(mr.random() + 1e-300) < cur - new:
                cur = new
            else:
                sigma[e] ^= 1
        if sweep >= burnin and (sweep - burnin) % thin == 0:
            n, k = estimator(sigma, ordered, N, T); mn.append(n); mk.append(k)
    corr_mcmc = float(np.corrcoef(mn, mk)[0, 1])

    print(f"\nconfound corr(n_node, kappa) under the latent label:")
    print(f"  sequential imputation (filter)   : {corr_seq:+.2f}   "
          f"(mean n_node {np.mean(sn):.3f})")
    print(f"  Metropolis-within-Gibbs (smoother): {corr_mcmc:+.2f}   "
          f"(mean n_node {np.mean(mn):.3f})")
    verdict = ("genuine information loss (survives smoothing)"
               if abs(corr_mcmc) > 0.25 else "largely a filtering artefact")
    print(f"  => the confound is {verdict}.")


if __name__ == "__main__":
    main()

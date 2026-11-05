#!/usr/bin/env python3
"""One-step demonstration of the two main results of the paper (about 10 s, no data needed).

1. With the initiator observed, node and tie memory separate: corr(n_node, kappa) = 0 (Eq. 1, Sec. II A).
2. With the initiator hidden, the two estimates are coupled through the missing label (Eq. 2, Sec. II B).
3. A memoryless renewal process with heavy-tailed gaps still gets a large fitted n_node, and the
   inter-event-order shuffle does not move it: R = n_shuffled / n_real is close to 1 and the
   memory coefficient M is close to 0 (Sec. II G).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))

import numpy as np

from generative_models import MechanismParams, TemporalNetworkModel, simulate, strip_initiators
from inference import fit_latent_initiator, fit_observed_initiator
from memory_stats import rank_memory_train


def separation(N=40, T=600.0, seed=7):
    truth = MechanismParams(0.5, 0.5, 20.0, 0.05)
    activities = 0.02 + 0.03 * np.random.default_rng(seed).random(N)
    events = simulate(TemporalNetworkModel(N, activities, truth), T, rng=np.random.default_rng(seed))
    obs = fit_observed_initiator(events, N, T)
    print(f"{len(events)} events; truth n_node = {truth.n_node}, kappa = {truth.kappa}")
    print(f"  initiator observed: n_node = {obs.n_node:.3f}, kappa = {obs.kappa:.2f}, corr = {obs.corr_n_kappa:+.2f}")
    lat = fit_latent_initiator(strip_initiators(events), N, T, n_imputations=4, n_em=6, seed=seed + 1)
    print(f"  initiator hidden:   n_node = {lat.n_node:.3f}, kappa = {lat.kappa:.2f}, corr = {lat.corr_n_kappa:+.2f}")


def fitted_n(gaps):
    """Fit the timing block to one train, with a uniform partner so that only timing matters."""
    times = np.cumsum(gaps)
    events = [(float(t), 0, 1) for t in times]
    return fit_observed_initiator(events, 2, float(times[-1]) + 1.0).n_node


def renewal_artefact(n_events=3000, seed=11):
    rng = np.random.default_rng(seed)
    gaps = rng.lognormal(mean=0.0, sigma=2.0, size=n_events)
    real = fitted_n(gaps)
    shuffled = fitted_n(rng.permutation(gaps))
    M = rank_memory_train(np.cumsum(gaps))
    print(f"renewal train with no memory ({n_events} events): M = {M:+.3f}, "
          f"fitted n_node = {real:.2f}, R = {shuffled / real:.2f}")


if __name__ == "__main__":
    separation()
    renewal_artefact()

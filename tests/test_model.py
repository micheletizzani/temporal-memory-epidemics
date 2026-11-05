"""Correctness tests for the three-mechanism model and its estimator.

Run with:  python -m pytest tests -q     (from the experiment directory)
       or:  python tests/test_model.py    (no pytest required)

These are not smoke tests. Each one can fail for a real reason:

  * the likelihood is checked against a brute-force O(E^2) reimplementation that sums
    the kernel history directly, so a bug in the O(1) recursion shows up;
  * the simulator is checked against the theoretical event rate implied by the node
    branching ratio, so a wrong kernel normalisation shows up;
  * parameter recovery is checked with a tolerance derived from the estimator's own
    standard error rather than a hard-coded margin, so it does not silently pass when
    the estimator degrades;
  * a fixture is written for cross-checking the Julia port.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np


from generative_models import (  # noqa: E402
    ExcitationState, MechanismParams, TemporalNetworkModel, simulate,
    strip_initiators, summarise)
from inference import (  # noqa: E402
    _split_events, fit_observed_initiator, profile_activities, replay_selection,
    replay_timing, selection_nll, timing_nll)

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_fixture.json")


def _toy(seed: int = 3, N: int = 12, T: float = 300.0,
         n_node: float = 0.4, kappa: float = 8.0):
    rng = np.random.default_rng(seed)
    act = 0.02 + 0.04 * rng.random(N)
    params = MechanismParams(n_node, 0.4, kappa, 0.08)
    model = TemporalNetworkModel(N, act, params)
    events = simulate(model, T, rng=np.random.default_rng(seed + 1))
    return model, events, T


# ------------------------------------------------------------------ model validation

def test_params_reject_explosive():
    for bad in [1.0, 1.5, -0.1]:
        try:
            MechanismParams(bad, 1.0, 1.0, 1.0)
        except ValueError:
            continue
        raise AssertionError(f"n_node={bad} should have been rejected")


def test_attractiveness_normalised():
    m = TemporalNetworkModel(5, np.full(5, 0.1), MechanismParams(0.0, 1.0, 0.0, 1.0),
                             attractiveness=np.array([2.0, 4.0, 6.0, 8.0, 10.0]))
    assert abs(m.attractiveness.mean() - 1.0) < 1e-12, "b must be normalised to mean 1"


def test_model_rejects_bad_shapes():
    for kwargs in [dict(N=3, activities=np.ones(4)),
                   dict(N=3, activities=-np.ones(3)),
                   dict(N=3, activities=np.zeros(3))]:
        try:
            TemporalNetworkModel(params=MechanismParams(0.0, 1.0, 0.0, 1.0), **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected {kwargs}")


# ------------------------------------------------------------------ recursion vs brute force

def test_timing_recursion_matches_brute_force():
    """S_i(t_e^-) from the O(1) recursion must equal the direct kernel sum."""
    _model, events, T = _toy()
    contacts, initiator, _ = _split_events(events)
    gamma_n = 0.37
    stats = replay_timing(contacts, initiator, 12, gamma_n, T)

    hist = {}
    for e, (t, _i, _j) in enumerate(contacts):
        i = int(initiator[e])
        brute = sum(gamma_n * math.exp(-gamma_n * (t - tk)) for tk in hist.get(i, []))
        assert abs(brute - stats.S_at_event[e]) < 1e-9 * max(1.0, abs(brute)), (
            f"event {e}: recursion {stats.S_at_event[e]} vs brute force {brute}")
        hist.setdefault(i, []).append(t)


def test_selection_recursion_matches_brute_force():
    """m_ij(t_e^-) and the row sums M_i must equal direct kernel sums (unit peak)."""
    _model, events, _T = _toy()
    contacts, _initiator, _ = _split_events(events)
    gamma_e = 0.11
    sel = replay_selection(contacts, 12, gamma_e)

    hist = {}
    row = {}
    for e, (t, i, j) in enumerate(contacts):
        brute_pair = sum(math.exp(-gamma_e * (t - tk)) for tk in hist.get((i, j), []))
        assert abs(brute_pair - sel.m_pair[e]) < 1e-9 * max(1.0, brute_pair)
        for col, node in enumerate((i, j)):
            brute_row = sum(math.exp(-gamma_e * (t - tk)) for tk in row.get(node, []))
            assert abs(brute_row - sel.M_row[e, col]) < 1e-8 * max(1.0, brute_row), (
                f"event {e} node {node}: {sel.M_row[e, col]} vs {brute_row}")
        hist.setdefault((i, j), []).append(t)
        row.setdefault(i, []).append(t)
        row.setdefault(j, []).append(t)


# ------------------------------------------------------------------ likelihood

def test_profiled_activities_solve_the_score_equation():
    """The profiled a_i must satisfy sum_e 1/(a_i + n S_e) = T, or sit at a boundary."""
    _model, events, T = _toy()
    contacts, initiator, _ = _split_events(events)
    n_node = 0.4
    stats = replay_timing(contacts, initiator, 12, 0.4, T)
    a = profile_activities(stats, n_node, 12)
    for i in range(12):
        sel = stats.initiator == i
        if not sel.any():
            assert a[i] == 0.0
            continue
        score = float(np.sum(1.0 / (a[i] + n_node * stats.S_at_event[sel]))) - T
        at_boundary = a[i] < 1e-12 and score <= 1e-6
        assert abs(score) < 1e-5 * max(1.0, T) or at_boundary, (
            f"node {i}: score {score} not zero and not at the boundary (a={a[i]})")


def test_likelihood_is_minimised_near_the_truth():
    """The concentrated NLL at the truth must beat clearly wrong parameter values."""
    _model, events, T = _toy()
    contacts, initiator, side = _split_events(events)
    N = 12

    at_truth = timing_nll(0.4, 0.4, contacts, initiator, N, T)
    for n_bad in [0.0, 0.9]:
        assert timing_nll(n_bad, 0.4, contacts, initiator, N, T) > at_truth, (
            f"timing NLL at n_node={n_bad} should exceed the value at the truth")

    at_truth_s = selection_nll(8.0, 0.08, contacts, side, N)
    assert selection_nll(0.0, 0.08, contacts, side, N) > at_truth_s
    assert selection_nll(500.0, 0.08, contacts, side, N) > at_truth_s


def test_zero_memory_reduces_to_poisson_mle():
    """With n_node = 0 the profiled activity must be exactly K_i / T."""
    _model, events, T = _toy(n_node=0.0, kappa=0.0)
    contacts, initiator, _ = _split_events(events)
    stats = replay_timing(contacts, initiator, 12, 1.0, T)
    a = profile_activities(stats, 0.0, 12)
    for i in range(12):
        assert abs(a[i] - stats.counts[i] / T) < 1e-9, (
            f"node {i}: profiled a={a[i]}, Poisson MLE={stats.counts[i] / T}")


# ------------------------------------------------------------------ simulator

def test_simulator_event_rate_matches_branching_theory():
    """Total rate must be sum(a_i) / (1 - n_node) in the stationary regime.

    This is the check that catches a wrong node-kernel normalisation: it only holds if
    the kernel has unit integral, so that n_node really is a branching ratio.
    """
    N, T = 30, 4000.0
    act = np.full(N, 0.02)
    for n_node in [0.0, 0.3, 0.6]:
        params = MechanismParams(n_node, 0.5, 0.0, 0.1)
        model = TemporalNetworkModel(N, act, params)
        events = simulate(model, T, rng=np.random.default_rng(11))
        observed = len(events) / T
        expected = act.sum() / (1.0 - n_node)
        rel = abs(observed - expected) / expected
        assert rel < 0.15, (
            f"n_node={n_node}: observed rate {observed:.4f} vs theory {expected:.4f} "
            f"({rel:.1%} off). A rate mismatch that grows with n_node means the node "
            f"kernel is not unit-integral.")


def test_kappa_increases_tie_repetition():
    """Larger kappa must produce more repeated dyads at identical timing."""
    N, T = 25, 800.0
    act = np.full(N, 0.05)
    reps = []
    for kappa in [0.0, 5.0, 50.0]:
        model = TemporalNetworkModel(N, act, MechanismParams(0.0, 0.5, kappa, 0.05))
        ev = simulate(model, T, rng=np.random.default_rng(5))
        reps.append(summarise(ev, N, T)["repeat_fraction"])
    assert reps[0] < reps[1] < reps[2], (
        f"repeat fraction should increase with kappa, got {reps}")


def test_strip_initiators_preserves_contacts():
    _model, events, _T = _toy()
    stripped = strip_initiators(events)
    assert len(stripped) == len(events)
    for (t1, i1, j1), (t2, i2, j2) in zip(events, stripped):
        assert t1 == t2 and i2 < j2 and {i1, j1} == {i2, j2}


def test_excitation_state_rejects_backwards_time():
    st = ExcitationState(4, 1.0, 1.0)
    st.advance_to(1.0)
    try:
        st.advance_to(0.5)
    except ValueError:
        return
    raise AssertionError("advancing to an earlier time should raise")


# ------------------------------------------------------------------ recovery

def test_recovery_within_estimator_own_error():
    """Estimates must land within ~3 of their own standard errors of the truth.

    Tolerance comes from the reported SE, not a fixed margin, so this test tightens
    automatically as the estimator improves and fails if the SE becomes dishonest.
    """
    N, T = 40, 1200.0
    truth = MechanismParams(0.5, 0.5, 20.0, 0.05)
    rng = np.random.default_rng(2)
    model = TemporalNetworkModel(N, 0.02 + 0.03 * rng.random(N), truth)
    events = simulate(model, T, rng=np.random.default_rng(9))
    res = fit_observed_initiator(events, N, T)
    for name, tv, est in [("n_node", truth.n_node, res.n_node),
                          ("kappa", truth.kappa, res.kappa)]:
        se = res.se[name]
        assert np.isfinite(se) and se > 0, f"{name}: no usable standard error"
        z = abs(est - tv) / se
        assert z < 3.0, f"{name}: estimate {est:.4f} is {z:.1f} se from truth {tv}"


def test_blocks_are_orthogonal_when_initiator_observed():
    """corr(n_node, kappa) must be exactly 0: the blocks share no parameters."""
    _model, events, T = _toy()
    res = fit_observed_initiator(events, 12, T)
    assert res.corr_n_kappa == 0.0


# ------------------------------------------------------------------ higher-order extension

def test_group_model_reduces_pathwise_to_dyadic():
    """With k_max = 2 the group simulator must reproduce `simulate` EVENT FOR EVENT.

    Distributional agreement would be far weaker and much harder to test. Pathwise
    agreement holds only because the size draw is skipped in the dyadic case, keeping the
    random stream synchronised; an earlier version consumed a variate there and the two
    paths diverged. This test exists to stop that regressing.
    """
    from generative_models import (GroupParams, group_events_to_dyads,
                                          simulate_group_events)
    N, T = 40, 600.0
    rng = np.random.default_rng(1)
    model = TemporalNetworkModel(N, 0.02 + 0.03 * rng.random(N),
                                 MechanismParams(0.5, 0.5, 20.0, 0.05))
    for seed in (9, 17, 23):
        dyadic = simulate(model, T, rng=np.random.default_rng(seed))
        viagroup = group_events_to_dyads(
            simulate_group_events(model, GroupParams(k_max=2), T,
                                  rng=np.random.default_rng(seed)))
        assert dyadic == viagroup, (
            f"seed {seed}: reduction is not pathwise "
            f"({len(dyadic)} vs {len(viagroup)} events)")


def test_group_memory_increases_exact_group_recurrence():
    """kappa_g must raise recurrence of the SAME group, at matched event counts."""
    from collections import Counter
    from generative_models import GroupParams, simulate_group_events
    N, T = 40, 600.0
    rng = np.random.default_rng(1)
    model = TemporalNetworkModel(N, 0.02 + 0.03 * rng.random(N),
                                 MechanismParams(0.5, 0.5, 20.0, 0.05))
    frac = []
    for kg in (0.0, 20.0):
        ev = simulate_group_events(
            model, GroupParams(kappa_g=kg, gamma_g=0.02, kappa_1p=2.0,
                               size_p=0.7, k_max=5), T, rng=np.random.default_rng(4))
        c = Counter(frozenset((i,) + s) for _t, i, s in ev)
        ho = [k for k in c if len(k) >= 3]
        frac.append(sum(1 for k in ho if c[k] > 1) / max(len(ho), 1))
    assert frac[1] > 2 * frac[0], (
        f"group memory did not raise exact-group recurrence: {frac}")


def test_group_params_reject_bad_values():
    from generative_models import GroupParams
    for kwargs in (dict(kappa_g=-1.0), dict(gamma_g=0.0), dict(size_p=0.0),
                   dict(size_p=1.5), dict(k_max=1)):
        try:
            GroupParams(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"GroupParams should have rejected {kwargs}")


# ------------------------------------------------------------------ Julia parity fixture

def write_parity_fixture(path: str = FIXTURE) -> dict:
    """Write a small deterministic fixture for the Julia port to reproduce.

    The Julia RNG differs from numpy's, so the fixture pins the EVENT SEQUENCE and the
    likelihood values on it, not the simulation. `julia/scripts/check_parity.jl` reads
    this file and must reproduce every number to 1e-8.
    """
    _model, events, T = _toy(seed=3, N=12, T=300.0, n_node=0.4, kappa=8.0)
    contacts, initiator, side = _split_events(events)
    N = 12
    grid = []
    for n_node in [0.0, 0.2, 0.5]:
        for gamma_n in [0.2, 0.7]:
            grid.append({"n_node": n_node, "gamma_n": gamma_n,
                         "timing_nll": timing_nll(n_node, gamma_n, contacts,
                                                  initiator, N, T)})
    sgrid = []
    for kappa in [0.0, 2.0, 15.0]:
        for gamma_e in [0.03, 0.2]:
            sgrid.append({"kappa": kappa, "gamma_e": gamma_e,
                          "selection_nll": selection_nll(kappa, gamma_e, contacts,
                                                         side, N)})
    payload = {
        "N": N, "T": T,
        "events": [[float(t), int(i), int(j)] for t, i, j in events],
        "timing_grid": grid, "selection_grid": sgrid,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    return payload


def test_parity_fixture_roundtrips():
    payload = write_parity_fixture()
    assert len(payload["events"]) > 50, "fixture too small to be a useful check"
    assert all(np.isfinite(g["timing_nll"]) for g in payload["timing_grid"])
    assert all(np.isfinite(g["selection_nll"]) for g in payload["selection_grid"])


# ------------------------------------------------------------------ runner

if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}\n        {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}\n        {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

"""Maximum-likelihood inference and identifiability diagnostics for the three-mechanism model.

Structure of the likelihood
---------------------------
With the initiator observed, the exact log-likelihood of an event sequence
{(t_e, i_e, j_e)} on [0, T] under the model in `models.generative_models` is

    log L = TIMING + SELECTION

    TIMING    = sum_e log( a_{i_e} + n_node * S_{i_e}(t_e^-) )
                - sum_i [ a_i T + n_node * C_i ]
    C_i       = sum_{t_k in H_i} ( 1 - exp(-gamma_n (T - t_k)) )

    SELECTION = sum_e [ log( b_{j_e} + kappa * m_{i_e j_e}(t_e^-) )
                        - log( (B - b_{i_e}) + kappa * M_{i_e}(t_e^-) ) ]

TIMING depends only on (a, n_node, gamma_n); SELECTION depends only on (b, kappa,
gamma_e). The two blocks share no parameters, so when the initiator is observed the
observed information matrix is exactly block-diagonal between node memory and tie
memory: the A and B mechanisms are estimated from disjoint sufficient statistics and
cannot trade off against each other. That is the identifiability result, and it is a
structural fact about this parametrisation rather than a numerical observation.

When the initiator is NOT observed -- undirected proximity data, where a contact
carries no label saying who approached whom -- the factorisation breaks: each event
contributes a mixture over the two possible labels, S_i depends on which label was
taken, and the blocks couple. `fit_latent_initiator` handles this by Monte Carlo EM
with sequential imputation; the resulting inflation of the (n_node, kappa) covariance
is the quantity the study is designed to measure.

Nuisance parameters
-------------------
The N baseline activities a_i are profiled out exactly. For fixed (n_node, gamma_n)
the timing log-likelihood is separable and strictly concave in each a_i, with
stationarity condition

    sum_{e: i_e = i} 1 / (a_i + n_node * S_i(t_e^-)) = T

whose unique root lies in [0, K_i / T] (K_i = number of events initiated by i). It is
found by bisection. Profiling rather than plugging in a moment estimate is the fix for
the scale degeneracy documented in Ideas/code_vs_assessment_review.md: there is no
longer any exogenous normalisation for a mechanism parameter to absorb.

The attractiveness b is pinned by mean(b) = 1 (enforced in the model). It may be held
uniform or estimated by an MM fixed point; see `estimate_attractiveness`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

Contact = Tuple[float, int, int]

_TINY = 1e-300


# --------------------------------------------------------------------------------------
# Sufficient-statistic replays. These are the only O(E) passes; everything else is cheap.
# --------------------------------------------------------------------------------------

@dataclass
class TimingStats:
    """Statistics for the TIMING block at fixed gamma_n, given initiator labels."""
    S_at_event: np.ndarray       # (E,) node excitation of the initiator at t_e^-
    initiator: np.ndarray        # (E,) int
    compensator: np.ndarray      # (N,) C_i
    counts: np.ndarray           # (N,) K_i
    T: float


def replay_timing(contacts: Sequence[Contact], initiator: Sequence[int], N: int,
                  gamma_n: float, T: float) -> TimingStats:
    """One O(E) pass computing S_{i_e}(t_e^-) and the compensator terms C_i.

    Depends on gamma_n and on the initiator labels, not on n_node or a.
    """
    if gamma_n <= 0.0:
        raise ValueError(f"gamma_n must be positive, got {gamma_n}")
    E = len(contacts)
    S = np.zeros(N)
    S_at = np.zeros(E)
    C = np.zeros(N)
    counts = np.zeros(N)
    t_last = np.zeros(N)
    for e, (t, _i, _j) in enumerate(contacts):
        i = int(initiator[e])
        S[i] *= math.exp(-gamma_n * (t - t_last[i]))
        t_last[i] = t
        S_at[e] = S[i]
        S[i] += gamma_n
        counts[i] += 1.0
        C[i] += 1.0 - math.exp(-gamma_n * (T - t))
    return TimingStats(S_at, np.asarray(initiator, dtype=int), C, counts, float(T))


@dataclass
class SelectionStats:
    """Statistics for the SELECTION block at fixed gamma_e.

    `m_pair` is label-free (dyad memory is symmetric), but `M_row` depends on which
    endpoint initiated, so both orientations are stored and the caller picks.
    """
    m_pair: np.ndarray           # (E,)  m_{i j}(t_e^-)
    M_row: np.ndarray            # (E,2) M at t_e^- for (endpoint0, endpoint1)
    endpoints: np.ndarray        # (E,2) int, as stored in the contact list


def replay_selection(contacts: Sequence[Contact], N: int, gamma_e: float) -> SelectionStats:
    """One O(E) pass computing dyad memory and dyad-memory row sums at each event.

    Label-free: it does not need to know who initiated, which is what allows the same
    replay to be reused across all imputation draws in the latent-initiator fit.
    """
    if gamma_e <= 0.0:
        raise ValueError(f"gamma_e must be positive, got {gamma_e}")
    E = len(contacts)
    M = np.zeros(N)
    t_last = np.zeros(N)
    nbr: List[Dict[int, Tuple[float, float]]] = [dict() for _ in range(N)]
    m_pair = np.zeros(E)
    M_row = np.zeros((E, 2))
    endpoints = np.zeros((E, 2), dtype=int)
    for e, (t, i, j) in enumerate(contacts):
        M[i] *= math.exp(-gamma_e * (t - t_last[i]))
        t_last[i] = t
        M[j] *= math.exp(-gamma_e * (t - t_last[j]))
        t_last[j] = t
        
        entry = nbr[i].get(j)
        m_now = 0.0 if entry is None else entry[0] * math.exp(-gamma_e * (t - entry[1]))
        m_pair[e] = m_now
        M_row[e, 0] = M[i]
        M_row[e, 1] = M[j]
        endpoints[e] = (i, j)
        m_new = m_now + 1.0          # unit-peak tie kernel, matches the simulator
        nbr[i][j] = (m_new, t)
        nbr[j][i] = (m_new, t)
        M[i] += 1.0
        M[j] += 1.0
    return SelectionStats(m_pair, M_row, endpoints)


# --------------------------------------------------------------------------------------
# TIMING block: profiled activities and the concentrated negative log-likelihood.
# --------------------------------------------------------------------------------------

def profile_activities(stats: TimingStats, n_node: float, N: int,
                       tol: float = 1e-12, max_iter: int = 200,
                       use_prior: bool = False, prior_shape: float = 4.0,
                       prior_abar: Optional[float] = None) -> np.ndarray:
    """Exact profile MLE (or MAP with Gamma prior) of baseline activities a_i at fixed (n_node, gamma_n)."""
    T = stats.T
    a = np.zeros(N)
    E = len(stats.initiator)
    if E == 0:
        return a
    order = np.argsort(stats.initiator, kind="stable")
    init_sorted = stats.initiator[order]
    S_sorted = stats.S_at_event[order]
    bounds = np.flatnonzero(np.diff(init_sorted)) + 1        # group start indices
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [E]))

    if use_prior:
        if prior_abar is None:
            prior_abar = float(np.mean(stats.counts[stats.counts > 0])) / T if np.any(stats.counts > 0) else 0.03
        rate = prior_shape / prior_abar
        for s, e in zip(starts, ends):
            i = int(init_sorted[s])
            K = stats.counts[i]
            if K == 0.0:
                continue
            base = n_node * S_sorted[s:e]
            def score(x: float) -> float:
                return float(np.sum(1.0 / (x + base))) - T + (prior_shape - 1.0) / x - rate
            lo, hi = 1e-9, max(K / T, prior_abar)
            while score(hi) > 0.0 and hi < 1e6:
                hi *= 2.0
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                if score(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < tol * max(1.0, hi):
                    break
            a[i] = 0.5 * (lo + hi)
        return a

    for s, e in zip(starts, ends):
        i = int(init_sorted[s])
        K = stats.counts[i]
        if K == 0.0:
            continue
        base = n_node * S_sorted[s:e]

        def deriv(x: float) -> float:
            return float(np.sum(1.0 / (x + base))) - T

        if base.min() > 0.0 and deriv(0.0) <= 0.0:
            a[i] = 0.0
            continue
        lo, hi = 0.0, K / T
        if deriv(hi) > 0.0:  # numerical safety; should not occur analytically
            a[i] = hi
            continue
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if deriv(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol * max(1.0, hi):
                break
        a[i] = 0.5 * (lo + hi)
    return a


def timing_nll(n_node: float, gamma_n: float, contacts: Sequence[Contact],
               initiator: Sequence[int], N: int, T: float,
               return_activities: bool = False,
               use_prior: bool = False, prior_shape: float = 4.0,
               prior_abar: Optional[float] = None):
    """Concentrated negative log-likelihood (or negative log-posterior) of the TIMING block."""
    if not (0.0 <= n_node < 1.0) or gamma_n <= 0.0:
        return (np.inf, None) if return_activities else np.inf
    stats = replay_timing(contacts, initiator, N, gamma_n, T)
    a = profile_activities(stats, n_node, N, use_prior=use_prior,
                           prior_shape=prior_shape, prior_abar=prior_abar)
    rate = a[stats.initiator] + n_node * stats.S_at_event
    if np.any(rate <= 0.0):
        return (np.inf, None) if return_activities else np.inf
    nll = (-float(np.sum(np.log(rate)))
           + float(np.sum(a) * T + n_node * np.sum(stats.compensator)))
    if use_prior:
        pos = a > 0.0
        abar = prior_abar if prior_abar is not None else (float(np.mean(a[pos])) if np.any(pos) else 0.03)
        logprior = float(np.sum((prior_shape - 1.0) * np.log(a[pos]) - (prior_shape / abar) * a[pos]))
        nll -= logprior
    if not np.isfinite(nll):
        return (np.inf, None) if return_activities else np.inf
    return (nll, a) if return_activities else nll


# --------------------------------------------------------------------------------------
# SELECTION block.
# --------------------------------------------------------------------------------------

def selection_nll_from_stats(kappa: float, sel: SelectionStats,
                             initiator_side: np.ndarray, b: np.ndarray) -> float:
    """Negative log-likelihood of the SELECTION block given precomputed statistics.

    `initiator_side` is 0 or 1 per event, indexing which column of `sel.endpoints` was
    the initiator.
    """
    if kappa < 0.0:
        return np.inf
    E = len(initiator_side)
    idx = np.arange(E)
    i_nodes = sel.endpoints[idx, initiator_side]
    j_nodes = sel.endpoints[idx, 1 - initiator_side]
    M_i = sel.M_row[idx, initiator_side]
    num = b[j_nodes] + kappa * sel.m_pair
    den = (float(b.sum()) - b[i_nodes]) + kappa * M_i
    if np.any(num <= 0.0) or np.any(den <= 0.0):
        return np.inf
    return -float(np.sum(np.log(num) - np.log(den)))


def estimate_attractiveness(kappa: float, sel: SelectionStats,
                            initiator_side: np.ndarray, N: int,
                            n_iter: int = 60, tol: float = 1e-10) -> np.ndarray:
    """MM fixed point for the attractiveness b, renormalised to mean 1 each sweep.

    Standard minorise-maximise step for a Plackett-Luce-style choice model with an
    additive memory offset. Monotonicity is checked: if a sweep fails to improve the
    objective the iteration stops and returns the last good iterate rather than
    silently diverging.
    """
    E = len(initiator_side)
    idx = np.arange(E)
    i_nodes = sel.endpoints[idx, initiator_side]
    j_nodes = sel.endpoints[idx, 1 - initiator_side]
    M_i = sel.M_row[idx, initiator_side]

    b = np.ones(N)
    prev = selection_nll_from_stats(kappa, sel, initiator_side, b)
    for _ in range(n_iter):
        num = b[j_nodes] + kappa * sel.m_pair
        den = (float(b.sum()) - b[i_nodes]) + kappa * M_i
        share = np.zeros(N)
        np.add.at(share, j_nodes, b[j_nodes] / np.maximum(num, _TINY))
        inv_den = 1.0 / np.maximum(den, _TINY)
        exposure = np.full(N, float(np.sum(inv_den)))
        np.add.at(exposure, i_nodes, -inv_den)
        with np.errstate(divide="ignore", invalid="ignore"):
            b_new = np.where(exposure > 0.0, share / exposure, b)
        b_new = np.maximum(b_new, 1e-12)
        b_new /= b_new.mean()
        cur = selection_nll_from_stats(kappa, sel, initiator_side, b_new)
        if not np.isfinite(cur) or cur > prev + 1e-9:
            break
        if abs(prev - cur) < tol * max(1.0, abs(prev)):
            b = b_new
            break
        b, prev = b_new, cur
    return b


def selection_nll(kappa: float, gamma_e: float, contacts: Sequence[Contact],
                  initiator_side: np.ndarray, N: int, fit_b: bool = False,
                  return_b: bool = False):
    """Concentrated negative log-likelihood of the SELECTION block."""
    if kappa < 0.0 or gamma_e <= 0.0:
        return (np.inf, None) if return_b else np.inf
    sel = replay_selection(contacts, N, gamma_e)
    b = estimate_attractiveness(kappa, sel, initiator_side, N) if fit_b else np.ones(N)
    nll = selection_nll_from_stats(kappa, sel, initiator_side, b)
    return (nll, b) if return_b else nll


# --------------------------------------------------------------------------------------
# Fitting, with observed information and profile likelihoods.
# --------------------------------------------------------------------------------------

@dataclass
class FitResult:
    n_node: float
    gamma_n: float
    kappa: float
    gamma_e: float
    activities: np.ndarray
    attractiveness: np.ndarray
    nll_timing: float
    nll_selection: float
    se: Dict[str, float]                 # asymptotic standard errors, may be nan
    corr_n_kappa: float                  # 0.0 exactly when the initiator is observed
    n_events: int
    T: float
    cov_matrix: Optional[np.ndarray] = None # 4x4 covariance over [n_node, gamma_n, kappa, gamma_e]
    at_bounds: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def nll(self) -> float:
        return self.nll_timing + self.nll_selection

    def summary(self) -> str:
        def pm(name: str, val: float) -> str:
            s = self.se.get(name, float("nan"))
            return f"{val:.4f} +/- {s:.4f}" if np.isfinite(s) else f"{val:.4f} (se n/a)"
        return (
            f"n_node  = {pm('n_node', self.n_node)}   [node branching ratio, A-memory]\n"
            f"gamma_n = {pm('gamma_n', self.gamma_n)}   [node memory decay, 1/time]\n"
            f"kappa   = {pm('kappa', self.kappa)}   [tie reinforcement, B-memory]\n"
            f"gamma_e = {pm('gamma_e', self.gamma_e)}   [tie memory decay, 1/time]\n"
            f"corr(n_node, kappa) = {self.corr_n_kappa:+.3f}\n"
            f"nll = {self.nll:.3f} on {self.n_events} events over T = {self.T:.1f}"
            + (f"\nnotes: {self.notes}" if self.notes else "")
        )


def _numerical_hessian(f, x: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Central-difference Hessian. Returns nan entries where f is not finite."""
    d = len(x)
    H = np.full((d, d), np.nan)
    f0 = f(x)
    if not np.isfinite(f0):
        return H
    for p in range(d):
        for q in range(p, d):
            if p == q:
                xp, xm = x.copy(), x.copy()
                xp[p] += step[p]
                xm[p] -= step[p]
                H[p, p] = (f(xp) - 2 * f0 + f(xm)) / step[p] ** 2
            else:
                xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
                xpp[p] += step[p]; xpp[q] += step[q]
                xpm[p] += step[p]; xpm[q] -= step[q]
                xmp[p] -= step[p]; xmp[q] += step[q]
                xmm[p] -= step[p]; xmm[q] -= step[q]
                H[p, q] = H[q, p] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (
                    4 * step[p] * step[q])
    return H


def _safe_cov(H: np.ndarray) -> np.ndarray:
    """Invert an observed-information matrix, returning nan if it is not usable."""
    if not np.all(np.isfinite(H)):
        return np.full_like(H, np.nan)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full_like(H, np.nan)
    if np.any(np.diag(cov) <= 0.0):
        return np.full_like(H, np.nan)
    return cov


def time_resolution(contacts: Sequence[Contact]) -> Tuple[float, int]:
    """Smallest strictly positive inter-event gap, and the number of exact ties.

    Both matter for real data. SocioPatterns timestamps live on a 20 s grid, so many
    events share a timestamp exactly. Exactly simultaneous events make the timing
    likelihood DIVERGE as gamma_n -> infinity: a second event at zero time lag sees
    self-excitation gamma_n * (prior events at that instant), so log-intensity grows
    without bound while the compensator stays finite. The optimiser will happily run
    gamma_n to its upper bound and report a spurious fit.

    The fix used throughout this module is to cap the decay rates at 1 / resolution.
    That is a statement about what the data can resolve, not a numerical hack: a memory
    kernel decaying faster than the sampling grid is not estimable from the grid.
    """
    ts = np.asarray([c[0] for c in contacts], dtype=float)
    d = np.diff(ts)
    n_ties = int(np.sum(d == 0.0))
    pos = d[d > 0.0]
    res = float(pos.min()) if pos.size else float("nan")
    return res, n_ties


def _gamma_bounds(contacts: Sequence[Contact], T: float,
                  verbose: bool = False) -> Tuple[float, float]:
    """(gamma_min, gamma_max) implied by the observation window and the resolution."""
    res, n_ties = time_resolution(contacts)
    gamma_min = 1.0 / (10.0 * T)
    if not np.isfinite(res) or res <= 0.0:
        gamma_max = 1.0 / max(T / len(contacts), 1e-12)
    else:
        gamma_max = 1.0 / res
    if gamma_max <= gamma_min:
        gamma_max = gamma_min * 1e3
    if verbose and n_ties:
        print(f"  [fit] {n_ties} exactly simultaneous events detected; capping decay "
              f"rates at 1/{res:g} = {gamma_max:g} per time unit. See time_resolution().")
    return gamma_min, gamma_max


def _coarse_start(f, grid_a: Sequence[float], grid_b: Sequence[float]) -> List[float]:
    """Cheapest point of a coarse 2-D grid, used to seed L-BFGS-B.

    Without this the optimiser fails silently on real data: if gamma is initialised at
    1 while the data live on a scale of hours, every memory term is numerically zero,
    the gradient vanishes, and the fit returns "no memory" regardless of the truth.
    """
    best, best_val = [grid_a[0], grid_b[0]], np.inf
    for va in grid_a:
        for vb in grid_b:
            val = f(va, vb)
            if np.isfinite(val) and val < best_val:
                best, best_val = [va, vb], val
    return best


def _split_events(events: Sequence[Contact]):
    """Split (t, initiator, partner) into ordered contacts plus label arrays."""
    contacts: List[Contact] = []
    initiator = np.zeros(len(events), dtype=int)
    initiator_side = np.zeros(len(events), dtype=int)
    for e, (t, i, j) in enumerate(events):
        initiator[e] = i
        if i < j:
            contacts.append((t, i, j))
            initiator_side[e] = 0
        else:
            contacts.append((t, j, i))
            initiator_side[e] = 1
    return contacts, initiator, initiator_side


def fit_observed_initiator(events: Sequence[Contact], N: int, T: float,
                           fit_b: bool = False,
                           x0: Optional[Sequence[float]] = None,
                           verbose: bool = False,
                           use_prior: bool = False,
                           prior_shape: float = 4.0,
                           prior_abar: Optional[float] = None) -> FitResult:
    """Exact MLE (or MAP with prior) when the initiator of every event is observed."""
    if len(events) < 10:
        raise ValueError(f"need at least 10 events to fit, got {len(events)}")
    T = float(T)
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")

    contacts, initiator, initiator_side = _split_events(events)
    g_lo, g_hi = _gamma_bounds(contacts, T, verbose=verbose)

    if use_prior and prior_abar is None:
        pilot = fit_observed_initiator(events, N, T, fit_b=fit_b, x0=x0, verbose=False, use_prior=False)
        pos = pilot.activities > 0
        prior_abar = float(np.mean(pilot.activities[pos])) if np.any(pos) else 0.03

    def f_t(n, g):
        return timing_nll(n, g, contacts, initiator, N, T, use_prior=use_prior,
                          prior_shape=prior_shape, prior_abar=prior_abar)

    def f_s(k, g):
        return selection_nll(k, g, contacts, initiator_side, N, fit_b=fit_b)

    if x0 is None:
        g_scan = np.geomspace(g_lo, g_hi, 9)
        s_t = _coarse_start(f_t, [0.05, 0.2, 0.4, 0.6, 0.8], g_scan)
        s_s = _coarse_start(f_s, [0.01, 0.3, 3.0, 30.0, 300.0], g_scan)
        x0 = [s_t[0], s_t[1], s_s[0], s_s[1]]
    else:
        x0 = list(x0)

    lg = (math.log(g_lo), math.log(g_hi))
    r_t = minimize(lambda z: f_t(z[0], math.exp(z[1])),
                   [min(max(x0[0], 0.0), 0.95),
                    min(max(math.log(max(x0[1], g_lo)), lg[0]), lg[1])],
                   bounds=[(0.0, 0.999), lg], method="L-BFGS-B")
    r_s = minimize(lambda z: f_s(math.exp(z[0]), math.exp(z[1])),
                   [math.log(max(x0[2], 1e-6)),
                    min(max(math.log(max(x0[3], g_lo)), lg[0]), lg[1])],
                   bounds=[(math.log(1e-8), math.log(1e6)), lg], method="L-BFGS-B")

    n_hat, gn_hat = float(r_t.x[0]), math.exp(float(r_t.x[1]))
    kp_hat, ge_hat = math.exp(float(r_s.x[0])), math.exp(float(r_s.x[1]))

    _, a_hat = timing_nll(n_hat, gn_hat, contacts, initiator, N, T, return_activities=True,
                          use_prior=use_prior, prior_shape=prior_shape, prior_abar=prior_abar)
    _, b_hat = selection_nll(kp_hat, ge_hat, contacts, initiator_side, N,
                             fit_b=fit_b, return_b=True)

    H_t = _numerical_hessian(
        lambda z: timing_nll(z[0], z[1], contacts, initiator, N, T),
        np.array([n_hat, gn_hat]),
        np.array([max(1e-3, 0.02 * max(n_hat, 1e-2)), 0.02 * gn_hat]))
    H_s = _numerical_hessian(
        lambda z: selection_nll(z[0], z[1], contacts, initiator_side, N, fit_b=fit_b),
        np.array([kp_hat, ge_hat]),
        np.array([0.02 * max(kp_hat, 1e-3), 0.02 * ge_hat]))
    cov_t, cov_s = _safe_cov(H_t), _safe_cov(H_s)
    
    cov = np.zeros((4, 4))
    cov[0:2, 0:2] = cov_t
    cov[2:4, 2:4] = cov_s

    def _se(cov_arr: np.ndarray, k: int) -> float:
        v = cov_arr[k, k]
        return math.sqrt(v) if np.isfinite(v) and v > 0 else float("nan")

    se = {"n_node": _se(cov_t, 0), "gamma_n": _se(cov_t, 1),
          "kappa": _se(cov_s, 0), "gamma_e": _se(cov_s, 1)}

    at_bounds = _check_bounds(n_hat, gn_hat, kp_hat, ge_hat, g_lo, g_hi)
    notes = "initiator observed; timing and selection blocks are exactly orthogonal"
    if at_bounds:
        notes += ("\n  *** PARAMETER(S) AT A BOUND: " + ", ".join(at_bounds) + ". "
                  "These are NOT estimates -- the optimiser ran to the wall and stopped. "
                  "A decay rate at 1/Delta_min means the kernel is unresolvable at the "
                  "data's time resolution; at 1/(10T) it means the kernel is flat over "
                  "the window, so the paired kappa or n_node is absorbing structure "
                  "rather than measuring memory. Do not report them.")

    return FitResult(
        n_node=n_hat, gamma_n=gn_hat, kappa=kp_hat, gamma_e=ge_hat,
        activities=a_hat, attractiveness=b_hat,
        nll_timing=float(r_t.fun), nll_selection=float(r_s.fun),
        se=se, corr_n_kappa=0.0, cov_matrix=cov, n_events=len(events), T=T,
        at_bounds=at_bounds, notes=notes,
    )


def _check_bounds(n_node: float, gamma_n: float, kappa: float, gamma_e: float,
                  g_lo: float, g_hi: float, rel_tol: float = 0.01) -> List[str]:
    """Flag parameters sitting on a constraint.

    This exists because a bound value was silently returned and reported as an estimate:
    on email-Eu a fit gave gamma_e = 1.25e-06 against a lower bound of 1/(10T) =
    1.251e-06, and gamma_n = 1 against an upper bound of 1/Delta_min = 1. Both were the
    wall, neither was an estimate, and nothing in the output said so.
    """
    hits: List[str] = []
    for name, val in (("gamma_n", gamma_n), ("gamma_e", gamma_e)):
        if val <= g_lo * (1.0 + rel_tol):
            hits.append(f"{name} at LOWER bound 1/(10T)={g_lo:.4g}")
        elif val >= g_hi * (1.0 - rel_tol):
            hits.append(f"{name} at UPPER bound 1/dt_min={g_hi:.4g}")
    if n_node >= 0.999 * (1.0 - rel_tol):
        hits.append("n_node at the stationarity boundary (explosive regime)")
    if kappa <= 1e-8 * (1.0 + rel_tol):
        hits.append("kappa at zero (gamma_e is then unidentified)")
    return hits


def _impute_initiators(ordered: Sequence[Contact], N: int, T: float,
                       theta: Sequence[float], rng: np.random.Generator) -> np.ndarray:
    """Draw an initiator label per contact by a forward pass at parameters `theta`.

    P(i initiated) is proportional to rho_i(t) * pi_{j|i}(t), evaluated with the state
    accumulated from the labels already drawn.
    """
    n_node, gamma_n, kappa, gamma_e = (float(theta[0]), float(theta[1]),
                                       float(theta[2]), float(theta[3]))
    S = np.zeros(N)
    M = np.zeros(N)
    nbr: List[Dict[int, Tuple[float, float]]] = [dict() for _ in range(N)]
    counts = np.zeros(N)
    for _t, i, j in ordered:
        counts[i] += 0.5
        counts[j] += 0.5
    a = np.maximum(counts / max(T, 1e-12), 1e-9)   # proxy for imputation only
    b_total = float(N)
    side = np.zeros(len(ordered), dtype=int)
    t_prev = 0.0
    for e, (t, i, j) in enumerate(ordered):
        dt = t - t_prev
        S *= math.exp(-gamma_n * dt)
        M *= math.exp(-gamma_e * dt)
        t_prev = t
        entry = nbr[i].get(j)
        m_now = 0.0 if entry is None else entry[0] * math.exp(-gamma_e * (t - entry[1]))
        num = 1.0 + kappa * m_now
        w_i = (a[i] + n_node * S[i]) * num / max((b_total - 1.0) + kappa * M[i], _TINY)
        w_j = (a[j] + n_node * S[j]) * num / max((b_total - 1.0) + kappa * M[j], _TINY)
        tot = w_i + w_j
        s = 0 if (tot <= 0.0 or rng.random() < w_i / tot) else 1
        side[e] = s
        init = i if s == 0 else j
        S[init] += gamma_n
        m_new = m_now + 1.0
        nbr[i][j] = (m_new, t)
        nbr[j][i] = (m_new, t)
        M[i] += 1.0
        M[j] += 1.0
    return side


def fit_latent_initiator(contacts: Sequence[Contact], N: int, T: float,
                         n_imputations: int = 6, n_em: int = 8,
                         fit_b: bool = False, seed: int = 0,
                         x0: Optional[Sequence[float]] = None,
                         use_prior: bool = False,
                         prior_shape: float = 4.0,
                         prior_abar: Optional[float] = None) -> FitResult:
    """Monte Carlo EM when only undirected contacts are observed."""
    if len(contacts) < 10:
        raise ValueError(f"need at least 10 contacts to fit, got {len(contacts)}")
    rng = np.random.default_rng(seed)
    ordered = [(t, i, j) if i < j else (t, j, i) for (t, i, j) in contacts]

    if x0 is None:
        flip = rng.integers(0, 2, size=len(ordered))
        seed_events = [(t, (i if s == 0 else j), (j if s == 0 else i))
                       for (t, i, j), s in zip(ordered, flip)]
        r0 = fit_observed_initiator(seed_events, N, T, fit_b=fit_b, verbose=True,
                                    use_prior=use_prior, prior_shape=prior_shape, prior_abar=prior_abar)
        theta = [r0.n_node, r0.gamma_n, r0.kappa, r0.gamma_e]
    else:
        theta = list(x0)
    draws: List[List[float]] = []
    ses: List[List[float]] = []
    last: Optional[FitResult] = None

    for sweep in range(n_em):
        sweep_theta, sweep_se = [], []
        for _rep in range(n_imputations):
            side = _impute_initiators(ordered, N, T, theta, rng)
            events = [(t, (i if s == 0 else j), (j if s == 0 else i))
                      for (t, i, j), s in zip(ordered, side)]
            res = fit_observed_initiator(events, N, T, fit_b=fit_b, x0=theta,
                                        use_prior=use_prior, prior_shape=prior_shape, prior_abar=prior_abar)
            sweep_theta.append([res.n_node, res.gamma_n, res.kappa, res.gamma_e])
            sweep_se.append([res.se["n_node"], res.se["gamma_n"],
                             res.se["kappa"], res.se["gamma_e"]])
            last = res
        theta = list(np.array(sweep_theta).mean(axis=0))
        if sweep >= n_em - 3:                       # pool the final sweeps
            draws.extend(sweep_theta)
            ses.extend(sweep_se)

    D = np.array(draws)
    Sm = np.array(ses)
    with np.errstate(invalid="ignore"):
        within = np.nanmean(Sm ** 2, axis=0)
    between = D.var(axis=0, ddof=1) if len(D) > 1 else np.zeros(4)
    total = within + (1.0 + 1.0 / max(len(D), 1)) * between
    names = ["n_node", "gamma_n", "kappa", "gamma_e"]
    se = {nm: (float(math.sqrt(v)) if np.isfinite(v) and v > 0 else float("nan"))
          for nm, v in zip(names, total)}

    if len(D) > 2 and D[:, 0].std() > 0 and D[:, 2].std() > 0:
        c = np.corrcoef(D[:, 0], D[:, 2])[0, 1]
        corr = float(c) if np.isfinite(c) else float("nan")
        cov_matrix = np.cov(D, rowvar=False)
    else:
        corr = float("nan")
        cov_matrix = np.zeros((4, 4)) * np.nan

    if last is None:
        raise RuntimeError("EM produced no fits; check n_em and n_imputations")
    return FitResult(
        n_node=float(D[:, 0].mean()), gamma_n=float(D[:, 1].mean()),
        kappa=float(D[:, 2].mean()), gamma_e=float(D[:, 3].mean()),
        activities=last.activities, attractiveness=last.attractiveness,
        nll_timing=last.nll_timing, nll_selection=last.nll_selection,
        se=se, corr_n_kappa=corr, cov_matrix=cov_matrix, n_events=len(contacts), T=float(T),
        at_bounds=last.at_bounds,
        notes=("initiator latent; Monte Carlo EM with sequential imputation "
               f"({n_em} sweeps x {n_imputations} draws), SEs pooled by Rubin's rules. "
               "corr(n_node, kappa) is the across-imputation correlation, i.e. the "
               "mechanism confound induced by the missing labels."),
    )


def profile_likelihood(events: Sequence[Contact], N: int, T: float, which: str,
                       grid: Sequence[float], fit_b: bool = False) -> np.ndarray:
    """Profile negative log-likelihood over `which` in {'n_node', 'kappa'}.

    All remaining parameters, including the nuisance activities, are re-optimised at
    every grid point. The curvature of this curve -- not a point estimate -- is what
    answers the identifiability question: a flat profile means the mechanism is not
    recoverable from the data at hand, however confident the argmax looks.
    """
    contacts, initiator, initiator_side = _split_events(events)
    out = np.empty(len(grid))
    if which == "n_node":
        for k, v in enumerate(grid):
            r = minimize(lambda z: timing_nll(v, math.exp(z[0]), contacts,
                                              initiator, N, T),
                         [0.0], bounds=[(math.log(1e-6), math.log(1e6))],
                         method="L-BFGS-B")
            out[k] = float(r.fun)
    elif which == "kappa":
        for k, v in enumerate(grid):
            r = minimize(lambda z: selection_nll(v, math.exp(z[0]), contacts,
                                                 initiator_side, N, fit_b=fit_b),
                         [0.0], bounds=[(math.log(1e-6), math.log(1e6))],
                         method="L-BFGS-B")
            out[k] = float(r.fun)
    else:
        raise ValueError(f"`which` must be 'n_node' or 'kappa', got {which!r}")
    return out


# --------------------------------------------------------------------------------------
# Goodness of fit.
#
# Everything above establishes that the estimator inverts its own generative process.
# Nothing above says the model describes the data. These are the checks that do.
# --------------------------------------------------------------------------------------

def rescaled_times(events: Sequence[Contact], N: int, T: float, n_node: float,
                   gamma_n: float, activities: np.ndarray) -> np.ndarray:
    """Time-rescaled inter-event intervals for the TIMING block.

    By the time-rescaling theorem, if the conditional intensity is correct then

        tau_k = Lambda_i(t_k) - Lambda_i(t_{k-1}),
        Lambda_i(t) = a_i t + n_node * sum_{t_l < t} [1 - exp(-gamma_n (t - t_l))]

    are i.i.d. Exp(1), pooled over nodes. This is the standard residual analysis for a
    point process and it is the check that distinguishes "the estimator converged" from
    "the model fits".

    The compensator is evaluated in O(E) using the identity

        sum_{t_l < t} exp(-gamma_n (t - t_l)) = S_i(t^-) / gamma_n ,

    so the same excitation recursion that serves the likelihood serves the residuals.
    """
    contacts, initiator, _ = _split_events(events)
    stats = replay_timing(contacts, initiator, N, gamma_n, T)

    taus: List[float] = []
    prev_t: Dict[int, float] = {}
    prev_S: Dict[int, float] = {}
    for e, (t, _u, _v) in enumerate(contacts):
        i = int(initiator[e])
        S_now = stats.S_at_event[e]
        if i not in prev_t:
            # First event of node i: no prior excitation, compensator is a_i * t.
            taus.append(activities[i] * t)
        else:
            dt = t - prev_t[i]
            dS = (S_now - prev_S[i]) / gamma_n
            taus.append(activities[i] * dt + n_node * (1.0 - dS))
        prev_t[i] = t
        prev_S[i] = S_now
    tau = np.asarray(taus, float)
    if np.any(tau < -1e-9):
        raise RuntimeError(
            "negative rescaled time; the compensator recursion disagrees with the "
            "likelihood, which indicates a bug rather than a bad fit")
    return np.maximum(tau, 0.0)


def selection_pit(events: Sequence[Contact], N: int, kappa: float, gamma_e: float,
                  attractiveness: np.ndarray,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomised probability integral transform for the SELECTION block.

    For a discrete choice the PIT must be randomised to be uniform: with alternatives
    in a fixed order and the realised partner j,

        U = F(<j) + V * pi_j ,   V ~ Uniform(0,1),

    is Uniform(0,1) under the true model. Testing U against uniformity checks the
    partner-choice mechanism independently of the timing mechanism, which is possible
    precisely because the two blocks are orthogonal (Proposition: block orthogonality).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    contacts, initiator, side = _split_events(events)
    sel = replay_selection(contacts, N, gamma_e)

    # Rebuild the per-event weight vector. O(E * N); acceptable as a diagnostic.
    state = ExcitationStateSelection(N, gamma_e)
    out = np.empty(len(contacts))
    for e, (t, u, v) in enumerate(contacts):
        state.advance_to(t)
        i = u if side[e] == 0 else v
        j = v if side[e] == 0 else u
        w = attractiveness.copy()
        if kappa > 0.0:
            for l, (val, t_last) in state.nbr[i].items():
                w[l] += kappa * val * math.exp(-gamma_e * (t - t_last))
        w[i] = 0.0
        tot = float(w.sum())
        if tot <= 0.0:
            raise RuntimeError(f"zero total partner weight at event {e}")
        p = w / tot
        out[e] = float(p[:j].sum() + rng.random() * p[j])
        state.register(t, i, j)
    return out


class ExcitationStateSelection:
    """Minimal dyad-memory state, used only by `selection_pit`.

    Duplicates the selection half of models.ExcitationState so that inference.py has no
    import dependency on the model module. The two are checked against each other in
    tests/test_model.py.
    """

    def __init__(self, N: int, gamma_e: float) -> None:
        self.N = N
        self.gamma_e = gamma_e
        self.t = 0.0
        self.nbr: List[Dict[int, Tuple[float, float]]] = [dict() for _ in range(N)]

    def advance_to(self, t: float) -> None:
        if t < self.t:
            raise ValueError("events must be time-ordered")
        self.t = t

    def register(self, t: float, i: int, j: int) -> None:
        entry = self.nbr[i].get(j)
        m = 0.0 if entry is None else entry[0] * math.exp(-self.gamma_e * (t - entry[1]))
        self.nbr[i][j] = (m + 1.0, t)
        self.nbr[j][i] = (m + 1.0, t)


def serial_correlation(events: Sequence[Contact], N: int, T: float, n_node: float,
                       gamma_n: float, activities: np.ndarray) -> Dict[str, float]:
    """Lag-1 serial correlation of the rescaled times, computed within nodes.

    Why this is needed
    ------------------
    The Kolmogorov-Smirnov test on the rescaled times checks only their MARGINAL
    distribution, and it is close to blind to kernel-shape misspecification. Fitting an
    exponential memory kernel to data generated with a power-law kernel produces
    residuals whose marginal is still nearly Exp(1) -- the fit absorbs the discrepancy
    into the baseline and the decay rate -- while the residuals become serially
    correlated, because the fitted kernel decays too fast at long lags and too slowly at
    short ones. We verified exactly this: a power-law-generated network fitted with the
    exponential model gives KS = 0.024 with p = 0.24, i.e. it passes.

    Under a correct conditional intensity the rescaled times are i.i.d., so the
    lag-1 correlation is zero with standard error 1/sqrt(n_pairs). Pairs are formed
    within each node's own residual sequence; pooling across nodes before pairing would
    destroy the ordering that carries the information.

    This is the standard companion to the KS test in point-process residual
    analysis, and on this model class it is the more powerful of the two.
    """
    contacts, initiator, _ = _split_events(events)
    tau = rescaled_times(events, N, T, n_node, gamma_n, activities)

    by_node: Dict[int, List[float]] = {}
    for e in range(len(contacts)):
        by_node.setdefault(int(initiator[e]), []).append(tau[e])

    x, y = [], []
    for _i, seq in by_node.items():
        if len(seq) < 3:
            continue
        x.extend(seq[:-1])
        y.extend(seq[1:])
    if len(x) < 30:
        return {"acf1": float("nan"), "acf1_se": float("nan"),
                "acf1_z": float("nan"), "n_pairs": float(len(x))}

    x_a, y_a = np.asarray(x), np.asarray(y)
    if x_a.std() == 0 or y_a.std() == 0:
        return {"acf1": float("nan"), "acf1_se": float("nan"),
                "acf1_z": float("nan"), "n_pairs": float(len(x))}
    r = float(np.corrcoef(x_a, y_a)[0, 1])
    se = 1.0 / math.sqrt(len(x))
    return {"acf1": r, "acf1_se": se, "acf1_z": r / se, "n_pairs": float(len(x))}


def _ks_uniform(u: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic of `u` against Uniform(0,1)."""
    x = np.sort(np.asarray(u, float))
    n = len(x)
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - x), np.max(x - (i - 1) / n)))


def goodness_of_fit(events: Sequence[Contact], N: int, T: float, fit: FitResult,
                    n_bootstrap: int = 0, seed: int = 0,
                    verbose: bool = True) -> Dict[str, float]:
    """Residual analysis for both blocks, with optional parametric-bootstrap calibration.

    Returns KS statistics for the timing residuals (against Exp(1), via the uniform
    transform) and for the selection PIT (against Uniform(0,1)).

    Why the bootstrap matters
    ------------------------
    The parameters were estimated from the same data, so the nominal KS null
    distribution is anti-conservative: it will reject too rarely, making a bad model look
    acceptable. With `n_bootstrap > 0` the null is rebuilt by simulating from the fitted
    model, refitting, and recomputing the statistic, which is the calibrated test. The
    nominal p-value is reported alongside and labelled as uncalibrated so the two are not
    confused.

    Interpretation
    --------------
    A large KS on the timing block means the node-activity mechanism -- baseline plus
    exponential self-excitation -- does not describe when nodes act. A large KS on the
    selection block means the tie-reinforcement mechanism does not describe whom they act
    with. The blocks fail independently, which is diagnostically useful: it localises the
    misspecification rather than merely flagging it.
    """
    from scipy.stats import kstwobign

    tau = rescaled_times(events, N, T, fit.n_node, fit.gamma_n, fit.activities)
    u_time = 1.0 - np.exp(-tau)                       # Exp(1) -> Uniform(0,1)
    ks_time = _ks_uniform(u_time)

    u_sel = selection_pit(events, N, fit.kappa, fit.gamma_e, fit.attractiveness,
                          rng=np.random.default_rng(seed))
    ks_sel = _ks_uniform(u_sel)

    n_t, n_s = len(u_time), len(u_sel)
    out = {
        "ks_timing": ks_time,
        "ks_selection": ks_sel,
        "n_timing": float(n_t),
        "n_selection": float(n_s),
        "p_timing_nominal": float(kstwobign.sf(ks_time * math.sqrt(n_t))),
        "p_selection_nominal": float(kstwobign.sf(ks_sel * math.sqrt(n_s))),
        "mean_tau": float(tau.mean()),      # should be 1.0 under a correct model
        "var_tau": float(tau.var()),        # should be 1.0 under a correct model
    }
    out.update(serial_correlation(events, N, T, fit.n_node, fit.gamma_n, fit.activities))

    if verbose:
        print(f"  timing    KS   = {ks_time:.4f} on {n_t} residuals "
              f"(uncalibrated p = {out['p_timing_nominal']:.3g})")
        print(f"  selection KS   = {ks_sel:.4f} on {n_s} residuals "
              f"(uncalibrated p = {out['p_selection_nominal']:.3g})")
        print(f"  mean(tau) = {out['mean_tau']:.3f}, var(tau) = {out['var_tau']:.3f} "
              "(both 1.0 under a correct model)")
        print(f"  lag-1 acf = {out['acf1']:+.4f} +/- {out['acf1_se']:.4f}  "
              f"(z = {out['acf1_z']:+.1f}) on {int(out['n_pairs'])} within-node pairs")
        if np.isfinite(out["acf1_z"]) and abs(out["acf1_z"]) > 3.0:
            print("    ^ residuals are serially correlated: the conditional intensity is")
            print("      misspecified in its TIME STRUCTURE even where the marginal KS")
            print("      passes. Leading suspects, in order: a non-stationary baseline")
            print("      (circadian rhythm) not represented by constant a_i; then the")
            print("      exponential kernel shape. Fit within days before blaming the")
            print("      kernel. See Ideas/future_work/01_open_items.md.")

    if n_bootstrap > 0:
        out.update(_bootstrap_gof(events, N, T, fit, n_bootstrap, seed, ks_time,
                                  ks_sel, verbose))
    return out


def _bootstrap_gof(events, N, T, fit, n_bootstrap, seed, ks_time_obs, ks_sel_obs,
                   verbose):
    """Parametric bootstrap null for the KS statistics.

    Imported lazily to keep inference.py free of a hard dependency on the model module
    at import time; the dependency is real but only for this diagnostic.
    """
    from generative_models import (MechanismParams, TemporalNetworkModel,
                                          simulate)

    ks_t_null, ks_s_null = [], []
    rng = np.random.default_rng(seed + 1)
    for b in range(n_bootstrap):
        params = MechanismParams(min(fit.n_node, 0.98), fit.gamma_n,
                                 fit.kappa, fit.gamma_e)
        model = TemporalNetworkModel(N, np.maximum(fit.activities, 1e-12), params,
                                     attractiveness=fit.attractiveness)
        try:
            sim = simulate(model, T, rng=np.random.default_rng(int(rng.integers(1 << 30))))
            if len(sim) < 50:
                continue
            # Warm-start from the observed fit. Without x0 each replicate repeats the
            # 45-point coarse grid scan over (n_node, gamma) x (kappa, gamma), which
            # dominates the cost: 50 replicates took ~17 min and timed out. The
            # replicates are simulated FROM this fit, so its parameters are the right
            # starting point and the scan buys nothing.
            refit = fit_observed_initiator(
                sim, N, T, x0=[fit.n_node, fit.gamma_n, fit.kappa, fit.gamma_e])
            tau_b = rescaled_times(sim, N, T, refit.n_node, refit.gamma_n,
                                   refit.activities)
            ks_t_null.append(_ks_uniform(1.0 - np.exp(-tau_b)))
            ks_s_null.append(_ks_uniform(selection_pit(
                sim, N, refit.kappa, refit.gamma_e, refit.attractiveness,
                rng=np.random.default_rng(seed + 100 + b))))
        except (RuntimeError, ValueError):
            continue
        if verbose and (b + 1) % 5 == 0:
            print(f"    bootstrap {b + 1}/{n_bootstrap}")

    res = {}
    if ks_t_null:
        arr = np.array(ks_t_null)
        res["p_timing_bootstrap"] = float((arr >= ks_time_obs).mean())
        res["ks_timing_null_median"] = float(np.median(arr))
    if ks_s_null:
        arr = np.array(ks_s_null)
        res["p_selection_bootstrap"] = float((arr >= ks_sel_obs).mean())
        res["ks_selection_null_median"] = float(np.median(arr))
    res["n_bootstrap_ok"] = float(len(ks_t_null))
    if verbose and ks_t_null:
        print(f"  calibrated p (timing)    = {res.get('p_timing_bootstrap', float('nan')):.3g}"
              f"   [null median KS {res['ks_timing_null_median']:.4f}]")
        print(f"  calibrated p (selection) = "
              f"{res.get('p_selection_bootstrap', float('nan')):.3g}"
              f"   [null median KS {res['ks_selection_null_median']:.4f}]")
    return res


def likelihood_ratio_test(nll_full: float, nll_restricted: float, df: int) -> float:
    """p-value of 2*(nll_restricted - nll_full) against chi2_df.

    Used for the pure-A null (kappa = 0) and the memoryless null (n_node = 0). Both
    nulls sit on the boundary of the parameter space, where the asymptotic null
    distribution is a chi-bar-squared mixture rather than chi2_df, so this p-value is
    CONSERVATIVE (roughly a factor of two too large for df = 1). Reported uncorrected
    rather than silently adjusted.
    """
    from scipy.stats import chi2
    stat = 2.0 * (nll_restricted - nll_full)
    if stat < 0.0:
        return float("nan")
    return float(chi2.sf(stat, df))

"""Initiator-resolved generative model for temporal networks with three mechanisms.

Motivation
----------
The previous parametrisation

    lambda_ij(t) = alpha * a_i a_j + (1 - alpha) * (mu + sum_k phi(t - t_k))

was not identified: alpha multiplies `a_i a_j`, whose overall scale was supplied
exogenously, so rescaling a -> c*a is exactly absorbed by alpha -> alpha/c^2. It also
conflated two distinct axes (node-vs-edge and memoryless-vs-memory) into one number,
and its "node" pole was a rank-1 Poisson graph with independent dyads, i.e. it had no
node activation event at all.

This module replaces it with an initiator-resolved process in which an event is a
triple (t, i, j) with i the initiator. Three mechanisms enter through structurally
distinct channels:

  A0  node baseline activity        a_i          -- memoryless activity-driven
  A1  node self-excitation          n_node       -- burstiness in *when* i acts
  B   tie reinforcement             kappa        -- memory in *which* tie i reuses

Node initiation intensity (H_i = past times at which i initiated):

    rho_i(t) = a_i + n_node * sum_{t_k in H_i} gamma_n * exp(-gamma_n (t - t_k))

The kernel is normalised to unit integral, so `n_node` IS the node branching ratio:
it is dimensionless, identified, and stationarity is exactly n_node < 1.

Partner choice given initiator i fires at time t (H_ij = past contacts of dyad ij):

    m_ij(t)     = sum_{t_k in H_ij} exp(-gamma_e (t - t_k))
    pi_{j|i}(t) = (b_j + kappa * m_ij(t)) / sum_{l != i} (b_l + kappa * m_il(t))

Note the two kernels are normalised differently, on purpose. The node kernel carries
unit integral so that n_node is a branching ratio (a rate quantity). The tie kernel
carries unit *peak*, so a partner contacted an instant ago contributes exactly
kappa * 1 of extra weight against a baseline attractiveness of b = 1: kappa reads
directly as "how many times more attractive a just-contacted partner is". Normalising
the tie kernel to unit integral instead would make kappa scale like 1/gamma_e and
badly condition the (kappa, gamma_e) profile.

pi is invariant under (b, kappa) -> (c*b, c*kappa), so `b` is pinned by the explicit
normalisation mean(b) = 1 enforced in `TemporalNetworkModel.__post_init__`. With that
normalisation kappa is identified. kappa = 0 is the pure-A pole (partner drawn with
probability proportional to attractiveness, no tie memory); large kappa is the B pole.

Why this fixes the confound
---------------------------
Node memory acts on event *timing*, tie memory acts on partner *selection*. When the
initiator is observed the log-likelihood separates exactly into a timing part and a
selection part (see inference.py), so the two mechanisms are informed by disjoint
sufficient statistics. When the initiator is NOT observed -- the SocioPatterns case,
where contacts are undirected and unlabelled -- the two channels mix and the estimator
must marginalise over the latent initiator. Quantifying the resulting loss of
identifiability is the point of the study, not a nuisance.

All kernels are exponential, which lets the simulator and the likelihood share an O(1)
recursion for the excitation state instead of re-summing the history at every event.

Conventions
-----------
* Dyads are stored as ordered tuples (i, j) with i < j.
* Events are (time, initiator, partner); the *contact* is undirected, the label records
  who initiated. Empirical data without initiator labels is handled in inference.py.
* Time is continuous, float, starting at 0.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

Event = Tuple[float, int, int]  # (time, initiator, partner)


@dataclass
class MechanismParams:
    """Free parameters of the three-mechanism model.

    Parameters
    ----------
    n_node:
        Node branching ratio, integral of the node self-excitation kernel. Must lie in
        [0, 1); n_node >= 1 is the explosive regime and is rejected.
    gamma_n:
        Node memory decay rate (1/time). Larger = shorter node memory.
    kappa:
        Tie-reinforcement strength, in units of the attractiveness normalisation
        mean(b) = 1. kappa = 0 is the pure node-driven partner choice.
    gamma_e:
        Tie memory decay rate (1/time).
    """

    n_node: float
    gamma_n: float
    kappa: float
    gamma_e: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.n_node < 1.0):
            raise ValueError(
                f"n_node must be in [0, 1) for a stationary process, got {self.n_node}. "
                "n_node >= 1 is the explosive regime."
            )
        if self.gamma_n <= 0.0 or self.gamma_e <= 0.0:
            raise ValueError(
                f"decay rates must be strictly positive, got gamma_n={self.gamma_n}, "
                f"gamma_e={self.gamma_e}"
            )
        if self.kappa < 0.0:
            raise ValueError(f"kappa must be non-negative, got {self.kappa}")

    def as_vector(self) -> np.ndarray:
        return np.array([self.n_node, self.gamma_n, self.kappa, self.gamma_e], float)

    @staticmethod
    def from_vector(x: Sequence[float]) -> "MechanismParams":
        return MechanismParams(float(x[0]), float(x[1]), float(x[2]), float(x[3]))


@dataclass
class TemporalNetworkModel:
    """N nodes, baseline activities, attractiveness, mechanism parameters.

    `attractiveness` is normalised to mean 1 on construction. This is the identifying
    normalisation for kappa; it is not cosmetic and must not be removed.
    """

    N: int
    activities: np.ndarray          # a_i, baseline initiation rates, shape (N,)
    params: MechanismParams
    attractiveness: np.ndarray = field(default=None)  # b_j, shape (N,), mean 1

    def __post_init__(self) -> None:
        if self.N < 2:
            raise ValueError(f"need at least 2 nodes, got N={self.N}")
        self.activities = np.asarray(self.activities, dtype=float)
        if self.activities.shape != (self.N,):
            raise ValueError(
                f"activities must have shape ({self.N},), got {self.activities.shape}"
            )
        if np.any(self.activities < 0.0):
            raise ValueError("activities must be non-negative")
        if not np.any(self.activities > 0.0):
            raise ValueError("at least one activity must be strictly positive")

        if self.attractiveness is None:
            self.attractiveness = np.ones(self.N, dtype=float)
        else:
            self.attractiveness = np.asarray(self.attractiveness, dtype=float)
            if self.attractiveness.shape != (self.N,):
                raise ValueError(
                    f"attractiveness must have shape ({self.N},), "
                    f"got {self.attractiveness.shape}"
                )
            if np.any(self.attractiveness < 0.0):
                raise ValueError("attractiveness must be non-negative")
            mean_b = float(self.attractiveness.mean())
            if mean_b <= 0.0:
                raise ValueError("attractiveness must have strictly positive mean")
            # Identifying normalisation for kappa.
            self.attractiveness = self.attractiveness / mean_b


class ExcitationState:
    """Recursive O(1) state for the exponential excitation terms.

    Holds

        S_i   = sum_{t_k in H_i}  gamma_n exp(-gamma_n (t - t_k))     (node, unit integral)
        m_ij  = sum_{t_k in H_ij}          exp(-gamma_e (t - t_k))    (dyad, unit peak)
        M_i   = sum_{l != i} m_il                                     (dyad row sums)

    all evaluated at the current time `t`. Advancing time multiplies every term by a
    common decay factor, so `S` and `M` are updated in O(N) with numpy and the sparse
    dyad memories are decayed lazily on access.

    Using the recursion rather than re-summing the history is what makes the likelihood
    O(E) instead of O(E * events_per_dyad); it is the reason the empirical fit no longer
    needs to truncate the dataset.
    """

    def __init__(self, N: int, gamma_n: float, gamma_e: float) -> None:
        self.N = N
        self.gamma_n = gamma_n
        self.gamma_e = gamma_e
        self.t = 0.0
        self.S = np.zeros(N, dtype=float)
        self.M = np.zeros(N, dtype=float)
        # nbr[i][j] = (value_at_t_last, t_last), stored symmetrically
        self.nbr: List[Dict[int, Tuple[float, float]]] = [dict() for _ in range(N)]

    def advance_to(self, t: float) -> None:
        """Decay all state to time `t`. Must be called with non-decreasing `t`."""
        dt = t - self.t
        if dt < 0.0:
            raise ValueError(
                f"events must be time-ordered; tried to advance from t={self.t} to {t}"
            )
        if dt == 0.0:
            return
        self.S *= math.exp(-self.gamma_n * dt)
        self.M *= math.exp(-self.gamma_e * dt)
        self.t = t

    def dyad_memory(self, i: int, j: int) -> float:
        """m_ij evaluated at the current time, with lazy decay."""
        entry = self.nbr[i].get(j)
        if entry is None:
            return 0.0
        value, t_last = entry
        return value * math.exp(-self.gamma_e * (self.t - t_last))

    def partner_weights(self, i: int, attractiveness: np.ndarray,
                        kappa: float) -> np.ndarray:
        """Unnormalised partner weights b_l + kappa*m_il for all l, with w_i = 0."""
        w = attractiveness.copy()
        if kappa > 0.0:
            for j, (value, t_last) in self.nbr[i].items():
                w[j] += kappa * value * math.exp(-self.gamma_e * (self.t - t_last))
        w[i] = 0.0
        return w

    def partner_normaliser(self, i: int, b_total: float, attractiveness: np.ndarray,
                           kappa: float) -> float:
        """Z_i = sum_{l != i} (b_l + kappa m_il), in O(1) from the cached row sum."""
        return (b_total - attractiveness[i]) + kappa * self.M[i]

    def register(self, t: float, i: int, j: int) -> None:
        """Record an event at time `t` with initiator i and partner j."""
        self.advance_to(t)
        self.S[i] += self.gamma_n     # unit-integral node kernel
        m_now = self.dyad_memory(i, j) + 1.0   # unit-peak tie kernel
        self.nbr[i][j] = (m_now, t)
        self.nbr[j][i] = (m_now, t)
        self.M[i] += 1.0
        self.M[j] += 1.0


def node_intensities(model: TemporalNetworkModel, state: ExcitationState) -> np.ndarray:
    """rho_i at the state's current time."""
    return model.activities + model.params.n_node * state.S


def simulate(model: TemporalNetworkModel, T_max: float,
             rng: np.random.Generator | None = None,
             max_events: int | None = None) -> List[Event]:
    """Simulate the process on [0, T_max] by Ogata thinning.

    The thinning bound is exact: between events every excitation term decays
    monotonically, so Lambda*(t) = sum_i rho_i(t) evaluated immediately after the last
    event dominates the total intensity until the next candidate point.

    Raises
    ------
    RuntimeError
        If `max_events` is exceeded, which indicates a near-explosive parameter choice
        rather than a finite sample. Failing loudly is deliberate: silently truncating
        would bias every downstream estimate.
    """
    if T_max <= 0.0:
        raise ValueError(f"T_max must be positive, got {T_max}")
    if rng is None:
        rng = np.random.default_rng()
    if max_events is None:
        max_events = 20_000_000

    p = model.params
    b = model.attractiveness
    b_total = float(b.sum())

    state = ExcitationState(model.N, p.gamma_n, p.gamma_e)
    events: List[Event] = []
    t = 0.0

    while t < T_max:
        state.advance_to(t)
        lam = node_intensities(model, state)
        lam_star = float(lam.sum())
        if lam_star <= 0.0:
            break

        t = t - math.log(rng.random()) / lam_star
        if t >= T_max:
            break

        state.advance_to(t)
        lam = node_intensities(model, state)
        lam_total = float(lam.sum())
        if lam_total > lam_star * (1.0 + 1e-9):
            raise RuntimeError(
                "thinning bound violated (lam_total > lam_star); the excitation state "
                "is not decaying monotonically, which indicates a bug"
            )
        if rng.random() * lam_star >= lam_total:
            continue  # thinned

        i = int(rng.choice(model.N, p=lam / lam_total))

        w = state.partner_weights(i, b, p.kappa)
        w_total = float(w.sum())
        if w_total <= 0.0:
            raise RuntimeError(
                f"node {i} has zero total partner weight at t={t}; check that "
                "attractiveness is not identically zero off-diagonal"
            )
        j = int(rng.choice(model.N, p=w / w_total))

        state.register(t, i, j)
        events.append((t, i, j))
        if len(events) > max_events:
            raise RuntimeError(
                f"exceeded max_events={max_events} at t={t:.3f} of T_max={T_max}. "
                f"n_node={p.n_node} is likely too close to 1 (explosive regime)."
            )

    return events


def strip_initiators(events: Sequence[Event]) -> List[Tuple[float, int, int]]:
    """Drop initiator labels, returning undirected contacts with i < j.

    This is the operation that empirical proximity data (SocioPatterns, call records)
    performs implicitly. Comparing inference on `events` against inference on
    `strip_initiators(events)` isolates exactly how much identifiability the missing
    initiator label costs.
    """
    out = []
    for t, i, j in events:
        a, b_ = (i, j) if i < j else (j, i)
        out.append((t, a, b_))
    return out


# ======================================================================================
# HIGHER-ORDER EXTENSION
#
# Motivated by a measured failure, not by generality for its own sake. Fitting the dyadic
# model above to face-to-face and to email data rejects it, and the rejection localises
# to one cause: a group interaction is recorded as several simultaneous dyads, so the
# initiator has inter-event times of exactly zero. On SocioPatterns High School the
# Kolmogorov-Smirnov statistic of the timing residuals was 0.094 against a tied-event
# fraction of 0.088; on email-Eu the two agree to four significant figures (0.4349 vs
# 404/929 = 0.4349). The marginal misfit IS the zero atom.
#
# The remedy is to make a group interaction ONE event with several partners. Below,
# `GroupParams` and `simulate_group_events` do that, and they are built so that
#
#     k == 2 for every event   =>   EXACTLY the dyadic model above.
#
# That exact reduction is the point. It is what makes the extension non-redundant: the
# higher-order machinery is a strict superset, `kappa_g = 0` recovers pairwise-only group
# formation, and a size distribution concentrated on k = 2 recovers the original model
# including its parametrisation and normalisations. `tests/test_model.py` checks the
# reduction numerically rather than trusting the claim.
# ======================================================================================

@dataclass
class GroupParams:
    """Higher-order parameters, layered on top of `MechanismParams`.

    kappa_g is the object of interest: kappa_g = 0 states that group memory is entirely
    reducible to pairwise memory, so the higher-order hypothesis is a single parameter at
    a boundary rather than a separate model.

    kappa_1p is the CONTROL. It carries reinforcement among already-recruited members.
    Without it, a node joining because it knows the current members -- ordinary triadic
    closure -- is misattributed to kappa_g. Omitting it is the most direct route to a
    false positive for higher-order memory, so it is not optional.
    """

    kappa_g: float = 0.0        # group-as-object memory
    gamma_g: float = 0.02       # group memory decay
    kappa_1p: float = 0.0       # reinforcement among recruited members (control term)
    size_p: float = 1.0         # truncated geometric on k >= 2; size_p = 1 => all k = 2
    k_max: int = 2

    def __post_init__(self) -> None:
        if self.kappa_g < 0 or self.kappa_1p < 0:
            raise ValueError("kappa_g and kappa_1p must be non-negative")
        if self.gamma_g <= 0:
            raise ValueError(f"gamma_g must be positive, got {self.gamma_g}")
        if not (0.0 < self.size_p <= 1.0):
            raise ValueError(f"size_p must be in (0, 1], got {self.size_p}")
        if self.k_max < 2:
            raise ValueError(f"k_max must be >= 2, got {self.k_max}")

    @property
    def is_dyadic(self) -> bool:
        """True when this reduces exactly to the pairwise model."""
        return self.k_max == 2 or self.size_p == 1.0

    def size_probs(self) -> np.ndarray:
        ks = np.arange(2, self.k_max + 1)
        w = self.size_p * (1.0 - self.size_p) ** (ks - 2)
        return w / w.sum()


class GroupMemory:
    """Sparse memory of subgroups that have convened, keyed by frozenset.

    Only subsets of order >= 3 are stored; orders 0 and 2 are already carried by the
    node and dyad states of `ExcitationState`, and duplicating them would double-count.

    Containment, not exact matching: an event on group G increments the memory of every
    subset of G of order >= 3. Exact-set matching would make large groups almost surely
    memoryless for combinatorial reasons -- for N = 300 there are ~3e8 possible 5-sets --
    whereas "this subgroup has convened before" is the quantity with meaning. |G| is small
    in practice (<= 9 in SocioPatterns), so enumerating 2^|G| subsets is cheap.
    """

    def __init__(self, gamma_g: float) -> None:
        if gamma_g <= 0:
            raise ValueError(f"gamma_g must be positive, got {gamma_g}")
        self.gamma_g = gamma_g
        self.store: Dict[frozenset, Tuple[float, float]] = {}

    def value(self, sigma: frozenset, t: float) -> float:
        e = self.store.get(sigma)
        if e is None:
            return 0.0
        return e[0] * math.exp(-self.gamma_g * (t - e[1]))

    def register(self, t: float, group: Sequence[int]) -> None:
        from itertools import combinations
        g = sorted(group)
        for order in range(3, len(g) + 1):
            for sub in combinations(g, order):
                key = frozenset(sub)
                self.store[key] = (self.value(key, t) + 1.0, t)


def simulate_group_events(model: TemporalNetworkModel, group: GroupParams,
                          T_max: float, rng: np.random.Generator | None = None,
                          max_events: int | None = None
                          ) -> List[Tuple[float, int, Tuple[int, ...]]]:
    """Simulate group events: (time, initiator, partners).

    Timing is unchanged from `simulate` -- the node initiation intensity is the same
    object, so the branching-ratio interpretation of n_node and the stationary rate of
    Eq. (rate) carry over untouched.

    Partner selection becomes SEQUENTIAL RECRUITMENT: the group accretes one member at a
    time, each drawn from a choice model over the O(N) remaining candidates. A choice
    model over subsets would need a normaliser summing C(N-1, k-1) alternatives, which is
    intractable; sequential recruitment reduces it to O(kN) and describes the process more
    naturally anyway.

    The normaliser is load-bearing. Larger groups recur less for purely combinatorial
    reasons, so a memory parameter estimated without dividing by the available
    alternatives measures the binomial coefficient rather than social memory.

    Reduction: with group.is_dyadic every event has k = 2, no recruitment beyond the
    first partner occurs, kappa_g and kappa_1p never enter, and the weights are exactly
    those of `ExcitationState.partner_weights`. The output is then the dyadic process.
    """
    if T_max <= 0.0:
        raise ValueError(f"T_max must be positive, got {T_max}")
    if rng is None:
        rng = np.random.default_rng()
    if max_events is None:
        max_events = 20_000_000

    p = model.params
    b = model.attractiveness
    state = ExcitationState(model.N, p.gamma_n, p.gamma_e)
    gmem = GroupMemory(group.gamma_g)
    size_probs = group.size_probs()
    ks = np.arange(2, group.k_max + 1)

    events: List[Tuple[float, int, Tuple[int, ...]]] = []
    t = 0.0
    while t < T_max:
        state.advance_to(t)
        lam = node_intensities(model, state)
        lam_star = float(lam.sum())
        if lam_star <= 0.0:
            break
        t = t - math.log(rng.random()) / lam_star
        if t >= T_max:
            break
        state.advance_to(t)
        lam = node_intensities(model, state)
        lam_total = float(lam.sum())
        if lam_total > lam_star * (1.0 + 1e-9):
            raise RuntimeError("thinning bound violated; excitation state is not decaying")
        if rng.random() * lam_star >= lam_total:
            continue

        i = int(rng.choice(model.N, p=lam / lam_total))
        # When the model is dyadic the size is deterministic, so no random draw is
        # made. This is not an optimisation: consuming a variate here would desynchronise
        # the random stream from `simulate` and the reduction to the dyadic model would
        # hold only in distribution. Skipping it makes the reduction hold PATHWISE, which
        # is testable to the last event.
        k = 2 if group.is_dyadic else min(int(rng.choice(ks, p=size_probs)), model.N)

        rho = [i]
        for _ in range(k - 1):
            w = _recruitment_weights(state, gmem, rho, i, model, group, t)
            tot = float(w.sum())
            if tot <= 0.0:
                break
            rho.append(int(rng.choice(model.N, p=w / tot)))
        if len(rho) < 2:
            continue

        # Register every dyad in the group, then the subgroup memory.
        from itertools import combinations
        state.advance_to(t)
        state.S[i] += p.gamma_n
        for u, v in combinations(sorted(rho), 2):
            m_now = state.dyad_memory(u, v) + 1.0
            state.nbr[u][v] = (m_now, t)
            state.nbr[v][u] = (m_now, t)
            state.M[u] += 1.0
            state.M[v] += 1.0
        if len(rho) >= 3:
            gmem.register(t, rho)

        events.append((t, i, tuple(x for x in rho if x != i)))
        if len(events) > max_events:
            raise RuntimeError(f"exceeded max_events at t={t:.3f}; n_node near 1?")
    return events


def _recruitment_weights(state: "ExcitationState", gmem: GroupMemory, rho: List[int],
                         initiator: int, model: TemporalNetworkModel,
                         group: GroupParams, t: float) -> np.ndarray:
    """w(j | rho) = [b_j + kappa*m_ij + kappa_1p*sum_{l in rho\\{i}} m_lj] * [1 + kappa_g*M]."""
    p = model.params
    w = state.partner_weights(initiator, model.attractiveness, p.kappa)
    if group.kappa_1p > 0.0:
        for l in rho:
            if l == initiator:
                continue
            for j in range(model.N):
                w[j] += group.kappa_1p * state.dyad_memory(l, j)
    if group.kappa_g > 0.0:
        base = frozenset(rho)
        for j in range(model.N):
            if j not in base:
                w[j] *= 1.0 + group.kappa_g * gmem.value(base | {j}, t)
    for l in rho:
        w[l] = 0.0
    return w


def group_events_to_dyads(events: Sequence[Tuple[float, int, Tuple[int, ...]]]
                          ) -> List[Event]:
    """Project group events onto simultaneous dyads -- the defective representation.

    Provided so the artefact can be reproduced deliberately: applying the dyadic
    goodness-of-fit machinery to this output regenerates the zero-inflation, and the
    resulting KS statistic should equal the fraction of events with k > 2 weighted by
    k - 2. That is the controlled version of what was observed on real data.
    """
    out: List[Event] = []
    for t, i, partners in events:
        for j in partners:
            out.append((t, i, j))
    return out


def summarise(events: Sequence[Event], N: int, T_max: float) -> Dict[str, float]:
    """Cheap descriptive statistics, useful for sanity-checking a simulation."""
    if len(events) == 0:
        return {"n_events": 0.0, "rate": 0.0, "n_distinct_dyads": 0.0,
                "repeat_fraction": 0.0, "dyad_saturation": 0.0}
    dyads = set()
    for t, i, j in events:
        dyads.add((i, j) if i < j else (j, i))
    return {
        "n_events": float(len(events)),
        "rate": len(events) / T_max,
        "n_distinct_dyads": float(len(dyads)),
        "repeat_fraction": 1.0 - len(dyads) / len(events),
        "dyad_saturation": len(dyads) / (0.5 * N * (N - 1)),
    }

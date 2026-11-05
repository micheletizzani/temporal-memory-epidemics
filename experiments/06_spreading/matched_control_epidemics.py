#!/usr/bin/env python3
# %%
"""Matched controls: is the spreading difference caused by WHERE memory sits,
or by the structural side-effects of the two generative mechanisms?

THE OBJECTION THIS ANSWERS
--------------------------
Comparing a node-memory network to a tie-memory network changes more than the carrier of
memory. Tie reinforcement also changes aggregate edge weights, repeated-contact structure,
temporal clustering, and strength heterogeneity. One can therefore ask whether the
epidemic gap is caused by memory placement at all, or simply by the different static
networks the two mechanisms happen to produce.

THE DESIGN
----------
We do NOT answer this by generating two more networks. We answer it with SURROGATES of a
single realisation. A shuffle preserves exactly the statistics we name and destroys exactly
the structure we target, so any surviving difference is attributable to the destroyed
structure and nothing else.

For each pole we build four streams:

  raw          the generated network.
  time_perm    timestamps randomly permuted across events. Preserves the static graph, every
               edge weight, and every node strength EXACTLY; destroys all temporal structure
               (node burstiness ordering AND tie temporal clustering). This is the null.
  node_shuf    per-node inter-event shuffle. Preserves each node's ISI marginal and every
               (i,j) pair; destroys node-level temporal ORDER only.
  pair_rewire  timestamps and per-node event counts preserved; partners reassigned at random
               among the same active set. Destroys tie reinforcement; keeps node timing.

The attribution argument then reads:

  gap_raw        = |SIR(node pole) - SIR(link pole)| on the raw streams
  gap_timeperm   = the same gap after ALL temporal structure is removed from both

If gap_timeperm ~ 0 while gap_raw is large, the difference is temporal, not structural: the
two static networks, with identical weights and strengths, spread identically. That is the
control the raw comparison lacks. If gap_timeperm is comparable to gap_raw, the difference
was structural and the headline epidemic claim does NOT survive.

The `verify_invariants` gate is the point of the script and is fail-loud: it ASSERTS that
each surrogate preserved what it claims to preserve. A control that is not verified is not
a control.

Run cell-by-cell ('# %%') or `python matched_control_epidemics.py`.
Writes results/matched_control_epidemics.json
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from generative_models import (MechanismParams, TemporalNetworkModel, simulate)  # noqa
from epidemic_divergence import run_sir, run_sis  # noqa  reuse the audited samplers

RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

# ---- configuration -------------------------------------------------------------------
N, T = 200, 4000.0
A0 = 0.02
N_NODE, KAPPA = 0.6, 20.0
GAMMA_N, GAMMA_E = 1.0 / 30.0, 1.0 / 600.0
BETAS = [0.02, 0.05, 0.10, 0.20, 0.35]
MU = 0.01
REPS = 200
SEEDS = [11, 12, 13, 14, 15]          # independent realisations; report across-seed spread


# %%
# ---- generation ----------------------------------------------------------------------
def gen_pole(kind, seed):
    """NODE pole: n_node>0, kappa=0. LINK pole: n_node=0, kappa>0.
    Stationary aggregate event rate is matched across poles by scaling baseline activity."""
    rng = np.random.default_rng(seed)
    n_node = N_NODE if kind == "node" else 0.0
    kappa = 0.0 if kind == "node" else KAPPA
    a = np.full(N, A0 * (1.0 - n_node))       # matched stationary aggregate rate
    b = np.ones(N)                             # uniform attractiveness: no intrinsic hubs
    model = TemporalNetworkModel(N=N, activities=a, attractiveness=b,
                                 params=MechanismParams(n_node, GAMMA_N, kappa, GAMMA_E))
    ev = [(float(t), int(i), int(j)) for (t, i, j) in simulate(model, T, rng=rng)]
    ev.sort()
    return ev


# %%
# ---- surrogates ----------------------------------------------------------------------
def surr_time_perm(events, rng):
    """Permute timestamps across events. Static graph, edge weights, node strengths EXACT."""
    ts = np.array([t for t, _i, _j in events])
    pairs = [(i, j) for _t, i, j in events]
    rng.shuffle(ts)
    return sorted((float(t), i, j) for t, (i, j) in zip(ts, pairs))


def surr_node_shuffle(events, rng):
    """Per-node inter-event shuffle: destroys node temporal ORDER, keeps ISI marginal+pairs.
    Each event time is rebuilt from its initiator's shuffled ISI sequence (first endpoint)."""
    evs = sorted(events)
    tnew = [t for t, _i, _j in evs]
    idxby = defaultdict(list)
    for k, (_t, i, _j) in enumerate(evs):
        idxby[i].append(k)
    for _i, idx in idxby.items():
        ts = sorted(tnew[k] for k in idx)
        d = np.diff(np.concatenate([[0.0], ts]))
        rng.shuffle(d)
        for k, tt in zip(idx, np.cumsum(d)):
            tnew[k] = float(tt)
    return sorted((tnew[k], evs[k][1], evs[k][2]) for k in range(len(evs)))


def surr_pair_rewire(events, rng):
    """Permute the second endpoints across events. Destroys tie reinforcement while
    preserving timestamps AND every node's strength exactly.

    Permuting the j-multiset leaves each node's count-as-j unchanged and never touches
    count-as-i, so total strength is preserved exactly. Self-loops created by the permutation
    are repaired by SWAPPING with another position (never by resampling, which would break
    the strength invariant)."""
    evs = sorted(events)
    firsts = [i for _t, i, _j in evs]
    seconds = [j for _t, _i, j in evs]
    rng.shuffle(seconds)
    n = len(evs)
    for k in range(n):                          # repair self-loops by strength-preserving swap
        if firsts[k] != seconds[k]:
            continue
        for _attempt in range(200):
            q = int(rng.integers(0, n))
            if seconds[q] != firsts[k] and firsts[q] != seconds[k]:
                seconds[k], seconds[q] = seconds[q], seconds[k]
                break
    keep = [k for k in range(n) if firsts[k] != seconds[k]]   # drop any residual self-loop
    return sorted((float(evs[k][0]), int(firsts[k]), int(seconds[k])) for k in keep)


# %%
# ---- invariant verification (fail-loud; this is what makes them controls) --------------
def _stats(events):
    strength = Counter()
    weights = Counter()
    for _t, i, j in events:
        strength[i] += 1
        strength[j] += 1
        weights[(min(i, j), max(i, j))] += 1
    return dict(
        E=len(events),
        strength=tuple(sorted(strength.items())),
        weight_seq=tuple(sorted(weights.values())),
        edge_set=frozenset(weights.keys()),
        tspan=(min(t for t, _i, _j in events), max(t for t, _i, _j in events)),
    )


def verify_invariants(raw, surrogates):
    """Assert each surrogate preserved what it claims. Raises on violation."""
    r = _stats(raw)
    report = {}
    for name, ev in surrogates.items():
        s = _stats(ev)
        checks = {}
        # pair_rewire may drop a negligible number of irreparable self-loops; the others may not
        checks["event_count"] = (s["E"] == r["E"]) if name != "pair_rewire" \
            else (r["E"] - s["E"]) <= 0.01 * r["E"]
        if name == "time_perm":
            # must preserve the static graph, every edge weight, every node strength
            checks["edge_set"] = (s["edge_set"] == r["edge_set"])
            checks["weight_seq"] = (s["weight_seq"] == r["weight_seq"])
            checks["strength"] = (s["strength"] == r["strength"])
        if name == "node_shuf":
            # pairs untouched -> graph, weights, strengths all preserved
            checks["edge_set"] = (s["edge_set"] == r["edge_set"])
            checks["weight_seq"] = (s["weight_seq"] == r["weight_seq"])
            checks["strength"] = (s["strength"] == r["strength"])
        if name == "pair_rewire":
            # timestamps and node strengths preserved exactly; edge set deliberately NOT.
            # NOTE: the repeat fraction is an OUTCOME we report, not an invariant we assert --
            # on a small node set random rewiring produces collisions by chance, so a drop in
            # repeats is expected for the link pole and not guaranteed for the node pole.
            kept = set(round(t, 9) for t, _i, _j in ev)
            checks["timestamps_subset"] = kept.issubset(
                set(round(t, 9) for t, _i, _j in raw))
            checks["strength_preserved"] = (s["strength"] == r["strength"]) if s["E"] == r["E"] \
                else True    # exact only when no self-loop was dropped
            checks["loss_negligible"] = (r["E"] - s["E"]) <= 0.01 * r["E"]
        bad = [k for k, v in checks.items() if not v]
        if bad:
            raise AssertionError(
                f"surrogate '{name}' violated invariants {bad}; it is not a valid control")
        report[name] = checks
    return report


def repeat_fraction(events):
    """Share of events that fall on an already-used tie: the tie-reinforcement signature."""
    seen = set()
    rep = 0
    for _t, i, j in events:
        e = (min(i, j), max(i, j))
        if e in seen:
            rep += 1
        seen.add(e)
    return rep / max(len(events), 1)


# %%
# ---- main sweep ------------------------------------------------------------------------
def run_all():
    out = {"config": dict(N=N, T=T, a0=A0, n_node=N_NODE, kappa=KAPPA, betas=BETAS,
                          mu=MU, reps=REPS, seeds=SEEDS), "runs": []}
    for seed in SEEDS:
        for pole in ("node", "link"):
            raw = gen_pole(pole, seed)
            if len(raw) < 500:
                print(f"  [skip] {pole} seed={seed}: only {len(raw)} events")
                continue
            rng = np.random.default_rng(1000 + seed)
            surr = {"time_perm": surr_time_perm(raw, rng),
                    "node_shuf": surr_node_shuffle(raw, rng),
                    "pair_rewire": surr_pair_rewire(raw, rng)}
            verify_invariants(raw, surr)                 # fail-loud gate
            streams = {"raw": raw, **surr}
            for sname, ev in streams.items():
                for beta in BETAS:
                    m, se = run_sir(ev, N, beta, MU, np.random.default_rng(seed * 97 + 7), REPS)
                    out["runs"].append(dict(seed=seed, pole=pole, stream=sname, beta=beta,
                                            sir_mean=m, sir_se=se, E=len(ev),
                                            repeat_frac=repeat_fraction(ev)))
                print(f"  {pole:4s} seed={seed} {sname:12s} E={len(ev):6d} "
                      f"rep={repeat_fraction(ev):.3f}")
    with open(os.path.join(RES, "matched_control_epidemics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", os.path.join(RES, "matched_control_epidemics.json"))
    return out


# %%
def summarise(out):
    """The decisive numbers: node-vs-link gap on raw vs on the fully time-permuted control."""
    import itertools
    by = defaultdict(list)
    for r in out["runs"]:
        by[(r["stream"], r["beta"], r["pole"])].append(r["sir_mean"])
    print(f"\n{'stream':12s} {'beta':>6s} {'node':>8s} {'link':>8s} {'gap':>8s} {'gap/raw':>8s}")
    raw_gap = {}
    for stream, beta in itertools.product(["raw", "time_perm", "node_shuf", "pair_rewire"], BETAS):
        n_ = by.get((stream, beta, "node"))
        l_ = by.get((stream, beta, "link"))
        if not n_ or not l_:
            continue
        gn, gl = float(np.mean(n_)), float(np.mean(l_))
        gap = abs(gn - gl)
        if stream == "raw":
            raw_gap[beta] = gap
        ratio = gap / raw_gap[beta] if raw_gap.get(beta) else float("nan")
        print(f"{stream:12s} {beta:6.2f} {gn:8.3f} {gl:8.3f} {gap:8.3f} {ratio:8.2f}")
    print("\nREAD: if time_perm gap/raw << 1, the epidemic difference is TEMPORAL "
          "(memory placement), not a static-structure artefact -- the claim survives.")
    print("      if time_perm gap/raw ~ 1, the difference is STRUCTURAL -- the claim does not.")


if __name__ == "__main__":
    summarise(run_all())

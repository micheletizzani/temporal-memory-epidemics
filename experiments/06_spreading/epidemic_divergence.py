#!/usr/bin/env python3
"""Do node-memory and link-memory networks spread epidemics differently?

The introduction claims node self-excitation and tie reinforcement are not interchangeable
for spreading. This experiment proves it. We generate two temporal networks from the pure
poles of the model, matched on the coarse statistics (same N, same baseline activities, so
the same aggregate event rate and comparable event counts):

  NODE pole : n_node>0, kappa=0  -- bursty node timing, partners chosen at random.
  LINK pole : n_node=0, kappa>0  -- memoryless (Poisson) timing, partners reinforced.

We then run many SIR realisations on each at a range of per-contact transmission
probabilities beta, and compare the final epidemic size. If the two curves diverge, the
mechanism a modeller assumes materially changes an epidemic forecast, so mis-identifying it
(the paper's warning) has real consequences.

Python reference implementation (small N); a Julia port scales it. Deliberately simple and
fail-loud so the Julia version can be checked against it.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import heapq
import os
import sys

import numpy as np

from generative_models import (MechanismParams, TemporalNetworkModel, simulate)  # noqa


def gen_pole(kind, N, T, a0, n_node, kappa, seed):
    rng = np.random.default_rng(seed)
    a = np.full(N, a0 * (1.0 - n_node))          # fix stationary aggregate rate across poles
    b = np.ones(N)                                # UNIFORM base attractiveness for both poles:
    #                                              the only difference is the memory carrier,
    #                                              not intrinsic hubs (matched degree base).
    model = TemporalNetworkModel(N=N, activities=a, attractiveness=b,
                                 params=MechanismParams(n_node, 1.0 / 30.0, kappa, 1.0 / 600.0))
    ev = [(float(t), int(i), int(j)) for (t, i, j) in simulate(model, T, rng=rng)]
    ev.sort()
    return ev


def run_sir(events, N, beta, mu, rng, reps):
    """Temporal-network SIR; returns mean final size (fraction ever infected)."""
    finals = np.zeros(reps)
    
    first_contact = np.full(N, np.inf)
    for (t, i, j) in events:
        if t < first_contact[i]: first_contact[i] = t
        if t < first_contact[j]: first_contact[j] = t

    for r in range(reps):
        state = np.zeros(N, np.int8)             # 0=S 1=I 2=R
        rec = np.full(N, np.inf)
        
        # Pick a valid seed node that actually has contacts
        valid_nodes = [n for n in range(N) if first_contact[n] < np.inf]
        if not valid_nodes: return 0.0, 0.0
        seed = int(rng.choice(valid_nodes))
        
        state[seed] = 1; rec[seed] = first_contact[seed] + rng.exponential(1.0 / mu)
        ever = 1
        for (t, i, j) in events:
            if state[i] == 1 and t >= rec[i]:
                state[i] = 2
            if state[j] == 1 and t >= rec[j]:
                state[j] = 2
            si, sj = state[i], state[j]
            if si == 1 and sj == 0 and rng.random() < beta:
                state[j] = 1; rec[j] = t + rng.exponential(1.0 / mu); ever += 1
            elif sj == 1 and si == 0 and rng.random() < beta:
                state[i] = 1; rec[i] = t + rng.exponential(1.0 / mu); ever += 1
        finals[r] = ever / N
    return float(finals.mean()), float(finals.std() / np.sqrt(reps))


def run_sis(events, N, beta, mu, rng, reps):
    """Temporal-network SIS; returns mean endemic prevalence (2nd 40% of the timeline).

    Event-driven with a recovery heap so recoveries are exact even under reinfection; a
    popped recovery fires only if it is the node's latest scheduled one (rec[n]).
    """
    t0, tend = events[0][0], events[-1][0]
    thalf = t0 + 0.6 * (tend - t0)
    prevs = np.zeros(reps)
    
    first_contact = np.full(N, np.inf)
    for (t, i, j) in events:
        if t < first_contact[i]: first_contact[i] = t
        if t < first_contact[j]: first_contact[j] = t

    for r in range(reps):
        state = np.zeros(N, np.int8)                 # 0 S, 1 I
        rec = np.full(N, np.inf)
        heap = []
        
        valid_nodes = [n for n in range(N) if first_contact[n] < np.inf]
        if not valid_nodes: return 0.0, 0.0
        seed = int(rng.choice(valid_nodes))
        
        state[seed] = 1; rec[seed] = first_contact[seed] + rng.exponential(1.0 / mu)
        heapq.heappush(heap, (rec[seed], seed))
        infc = 1; psum = 0.0; pn = 0
        for (t, i, j) in events:
            while heap and heap[0][0] <= t:          # apply due recoveries
                rt, n = heapq.heappop(heap)
                if state[n] == 1 and rt == rec[n]:
                    state[n] = 0; infc -= 1
            si, sj = state[i], state[j]
            if si == 1 and sj == 0 and rng.random() < beta:
                state[j] = 1; infc += 1
                rec[j] = t + rng.exponential(1.0 / mu); heapq.heappush(heap, (rec[j], j))
            elif sj == 1 and si == 0 and rng.random() < beta:
                state[i] = 1; infc += 1
                rec[i] = t + rng.exponential(1.0 / mu); heapq.heappush(heap, (rec[i], i))
            if t >= thalf:
                psum += infc; pn += 1
        prevs[r] = (psum / max(pn, 1)) / N
    return float(prevs.mean()), float(prevs.std() / np.sqrt(reps))


def extract_triangles(events, N):
    adj = [set() for _ in range(N)]
    for (t, i, j) in events:
        adj[i].add(j)
        adj[j].add(i)
    triangles_per_edge = {}
    for i in range(N):
        for j in adj[i]:
            common = adj[i].intersection(adj[j])
            triangles_per_edge[(i, j)] = list(common)
    return triangles_per_edge


def run_simplicial(events, N, beta, mu, rng, reps):
    """Temporal-network SIS Simplicial Contagion (Group interaction).
    beta is beta1. beta2 is set to 2*beta1.
    """
    beta1 = beta
    beta2 = min(1.0, 2.0 * beta)
    triangles = extract_triangles(events, N)
    t0, tend = events[0][0], events[-1][0]
    thalf = t0 + 0.6 * (tend - t0)
    prevs = np.zeros(reps)
    for r in range(reps):
        state = np.zeros(N, np.int8)                 # 0 S, 1 I
        rec = np.full(N, np.inf)
        heap = []
        seed = int(rng.integers(N))
        state[seed] = 1; rec[seed] = t0 + rng.exponential(1.0 / mu)
        heapq.heappush(heap, (rec[seed], seed))
        infc = 1; psum = 0.0; pn = 0
        for (t, i, j) in events:
            while heap and heap[0][0] <= t:
                rt, n = heapq.heappop(heap)
                if state[n] == 1 and rt == rec[n]:
                    state[n] = 0; infc -= 1
            si, sj = state[i], state[j]
            
            def attempt_infection(target, source):
                nonlocal infc
                m = sum(1 for k in triangles.get((source, target), []) if state[k] == 1)
                prob = 1.0 - (1.0 - beta1) * ((1.0 - beta2) ** m)
                if rng.random() < prob:
                    state[target] = 1; infc += 1
                    rec[target] = t + rng.exponential(1.0 / mu)
                    heapq.heappush(heap, (rec[target], target))
                    
            if si == 1 and sj == 0:
                attempt_infection(j, i)
            elif sj == 1 and si == 0:
                attempt_infection(i, j)
            if t >= thalf:
                psum += infc; pn += 1
        prevs[r] = (psum / max(pn, 1)) / N
    return float(prevs.mean()), float(prevs.std() / np.sqrt(reps))


def run_threshold(events, N, beta, mu, rng, reps):
    """Temporal-network SIS Threshold Contagion.
    Dose increases by beta per contact with I, decays over time. Threshold = 1.0.
    """
    decay = 1.0 / 30.0
    threshold = 1.0
    jump = beta * 2.0  # scale jump with beta
    t0, tend = events[0][0], events[-1][0]
    thalf = t0 + 0.6 * (tend - t0)
    prevs = np.zeros(reps)
    for r in range(reps):
        state = np.zeros(N, np.int8)
        rec = np.full(N, np.inf)
        dose = np.zeros(N)
        last_t = np.full(N, t0)
        heap = []
        seed = int(rng.integers(N))
        state[seed] = 1; rec[seed] = t0 + rng.exponential(1.0 / mu)
        heapq.heappush(heap, (rec[seed], seed))
        infc = 1; psum = 0.0; pn = 0
        for (t, i, j) in events:
            while heap and heap[0][0] <= t:
                rt, n = heapq.heappop(heap)
                if state[n] == 1 and rt == rec[n]:
                    state[n] = 0; infc -= 1
                    dose[n] = 0.0
            
            def update_dose(target, source):
                nonlocal infc
                dt = t - last_t[target]
                dose[target] = dose[target] * np.exp(-decay * dt)
                last_t[target] = t
                if state[source] == 1:
                    dose[target] += jump
                    if dose[target] >= threshold:
                        state[target] = 1; infc += 1
                        rec[target] = t + rng.exponential(1.0 / mu)
                        heapq.heappush(heap, (rec[target], target))
                        dose[target] = 0.0
            
            if state[j] == 0:
                update_dose(j, i)
            if state[i] == 0:
                update_dose(i, j)
                
            if t >= thalf:
                psum += infc; pn += 1
        prevs[r] = (psum / max(pn, 1)) / N
    return float(prevs.mean()), float(prevs.std() / np.sqrt(reps))

def main():
    N, T, a0 = 100, 1200.0, 0.12                 # small: Python reference (Julia scales it)
    betas = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6])
    mu = 1.0 / 120.0                             # recovery timescale
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 400      # replicates per (model, pole, beta); 40 gave SE ~0.07

    node_ev = gen_pole("node", N, T, a0, n_node=0.8, kappa=0.0, seed=1)
    link_ev = gen_pole("link", N, T, a0, n_node=0.0, kappa=40.0, seed=2)
    print(f"NODE pole: {len(node_ev)} events; LINK pole: {len(link_ev)} events "
          f"(N={N}, T={T:.0f})")

    rng = np.random.default_rng(0)
    out = {"betas": betas.tolist()}
    for model_name, runner in [("sir", run_sir), ("sis", run_sis), 
                               ("simplicial", run_simplicial), ("threshold", run_threshold)]:
        for name, ev in [("node", node_ev), ("link", link_ev)]:
            means, ses = [], []
            for be in betas:
                m, s = runner(ev, N, float(be), mu, rng, reps)
                means.append(m); ses.append(s)
            out[f"{model_name}_{name}"] = means; out[f"{model_name}_{name}_se"] = ses
            lab = "final size" if model_name == "sir" else "prevalence"
            print(f"  {model_name} {name} ({lab}) = " + ", ".join(f"{m:.2f}" for m in means))

    import json
    resdir = os.path.join(ROOT, "results"); os.makedirs(resdir, exist_ok=True)
    json.dump(out, open(os.path.join(resdir, "epidemic_divergence.json"), "w"), indent=2)
    _plot(out)



def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pass
    import plotstyle  # noqa: E402
    plotstyle.apply()
    outdir = os.path.abspath(os.path.join(ROOT, "figures"))
    betas = np.array(out["betas"])
    fig, axes = plt.subplots(2, 2, figsize=(plotstyle.FULL, 5.4))
    ((aS, aE), (aSim, aThr)) = axes
    poles = [("node", "#82A0C2", r"node self-excitation ($\kappa=0$)"),
             ("link", "#85496F", r"tie reinforcement ($n_{\rm node}=0$)")]
    panels = [(aS, "sir", "final epidemic size", "SIR", "a"),
              (aE, "sis", "endemic prevalence", "SIS", "b"),
              (aSim, "simplicial", "endemic prevalence", "simplicial contagion", "c"),
              (aThr, "threshold", "endemic prevalence", "threshold contagion", "d")]
    for ax, key, ylab, title, plab in panels:
        for name, col, lab in poles:
            m = np.array(out[f"{key}_{name}"]); s = np.array(out[f"{key}_{name}_se"])
            ax.plot(betas, m, "o-", color=col, lw=1.4, ms=4.5, markeredgecolor="#222",
                    markeredgewidth=0.4, label=lab)
            ax.fill_between(betas, m - s, m + s, color=col, alpha=0.18)
        ax.set_xlabel(r"per-contact transmission  $\beta$")
        ax.set_ylabel(ylab); ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.text(-0.16, 1.04, plab, transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="bottom", ha="right")
    h, l = aS.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=6)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plotstyle.save(fig, os.path.join(outdir, "epidemic_divergence"))
    plt.close(fig)
    print("wrote epidemic_divergence.pdf/.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "replot":
        import json
        cache = os.path.join(ROOT, "results", "epidemic_divergence.json")
        _plot(json.load(open(cache)))
    else:
        main()

#!/usr/bin/env python3
"""Which memory builds the recurring groups that COMPLEX social contagion rides on?

Simple epidemics (Fig. epidemic) spread along single contacts. Social contagion is
different: adoption of a behaviour typically needs *reinforcement* -- exposure to several
adopters at once -- a genuinely higher-order, group phenomenon (Centola & Macy 2007;
Iacopini et al. 2019). Complex contagion therefore needs the SAME adopters to co-occur with
a susceptible repeatedly, i.e. it needs recurring GROUPS. The question is which kind of
memory manufactures that recurrence.

The higher-order model has two memory channels:
  * kappa   -- pairwise tie reinforcement (recently-contacted partners are re-recruited).
  * kappa_g -- group-as-object memory (a subgroup that convened is re-convened as a unit).
We cross them in a 2x2 (each off/on) on group temporal networks with memoryless node timing
and uniform attractiveness, so memory is the only thing that changes.

Diagnostic (measured, not assumed): exact size>=3 group recurrence is
  none 0.00 | tie-only 0.04 | group-only 0.02 | both 0.64.
Neither channel ALONE reconvenes the same triad -- tie memory keeps *pairs* meeting but not
as a threesome; group memory favours the unit but the pairwise ties drift -- only the two
together lock a recurring higher-order structure in place.

Prediction. theta=2 (complex) contagion is accelerated ONLY in the "both" cell, where
recurring groups exist; theta=1 (simple) needs no reinforcement and is insensitive to which
memory is on. This is a higher-order effect: it is invisible to either memory channel alone.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
from generative_models import (MechanismParams, TemporalNetworkModel,  # noqa
                                      GroupParams, simulate_group_events)

N, T, A0 = 150, 400.0, 0.15
CAP = 9000
KAPPA, KAPPA_G = 50.0, 15.0
CONDS = [("none", 0.0, 0.0), ("tie", KAPPA, 0.0),
         ("group", 0.0, KAPPA_G), ("both", KAPPA, KAPPA_G)]


def gen_group_net(kappa, kappa_g, seed):
    rng = np.random.default_rng(seed)
    model = TemporalNetworkModel(
        N=N, activities=np.full(N, A0), attractiveness=np.ones(N),
        params=MechanismParams(n_node=0.0, gamma_n=1.0 / 30.0, kappa=kappa, gamma_e=1.0 / 600.0))
    grp = GroupParams(kappa_g=kappa_g, gamma_g=1.0 / 2000.0, kappa_1p=0.0,
                      size_p=0.30, k_max=6)
    ev = simulate_group_events(model, grp, T, rng=rng)
    return [[int(i)] + [int(x) for x in p] for (t, i, p) in ev][:CAP]


def recur3(gr):
    big = [frozenset(G) for G in gr if len(G) >= 3]
    return sum(v for v in Counter(big).values() if v > 1) / max(len(big), 1)


def build_cache():
    nets = {}
    for k, (name, kap, kg) in enumerate(CONDS):
        gr = gen_group_net(kap, kg, seed=10 + k)
        nets[name] = gr
        print(f"{name:6s}: {len(gr)} groups  size>=3 recurrence={recur3(gr):.3f}")
    m = min(len(v) for v in nets.values())
    nets = {k: v[:m] for k, v in nets.items()}
    pickle.dump(nets, open(os.path.join(RES, "sc_nets.pkl"), "wb"))
    return nets


def run_threshold(groups, theta, p, rng, reps):
    finals = np.zeros(reps)
    for r in range(reps):
        adopted = np.zeros(N, bool)
        adopted[rng.choice(N, size=max(2, N // 20), replace=False)] = True   # 5% seed
        for G in groups:
            if sum(adopted[n] for n in G) >= theta:
                for n in G:
                    if not adopted[n] and rng.random() < p:
                        adopted[n] = True
        finals[r] = adopted.mean()
    return float(finals.mean()), float(finals.std() / np.sqrt(reps))


def main():
    ps = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.08])
    reps = 25
    pkl = os.path.join(RES, "sc_nets.pkl")
    nets = pickle.load(open(pkl, "rb")) if os.path.exists(pkl) else build_cache()
    rec = {name: recur3(gr) for name, gr in nets.items()}

    rng = np.random.default_rng(0)
    out = {"ps": ps.tolist(), "recur3": rec, "conds": [c[0] for c in CONDS]}
    for theta in (1, 2):
        for name in nets:
            ms, ss = [], []
            for p in ps:
                mn, se = run_threshold(nets[name], theta, float(p), rng, reps)
                ms.append(mn); ss.append(se)
            out[f"t{theta}_{name}"] = ms; out[f"t{theta}_{name}_se"] = ss
            print(f"theta={theta} {name:6s}: " + ", ".join(f"{x:.2f}" for x in ms))
    json.dump(out, open(os.path.join(RES, "social_contagion.json"), "w"), indent=2)
    _plot(out)


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pass
    import plotstyle
    from palette import TIE, POISSON
    plotstyle.apply()
    outdir = os.path.abspath(os.path.join(ROOT, "figures"))
    ps = np.array(out["ps"])
    style = {"none": (POISSON, "no memory"), "tie": (TIE, r"tie memory ($\kappa$)"),
             "group": ("#C88A3A", r"group memory ($\kappa_g$)"), "both": ("#2F2F2F", "both")}
    fig, axs = plt.subplots(1, 2, figsize=(plotstyle.FULL * 0.86, 2.2), sharey=True,
                            gridspec_kw=dict(wspace=0.12))
    for k, (ax, theta, title) in enumerate(((axs[0], 1, r"simple contagion ($\theta=1$)"),
                                            (axs[1], 2, r"complex contagion ($\theta=2$)"))):
        for name, (col, lab) in style.items():
            m = np.array(out[f"t{theta}_{name}"]); se = np.array(out[f"t{theta}_{name}_se"])
            ax.fill_between(ps, m - se, m + se, color=col, alpha=0.15, lw=0)
            ax.plot(ps, m, "o-", color=col, lw=1.1, ms=3.0, markeredgecolor="white",
                    markeredgewidth=0.4, label=lab)
        ax.set_xlabel(r"adoption probability $p$")
        ax.set_title(title, loc="left", fontsize=7, color="#555555")
        plotstyle.panel(ax, "ab"[k], x=-0.18 if k == 0 else -0.06, y=1.03)
    axs[0].set_ylabel("final adopted fraction")
    axs[1].legend(loc="lower right")
    plotstyle.save(fig, os.path.join(outdir, "social_contagion"))
    plt.close(fig)
    print("wrote social_contagion.pdf/.png")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        _plot(json.load(open(os.path.join(RES, "social_contagion.json"))))
    else:
        main()

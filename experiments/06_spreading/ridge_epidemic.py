#!/usr/bin/env python3
"""Epidemic uncertainty along the confound ridge of real datasets for SIR and SIS dynamics.

This script evaluates how epidemic forecasts (SIR final size and SIS endemic prevalence)
move along the parameter ridge between pure node reading (n_node > 0, kappa = 0) and pure link
reading (n_node = 0, kappa > 0), for both High School human proximity and Baboons animal proximity.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from generative_models import MechanismParams, TemporalNetworkModel, simulate  # noqa: E402
from semi_synthetic import real_activities  # noqa: E402
from epidemic_divergence import run_sir, run_sis  # noqa: E402
import plotstyle  # noqa: E402


def load_baboons_activities():
    DATA = os.path.abspath(os.path.join(ROOT, "data"))
    if not os.path.exists(DATA):
        DATA = os.path.abspath(os.path.join(ROOT, "data"))
    b_raw = []
    with open(os.path.join(DATA, "baboons_proximity_data.txt")) as f:
        _header = next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                b_raw.append((float(parts[0]), parts[1], parts[2]))
    nodes = sorted({i for _, i, _ in b_raw} | {j for _, _, j in b_raw})
    rm = {n: k for k, n in enumerate(nodes)}
    N_bab = len(nodes)
    K_bab = np.zeros(N_bab)
    for _, i, j in b_raw:
        K_bab[rm[i]] += 1
        K_bab[rm[j]] += 1
    T_bab_raw = b_raw[-1][0] - b_raw[0][0]
    rate_bab = K_bab / T_bab_raw
    return rate_bab, N_bab


def gen_ridge(act, T, n_node, kappa, seed, max_events=400000):
    """Generator matched on per-node aggregate rate `act`; memory split set by (n_node,kappa)."""
    N = len(act)
    a = act * (1.0 - n_node)
    model = TemporalNetworkModel(N=N, activities=a, attractiveness=np.ones(N),
                                 params=MechanismParams(n_node, 1 / 30.0, kappa, 1 / 600.0))
    ev = [(float(t), int(i), int(j)) for (t, i, j) in
          simulate(model, T, rng=np.random.default_rng(seed), max_events=max_events)]
    ev.sort()
    return ev


def run_dataset_ridge(dataset_name, act, N, n_max, kappa_max, beta_sir, beta_sis, mu, ss, reps=30, n_gen=6):
    target_per_node = 160.0
    T = 4000.0
    act_norm = act / act.mean() * (target_per_node / T)
    rng = np.random.default_rng(42)

    res = {
        "s": ss.tolist(),
        "n_node": [],
        "kappa": [],
        "sir": [], "sir_se": [],
        "sis": [], "sis_se": [],
        "beta_sir": beta_sir,
        "beta_sis": beta_sis
    }

    for s in ss:
        n_node = n_max * (1.0 - s)
        kappa = kappa_max * s
        sir_gens, sis_gens = [], []
        for g in range(n_gen):
            ev = gen_ridge(act_norm, T, n_node, kappa, seed=100 + int(1000 * s) + 31 * g)
            m_sir, _ = run_sir(ev, N, beta_sir, mu, rng, reps)
            m_sis, _ = run_sis(ev, N, beta_sis, mu, rng, reps)
            sir_gens.append(m_sir)
            sis_gens.append(m_sis)

        sir_arr, sis_arr = np.array(sir_gens), np.array(sis_gens)
        res["n_node"].append(round(float(n_node), 3))
        res["kappa"].append(round(float(kappa), 2))
        res["sir"].append(float(sir_arr.mean()))
        res["sir_se"].append(float(sir_arr.std(ddof=1) / np.sqrt(n_gen)))
        res["sis"].append(float(sis_arr.mean()))
        res["sis_se"].append(float(sis_arr.std(ddof=1) / np.sqrt(n_gen)))
        print(f"[{dataset_name}] s={s:.2f} n_node={n_node:.2f} kappa={kappa:4.1f} | SIR={res['sir'][-1]:.3f} SIS={res['sis'][-1]:.3f}")

    return res


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(ROOT, "results")
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, "ridge_epidemic.json")

    if "--plot-only" in sys.argv and os.path.exists(json_path):
        with open(json_path) as f:
            all_res = json.load(f)
        _plot(all_res, outdir)
        return

    ss = np.linspace(0.0, 1.0, 9)
    mu = 1.0 / 120.0

    # 1. High School
    rate_hs, N_hs, _ = real_activities(n_keep=50)
    res_hs = run_dataset_ridge("High School", rate_hs, N_hs, n_max=0.75, kappa_max=30.0,
                               beta_sir=0.20, beta_sis=0.25, mu=mu, ss=ss)

    # 2. Baboons
    rate_bab, N_bab = load_baboons_activities()
    res_bab = run_dataset_ridge("Baboons", rate_bab, N_bab, n_max=0.80, kappa_max=40.0,
                                beta_sir=0.20, beta_sis=0.25, mu=mu, ss=ss)

    all_res = {"high_school": res_hs, "baboons": res_bab}
    with open(json_path, "w") as f:
        json.dump(all_res, f, indent=2)

    _plot(all_res, outdir)


def _plot(all_res, outdir):
    plotstyle.apply()
    figdir = os.path.abspath(os.path.join(ROOT, "figures"))

    fig, axes = plt.subplots(1, 2, figsize=(plotstyle.FULL * 0.86, 2.2), sharex=True,
                             gridspec_kw=dict(wspace=0.32))
    for k, (key, title) in enumerate((("high_school", "high school ($N=50$)"), ("baboons", "baboons ($N=13$)"))):
        ax = axes[k]; d = all_res[key]; nn = np.array(d["n_node"])
        for m, se, col, ls, lab in ((d["sir"], d["sir_se"], "#2F2F2F", "-", rf"SIR final size, $\beta={d['beta_sir']}$"),
                                    (d["sis"], d["sis_se"], "#8A8A8A", "--", rf"SIS prevalence, $\beta={d['beta_sis']}$")):
            m, se = np.array(m), np.array(se)
            ax.fill_between(nn, m - se, m + se, color=col, alpha=0.15, lw=0)
            ax.plot(nn, m, "o" + ls, color=col, lw=1.1, ms=3.0, markeredgecolor="white", markeredgewidth=0.4, label=lab)
        ax.set_title(title, loc="left", fontsize=7, color="#555555")
        ax.set_xlabel(r"$n_{\rm node}$ attributed to the node channel")
        plotstyle.panel(ax, "ab"[k], x=-0.20, y=1.03)
    axes[0].set_ylabel("fraction of nodes")
    axes[0].legend(loc="upper left")
    plotstyle.save(fig, os.path.join(figdir, "ridge_epidemic"))
    plt.close(fig)
    print("Updated ridge_epidemic figure saved cleanly.")


if __name__ == "__main__":
    main()


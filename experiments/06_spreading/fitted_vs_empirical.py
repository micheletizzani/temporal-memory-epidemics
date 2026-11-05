#!/usr/bin/env python3
"""Practical spreading error: does the FITTED (artefactual) model forecast the real outbreak?

The pure-pole experiment bounds the mechanism effect at the extremes. A practitioner does not
use a pole; they fit the model to the record and simulate from the fit. On proximity data that
fit returns n_node ~ 0.9 -- a marginal artefact, not real node self-excitation. This script
measures the downstream cost directly by running temporal-network SIR on three sequences at
matched activity:
  (i)   REAL    : the empirical contact sequence (ground truth);
  (ii)  FITTED  : simulated from the naive fit (n_node~0.9, kappa), i.e. taking the artefact
                  at face value -- it injects spurious node burstiness the data do not have;
  (iii) CORRECTED: simulated from a model that heeds the diagnosis (n_node=0, memoryless
                  timing, same identifiable kappa, activities rescaled to match the rate).
If FITTED diverges from REAL while CORRECTED tracks it, the artefact mis-forecasts the outbreak
and diagnosing it (rather than fitting it) is what a modeller should do.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import sys

import numpy as np

import inference as inf  # noqa: E402
from generative_models import MechanismParams, TemporalNetworkModel, simulate  # noqa
from sociopatterns_loader import load_sociopatterns, split_by_window  # noqa: E402
from epidemic_divergence import run_sir  # noqa: E402
import plotstyle  # noqa: E402
DATA = os.path.abspath(os.path.join(ROOT, "data"))



def load_real(dataset="school", n_keep=55):
    if dataset == "baboons":
        baboon_path = os.path.join(ROOT, "data", "baboons_proximity_data.txt")
        with open(baboon_path) as f:
            lines = f.readlines()[1:]
        b_raw = []
        for l in lines:
            parts = l.strip().split("\t")
            if len(parts) >= 3:
                b_raw.append((float(parts[0]), parts[1], parts[2]))
        # keep top nodes by degree
        deg = {}
        for _t, i, j in b_raw:
            deg[i] = deg.get(i, 0) + 1; deg[j] = deg.get(j, 0) + 1
        keep = set(sorted(deg, key=deg.get, reverse=True)[:n_keep])
        b_raw = [(t, i, j) for (t, i, j) in b_raw if i in keep and j in keep]
        nodes = sorted({i for _, i, _ in b_raw} | {j for _, _, j in b_raw})
        rm = {n: k for k, n in enumerate(nodes)}
        t0 = b_raw[0][0]
        span = b_raw[-1][0] - t0
        T_TARGET = 1200.0
        sc = T_TARGET / span
        ev = [((float(t) - t0) * sc, rm[i], rm[j]) for (t, i, j) in b_raw]
        T = ev[-1][0] + T_TARGET * 0.01
        return ev, len(nodes), float(T)
    else:
        d = load_sociopatterns(os.path.join(DATA, "HighSchool2013_proximity_net.csv"),
                               time_col=0, i_col=1, j_col=2, sessionize_gap=60.0)
        blk = split_by_window(d, 86400.0, min_contacts=5)[0]
        ev = list(blk.contacts)
        deg = {}
        for _t, i, j in ev:
            deg[i] = deg.get(i, 0) + 1; deg[j] = deg.get(j, 0) + 1
        keep = set(sorted(deg, key=deg.get, reverse=True)[:n_keep])
        ev = [(t, i, j) for (t, i, j) in ev if i in keep and j in keep]
        nodes = sorted({i for _t, i, _j in ev} | {j for _t, _i, j in ev})
        rm = {n: k for k, n in enumerate(nodes)}
        t0 = ev[0][0]
        span = ev[-1][0] - t0
        T_TARGET = 1200.0
        sc = T_TARGET / span
        ev = [((float(t) - t0) * sc, rm[i], rm[j]) for (t, i, j) in ev]
        T = ev[-1][0] + T_TARGET * 0.01
        return ev, len(nodes), float(T)


def sim_seq(model, T_gen, seed, cap, n_target):
    """Generate over a longer window, then trim to exactly n_target events so contact COUNT
    is matched across models and only the temporal structure differs."""
    ev = [(float(t), int(i), int(j)) for (t, i, j) in
          simulate(model, T_gen, rng=np.random.default_rng(seed), max_events=cap)]
    ev.sort()
    return ev[:n_target]


def sir_over_gens(gen_fn, G, N, beta, mu, rng, reps):
    vals = [run_sir(gen_fn(g), N, beta, mu, rng, reps)[0] for g in range(G)]
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(max(G, 1)))


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "school"
    outdir = os.path.join(ROOT, "results"); os.makedirs(outdir, exist_ok=True)
    ev_real, N, T = load_real(dataset=dataset, n_keep=55)
    print(f"real {dataset} subnetwork: {len(ev_real)} contacts, N={N}, T={T:.0f}")
    fitL = inf.fit_latent_initiator(ev_real, N, T, n_imputations=6, n_em=8, fit_b=True, seed=0)
    nn, gn, k, ge = fitL.n_node, fitL.gamma_n, fitL.kappa, fitL.gamma_e
    a, b = np.asarray(fitL.activities), np.asarray(fitL.attractiveness)
    print(f"fitted model ({dataset}): n_node={nn:.3f} kappa={k:.2f}")

    ntgt = len(ev_real); Tg = 2.5 * T; cap = int(12 * ntgt)
    fitted = TemporalNetworkModel(N, a, MechanismParams(nn, gn, k, ge), attractiveness=b)
    a_corr = a / max(1.0 - nn, 1e-3)                       # rescale so Poisson rate matches
    corrected = TemporalNetworkModel(N, a_corr, MechanismParams(0.0, gn, k, ge), attractiveness=b)
    nfit = len(sim_seq(fitted, Tg, 0, cap, ntgt)); ncorr = len(sim_seq(corrected, Tg, 0, cap, ntgt))
    print(f"event counts (all trimmed to real) -- real {ntgt}, fitted {nfit}, corrected {ncorr}")

    if dataset == "baboons":
        betas = np.array([0.001, 0.002, 0.004, 0.008, 0.015, 0.03])
    else:
        betas = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.45])
    mu, reps, G = 1.0 / 120.0, 100, 4   # replicates per arm: 100 (25 gave SE up to 0.08 on the baboon record)
    rng = np.random.default_rng(0)
    out = {"dataset": dataset, "betas": betas.tolist(), "real": [], "real_se": [], "fit": [], "fit_se": [],
           "corr": [], "corr_se": [], "n_node": nn, "kappa": k}
    for be in betas:
        r_m, r_s = run_sir(ev_real, N, float(be), mu, rng, reps)
        f_m, f_s = sir_over_gens(lambda g: sim_seq(fitted, Tg, 10 + g, cap, ntgt), G, N, float(be), mu, rng, reps)
        c_m, c_s = sir_over_gens(lambda g: sim_seq(corrected, Tg, 20 + g, cap, ntgt), G, N, float(be), mu, rng, reps)
        out["real"].append(r_m); out["real_se"].append(r_s)
        out["fit"].append(f_m); out["fit_se"].append(f_s)
        out["corr"].append(c_m); out["corr_se"].append(c_s)
        print(f"  beta={be:<6g} | REAL={r_m:.3f}  FITTED={f_m:.3f}  CORRECTED={c_m:.3f}")

    fname = f"fitted_vs_empirical_{dataset}" if dataset != "school" else "fitted_vs_empirical"
    import json
    json.dump(out, open(os.path.join(outdir, f"{fname}.json"), "w"), indent=2)
    _plot(out, fname)


def _plot(out, fname="fitted_vs_empirical"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plotstyle.apply()
    figdir = os.path.abspath(os.path.join(ROOT, "figures"))
    be = np.array(out["betas"])
    pass
    from palette import NODE, TIE
    fig, ax = plt.subplots(figsize=(plotstyle.COL, 2.3))
    for key, col, lab, mk in (("real", "#2F2F2F", "recorded contacts", "o"),
                              ("fit", NODE, r"naive fit ($\hat n_{\rm node}=%.2f$)" % out["n_node"], "s"),
                              ("corr", TIE, r"corrected ($n_{\rm node}=0$, $\hat\kappa$ kept)", "^")):
        m = np.array(out[key]); se = np.array(out[key + "_se"])
        ax.fill_between(be, m - se, m + se, color=col, alpha=0.15, lw=0)
        ax.plot(be, m, mk + "-", color=col, lw=1.1, ms=3.0, markeredgecolor="white", markeredgewidth=0.4, label=lab)
    ax.set_xlabel(r"transmission probability $\beta$"); ax.set_ylabel("SIR final size")
    name = {"school": "high school", "baboons": "baboons"}.get(out.get("dataset", "school"), out.get("dataset"))
    ax.set_title(name, loc="left", fontsize=7, color="#555555")
    ax.legend(loc="lower right" if out.get("dataset") == "baboons" else "upper left",
              bbox_to_anchor=(1.0, 0.03) if out.get("dataset") == "baboons" else (0.0, 0.97))
    plotstyle.save(fig, os.path.join(figdir, fname))
    plt.close(fig)
    print(f"wrote {fname}.pdf/.png")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        import json
        for fn in ("fitted_vs_empirical", "fitted_vs_empirical_baboons"):
            _plot(json.load(open(os.path.join(ROOT, "results", fn + ".json"))), fn)
    else:
        main()

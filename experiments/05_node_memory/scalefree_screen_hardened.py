# %%
"""Hardened scale-free identifiability screen: is the node-memory collapse a KERNEL artefact?

Adds to scalefree_identifiability.py the rigor needed before this can carry a paper claim:
  * ERROR BARS -- R_exp, R_mix as mean +/- 5-95% over B subsamples x S shuffle seeds;
  * a NEGATIVE CONTROL -- a synthetic heavy-tailed RENEWAL process (no order): R_mix MUST stay
    ~1, else the flexible mixture kernel is manufacturing order (overfitting the marginal);
  * a POSITIVE CONTROL -- synthetic power-law self-excitation: R_mix should drop below R_exp;
  * a CONVERGENCE diagnostic -- the mixture log-likelihood gain over the single exponential
    (ll_gain; if ~0 the mixture adds nothing and any R_mix<1 is suspect), and the mixture fit is
    SEEDED at the single-exponential MLE for reliable Nelder-Mead convergence;
  * JSON CACHING -- the slow fits run once into results/scalefree_screen.json.

Reads only what the caller lists in DATASETS. R<0.  Run '# %%' cell-by-cell or as a script.

Read the table: R_mix << 1 with a clear ll_gain and a renewal control at ~1  => genuine order,
node memory resolvable with the right kernel (kernel artefact). R_mix ~ 1 (or ~ the renewal
control) => marginal/renewal burstiness, negative result kernel-independent. R_mix ~ 0.8 is a
WEAK residual, not resolution (resolved is R -> 0).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

import kernel_phase as kp                                             # noqa: E402
from kernel_families import powerlaw_mixture                         # noqa: E402
from sum_exp_kernel import multi_replay, nll_multi, fit_mixture, load_directed  # noqa: E402

GAMMAS = np.geomspace(1e-3, 1e1, 8)
B_SUB, S_SHUF, SUB = 4, 3, 4000          # subsamples, shuffle seeds/subsample, events/subsample
RESULTS = os.path.join(ROOT, "results", "scalefree_screen.json")


# %%
def fit_nR(ev, N, T, gammas=GAMMAS):
    """(n_exp, ll_exp, n_mix, ll_mix). Mixture SEEDED at the single-exp MLE for convergence."""
    init = [i for _t, i, _j in ev]; R = len(gammas)
    S, C, counts = multi_replay(ev, init, N, gammas, T)
    best = (-1e18, 0.0, 0)
    for r in range(R):
        w = np.zeros(R); w[r] = 1.0
        o = minimize(lambda x: nll_multi(1 / (1 + math.exp(-x[0])), w, S, C, counts, init, N, T)[0],
                     [0.0], method="Nelder-Mead", options=dict(maxiter=250, xatol=1e-3, fatol=1e-3))
        n = 1 / (1 + math.exp(-o.x[0]))
        if -o.fun > best[0]:
            best = (-o.fun, n, r)
    ll_exp, n_exp, r_star = best
    z0 = np.full(R, -3.0); z0[r_star] = 3.0                          # seed weights at the exp rate
    x0 = np.concatenate([[math.log(n_exp / (1 - n_exp + 1e-9))], z0])
    n_mix, _w, ll_mix = fit_mixture(S, C, counts, init, N, T, R, x0=x0)
    return n_exp, ll_exp, n_mix, ll_mix


def per_node_shuffle(evs, rng):
    """Destroy per-node timing order, keep the marginal (and the (i,j) pairing)."""
    tnew = [t for t, _i, _j in evs]; idxby = {}
    for k, (t, i, j) in enumerate(evs):
        idxby.setdefault(i, []).append(k)
    for i, idx in idxby.items():
        ts = sorted(tnew[k] for k in idx)
        d = np.diff([0.0] + ts); rng.shuffle(d); cs = np.cumsum(d)
        for k, tt in zip(idx, cs):
            tnew[k] = float(tt)
    return sorted((tnew[k], evs[k][1], evs[k][2]) for k in range(len(evs)))


def screen(ev, N, T, tag, b_sub=B_SUB, s_shuf=S_SHUF, sub=SUB, seed=0):
    """R_exp, R_mix mean +/- 5-95% over subsamples x shuffles, plus mixture ll_gain."""
    evs = sorted(ev); E = len(evs); rng = np.random.default_rng(seed)
    Re, Rm, gains = [], [], []
    for _b in range(b_sub):
        if E > sub:
            s = int(rng.integers(0, E - sub)); window = evs[s:s + sub]
        else:
            window = evs
        Nw = 1 + max(max(i, j) for _t, i, j in window)
        Tw = window[-1][0] - window[0][0] + 1e-9
        w0 = [(t - window[0][0], i, j) for t, i, j in window]
        ne, lle, nm, llm = fit_nR(w0, Nw, Tw)
        gains.append(llm - lle)
        for s2 in range(s_shuf):
            wsh = per_node_shuffle([tuple(e) for e in w0], np.random.default_rng(1000 * _b + s2))
            nes, _, nms, _ = fit_nR(wsh, Nw, Tw)
            Re.append(nes / max(ne, 1e-3)); Rm.append(nms / max(nm, 1e-3))
    q = lambda a: (float(np.mean(a)), float(np.percentile(a, 5)), float(np.percentile(a, 95)))
    return dict(tag=tag, E=E, N=N, R_exp=q(Re), R_mix=q(Rm),
                ll_gain=float(np.mean(gains)), n_mix_real=float(nm), n_exp_real=float(ne))


# %%
# ---- synthetic controls (positive: power-law self-excitation; negative: heavy-tailed renewal) ----
def sim_powerlaw(seed, T=2500.0, n=0.5, theta=0.4):
    rng = np.random.default_rng(seed); nws, ngm = powerlaw_mixture(theta)
    ts = kp.hawkes_mix_sim(1.0, n, nws, ngm, T, rng)
    return [(float(t), 0, 1) for t in ts], 2, T


def sim_renewal(seed, T=2500.0, sigma_ln=1.6, rate=1.2):
    rng = np.random.default_rng(seed); mu = math.log(1.0 / rate) - 0.5 * sigma_ln ** 2
    t = 0.0; ts = []
    while True:
        t += float(rng.lognormal(mu, sigma_ln))
        if t >= T:
            break
        ts.append(t)
    return [(float(x), 0, 1) for x in ts], 2, T


# %%
if os.path.exists(RESULTS):
    out = json.load(open(RESULTS))
else:
    out = {}

# controls
if "POS power-law self-excite" not in out:
    ev, N, T = sim_powerlaw(1);  out["POS power-law self-excite"] = screen(ev, N, T, "pos")
if "NEG heavy-tail renewal" not in out:
    ev, N, T = sim_renewal(1);   out["NEG heavy-tail renewal"] = screen(ev, N, T, "neg")

# real datasets: (name, path, tcol, icol, jcol). Directed => no label confound.
DATASETS = [
    ("Email (EU)",       os.path.join(ROOT, "data", "email-Eu-core-temporal.txt"), 2, 0, 1),
    ("Stack Overflow",   os.path.join(ROOT, "data", "so_core_events.txt"),         0, 1, 2),
    ("Wikibooks (Bots)", os.path.join(ROOT, "data", "wikimedia", "wikibooks_bots_events.txt"), 0, 1, 2),  # rebuilt (build_wikibooks_bots.py); legacy file was not bots
    ("Alibaba micro",    os.path.join(ROOT, "data", "alibaba_microservices_sample.txt"), 0, 1, 2),
]
for name, path, tc, ic, jc in DATASETS:
    if name in out:
        print(f"cached {name}"); continue
    if not os.path.exists(path):
        print(f"(missing {name})"); continue
    ev, N, T = load_directed(path, tc, ic, jc, 10000)
    out[name] = screen(ev, N, T, name)
    print(f"done {name}")

# ---- PROXIMITY (latent initiator) -- CAVEAT: this is a WEAK add, read it with care --------
# The initiator is latent, so we must impute it; but the imputation uses a TIMING model, so
# R_mix here is confounded with the label imputation on TOP of the already-weak R_mix instrument
# (see the POS control: even genuine order only resolves to ~0.85). Two honest fixes vs the naive
# single-fixed-theta draw: (i) fit theta from the data (Monte-Carlo EM), not a hand-picked value;
# (ii) AVERAGE R over several imputation draws and report the spread. Even so, treat proximity
# R_mix as indicative only -- the directed screen + the two synthetic controls carry the result.
import inference as inf
from sociopatterns_loader import load_sociopatterns, split_by_window
wsdata = os.path.abspath(os.path.join(ROOT, "data"))
hs_path = os.path.join(ROOT, "data", "HighSchool2013_proximity_net.csv") if os.path.exists(os.path.join(ROOT, "data", "HighSchool2013_proximity_net.csv")) else os.path.join(wsdata, "HighSchool2013_proximity_net.csv")
PROX_DATASETS = [
    ("High School 2013", hs_path),
    ("Science Gallery",  os.path.join(wsdata, "sciencegallery.txt")),
]
for name, path in PROX_DATASETS:
    if name in out:
        print(f"cached {name}"); continue
    if not os.path.exists(path):
        print(f"(missing {name})"); continue
    d = load_sociopatterns(path, time_col=0, i_col=1, j_col=2, sessionize_gap=60.0)
    blk = split_by_window(d, 86400.0, min_contacts=5)[0]           # one day
    contacts = blk.contacts[:8000]
    if not contacts:
        continue
    T_w = contacts[-1][0] - contacts[0][0] + 1e-9
    c_w = [(t - contacts[0][0], i, j) for t, i, j in contacts]
    N_w = 1 + max(max(i, j) for _t, i, j in c_w)
    # (i) data-fitted theta from the latent MLE (not a hand-picked default)
    rL = inf.fit_latent_initiator(c_w, N_w, T_w, n_imputations=4, n_em=5, seed=0)
    theta = [rL.n_node, rL.gamma_n, rL.kappa, rL.gamma_e]
    # (ii) average R over several imputation draws under the fitted theta
    Rmix_draws = []
    for s in range(3):
        rng = np.random.default_rng(100 + s)
        side = inf._impute_initiators([(t, i, j) if i < j else (t, j, i) for t, i, j in c_w],
                                      N_w, T_w, theta, rng)
        ordered = [(t, i, j) if i < j else (t, j, i) for t, i, j in c_w]
        ev_imp = [(t, (i if sd == 0 else j), (j if sd == 0 else i))
                  for (t, i, j), sd in zip(ordered, side)]
        Rmix_draws.append(screen(ev_imp, N_w, T_w, name, b_sub=2, s_shuf=2)["R_mix"][0])
    out[name] = dict(tag="proximity(imputed)", E=len(c_w), N=N_w,
                     R_mix_mean=float(np.mean(Rmix_draws)), R_mix_sd=float(np.std(Rmix_draws)),
                     n_lat=float(rL.n_node), kappa_lat=float(rL.kappa),
                     note="latent-initiator: R_mix confounded by imputation; indicative only")
    print(f"done {name} (imputed): R_mix = {np.mean(Rmix_draws):.2f} +/- {np.std(Rmix_draws):.2f} "
          f"across {len(Rmix_draws)} imputations")

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
json.dump(out, open(RESULTS, "w"), indent=2)


# %%
# ---- table ----
print(f"\n{'dataset':26s} {'E':>7s} {'R_exp [5-95]':>18s} {'R_mix [5-95]':>18s} {'ll_gain':>8s}")
for name, d in out.items():
    if "R_exp" in d:                         # directed / synthetic entries (full schema)
        re, rm = d["R_exp"], d["R_mix"]
        print(f"{name:26s} {d['E']:>7d}  {re[0]:.2f} [{re[1]:.2f},{re[2]:.2f}]   "
              f"{rm[0]:.2f} [{rm[1]:.2f},{rm[2]:.2f}]  {d['ll_gain']:>8.0f}")
    else:                                    # proximity (imputed) entries: R_mix mean +/- sd only
        print(f"{name:26s} {d['E']:>7d}  {'(latent)':>18s}   "
              f"{d['R_mix_mean']:.2f} +/- {d['R_mix_sd']:.2f} (imputed)")
print("\nread: trust R_mix<1 only if (i) NEG renewal control is ~1 (no false order), (ii) ll_gain")
print(">> 0 (mixture genuinely fits better), (iii) the R_mix CI excludes 1. R_mix~0.8 = weak")
print("residual, NOT resolution (resolved is R->0, cf. POS control).")

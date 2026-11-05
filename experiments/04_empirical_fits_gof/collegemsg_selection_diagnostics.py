# %%
"""Why does CollegeMsg return kappa ~ 0.001 although 77% of its messages repeat a tie?

Four blocks, all on the full CollegeMsg record (59,835 messages, 1,899 users):
  A. REPEAT STRUCTURE  -- exact duplicates, simultaneous events, same-direction repeats vs replies,
                          lag to the previous contact on the same pair.
  B. SELECTION PROFILE -- nll(kappa, gamma_e) on a dense grid (b uniform and b fitted), the
                          profile maximum, and the likelihood-ratio statistic against kappa = 0.
                          If the profile has an interior maximum at large kappa, the published
                          kappa ~ 0.001 is an optimiser failure, not a property of the data.
  C. REFITS            -- fit_observed_initiator as originally run, then with (i) the decay-rate
                          ceiling set by the 1 s timestamp resolution instead of the 1e-4 s tie
                          jitter, (ii) exact duplicates removed, (iii) seeded at the profile optimum.
  D. REPEAT TESTS      -- the additional test. Two surrogate nulls and a predictive check:
       D1 strength-preserving target permutation: keeps every node's in- and out-strength,
          destroys pair identity -> do repeats exceed what popularity explains?
       D2 timestamp permutation: keeps every pair's count, destroys temporal clustering -> are
          repeats clustered in time (decaying tie memory) or only a static pair preference?
       D3 predictive check: simulate targets from the fitted selection model given the observed
          initiators and times, and compare the repeat fraction and repeat-lag quantiles with the
          data. A selection model that passes reproduces both.
Writes results/collegemsg_selection_diagnostics.json.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

import inference as inf  # noqa: E402

DATA = os.path.abspath(os.path.join(ROOT, "data", "CollegeMsg.txt"))
OUT = os.path.join(ROOT, "results", "collegemsg_selection_diagnostics.json")
LAGS = [60, 600, 3600, 86400, 604800]          # 1 min, 10 min, 1 h, 1 d, 1 wk
KAPPAS = np.geomspace(1e-3, 1e5, 33)
TAUS = np.geomspace(10.0, 1e7, 25)              # tie-memory times 1/gamma_e, seconds
N_SURR = 5


# %%
def load(dedup=False):
    d = np.loadtxt(DATA, dtype=np.int64)
    i, j, t = d[:, 0], d[:, 1], d[:, 2].astype(float)
    o = np.argsort(t, kind="stable"); i, j, t = i[o], j[o], t[o]
    if dedup:
        seen, keep = set(), []
        for k, key in enumerate(zip(t, i, j)):
            if key not in seen:
                keep.append(k); seen.add(key)
        i, j, t = i[keep], j[keep], t[keep]
    t = t - t[0]
    ids = sorted(set(i.tolist()) | set(j.tolist()))
    rm = {u: k for k, u in enumerate(ids)}
    return [(float(a), rm[int(b)], rm[int(c)]) for a, b, c in zip(t, i, j)], len(ids)


def jitter(ev, eps):
    out, last = [], -1.0
    for t, i, j in ev:
        if t <= last:
            t = last + eps
        out.append((t, i, j)); last = t
    return out, out[-1][0] + 0.1


def repeat_stats(ev):
    last_pair, seen_dir = {}, set()
    new = same = recip = 0
    lags = []
    for t, i, j in ev:
        p = (min(i, j), max(i, j))
        if p in last_pair:
            lags.append(t - last_pair[p])
            if (i, j) in seen_dir:
                same += 1
            else:
                recip += 1
        else:
            new += 1
        last_pair[p] = t; seen_dir.add((i, j))
    E = len(ev); lags = np.array(lags)
    return dict(repeat=float((same + recip) / E), same_direction=float(same / E),
                reply=float(recip / E),
                within={str(s): float(np.mean(lags <= s)) for s in LAGS} if len(lags) else {},
                lag_q=[float(x) for x in np.percentile(lags, [10, 25, 50, 75, 90])] if len(lags) else [])


# %%
def profile(ev, N, T, fit_b):
    """Grid profile; with fitted b the grid is thinned 2x per axis (each point runs an MM fit)."""
    contacts, _init, side = inf._split_events(ev)
    taus, kappas = (TAUS[::2], KAPPAS[::2]) if fit_b else (TAUS, KAPPAS)
    grid = np.full((len(taus), len(kappas)), np.nan)
    for a, tau in enumerate(taus):
        sel = inf.replay_selection(contacts, N, 1.0 / tau)
        for c, k in enumerate(kappas):
            b = inf.estimate_attractiveness(k, sel, side, N) if fit_b else np.ones(N)
            grid[a, c] = inf.selection_nll_from_stats(k, sel, side, b)
    sel0 = inf.replay_selection(contacts, N, 1.0 / TAUS[0])
    b0 = inf.estimate_attractiveness(0.0, sel0, side, N) if fit_b else np.ones(N)
    nll0 = inf.selection_nll_from_stats(0.0, sel0, side, b0)
    a, c = np.unravel_index(np.nanargmin(grid), grid.shape)
    return dict(fit_b=fit_b, kappa_hat=float(kappas[c]), tau_e_hat=float(taus[a]),
                kappa_grid=[float(x) for x in kappas],
                nll_min=float(grid[a, c]), nll_kappa0=float(nll0),
                LR=float(2 * (nll0 - grid[a, c])),
                profile_over_kappa=[float(x) for x in np.nanmin(grid, axis=0)])


def refit(ev, N, T, label, gamma_cap=None, x0=None):
    orig = inf._gamma_bounds
    if gamma_cap is not None:
        inf._gamma_bounds = lambda contacts, T, verbose=False: (1.0 / (10.0 * T), gamma_cap)
    try:
        f = inf.fit_observed_initiator(ev, N, T, fit_b=True, x0=x0)
    finally:
        inf._gamma_bounds = orig
    r = dict(label=label, n_node=f.n_node, tau_n=1 / f.gamma_n, kappa=f.kappa,
             tau_e=1 / f.gamma_e, nll_selection=f.nll_selection, at_bounds=f.at_bounds)
    print(f"  [{label}] n={f.n_node:.3f} kappa={f.kappa:.4g} 1/gamma_e={1/f.gamma_e:.4g}s "
          f"nll_sel={f.nll_selection:.1f} bounds={f.at_bounds}", flush=True)
    return r, f


# %%
def surrogate_targets(ev, rng):
    """D1: permute targets across events, keeping out- and in-strength; resample self-loops."""
    t = [e[0] for e in ev]; i = np.array([e[1] for e in ev]); j = np.array([e[2] for e in ev])
    j = j[rng.permutation(len(j))]
    for _ in range(50):
        bad = np.where(i == j)[0]
        if len(bad) == 0:
            break
        swap = rng.integers(0, len(j), len(bad))
        j[bad], j[swap] = j[swap], j[bad].copy()
    return [(t[k], int(i[k]), int(j[k])) for k in range(len(t))]


def surrogate_times(ev, rng):
    """D2: permute the pair sequence against the timestamps, keeping every pair's count."""
    t = [e[0] for e in ev]; pairs = [(e[1], e[2]) for e in ev]
    perm = rng.permutation(len(pairs))
    return [(t[k], pairs[perm[k]][0], pairs[perm[k]][1]) for k in range(len(t))]


def predictive(ev, N, kappa, gamma_e, b, rng):
    """D3: draw each target from the fitted pi_{j|i}(t) given the simulated tie history."""
    B = float(b.sum()); cdf_b = np.cumsum(b) / B
    mem = defaultdict(dict)                        # mem[i][l] = (m_il at last update, t_last)
    out = []
    for t, i, _j in ev:
        row = mem[i]
        ls = list(row.keys())
        m = np.array([row[l][0] * math.exp(-gamma_e * (t - row[l][1])) for l in ls]) if ls else np.zeros(0)
        Mi = float(m.sum()); base = B - b[i]
        if ls and rng.random() < kappa * Mi / (base + kappa * Mi):
            j = ls[int(np.searchsorted(np.cumsum(m) / Mi, rng.random()))]
        else:
            while True:
                j = int(np.searchsorted(cdf_b, rng.random()))
                if j != i:
                    break
        out.append((t, i, j))
        for u, v in ((i, j), (j, i)):                # pair memory is symmetric (label-free H_ij)
            old = mem[u].get(v)
            val = (old[0] * math.exp(-gamma_e * (t - old[1])) if old else 0.0) + 1.0
            mem[u][v] = (val, t)
    return out


# %%
if __name__ == "__main__":
    res = {}
    ev_raw, N = load()
    print(f"CollegeMsg: {len(ev_raw)} events, {N} users")
    trip = Counter(ev_raw)
    res["A_structure"] = dict(
        events=len(ev_raw), users=N,
        exact_duplicates=int(sum(c - 1 for c in trip.values() if c > 1)),
        shared_timestamp=int(sum(c for c in Counter(e[0] for e in ev_raw).values() if c > 1)),
        shared_t_initiator=int(sum(c for c in Counter((e[0], e[1]) for e in ev_raw).values() if c > 1)),
        **repeat_stats(ev_raw))
    print("A", res["A_structure"], flush=True)

    ev, T = jitter(ev_raw, 1e-4)
    res["B_profile"] = [profile(ev, N, T, False), profile(ev, N, T, True)]
    for p in res["B_profile"]:
        print(f"B fit_b={p['fit_b']}: kappa*={p['kappa_hat']:.4g} 1/gamma_e*={p['tau_e_hat']:.4g}s "
              f"LR(kappa=0)={p['LR']:.1f}", flush=True)

    print("C refits:")
    C = []
    r, f_orig = refit(ev, N, T, "original (jitter 1e-4 s, ceiling 1e4 /s)"); C.append(r)
    r, _ = refit(ev, N, T, "ceiling at 1 s resolution", gamma_cap=1.0); C.append(r)
    ev_d, N_d = load(dedup=True); ev_d, T_d = jitter(ev_d, 1e-4)
    r, _ = refit(ev_d, N_d, T_d, "duplicates removed + 1 s ceiling", gamma_cap=1.0); C.append(r)
    pb = res["B_profile"][1]
    r, f_best = refit(ev, N, T, "seeded at profile optimum + 1 s ceiling", gamma_cap=1.0,
                      x0=[0.5, 1e-3, pb["kappa_hat"], 1.0 / pb["tau_e_hat"]]); C.append(r)
    res["C_refits"] = C
    best = min([f_best], key=lambda f: f.nll_selection)

    rng = np.random.default_rng(7)
    D1, D2 = [], []
    for s in range(N_SURR):
        e1 = surrogate_targets(ev, rng); e2 = surrogate_times(ev, rng)
        D1.append(dict(**repeat_stats(e1), **{k: v for k, v in profile(e1, N, T, False).items()
                                              if k in ("kappa_hat", "tau_e_hat", "LR")}))
        D2.append(dict(**repeat_stats(e2), **{k: v for k, v in profile(e2, N, T, False).items()
                                              if k in ("kappa_hat", "tau_e_hat", "LR")}))
        print(f"D surrogate {s}: strength-null repeat={D1[-1]['repeat']:.3f} LR={D1[-1]['LR']:.0f} | "
              f"time-null repeat<=10min={D2[-1]['within'].get('600', float('nan')):.3f} "
              f"tau_e*={D2[-1]['tau_e_hat']:.3g}", flush=True)
    res["D1_strength_null"] = D1; res["D2_time_null"] = D2

    D3 = [repeat_stats(predictive(ev, N, best.kappa, best.gamma_e, best.attractiveness, rng))
          for _ in range(3)]
    res["D3_predictive"] = dict(kappa=best.kappa, tau_e=1 / best.gamma_e, sims=D3)
    print("D3 predictive repeat:", [round(x["repeat"], 3) for x in D3],
          "observed", round(res["A_structure"]["repeat"], 3), flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)

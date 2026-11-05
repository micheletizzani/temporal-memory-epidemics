# %%
"""One protocol, every record: (B, M, R_exp with 5-95% interval) for the node-timing channel, and
kappa for the records that lack a selection fit. Single source for Fig. 3(d), Fig. 5 and the text.

Replaces the hard-coded REAL list of phase_combine.py (every R written as 1.00).

Timing trains
  directed records : initiator trains (the sender / caller / presynaptic neuron is observed)
  proximity records: participation trains, day 0, sessionised at 60 s, activity filter 5 contacts
                     (each contact is an event of both endpoints; the initiator is not observed)
R protocol (scalefree_screen_hardened): B_SUB contiguous subsamples of SUB events x S_SHUF per-node
inter-event-order shuffles, exponential kernel. Records already screened with this exact protocol
are read from their caches; M and B are always recomputed here, with the pooled lag-one estimator
of phase_combine.burstiness_memory, on the same event set the R value was computed on.
Writes results/real_record_screen.json.
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

DATA = os.path.abspath(os.path.join(ROOT, "data"))
import inference as inf  # noqa: E402
from sociopatterns_loader import load_sociopatterns, split_by_window, _sessionize  # noqa: E402
from sum_exp_kernel import multi_replay, nll_multi, load_directed  # noqa: E402
from screen_neural import per_node_shuffle  # noqa: E402

GAMMAS = np.geomspace(1e-3, 1e1, 8)
B_SUB, S_SHUF, SUB = 4, 3, 4000
OUT = os.path.join(ROOT, "results", "real_record_screen.json")


# %%
def n_exp(ev, N, T):
    """Single-exponential branching ratio, best over the rate grid (as in the hardened screen)."""
    init = [i for _t, i, _j in ev]
    S, C, counts = multi_replay(ev, init, N, GAMMAS, T)
    best = (-1e18, 0.0)
    for r in range(len(GAMMAS)):
        w = np.zeros(len(GAMMAS)); w[r] = 1.0
        o = minimize(lambda x: nll_multi(1 / (1 + math.exp(-x[0])), w, S, C, counts, init, N, T)[0],
                     [0.0], method="Nelder-Mead", options=dict(maxiter=250, xatol=1e-3, fatol=1e-3))
        if -o.fun > best[0]:
            best = (-o.fun, 1 / (1 + math.exp(-o.x[0])))
    return best[1]


def screen_R(ev, seed=0):
    evs = sorted(ev); E = len(evs); rng = np.random.default_rng(seed); R = []
    for b in range(B_SUB):
        s = int(rng.integers(0, E - SUB)) if E > SUB else 0
        w = evs[s:s + SUB]; t0 = w[0][0]
        w = [(t - t0, i, j) for t, i, j in w]
        N = 1 + max(max(i, j) for _t, i, j in w); T = w[-1][0] + 1e-9
        ne = n_exp(w, N, T)
        R += [n_exp(per_node_shuffle(list(w), np.random.default_rng(1000 * b + k)), N, T) / max(ne, 1e-3)
              for k in range(S_SHUF)]
    return (float(np.mean(R)), float(np.percentile(R, 5)), float(np.percentile(R, 95)))


def burstiness_memory(events):
    """Goh-Barabasi burstiness B (pooled intervals) and the headline memory coefficient M: the within-node rank
    lag-one correlation of memory_stats.py. The pooled Pearson version of earlier drafts is returned as a third value."""
    from memory_stats import node_gaps, rank_memory
    g = node_gaps(events)
    a = np.concatenate(list(g.values()))
    xs, ys = [], []
    for v in g.values():
        xs += list(v[:-1]); ys += list(v[1:])
    return (float((a.std() - a.mean()) / (a.std() + a.mean())), rank_memory(g, 1),
            float(np.corrcoef(xs, ys)[0, 1]))


# %%
def participation(contacts):
    out = []
    for t, i, j in contacts:
        out += [(t, i, j), (t, j, i)]
    return sorted(out)


def proximity_day0(path, **kw):
    d = load_sociopatterns(path, sessionize_gap=60.0, **kw)
    return split_by_window(d, 86400.0, min_contacts=5)[0].contacts


def baboons_day0():
    rows, ids = [], {}
    with open(os.path.join(DATA, "baboons_proximity_data.txt")) as fh:
        next(fh)
        for line in fh:
            p = line.split()
            if len(p) >= 3:
                a, b = ids.setdefault(p[1], len(ids)), ids.setdefault(p[2], len(ids))
                rows.append((float(p[0]), min(a, b), max(a, b)))
    rows.sort(); t0 = rows[0][0]
    rows = _sessionize([(t - t0, i, j) for t, i, j in rows], 60.0)
    return [r for r in rows if r[0] < 86400.0]


def directed(path, tc, ic, jc, max_ev):
    return load_directed(path, tc, ic, jc, max_ev)[0]


# name -> (loader returning timing events, cached R source or None)
def cached(path, key):
    d = json.load(open(path)); r = d[key] if key else d
    return tuple(r["R_exp"])


RECORDS = {
    "Hospital ward": (lambda: participation(proximity_day0(os.path.join(ROOT, "data", "hospital_lyon_contacts.dat"))), None),
    "High school": (lambda: participation(proximity_day0(os.path.join(ROOT, "data", "HighSchool2013_proximity_net.csv"))), None),
    "Conference (SFHH)": (lambda: participation(proximity_day0(os.path.join(DATA, "SFHH_tij.dat"))), None),
    "Baboons": (lambda: participation(baboons_day0()), None),
    "Email (EU institution)": (lambda: directed(os.path.join(ROOT, "data", "email-Eu-core-temporal.txt"), 2, 0, 1, 10000),
                               lambda: cached(os.path.join(ROOT, "results", "scalefree_screen.json"), "Email (EU)")),
    "CollegeMsg (Direct Msgs)": (lambda: directed(os.path.join(DATA, "CollegeMsg.txt"), 2, 0, 1, None), None),
    "Stack Overflow": (lambda: directed(os.path.join(ROOT, "data", "so_core_events.txt"), 0, 1, 2, 10000),
                       lambda: cached(os.path.join(ROOT, "results", "scalefree_screen.json"), "Stack Overflow")),
    "Wikibooks": (lambda: directed(os.path.join(ROOT, "data", "wikibooks_core_events.txt"), 0, 1, 2, 60000), None),
    "Wikibooks bots": (lambda: directed(os.path.join(DATA, "wikimedia", "wikibooks_bots_events.txt"), 0, 1, 2, None),
                       lambda: cached(os.path.join(ROOT, "results", "wikibooks_bots_screen.json"), None)),
    "Microservices": (lambda: directed(os.path.join(DATA, "alibaba_microservices_sample.txt"), 0, 1, 2, 10000),
                      lambda: cached(os.path.join(ROOT, "results", "scalefree_screen.json"), "Alibaba micro")),
    "SNN (bursting)": (lambda: directed(os.path.join(DATA, "snn_bursty_network.txt"), 0, 1, 2, 60000),
                       lambda: cached(os.path.join(ROOT, "results", "neural_screen.json"), "SNN (bursting)")),
}
KAPPA_FITS = {"Wikibooks bots": 60000, "Microservices": 60000}   # records without a selection fit


# %%
def refresh_memory():
    """Recompute B, M (rank) and the pooled value of every cached record on the same event sets; R untouched."""
    out = json.load(open(OUT))
    for name, (load, _r) in RECORDS.items():
        B, M, Mp = burstiness_memory(load())
        out[name].update(B=B, M=M, M_pooled=Mp)
        print(f"{name:26s} B={B:+.3f} M={M:+.3f} pooled {Mp:+.3f}", flush=True)
    c = json.load(open(os.path.join(ROOT, "results", "allen_rank_memory.json")))
    out["Cortex"]["M_pooled"] = out["Cortex"]["M"]; out["Cortex"]["M"] = c["M_rank"][0]
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    if "--refresh-memory" in sys.argv:
        refresh_memory(); sys.exit(0)
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name, (load, R_src) in RECORDS.items():
        if name in out:
            print("cached", name); continue
        ev = load()
        B, M, Mp = burstiness_memory(ev)
        R = R_src() if R_src else screen_R(ev)
        out[name] = dict(B=B, M=M, M_pooled=Mp, R_exp=R, events=len(ev), R_source="cache" if R_src else "computed")
        print(f"{name:26s} E={len(ev):7d} B={B:+.2f} M={M:+.3f} (pooled {Mp:+.3f}) R={R[0]:.3f} [{R[1]:.3f},{R[2]:.3f}]", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    a = json.load(open(os.path.join(ROOT, "results", "allen_spontaneous_screen.json")))
    loc = json.load(open(os.path.join(ROOT, "results", "allen_local_shuffle_screen.json")))
    out["Cortex"] = dict(B=a["B"][0], M=a["M"][0], R_exp=a["R_exp"], R_exp_local=loc["R_exp_local"],
                         R_source="allen_spontaneous_screen.json")
    for name, mx in KAPPA_FITS.items():
        if "kappa" in out.get(name, {}):
            continue
        ev = RECORDS[name][0]()[:mx]
        t0 = ev[0][0]; ev = [(t - t0, i, j) for t, i, j in ev]
        N = 1 + max(max(i, j) for _t, i, j in ev); T = ev[-1][0] + 1.0
        f = inf.fit_observed_initiator(ev, N, T, fit_b=True)
        out[name].update(kappa=f.kappa, tau_e=1 / f.gamma_e, n_node_fit=f.n_node)
        print(f"{name}: kappa={f.kappa:.4g} 1/gamma_e={1 / f.gamma_e:.4g} n={f.n_node:.3f}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)

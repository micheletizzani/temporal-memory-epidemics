"""Record-matched controls for the shuffle ratio R (resolves the hospital and baboon tensions).

The phase map (hawkes_phase_space.py) is a single homogeneous node in continuous time.
A record has heterogeneous nodes, tied timestamps, a finite number of events and slow rate drift, so
its R need not sit on that curve. Here the identical screen (real_record_screen.screen_R protocol:
same windows, per-node order shuffle, single-exponential estimator) is applied to
  real          the participation train of the record (reproduces real_record_screen.json),
  local         the real record under a block shuffle (K consecutive gaps per node) that keeps slow drift,
  hawkes        exponential self-excitation at the fitted (n_hat, gamma_hat) of the screen estimator,
                with per-node baselines giving the record's own event counts, times quantised to the
                record's resolution, n_sur surrogates,
  null          the same without self-excitation (n = 0).
Also reports the node-normalised memory coefficient M_within (memory_artifact_check.py) of each stream.
Usage: record_matched_surrogate.py "<record name>"   -> results/matched_<slug>.json
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, math, os, re, sys, time
import numpy as np
from scipy.optimize import minimize
import real_record_screen as rs  # noqa: E402
from real_record_screen import GAMMAS, B_SUB, S_SHUF, SUB, multi_replay, nll_multi  # noqa: E402
from screen_neural import per_node_shuffle  # noqa: E402
from memory_artifact_check import measures  # noqa: E402

K_BLOCK, N_SUR = 20, 3


def n_gamma(ev, N, T):
    """As real_record_screen.n_exp, but also returns the best rate."""
    init = [i for _t, i, _j in ev]
    S, C, counts = multi_replay(ev, init, N, GAMMAS, T)
    best = (-1e18, 0.0, GAMMAS[0])
    for r in range(len(GAMMAS)):
        w = np.zeros(len(GAMMAS)); w[r] = 1.0
        o = minimize(lambda x: nll_multi(1 / (1 + math.exp(-x[0])), w, S, C, counts, init, N, T)[0],
                     [0.0], method="Nelder-Mead", options=dict(maxiter=250, xatol=1e-3, fatol=1e-3))
        if -o.fun > best[0]:
            best = (-o.fun, 1 / (1 + math.exp(-o.x[0])), float(GAMMAS[r]))
    return best[1], best[2]


def local_shuffle(ev, rng, k=K_BLOCK):
    by = {}
    for t, i, j in ev:
        by.setdefault(i, []).append((t, j))
    out = []
    for i, s in by.items():
        ts = np.array([t for t, _ in s]); d = np.diff(np.concatenate([[0.0], ts]))
        for a in range(0, len(d), k):
            blk = d[a:a + k]; rng.shuffle(blk); d[a:a + k] = blk
        out += [(float(x), i, j) for x, (_, j) in zip(np.cumsum(d), s)]
    return sorted(out)


def screen(ev, shuffler, seed=0):
    """screen_R with a pluggable shuffle; same windows as real_record_screen.screen_R."""
    evs = sorted(ev); E = len(evs); rng = np.random.default_rng(seed); R, ns, gs = [], [], []
    for b in range(B_SUB):
        s = int(rng.integers(0, E - SUB)) if E > SUB else 0
        w = evs[s:s + SUB]; t0 = w[0][0]
        w = [(t - t0, i, j) for t, i, j in w]
        N = 1 + max(max(i, j) for _t, i, j in w); T = w[-1][0] + 1e-9
        ne, ge = n_gamma(w, N, T); ns.append(ne); gs.append(ge)
        R += [n_gamma(shuffler(list(w), np.random.default_rng(1000 * b + k)), N, T)[0] / max(ne, 1e-3)
              for k in range(S_SHUF)]
    return dict(R=float(np.mean(R)), lo=float(np.percentile(R, 5)), hi=float(np.percentile(R, 95)),
                n_hat=float(np.mean(ns)), gamma_hat=float(np.exp(np.mean(np.log(gs)))))


def hawkes_node(mu, n, gamma, T, rng):
    ts = list(rng.uniform(0, T, rng.poisson(mu * T))); out = []
    while ts:
        t = ts.pop()
        if t > T:
            continue
        out.append(t)
        ts += list(t + rng.exponential(1.0 / gamma, rng.poisson(n)))
    return out


def surrogate(counts, T, n, gamma, delta, rng):
    ev = []
    for i, c in enumerate(counts):
        for t in hawkes_node((1 - n) * c / T, n, gamma, T, rng):
            ev.append((float(math.floor(t / delta) * delta), i, (i + 1) % len(counts)))
    return sorted(ev)


def curve(g, fit):
    return fit["R0"] + (1 - fit["R0"]) / (1 + (fit["g0"] / g) ** fit["a"])


if __name__ == "__main__":
    name = sys.argv[1]; slug = re.sub(r"\W+", "_", name).strip("_")
    out_path = os.path.join(ROOT, "results", f"matched_{slug}.json")
    ev = sorted(rs.RECORDS[name][0]())
    N = 1 + max(max(i, j) for _t, i, j in ev); T = ev[-1][0] - ev[0][0] + 1e-9
    t0 = ev[0][0]; ev = [(t - t0, i, j) for t, i, j in ev]
    pos = np.diff(sorted({t for t, _i, _j in ev})); delta = float(pos[pos > 0].min())
    counts = np.bincount([i for _t, i, _j in ev], minlength=N)
    res = dict(record=name, events=len(ev), nodes=int(N), T=T, resolution=delta)
    t = time.time()
    res["real"] = screen(ev, per_node_shuffle); print("real", res["real"], int(time.time() - t), flush=True)
    res["local"] = screen(ev, local_shuffle); print("local", res["local"], int(time.time() - t), flush=True)
    n_hat, g_hat = res["real"]["n_hat"], res["real"]["gamma_hat"]
    mu_med = float(np.median((1 - n_hat) * counts[counts >= 5] / T))
    res["g_screen"] = g_hat / mu_med
    fit = json.load(open(os.path.join(ROOT, "results", "hawkes_crossover_fit.json")))
    res["R_curve_at_g_screen"] = curve(res["g_screen"], fit)
    res["M_within_real"] = measures(ev)[1]
    for tag, nn in (("hawkes", n_hat), ("null", 0.0)):
        rows, mw = [], []
        for k in range(N_SUR):
            sev = surrogate(counts, T, nn, g_hat, delta, np.random.default_rng(7 + k))
            rows.append(screen(sev, per_node_shuffle, seed=k)); mw.append(measures(sev)[1])
            print(tag, k, rows[-1], int(time.time() - t), flush=True)
        res[tag] = dict(R=[r["R"] for r in rows], R_mean=float(np.mean([r["R"] for r in rows])),
                        n_hat=[r["n_hat"] for r in rows], M_within=float(np.mean(mw)))
        json.dump(res, open(out_path, "w"), indent=1)
    json.dump(res, open(out_path, "w"), indent=1); print("wrote", out_path)

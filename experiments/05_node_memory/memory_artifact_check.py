"""Is the pooled lag-one memory coefficient M an order effect or a between-node scale effect?

M of real_record_screen.burstiness_memory pools consecutive-gap pairs over nodes without removing each
node's mean gap, so a heterogeneous population of memoryless nodes already gives M > 0, and a per-node
shuffle (which keeps every node's gap multiset) cannot remove that component. For every record:
  M_pool         pooled lag-one correlation (as in the screen)
  M_pool_shuf    the same after a per-node inter-event-order shuffle (mean over S shuffles)
  M_within       lag-one correlation after dividing each node's gaps by its own mean gap
  M_within_shuf  the same after the shuffle
  M_rank         within-node rank (Spearman-type) lag-one correlation, robust to heavy-tailed gaps
  M_rank_shuf    the same after the shuffle
The order-carried part is M - M_shuf. Writes results/memory_artifact_check.json.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, sys
import numpy as np
import real_record_screen as rs  # noqa: E402
from screen_neural import per_node_shuffle  # noqa: E402

OUT = os.path.join(ROOT, "results", "memory_artifact_check.json")
S = 6


def gaps(events):
    seq, last = {}, {}
    for t, i, _j in events:
        if i in last:
            seq.setdefault(i, []).append(t - last[i])
        last[i] = t
    return {i: np.array([x for x in v if x > 0]) for i, v in seq.items() if sum(x > 0 for x in v) >= 5}


def lag1(vs, normalise):
    xs, ys = [], []
    for v in vs.values():
        if normalise:
            v = v / v.mean()
        xs += list(v[:-1]); ys += list(v[1:])
    return float(np.corrcoef(xs, ys)[0, 1])


def lag1_rank(vs):
    from scipy.stats import rankdata
    xs, ys = [], []
    for v in vs.values():
        r = rankdata(v) / len(v)
        xs += list(r[:-1]); ys += list(r[1:])
    return float(np.corrcoef(xs, ys)[0, 1])


def measures(ev):
    g = gaps(ev)
    return lag1(g, False), lag1(g, True), len(g)


def rank_measure(ev):
    return lag1_rank(gaps(ev))


if __name__ == "__main__":
    out = {}
    for name, (load, _r) in rs.RECORDS.items():
        ev = sorted(load())
        mp, mw, nn = measures(ev)
        sp, sw, sr = [], [], []
        for k in range(S):
            sh = per_node_shuffle(list(ev), np.random.default_rng(k))
            a, b, _ = measures(sh)
            sp.append(a); sw.append(b); sr.append(rank_measure(sh))
        out[name] = dict(M_pool=mp, M_pool_shuf=float(np.mean(sp)), M_within=mw,
                         M_within_shuf=float(np.mean(sw)), M_within_shuf_sd=float(np.std(sw)),
                         M_rank=rank_measure(ev), M_rank_shuf=float(np.mean(sr)), nodes=nn, events=len(ev))
        o = out[name]
        print(f"{name:26s} pool {mp:+.3f} -> shuf {o['M_pool_shuf']:+.3f} | within {mw:+.3f} -> shuf "
              f"{o['M_within_shuf']:+.3f} (sd {o['M_within_shuf_sd']:.3f})  nodes {nn}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)

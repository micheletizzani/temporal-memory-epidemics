"""Positive control for the memory coefficients: exponential node memory (1/gamma = 30 s) injected into the
high-school activity pattern. Reports the pooled Pearson M (as in the main text), the within-node Pearson M,
and the within-node rank M, each for the simulated record and after a per-node order shuffle.
Compare with results of memory_artifact_check.py for the real records."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os, sys
import numpy as np
from scipy.stats import rankdata
import semi_synthetic as ss
from generative_models import MechanismParams, TemporalNetworkModel, simulate


def stats(events, shuffle_seed=None):
    seq, last = {}, {}
    for t, i, _j in events:
        if i in last:
            seq.setdefault(i, []).append(t - last[i])
        last[i] = t
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    P, W, R = ([[], []], [[], []], [[], []])
    for v in seq.values():
        v = np.array([x for x in v if x > 0])
        if len(v) < 5:
            continue
        if rng is not None:
            rng.shuffle(v)
        for (xs, ys), w in ((P, v), (W, v / v.mean()), (R, rankdata(v) / len(v))):
            xs += list(w[:-1]); ys += list(w[1:])
    return [float(np.corrcoef(*q)[0, 1]) for q in (P, W, R)]


rate, N, T = ss.real_activities(n_keep=40)
rate = rate * (600.0 / (rate.mean() * T))
import json
rows = []
print("n_node | pooled M  within M  rank M   || after per-node shuffle: pooled  within  rank")
for true_n in [0.0, 0.2, 0.4, 0.6, 0.8]:
    a = rate * (1.0 - true_n)
    model = TemporalNetworkModel(N=N, activities=a, attractiveness=np.ones(N),
        params=MechanismParams(true_n, 1.0 / 30.0, 0.0, 1.0 / 3000.0))
    ev = list(simulate(model, T, rng=np.random.default_rng(1 + int(100 * true_n))))
    m = stats(ev); s = stats(ev, shuffle_seed=0)
    gp = np.concatenate([np.array([x for x in np.diff(sorted(t for t, i, _ in ev if i == u)) if x > 0]) for u in set(i for _, i, _ in ev)])
    Bv = float((gp.std() - gp.mean()) / (gp.std() + gp.mean()))
    rows.append(dict(true_n=true_n, B=Bv, M_rank=m[2], M_within=m[1], M_pooled=m[0], M_rank_shuf=s[2], M_pooled_shuf=s[0]))
    print(f"{true_n:4.1f}   | {m[0]:+.3f}  {m[1]:+.3f}  {m[2]:+.3f}   ||   {s[0]:+.3f}  {s[1]:+.3f}  {s[2]:+.3f}")

json.dump(rows, open(os.path.join(ROOT, "results", "poscon_memory_rank.json"), "w"), indent=1)

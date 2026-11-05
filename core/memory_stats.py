"""The memory coefficient used throughout the paper (single definition).

M_k = Pearson correlation of the within-node RANKS of intervals k apart, pooled over nodes with at least
five positive intervals. Ranks are taken within each node (rank / number of intervals), so M_k is
(i) free of between-node differences in scale and (ii) not dominated by a few very long intervals.
A Poisson process, and any renewal process, has M_k = 0 in expectation. For a single train it is the
Spearman-type correlation of consecutive intervals.
"""
import numpy as np
from scipy.stats import rankdata


def node_gaps(events, min_gaps=5):
    """events: iterable of (t, i, j); returns {i: positive inter-event times of the initiator train i}."""
    seq, last = {}, {}
    for t, i, _j in sorted(events):
        if i in last:
            seq.setdefault(i, []).append(t - last[i])
        last[i] = t
    out = {}
    for i, v in seq.items():
        v = np.array([x for x in v if x > 0])
        if len(v) >= min_gaps:
            out[i] = v
    return out


def rank_memory(gaps, lag=1):
    """Pooled within-node rank correlation at the given lag; `gaps` as returned by node_gaps."""
    xs, ys = [], []
    for v in gaps.values():
        if len(v) <= lag:
            continue
        r = rankdata(v) / len(v)
        xs += list(r[:-lag]); ys += list(r[lag:])
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def rank_memory_events(events, lag=1):
    return rank_memory(node_gaps(events), lag)


def rank_memory_train(ts, lag=1):
    """Single train of event times."""
    d = np.diff(np.sort(np.asarray(ts)))
    d = d[d > 0]
    if len(d) <= lag + 2:
        return float("nan")
    return rank_memory({0: d}, lag)

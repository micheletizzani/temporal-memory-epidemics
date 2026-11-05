"""Is the within-node lag-one order structure a common slow modulation of activity, or node memory?

Each record's event times are mapped through the cumulative smoothed global event rate
(t -> Lambda(t), rate estimated in bins of width T/nbins), which removes any activity profile shared by
all nodes (daily cycle, sessions). The within-node rank lag-one correlation is recomputed on the
rescaled times. If it falls to zero, a common profile explains the short-gap persistence; if it stays,
the order is node-specific. Writes results/memory_modulation_check.json.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, sys
import numpy as np
import real_record_screen as rs  # noqa: E402
from memory_artifact_check import rank_measure, measures  # noqa: E402

OUT = os.path.join(ROOT, "results", "memory_modulation_check.json")
NBINS = (25, 100, 400)


def rescale(ev, nbins):
    t = np.array([e[0] for e in ev]); t0, t1 = t.min(), t.max() + 1e-9
    edges = np.linspace(t0, t1, nbins + 1)
    cnt, _ = np.histogram(t, edges)
    cum = np.concatenate([[0.0], np.cumsum(cnt + 0.0)])          # Lambda at the bin edges (events)
    lam = np.interp(t, edges, cum)
    return [(float(l), e[1], e[2]) for l, e in zip(lam, ev)]


if __name__ == "__main__":
    out = {}
    for name, (load, _r) in rs.RECORDS.items():
        ev = sorted(load())
        row = dict(rank_raw=rank_measure(ev), within_raw=measures(ev)[1])
        for nb in NBINS:
            r = rescale(ev, nb)
            row[f"rank_{nb}"] = rank_measure(r); row[f"within_{nb}"] = measures(r)[1]
        out[name] = row
        print(f"{name:26s} rank raw {row['rank_raw']:+.3f} | " +
              " ".join(f"{nb} bins {row[f'rank_{nb}']:+.3f}" for nb in NBINS), flush=True)
        json.dump(out, open(OUT, "w"), indent=1)

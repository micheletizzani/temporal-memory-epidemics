"""Does the within-node rank lag-one memory survive removal of repeated consecutive partners?

A fragmented episode with one partner makes several quick events of both endpoints: tie reuse then
shows up as node-level short-gap persistence. Here every node's train keeps an event only when its
partner differs from the partner of the node's previous kept event (repeats of the same tie collapsed),
and the rank lag-one correlation is recomputed. Writes results/memory_tie_check.json.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, sys
import numpy as np
import real_record_screen as rs  # noqa: E402
from memory_artifact_check import rank_measure, measures  # noqa: E402

OUT = os.path.join(ROOT, "results", "memory_tie_check.json")


def collapse_repeats(ev):
    last, out = {}, []
    for t, i, j in sorted(ev):
        if last.get(i) == j:
            continue
        last[i] = j; out.append((t, i, j))
    return out


if __name__ == "__main__":
    res = {}
    for name, (load, _r) in rs.RECORDS.items():
        ev = sorted(load()); c = collapse_repeats(ev)
        try:
            row = dict(events=len(ev), kept=len(c), rank_raw=rank_measure(ev), rank_collapsed=rank_measure(c),
                       within_raw=measures(ev)[1], within_collapsed=measures(c)[1])
        except Exception as e:                       # too few gaps left
            row = dict(events=len(ev), kept=len(c), error=str(e))
        res[name] = row
        print(f"{name:26s} kept {row['kept']:6d}/{row['events']:6d}  rank {row.get('rank_raw', float('nan')):+.3f} -> "
              f"{row.get('rank_collapsed', float('nan')):+.3f}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

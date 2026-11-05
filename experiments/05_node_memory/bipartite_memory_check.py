"""Initiator-only versus both-endpoint memory (SM 'digital datasets'): within-node rank M_1 on the full core files."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, sys
import numpy as np
from memory_stats import rank_memory_events  # noqa: E402
out = {}
for name, f in (("Stack Overflow", "so_core_events.txt"), ("Wikibooks", "wikibooks_core_events.txt")):
    d = np.loadtxt(os.path.join(ROOT, "data", f)); ev = [(float(r[0]), int(r[1]), int(r[2])) for r in d]
    both = ev + [(t, j, i) for t, i, j in ev]
    ini, tgt = {i for _, i, _ in ev}, {j for _, _, j in ev}
    out[name] = dict(events=len(ev), initiators=len(ini), targets=len(tgt), overlap=len(ini & tgt),
                     M1_initiator=rank_memory_events(ev), M1_both=rank_memory_events(both))
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "bipartite_memory_check.json"), "w"), indent=1)

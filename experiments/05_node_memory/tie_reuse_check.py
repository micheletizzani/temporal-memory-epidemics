"""Repeated-tie fraction and mean uses per tie of the proximity records (day 0, sessionised contacts)."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, sys
import record_fits as rf  # noqa: E402
out = {}
for name in ("High school", "Hospital ward", "Conference (SFHH)"):
    c = sorted(rf.contacts_day0(name)); seen, rep = set(), 0
    for t, i, j in c:
        k = (min(i, j), max(i, j)); rep += k in seen; seen.add(k)
    out[name] = dict(contacts=len(c), repeat_fraction=rep / len(c), uses_per_tie=len(c) / len(seen), ties=len(seen))
    print(name, out[name])
json.dump(out, open(os.path.join(ROOT, "results", "tie_reuse_check.json"), "w"), indent=1)

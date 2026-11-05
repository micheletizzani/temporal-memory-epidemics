"""Is group simultaneity the cause of a small M and R ~ 1?  Each node's coincident contacts are collapsed into a single
activation (one event per node and time stamp) and M (within-node rank) and R (screen protocol) are recomputed.
Writes results/group_collapse_check.json."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, sys
import real_record_screen as rs  # noqa: E402
from memory_stats import rank_memory_events  # noqa: E402

out = {}
for name in ("Hospital ward", "High school", "Conference (SFHH)"):
    ev = sorted(rs.RECORDS[name][0]()); seen, col = set(), []
    for t, i, j in ev:
        if (t, i) not in seen:
            seen.add((t, i)); col.append((t, i, j))
    R = rs.screen_R(col)
    out[name] = dict(events=len(ev), collapsed=len(col), M_raw=rank_memory_events(ev), M_collapsed=rank_memory_events(col),
                     R_collapsed=list(R))
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items() if k != "R_collapsed"}, [round(x, 3) for x in R], flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "group_collapse_check.json"), "w"), indent=1)

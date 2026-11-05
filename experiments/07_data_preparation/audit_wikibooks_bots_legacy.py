# %%
"""Audit of the legacy file data/wikibooks_bots_only.txt, whose extraction script was lost.

Matches its rows to the KONECT edit-enwikibooks record through timestamps that are unique in the
source, recovers the (re-indexed) user mapping, and asks which users it contains. Finding: it keeps
every edit of 2,892 users, selected by activity (inclusion rises from 0 below 18 edits to ~1 above
60) rather than by any bot criterion, which the anonymised KONECT ids could not support. The file
is therefore an active-editor core, not a bot record; build_wikibooks_bots.py replaces it.
Writes results/wikibooks_bots_legacy_audit.json.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os
from collections import Counter, defaultdict

import numpy as np

DATA = os.path.abspath(os.path.join(ROOT, "data"))
SRC = os.path.join(DATA, "edit-enwikibooks_x", "edit-enwikibooks.edges")
LEG = os.path.join(DATA, "wikibooks_bots_only.txt")
OUT = os.path.join(ROOT, "results", "wikibooks_bots_legacy_audit.json")
BINS = [1, 2, 5, 10, 15, 18, 20, 25, 30, 40, 50, 60, 80, 100, 135, 200, 500, 10 ** 9]

# %%
E = np.loadtxt(SRC, comments="%", dtype=np.int64)
u, t = E[:, 0], E[:, 3]
L = np.loadtxt(LEG, dtype=np.int64)
lt, lu = L[:, 0], L[:, 1]
tc = Counter(t.tolist())
uniq = {a: b for a, b in zip(t.tolist(), u.tolist()) if tc[a] == 1}
amap = defaultdict(Counter)
for a, b in zip(lt.tolist(), lu.tolist()):
    if a in uniq:
        amap[b][uniq[a]] += 1
consistent = all(len(v) == 1 for v in amap.values())
inc = {v.most_common(1)[0][0] for v in amap.values()}
cnt = Counter(u.tolist())
lcnt = Counter(lu.tolist())
kept_all = all(lcnt[b] == cnt[v.most_common(1)[0][0]] for b, v in amap.items())
by_bin = []
for lo, hi in zip(BINS[:-1], BINS[1:]):
    us = [x for x, c in cnt.items() if lo <= c < hi]
    if us:
        by_bin.append(dict(edits_lo=lo, edits_hi=hi, users=len(us),
                           included=int(sum(x in inc for x in us))))
res = dict(source_edits=int(len(u)), source_users=len(cnt), legacy_rows=int(len(lt)),
           legacy_users=len(inc), legacy_share_of_edits=float(len(lt) / len(u)),
           id_mapping_consistent=consistent, all_edits_of_included_users_kept=kept_all,
           inclusion_by_edit_count=by_bin,
           conclusion="activity-selected editor core, not a bot subset")
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "inclusion_by_edit_count"}, indent=1))
for b in by_bin:
    print(f"  edits [{b['edits_lo']},{b['edits_hi']}): {b['included']}/{b['users']}")

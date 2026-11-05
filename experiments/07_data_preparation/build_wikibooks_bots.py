# %%
"""Rebuild the Wikibooks bot-edit record from the Wikimedia edit history.

The original extraction script was lost, and the file it left (data/wikibooks_bots_only.txt) is not
a bot subset: audit_wikibooks_bots_legacy.py shows it holds every edit of the 2,892 most active
KONECT editors (roughly >= 50 edits), 83% of all edits. The KONECT record cannot identify bots
at all, because its user ids are anonymised. This script therefore starts from the Wikimedia
stub-meta-history dump, which carries usernames.

Bot rule (documented, reviewable):
  1. NAME: the username follows the Wikimedia bot naming convention, i.e. contains "bot" as a
     word or a suffix (regex BOT_NAME, case-insensitive), minus the names in NOT_BOTS.
  2. FLAG: every account ever granted the local "bot" group (rights log) or currently in the
     global-bot group is recorded. Flagged accounts whose name fails rule 1 are humans who held
     the flag temporarily (e.g. administrators doing mass edits); they are listed in the manifest
     and excluded, because their edits mix manual and automated work.
  3. LOCAL: usernames containing ">" (e.g. "w>", "imported>", "strategywiki>") are revisions
     imported from other wikis, timestamped by their original wiki; they are not Wikibooks
     activity and are dropped before either rule is applied.
  A bot is a local account that passes rule 1. The manifest records, for every candidate, whether it
  was ever flagged, so a stricter set (rule 1 AND flagged) can be used as a robustness check.

Window: edits up to T_END, the last timestamp of the KONECT edit-enwikibooks record, so the bot
record covers the same period as the human Wikibooks record. Only registered accounts are kept.

Outputs (data/wikimedia/):
  enwikibooks_edits_registered.tsv   t  user  page  ns   (all registered edits in the window; cache)
  wikibooks_bots_events.txt          t  user_idx  page_idx   (bot edits, 0-indexed, time-sorted)
  wikibooks_bots_manifest.json       rule, counts, per-account table, validation against KONECT
Run: python build_wikibooks_bots.py   (parsing the 0.3 GB dump takes several minutes)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import gzip
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

DATA = os.path.abspath(os.path.join(ROOT, "data"))
WM = os.path.join(DATA, "wikimedia")
DUMP = os.path.join(WM, "enwikibooks-latest-stub-meta-history.xml.gz")
EDITS = os.path.join(WM, "enwikibooks_edits_registered.tsv")
OUT_EV = os.path.join(WM, "wikibooks_bots_events.txt")
OUT_MAN = os.path.join(WM, "wikibooks_bots_manifest.json")
KONECT = os.path.join(DATA, "edit-enwikibooks_x", "edit-enwikibooks.edges")
T_END = 1283031674                       # last timestamp in KONECT edit-enwikibooks (2010-08-28)
API = "https://en.wikibooks.org/w/api.php"
META = "https://meta.wikimedia.org/w/api.php"
UA = {"User-Agent": "temporal-memory-research/1.0 (micti@dtu.dk)"}

BOT_NAME = re.compile(r"(bot\b|bot$|bot[\s_\-.\d(]|\(bot\))", re.IGNORECASE)
NOT_BOTS = {"Talbot", "Abbot", "Cabot"}  # surnames that match the pattern; extend after review


# %%
def api_all(url, params, key):
    out, cont = [], {}
    while True:
        q = urllib.parse.urlencode({**params, **cont, "format": "json"})
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"{url}?{q}", headers=UA), timeout=60))
        out += d["query"][key]
        if "continue" not in d:
            return out
        cont = d["continue"]


def flagged_accounts():
    """Accounts ever granted the local bot group, and current global bots."""
    local = {}
    for e in api_all(API, dict(action="query", list="logevents", letype="rights", lelimit="500",
                               leprop="title|timestamp|details"), "logevents"):
        p = e.get("params", {})
        if "bot" in p.get("newgroups", []) and "bot" not in p.get("oldgroups", []):
            local.setdefault(e["title"].split(":", 1)[-1], e["timestamp"])
    current = {u["name"] for u in api_all(API, dict(action="query", list="allusers", augroup="bot",
                                                    aulimit="500"), "allusers")}
    try:
        glob = {u["name"] for u in api_all(META, dict(action="query", list="globalallusers",
                                                       agugroup="global-bot", agulimit="500"),
                                           "globalallusers")}
    except Exception as exc:                 # meta API unavailable: record it, do not guess
        print(f"  (global-bot list unavailable: {exc})")
        glob = set()
    return local, current, glob


# %%
def parse_dump():
    """Stream the dump once; cache (t, user, page_id, ns) for registered edits up to T_END."""
    if os.path.exists(EDITS):
        return
    from datetime import datetime, timezone
    n = 0
    with gzip.open(DUMP, "rb") as fh, open(EDITS + ".part", "w") as out:
        page_id = ns = None
        for _ev, el in ET.iterparse(fh, events=("end",)):
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "ns" and page_id is None:
                ns = el.text
            elif tag == "id" and page_id is None and el.text and ns is not None:
                page_id = el.text
            elif tag == "revision":
                ts = el.findtext("{*}timestamp")
                user = el.findtext("{*}contributor/{*}username")
                if ts and user:
                    t = int(datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
                            .replace(tzinfo=timezone.utc).timestamp())
                    if t <= T_END:
                        out.write(f"{t}\t{user}\t{page_id}\t{ns}\n"); n += 1
                el.clear()
            elif tag == "page":
                page_id = ns = None
                el.clear()
    os.replace(EDITS + ".part", EDITS)
    print(f"  parsed {n} registered edits up to T_END")


def load_edits():
    rows = []
    with open(EDITS) as fh:
        for line in fh:
            t, u, p, ns = line.rstrip("\n").split("\t")
            rows.append((int(t), u, p, ns))
    rows.sort(key=lambda r: r[0])
    return rows


def konect_counts():
    users, n = set(), 0
    with open(KONECT) as fh:
        for line in fh:
            if line.startswith("%"):
                continue
            p = line.split(); users.add(p[0]); n += 1
    return n, len(users)


# %%
if __name__ == "__main__":
    if not os.path.exists(DUMP):
        sys.exit(f"missing {DUMP}: download enwikibooks-latest-stub-meta-history.xml.gz from "
                 "https://dumps.wikimedia.org/enwikibooks/latest/")
    parse_dump()
    rows_all = load_edits()
    rows = [r for r in rows_all if ">" not in r[1]]     # rule 3: drop interwiki-imported revisions
    edits_by_user = Counter(r[1] for r in rows)
    local, current, glob = flagged_accounts()
    flagged = set(local) | current | glob

    def is_bot(name):
        return bool(BOT_NAME.search(name)) and name not in NOT_BOTS

    bots = {u for u in edits_by_user if is_bot(u)}
    flagged_human_named = sorted(u for u in flagged if u in edits_by_user and not is_bot(u))
    bot_rows = [r for r in rows if r[1] in bots]
    uidx = {u: k for k, u in enumerate(sorted({r[1] for r in bot_rows}))}
    pidx = {p: k for k, p in enumerate(sorted({r[2] for r in bot_rows}, key=int))}
    with open(OUT_EV, "w") as fh:
        for t, u, p, _ns in bot_rows:
            fh.write(f"{t} {uidx[u]} {pidx[p]}\n")

    kn, ku = konect_counts()
    man = dict(
        rule=dict(name_regex=BOT_NAME.pattern, not_bots=sorted(NOT_BOTS), window_end=T_END,
                  registered_only=True,
                  note="bot = name rule; flagged accounts with human names excluded and listed"),
        validation=dict(dump_registered_edits_incl_imported=len(rows_all),
                        imported_edits_dropped=len(rows_all) - len(rows),
                        dump_registered_edits=len(rows), dump_registered_users=len(edits_by_user),
                        konect_edits=kn, konect_users=ku),
        counts=dict(bot_accounts=len(bots), bot_edits=len(bot_rows),
                    bot_accounts_flagged=len(bots & flagged), pages=len(pidx)),
        flagged_human_named_excluded=[dict(name=u, edits=edits_by_user[u]) for u in flagged_human_named],
        accounts=sorted([dict(name=u, idx=uidx[u], edits=edits_by_user[u],
                              ever_flagged=u in flagged, local_flag_first=local.get(u))
                         for u in bots], key=lambda d: -d["edits"]),
    )
    json.dump(man, open(OUT_MAN, "w"), indent=1)
    print(json.dumps({k: man[k] for k in ("validation", "counts")}, indent=1))
    print("top bot accounts:", [(a["name"], a["edits"], a["ever_flagged"]) for a in man["accounts"][:12]])
    print("flagged but human-named (excluded):", [(d["name"], d["edits"]) for d in man["flagged_human_named_excluded"]])

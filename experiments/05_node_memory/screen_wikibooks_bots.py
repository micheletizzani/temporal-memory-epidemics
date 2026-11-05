# %%
"""Shuffle-ratio screen (protocol of scalefree_screen_hardened) on the rebuilt Wikibooks bot record.

Replaces the 'Wikibooks (Bots)' entry of results/scalefree_screen.json, which was computed on the
legacy file data/wikibooks_bots_only.txt -- an activity-selected editor core, not bots
(audit_wikibooks_bots_legacy.py). Input: data/wikimedia/wikibooks_bots_events.txt from
build_wikibooks_bots.py. Writes results/wikibooks_bots_screen.json.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os

from screen_neural import screen, load_directed, ROOT, DATA

PATH = os.path.join(DATA, "wikimedia", "wikibooks_bots_events.txt")
OUT = os.path.join(ROOT, "results", "wikibooks_bots_screen.json")

# %%
if __name__ == "__main__":
    ev, N, _T = load_directed(PATH, 0, 1, 2, None)
    r = screen(ev, N, "Wikibooks bots (rebuilt)")
    json.dump(r, open(OUT, "w"), indent=1)
    print(f"R_exp={r['R_exp'][0]:.3f} [{r['R_exp'][1]:.3f},{r['R_exp'][2]:.3f}] "
          f"R_mix={r['R_mix'][0]:.3f} [{r['R_mix'][1]:.3f},{r['R_mix'][2]:.3f}] n_exp={r['n_exp_real']:.3f}")

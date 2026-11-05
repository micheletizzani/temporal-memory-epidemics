"""Active cores of the Wikibooks and Stack Overflow records (paper, Sec. IV E).

wikibooks:     keep edits whose editor has >= 80 edits and whose page has >= 40 (one pass).
stackoverflow: from the output of preprocess_directed.py (--min-events 1000 --cap-nodes 400), keep
               events whose two endpoints each have >= 40 events in that file (one pass).

Output: t i j, time sorted, node ids relabelled from 0.

Usage:
  python build_core_subgraphs.py wikibooks data/edit-enwikibooks.edges
  python build_core_subgraphs.py stackoverflow data/stackoverflow_events.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path


def relabel(values):
    return np.unique(values, return_inverse=True)[1]


def write(out, t, i, j):
    order = np.argsort(t, kind="stable")
    np.savetxt(out, np.c_[t[order], i[order], j[order]], fmt=["%.6f", "%d", "%d"])
    nodes = len(set(i.tolist()) | set(j.tolist()))
    print(f"wrote {out}: {len(t)} events, {nodes} nodes, {len(set(i.tolist()))} initiators")


def wikibooks(path):
    rows = np.array([line.split() for line in open(path) if line[0] not in "%#"], dtype=float)
    editor, page, t = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 3]
    keep = (np.bincount(editor)[editor] >= 80) & (np.bincount(page)[page] >= 40)
    e = relabel(editor[keep])
    write(os.path.join(ROOT, "data", "wikibooks_core_events.txt"), t[keep], e, relabel(page[keep]) + e.max() + 1)


def stackoverflow(path):
    rows = np.loadtxt(path)
    t, i, j = rows[:, 0], rows[:, 1].astype(int), rows[:, 2].astype(int)
    counts = np.bincount(np.r_[i, j])
    keep = (counts[i] >= 40) & (counts[j] >= 40)
    ids = relabel(np.r_[i[keep], j[keep]])
    n = int(keep.sum())
    write(os.path.join(ROOT, "data", "so_core_events.txt"), t[keep], ids[:n], ids[n:])


if __name__ == "__main__":
    {"wikibooks": wikibooks, "stackoverflow": stackoverflow}[sys.argv[1]](sys.argv[2])

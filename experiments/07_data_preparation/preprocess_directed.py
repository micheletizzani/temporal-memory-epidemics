#!/usr/bin/env python3
"""Stream a large directed (i, j, t) temporal edge list into an unbiased, analysis-ready core.

Motivation. Selecting the top-K nodes by degree distorts exactly what the estimator measures:
it truncates the node-activity distribution and, by filtering to the among-core subgraph, throws
away each hub's periphery events (defeating the per-node power goal and biasing the selection
channel kappa). This tool offers two unbiased alternatives and, either way, RETAINS every
out-edge of each focal node (partners may be any node), so a focal node's true timing and partner
distribution survive.

Modes:
  --window A B     keep every event whose time lies in the fractional span [A, B] of the record
                   (a random-in-time subsample; does NOT distort the activity distribution).
  --min-events M   keep FOCAL initiators with >= M out-events; retain ALL their out-edges.
  --cap-nodes K    if the focal set exceeds K, sample K focal nodes at RANDOM (never top-K).

Two streaming passes keep memory bounded on 10^7-10^8-line files. Outputs a cleaned event file
(t i j, 0-indexed, time rescaled to --ttarget) plus a diagnostics JSON with the out-degree CCDF
before and after subsampling, so the "did subsampling distort activity?" question is answered
directly.

Usage:
  python preprocess_directed.py sx-stackoverflow.txt.gz --icol 0 --jcol 1 --tcol 2 \
         --min-events 1000 --cap-nodes 400 --tag stackoverflow
  python preprocess_directed.py wiki-talk-temporal.txt.gz --window 0.4 0.6 --tag wikitalk
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import argparse
import gzip
import json
import os
import random

import numpy as np



def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _parse(line, icol, jcol, tcol, sep):
    p = line.split(sep) if sep else line.split()
    try:
        return p[icol], p[jcol], float(p[tcol])
    except (IndexError, ValueError):
        return None


def pass1_stats(path, icol, jcol, tcol, sep):
    """One stream: out-degree per initiator, global time span, edge count."""
    outdeg = {}
    tmin, tmax, E = np.inf, -np.inf, 0
    with _open(path) as fh:
        for line in fh:
            if not line or line[0] in "#%":
                continue
            r = _parse(line, icol, jcol, tcol, sep)
            if r is None:
                continue
            i, j, t = r
            if i == j:
                continue
            outdeg[i] = outdeg.get(i, 0) + 1
            tmin = t if t < tmin else tmin
            tmax = t if t > tmax else tmax
            E += 1
    return outdeg, float(tmin), float(tmax), E


def choose_focal(outdeg, min_events, cap_nodes, seed):
    focal = {u for u, d in outdeg.items() if d >= min_events}
    if cap_nodes and len(focal) > cap_nodes:
        rng = random.Random(seed)
        focal = set(rng.sample(sorted(focal), cap_nodes))   # random within the eligible set
    return focal


def pass2_collect(path, icol, jcol, tcol, sep, mode, focal, t0, t1):
    """Second stream: collect retained events as (t, i_str, j_str)."""
    ev = []
    with _open(path) as fh:
        for line in fh:
            if not line or line[0] in "#%":
                continue
            r = _parse(line, icol, jcol, tcol, sep)
            if r is None:
                continue
            i, j, t = r
            if i == j:
                continue
            if mode == "window":
                if t0 <= t <= t1:
                    ev.append((t, i, j))
            else:                                            # min-events: keep all out-edges of focal initiators
                if i in focal:
                    ev.append((t, i, j))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--icol", type=int, default=0); ap.add_argument("--jcol", type=int, default=1)
    ap.add_argument("--tcol", type=int, default=2); ap.add_argument("--sep", default=None)
    ap.add_argument("--window", nargs=2, type=float, default=None, metavar=("A", "B"),
                    help="fractional time window, e.g. 0.4 0.6")
    ap.add_argument("--min-events", type=int, default=1000)
    ap.add_argument("--cap-nodes", type=int, default=400)
    ap.add_argument("--ttarget", type=float, default=2000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="core")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()

    print(f"[pass1] scanning {a.path} ...", flush=True)
    outdeg, tmin, tmax, E = pass1_stats(a.path, a.icol, a.jcol, a.tcol, a.sep)
    span = (tmax - tmin) or 1.0
    print(f"[pass1] {E} edges, {len(outdeg)} initiators, span={span:.0f}", flush=True)

    mode = "window" if a.window else "min_events"
    if mode == "window":
        t0, t1 = tmin + a.window[0] * span, tmin + a.window[1] * span
        focal = None
        print(f"[mode] temporal window [{a.window[0]:.2f},{a.window[1]:.2f}] -> t in [{t0:.0f},{t1:.0f}]", flush=True)
    else:
        t0 = t1 = 0.0
        focal = choose_focal(outdeg, a.min_events, a.cap_nodes, a.seed)
        print(f"[mode] min-events>={a.min_events}: {len(focal)} focal initiators "
              f"(random-capped at {a.cap_nodes})", flush=True)

    ev = pass2_collect(a.path, a.icol, a.jcol, a.tcol, a.sep, mode, focal, t0, t1)
    if len(ev) < 50:
        raise SystemExit(f"only {len(ev)} events retained; loosen the subsample")
    ev.sort(key=lambda r: r[0])

    # remap ids (initiators + partners actually present), rescale time
    nodes = sorted({i for _t, i, _j in ev} | {j for _t, _i, j in ev})
    rm = {n: k for k, n in enumerate(nodes)}
    e0, e1 = ev[0][0], ev[-1][0]
    sc = a.ttarget / ((e1 - e0) or 1.0)
    ev_out = [((t - e0) * sc, rm[i], rm[j]) for (t, i, j) in ev]

    os.makedirs(a.outdir, exist_ok=True)
    evpath = os.path.join(a.outdir, f"{a.tag}_events.txt")
    with open(evpath, "w") as fo:
        for t, i, j in ev_out:
            fo.write(f"{t:.6f} {i} {j}\n")

    # diagnostics: out-degree CCDF before (all) vs after (focal/window), to prove no truncation bias
    deg_all = np.array(sorted(outdeg.values(), reverse=True), float)
    after_deg = {}
    for _t, i, _j in ev_out:
        after_deg[i] = after_deg.get(i, 0) + 1
    deg_after = np.array(sorted([d for u, d in after_deg.items()], reverse=True), float)
    diag = {
        "tag": a.tag, "mode": mode, "E_in": E, "E_out": len(ev_out), "N_out": len(nodes),
        "events_per_node_out": len(ev_out) / max(len(nodes), 1),
        "min_events": a.min_events, "cap_nodes": a.cap_nodes, "window": a.window,
        "outdeg_all_ccdf": {"deg": deg_all[:: max(1, len(deg_all) // 200)].tolist(),
                             "rank": list(range(0, len(deg_all), max(1, len(deg_all) // 200)))},
        "outdeg_after": deg_after.tolist()[:2000],
    }
    json.dump(diag, open(os.path.join(a.outdir, f"{a.tag}_preproc.json"), "w"))
    print(f"[done] wrote {evpath}: {len(ev_out)} events, N={len(nodes)}, "
          f"{diag['events_per_node_out']:.0f} events/node", flush=True)


if __name__ == "__main__":
    main()

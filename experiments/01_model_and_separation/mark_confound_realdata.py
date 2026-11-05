#!/usr/bin/env python3
"""Real-data test of Proposition 2 (weighted): does a real event mark join the node/tie
missing-information confound when the initiator is masked?

Two dataset roles, one script:

  --directed : an observed-initiator, marked event log (e.g. email with body length, calls with
               duration). The sender is the true initiator and the mark is treated as
               initiator-attributed (H_i). We (1) GATE on node-memory identifiability, then
               (2) mask a fraction (1-rho) of the true initiators and track the 3x3
               across-imputation confound of (n_node, kappa, c_W) as rho -> 0.
               PREDICTION: corr(n,c_W), corr(kappa,c_W) and the spread sigma(c_W) grow as labels
               are removed -- but only if node memory is genuinely identifiable (the gate).

  --undirected : a latent-initiator log with a pair-attributed mark (e.g. SocioPatterns session
               duration, H_ij). Fitting the mark with the label-free covariate leaves c_W
               invariant to the imputation -- the Corollary negative control.

Input is a whitespace/comma-separated event file; give the column indices. Node set is reduced
to the --nkeep most active nodes and time is rescaled to --ttarget so the fit is well-conditioned
(a monotone rescale preserves event order and the memory structure).

Usage (background-friendly, writes results/mark_real_<tag>.json and prints progress):
  python mark_confound_realdata.py PATH --tcol 0 --icol 1 --jcol 2 --wcol 3 \
         --directed --nkeep 80 --rhos 0,0.25,0.5,0.75,1.0 --tag enron --reps 4 > mark_real.log 2>&1 &
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import argparse
import json
import os
import sys

import numpy as np

import inference as inf                       # noqa: E402
from joint_kernel_map import shuffle_node_time          # noqa: E402
from mark_confound import confound_3x3, _corr           # noqa: E402


def load_events(path, tcol, icol, jcol, wcol, nkeep, ttarget, logmark=True, sep=None):
    """Parse an event log into ev=[(t, i, j)] (i = initiator/sender), marks w, N, T.
    Keeps the nkeep most active nodes; rescales time to [0, ttarget]."""
    raw = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#%":
                continue
            p = line.split(sep) if sep else line.split()
            try:
                t = float(p[tcol]); i = p[icol]; j = p[jcol]
                w = float(p[wcol])
            except (IndexError, ValueError):
                continue
            if i == j or not np.isfinite(w):
                continue
            raw.append((t, i, j, w))
    if not raw:
        raise SystemExit(f"no parseable events in {path}")
    raw.sort(key=lambda r: r[0])
    deg = {}
    for _t, i, j, _w in raw:
        deg[i] = deg.get(i, 0) + 1; deg[j] = deg.get(j, 0) + 1
    keep = set(sorted(deg, key=deg.get, reverse=True)[:nkeep])
    raw = [r for r in raw if r[0] is not None and r[1] in keep and r[2] in keep]
    nodes = sorted({i for _t, i, _j, _w in raw} | {j for _t, _i, j, _w in raw})
    rm = {n: k for k, n in enumerate(nodes)}
    t0, t1 = raw[0][0], raw[-1][0]
    span = (t1 - t0) or 1.0
    sc = ttarget / span
    ev = [((t - t0) * sc, rm[i], rm[j]) for (t, i, j, _w) in raw]
    w = np.array([r[3] for r in raw], float)
    w = np.clip(w, 1e-9, None)
    if logmark:                                         # marks are heavy-tailed; model log-mark
        w = np.exp((np.log(w) - np.log(w).mean()))      # center in log-space, keep positivity
    T = ev[-1][0] + ttarget * 0.01
    return ev, w, len(nodes), float(T)


def node_memory_gate(ev, N, T, seed=0):
    """Fit observed-initiator (n_hat, kappa_hat) and the shuffle ratio R_node = n_shuf/n_real.
    R_node << 1 => genuine, identifiable node memory (positive control is viable);
    R_node ~ 1  => node memory is a marginal artefact (masking cannot reveal a clean confound)."""
    r = inf.fit_observed_initiator(ev, N, float(T), fit_b=False)
    rng = np.random.default_rng(seed)
    r_sh = inf.fit_observed_initiator(shuffle_node_time(ev, rng), N, float(T), fit_b=False)
    R = float(r_sh.n_node / r.n_node) if r.n_node > 5e-2 else float("nan")
    # R_node < 0.7 is strong evidence of genuine node memory, but the shuffle test needs
    # ~1e3 events/node to have power, so R_node ~ 1 at low per-node counts is INCONCLUSIVE,
    # not proof of an artefact. The operative check is sigma(n) across imputations at rho=0
    # (reported per-curve below): if it is non-negligible the node channel is live.
    return dict(n_hat=float(r.n_node), kappa_hat=float(r.kappa), gamma_n=float(r.gamma_n),
                R_node=R, per_node=None, strong_memory=bool(np.isfinite(R) and R < 0.7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tcol", type=int, default=0); ap.add_argument("--icol", type=int, default=1)
    ap.add_argument("--jcol", type=int, default=2); ap.add_argument("--wcol", type=int, default=3)
    ap.add_argument("--sep", default=None)
    ap.add_argument("--nkeep", type=int, default=80); ap.add_argument("--ttarget", type=float, default=2000.0)
    ap.add_argument("--rhos", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--reps", type=int, default=4, help="realisations of the mask RNG, averaged")
    ap.add_argument("--n_em", type=int, default=3); ap.add_argument("--n_imp", type=int, default=6)
    ap.add_argument("--directed", action="store_true", help="observed initiator (positive control)")
    ap.add_argument("--undirected", action="store_true", help="latent initiator (pair-mark negative control)")
    ap.add_argument("--tag", default="real")
    a = ap.parse_args()
    rhos = [float(x) for x in a.rhos.split(",")]
    mark_models = ["init", "pair"] if a.directed else ["pair"]

    ev, w, N, T = load_events(a.path, a.tcol, a.icol, a.jcol, a.wcol, a.nkeep, a.ttarget, sep=a.sep)
    print(f"[load] {a.tag}: E={len(ev)} N={N} T={T:.0f}", flush=True)
    gate = node_memory_gate(ev, N, T)
    epn = len(ev) / max(N, 1)
    print(f"[gate] n_hat={gate['n_hat']:.3f} kappa_hat={gate['kappa_hat']:.2f} R_node={gate['R_node']:.3f} "
          f"({epn:.0f} events/node) -> "
          f"{'strong node memory' if gate['strong_memory'] else 'R_node inconclusive at this per-node count; see sigma(n) below'}",
          flush=True)

    out = {"tag": a.tag, "E": len(ev), "N": N, "gate": gate, "rhos": rhos, "curves": {}}
    for mm in mark_models:
        rows = []
        for rho in rhos:
            acc = {k: [] for k in ("sig", "sig_n", "cnk", "cncw", "ckcw")}
            for rep in range(a.reps):
                D = confound_3x3(ev, w, N, T, rho, np.random.default_rng(100 + rep),
                                 mark_model=mm, n_em=a.n_em, n_imp=a.n_imp, seed=rep)
                acc["sig"].append(float(D[:, 2].std()))
                acc["sig_n"].append(float(D[:, 0].std()))     # node-channel liveness (pinned => ~0)
                # correlations are meaningful only while the estimates actually spread
                if D[:, 2].std() > 1e-6 and D[:, 0].std() > 1e-6:
                    acc["cnk"].append(_corr(D, 0, 1)); acc["cncw"].append(_corr(D, 0, 2)); acc["ckcw"].append(_corr(D, 1, 2))
            m = {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in acc.items()}
            rows.append({"rho": rho, **m})
            print(f"[{mm}] rho={rho:.2f}  sigma(cW)={m['sig']:.3f}  sigma(n)={m['sig_n']:.3f}  "
                  f"corr(n,cW)={m['cncw']:+.2f}  corr(k,cW)={m['ckcw']:+.2f}", flush=True)
        out["curves"][mm] = rows
    outdir = os.path.join(ROOT, "results"); os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"mark_real_{a.tag}.json")
    json.dump(out, open(fn, "w"), indent=2)
    print(f"[done] wrote {fn}", flush=True)


if __name__ == "__main__":
    main()

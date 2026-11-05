#!/usr/bin/env python3
"""The 2x2 node/tie confound on real DIRECTED data, driven by the label-entropy order parameter.

This is the main-text confound (node self-excitation n_node vs tie reinforcement kappa) tested on
a real directed sequence. The initiator is observed; we mask a fraction (1-rho) of the true
initiators and impute the rest, and for each imputation refit (n_hat, kappa_hat). The reported
discriminators are the ACROSS-IMPUTATION SPREADS sigma(n_hat), sigma(kappa_hat) -- the
missing-information magnitude -- which should rise as rho->0 and vanish at rho=1. The correlation
corr(n,kappa) is reported ONLY where the spread is non-negligible (it is ill-conditioned once the
spread collapses, so it is not the primary readout).

Gate first: if R_node < 0.7 the data carries genuine, identifiable node self-excitation (a real
directed dataset with a live node channel -- itself a useful finding). If sigma(n_hat) stays flat
across rho, the direction is inferable when masked (low label entropy) and the confound does not
arise -- an honest negative, not a failure.

Usage:
  python confound_2x2.py data/stackoverflow_events.txt --rhos 0.0,0.25,0.5,0.75,1.0 --reps 4 --tag so
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

from concurrent.futures import ProcessPoolExecutor      # noqa: E402
from mark_confound import confound_3x3, _corr            # noqa: E402
from mark_confound_realdata import node_memory_gate       # noqa: E402


def load_events(path):
    ev = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 3:
                continue
            ev.append((float(p[0]), int(p[1]), int(p[2])))
    ev.sort(key=lambda r: r[0])
    N = 1 + max(max(i, j) for _t, i, j in ev)
    T = ev[-1][0] * 1.01
    return ev, N, float(T)


# ---- parallel worker (one (rho, rep) task; events loaded once per worker) -----------------
_G = {}


def _init_worker(path, n_em, n_imp):
    _G["ev"], _G["N"], _G["T"] = load_events(path)
    _G["w"] = np.ones(len(_G["ev"])); _G["n_em"] = n_em; _G["n_imp"] = n_imp


def _work(task):
    rho, rep = task
    D = confound_3x3(_G["ev"], _G["w"], _G["N"], _G["T"], rho, np.random.default_rng(100 + rep),
                     mark_model="pair", n_em=_G["n_em"], n_imp=_G["n_imp"], seed=rep)
    cnk = _corr(D, 0, 1) if (D[:, 0].std() > 1e-6 and D[:, 1].std() > 1e-6) else np.nan
    return (rho, float(D[:, 0].std()), float(D[:, 1].std()), float(D[:, 0].mean()),
            float(D[:, 1].mean()), cnk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--rhos", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--n_em", type=int, default=3); ap.add_argument("--n_imp", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers over (rho,rep) tasks")
    ap.add_argument("--tag", default="real2x2")
    a = ap.parse_args()
    rhos = [float(x) for x in a.rhos.split(",")]

    ev, N, T = load_events(a.path)
    Etot = len(ev)
    print(f"[load] {a.tag}: E={Etot} N={N} events/node={Etot/N:.0f}", flush=True)
    gate = node_memory_gate(ev, N, T)
    print(f"[gate] n_hat={gate['n_hat']:.3f} kappa_hat={gate['kappa_hat']:.2f} R_node={gate['R_node']:.3f} -> "
          f"{'GENUINE node memory' if gate['strong_memory'] else 'R_node inconclusive (see sigma(n) vs rho)'}",
          flush=True)
    del ev                                                 # workers reload once each; free the driver copy

    tasks = [(rho, rep) for rho in rhos for rep in range(a.reps)]
    acc = {rho: {k: [] for k in ("sn", "sk", "cnk", "nhat", "khat")} for rho in rhos}
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init_worker,
                             initargs=(a.path, a.n_em, a.n_imp)) as ex:
        for rho, sn, sk, nhat, khat, cnk in ex.map(_work, tasks):
            acc[rho]["sn"].append(sn); acc[rho]["sk"].append(sk)
            acc[rho]["nhat"].append(nhat); acc[rho]["khat"].append(khat)
            if np.isfinite(cnk):
                acc[rho]["cnk"].append(cnk)
    rows = []
    for rho in rhos:
        m = {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in acc[rho].items()}
        rows.append({"rho": rho, **m})
        print(f"[2x2] rho={rho:.2f}  sigma(n)={m['sn']:.4f}  sigma(k)={m['sk']:.3f}  "
              f"corr(n,k)={m['cnk']:+.2f}  |  nhat={m['nhat']:.3f} khat={m['khat']:.2f}", flush=True)

    out = {"tag": a.tag, "E": Etot, "N": N, "gate": gate, "rows": rows}
    outdir = os.path.join(ROOT, "results"); os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, f"confound2x2_{a.tag}.json"), "w"), indent=2)
    print(f"[done] wrote results/confound2x2_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()

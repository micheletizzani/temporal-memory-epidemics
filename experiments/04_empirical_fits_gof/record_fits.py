# %%
"""One fit pipeline for every record: n_hat, kappa_hat and the node-level memory position g.

Events are the same as real_record_screen.RECORDS (the event set the shuffle screen uses; proximity
records: day 0 with sessionisation at 60 s and the activity filter of five contacts; directed records:
the first events of the record, capped as in RECORDS). Every record is fitted with the same estimator:
observed-initiator maximum likelihood with per-node baseline activities profiled out
(inference.fit_observed_initiator, fit_b=True). Writes results/record_fits.json:
  n_node, gamma_n, kappa, gamma_e          fitted parameters
  g, g_lo, g_hi                            memory-to-baseline ratio gamma_n / a_i over nodes with a_i > 0
                                           (median and inter-quartile range)
  at_bounds                                parameters at the admissible bounds (flags an unreliable fit)
  events, nodes                            size of the event set
The day-by-day fits of the SocioPatterns records are in results/record_perday.json (record_perday.py).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os
import sys
import time

import numpy as np

import real_record_screen as rs  # noqa: E402
import inference as inf  # noqa: E402

OUT = os.path.join(ROOT, "results", "record_fits.json")


PROXIMITY = {"Hospital ward", "High school", "Conference (SFHH)", "Baboons"}


def contacts_day0(name):
    """Undirected contacts (sessionised, activity-filtered) -- the input of the latent-initiator fit."""
    if name == "Baboons":
        return rs.baboons_day0()
    path = {"Hospital ward": "hospital_lyon_contacts.dat",
            "High school": "HighSchool2013_proximity_net.csv",
            "Conference (SFHH)": "SFHH_tij.dat"}[name]
    return rs.proximity_day0(os.path.join(rs.ROOT, "data", path))


def fit_record(name, ev):
    if name in PROXIMITY:
        # no initiator is recorded: MC-EM latent-initiator fit on the contacts (as in the paper's Methods)
        c = contacts_day0(name)
        t0 = c[0][0]; c = [(t - t0, i, j) for t, i, j in c]
        N = 1 + max(max(i, j) for _t, i, j in c); T = c[-1][0] + 1.0
        f = inf.fit_latent_initiator(c, N, T, n_imputations=6, n_em=8, fit_b=True, seed=0)
        ev = c
    else:
        t0 = ev[0][0]
        w = [(t - t0, i, j) for t, i, j in sorted(ev)]
        N = 1 + max(max(i, j) for _t, i, j in w)
        T = w[-1][0] + 1.0
        f = inf.fit_observed_initiator(w, N, T, fit_b=True)
        ev = w
    a = np.asarray(f.activities, float); a = a[a > 0]
    g = f.gamma_n / a
    return dict(n_node=float(f.n_node), gamma_n=float(f.gamma_n), kappa=float(f.kappa),
                gamma_e=float(f.gamma_e), g=float(np.median(g)), g_lo=float(np.percentile(g, 25)),
                g_hi=float(np.percentile(g, 75)), at_bounds=list(f.at_bounds),
                estimator="latent" if name in PROXIMITY else "observed",
                events=len(ev), nodes=int(N))


if __name__ == "__main__":
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name, (load, _r) in rs.RECORDS.items():
        if name in out:
            print("cached", name, flush=True); continue
        t = time.time()
        out[name] = fit_record(name, load())
        print(f"{name:26s} n={out[name]['n_node']:.3f} kappa={out[name]['kappa']:.4g} "
              f"g={out[name]['g']:.3g} [{out[name]['g_lo']:.3g},{out[name]['g_hi']:.3g}] "
              f"bounds={out[name]['at_bounds']} ({time.time() - t:.0f}s)", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)

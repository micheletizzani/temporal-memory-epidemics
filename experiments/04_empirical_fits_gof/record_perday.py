# %%
"""Per-day latent-initiator fits for the three SocioPatterns records, with the same estimator and
settings as record_fits.py (day 0 there is the first entry). Replaces the unsourced
results/perday_summary.json, whose producer is not in the repository.
Writes results/record_perday.json: {name: [{day, n_node, kappa, gamma_n, gamma_e, events, at_bounds}]}.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import os
import sys

import real_record_screen as rs  # noqa: E402
import inference as inf  # noqa: E402
from sociopatterns_loader import load_sociopatterns, split_by_window, _sessionize  # noqa: E402,F401

OUT = os.path.join(ROOT, "results", "record_perday.json")
FILES = {"Hospital ward": "hospital_lyon_contacts.dat",
         "High school": "HighSchool2013_proximity_net.csv",
         "Conference (SFHH)": "SFHH_tij.dat"}

# %%
if __name__ == "__main__":
    out = {}
    for name, fn in FILES.items():
        d = load_sociopatterns(os.path.join(rs.ROOT, "data", fn), sessionize_gap=60.0)
        days = split_by_window(d, 86400.0, min_contacts=5)
        out[name] = []
        for k, blk in enumerate(days):
            c = blk.contacts
            if len(c) < 50:
                continue
            t0 = c[0][0]; c = [(t - t0, i, j) for t, i, j in c]
            N = 1 + max(max(i, j) for _t, i, j in c); T = c[-1][0] + 1.0
            f = inf.fit_latent_initiator(c, N, T, n_imputations=6, n_em=8, fit_b=True, seed=0)
            out[name].append(dict(day=k, n_node=float(f.n_node), kappa=float(f.kappa),
                                  gamma_n=float(f.gamma_n), gamma_e=float(f.gamma_e),
                                  events=len(c), at_bounds=list(f.at_bounds)))
            print(f"{name} day {k}: n={f.n_node:.3f} kappa={f.kappa:.4g} events={len(c)} bounds={f.at_bounds}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)

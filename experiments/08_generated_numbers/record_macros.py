# %%
"""Macros of the proximity fits (school, conference, hospital) from the unified pipeline.

Day-0 fits with standard errors and the across-imputation correlation (same estimator and settings
as record_fits.py), per-day spread over the days with at least 1000 events (record_perday.json), and the
half-day value. Writes results/macros_fits.json; the values are spliced into generated/macros.tex by
update_macros.py.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import math
import os
import sys

import numpy as np

import real_record_screen as rs  # noqa: E402
import inference as inf  # noqa: E402
from sociopatterns_loader import load_sociopatterns, split_by_window  # noqa: E402

OUT = os.path.join(ROOT, "results", "macros_fits.json")
FILES = {"school": "HighSchool2013_proximity_net.csv", "conf": "SFHH_tij.dat",
         "hosp": "hospital_lyon_contacts.dat"}


def day0(fn):
    c = rs.proximity_day0(os.path.join(rs.ROOT, "data", fn))
    t0 = c[0][0]; c = [(t - t0, i, j) for t, i, j in c]
    N = 1 + max(max(i, j) for _t, i, j in c); T = c[-1][0] + 1.0
    f = inf.fit_latent_initiator(c, N, T, n_imputations=6, n_em=8, fit_b=True, seed=0)
    return f, len(c)


def se_of(f, key):
    try:
        v = f.se[key]
        return float(v) if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


if __name__ == "__main__":
    out = {}
    for key, fn in FILES.items():
        f, nc = day0(fn)
        out[key] = dict(n=float(f.n_node), n_se=se_of(f, "n_node"), kappa=float(f.kappa),
                        kappa_se=se_of(f, "kappa"), mem_n_s=1 / float(f.gamma_n), mem_e_s=1 / float(f.gamma_e),
                        corr=float(f.corr_n_kappa), contacts=nc, at_bounds=list(f.at_bounds))
        print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[key].items()}, flush=True)
    perday = json.load(open(os.path.join(ROOT, "results", "record_perday.json")))
    for name, key in (("High school", "hs"), ("Conference (SFHH)", "sf"), ("Hospital ward", "ho")):
        days = [d for d in perday[name] if d["events"] >= 1000]
        n = np.array([d["n_node"] for d in days]); k = np.array([d["kappa"] for d in days])
        out[f"perday_{key}"] = dict(n_mean=float(n.mean()), n_sd=float(n.std(ddof=1)) if len(n) > 1 else 0.0,
                                    k_mean=float(k.mean()), k_sd=float(k.std(ddof=1)) if len(k) > 1 else 0.0,
                                    days=len(days), n_days_total=len(perday[name]))
        print(key, out[f"perday_{key}"], flush=True)
    hs4 = [d for d in perday["High school"] if d["day"] == 4][0]
    out["school_day4"] = dict(n=hs4["n_node"], events=hs4["events"])
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)

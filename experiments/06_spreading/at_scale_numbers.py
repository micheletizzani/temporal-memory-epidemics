"""Numbers quoted for the at-scale contagion experiments (SM Sec. S13, main text 'At scale') from results/poles_*.json
(written by epidemic_poles.jl).  Usage: at_scale_numbers.py [results_dir]  ->  prints and writes at_scale_numbers.json."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, sys
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
def load(model, N, kind):
    return json.load(open(os.path.join(D, f"poles_{model}_N{N}_{kind}.json")))
out = {}
b30 = None
for N in (90, 1000, 10000):
    for kind in ("node", "link", "poisson"):
        d = load("sir", N, kind); b = np.array(d["betas"]); i = int(np.argmin(abs(b - 0.30))); out[f"sir_N{N}_{kind}_b0.30"] = float(d["final_size"][i])
for kind in ("node", "link", "poisson"):
    pk = {}
    for N in (500, 1000, 10000):
        d = load("sis", N, kind); b = np.array(d["betas"]); q = np.array(d["qs_chi"], float); i = int(np.nanargmax(q))
        pk[N] = (float(b[i]), float(q[i])); out[f"sis_qs_peak_beta_N{N}_{kind}"] = pk[N][0]; out[f"sis_qs_peak_N{N}_{kind}"] = pk[N][1]
        m = np.array(d["mean"], float); out[f"sis_onset_beta_N{N}_{kind}"] = float(b[np.argmax(m > 0.05)]) if (m > 0.05).any() else None
        c = np.array(d["chi"], float); out[f"sis_chi_over_N_max_N{N}_{kind}"] = float(np.nanmax(c) / N)
    Ns = np.array(list(pk)); vals = np.array([pk[n][1] for n in Ns]); out[f"sis_qs_alpha_{kind}"] = float(np.polyfit(np.log(Ns), np.log(vals), 1)[0])
for k, v in out.items(): print(k, None if v is None else round(v, 3))
json.dump(out, open(os.path.join(ROOT, "results", "at_scale_numbers.json" if D == os.path.join(ROOT, "results") else "at_scale_numbers_saved.json"), "w"), indent=1)

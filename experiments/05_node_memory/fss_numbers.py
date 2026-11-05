"""Numbers of the finite-size scaling of R(M) quoted in the text, from results/phase2_T*.npz.
Writes results/fss_numbers.json: per size E, R at M = 0.1 and 0.6 (drop), width of the decline (R: 0.9 -> 0.7), slope."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import glob, json, os
import numpy as np
from phase_width import width_at
RES = os.path.join(ROOT, "results")
files = sorted(glob.glob(os.path.join(RES, "phase2_T*.npz")), key=lambda p: float(np.load(p)["Emean"]))
rows, rng = [], np.random.default_rng(0)
for f in files:
    d = np.load(f); M, R, S = d["M"], d["R"], np.nan_to_num(d["Rsem"])
    ok = np.isfinite(M) & np.isfinite(R) & (M >= 0); o = np.argsort(M[ok]); m, r = M[ok][o], R[ok][o]
    boot = [width_at(M, R + rng.normal(0, S)) for _ in range(400)]
    rows.append(dict(T=float(d["T"]), E=float(d["Emean"]), R01=float(np.interp(0.1, m, r)), R06=float(np.interp(0.6, m, r)),
                     R08=float(np.interp(0.8, m, r)), R028=float(np.interp(0.28, m, r)), width=width_at(M, R), width_err=float(np.nanstd(boot)),
                     Rmin=float(np.nanmin(r)), Mmax=float(m.max())))
    rows[-1]["drop"] = rows[-1]["R01"] - rows[-1]["R06"]
E = np.array([r["E"] for r in rows]); W = np.array([r["width"] for r in rows])
a, b = np.polyfit(np.log10(E), np.log10(W), 1)
# bootstrap the slope from the per-size width errors
sl = [np.polyfit(np.log10(E), np.log10(np.clip(W + rng.normal(0, [r["width_err"] for r in rows]), 1e-3, None)), 1)[0] for _ in range(1000)]
out = dict(sizes=rows, slope=float(a), slope_ci=[float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))])
json.dump(out, open(os.path.join(RES, "fss_numbers.json"), "w"), indent=1)
for r in rows: print({k: round(v, 3) for k, v in r.items()})
print("slope", round(a, 3), "95% CI", [round(x, 3) for x in out["slope_ci"]])

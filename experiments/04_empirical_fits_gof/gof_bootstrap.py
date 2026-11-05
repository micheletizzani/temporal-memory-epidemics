"""Parametric-bootstrap calibration of the goodness-of-fit KS statistics (SM Table S2).
Usage: gof_bootstrap.py <record|hs-window> [n_bootstrap]. Fit and imputation as record_gof.py; the bootstrap is
inference.goodness_of_fit(..., n_bootstrap): simulate from the fit, refit (warm start), recompute the KS statistics.
Writes results/gof_boot_<slug>.json."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, re, sys
import record_gof as rg  # noqa: E402
import record_fits as rf  # noqa: E402
import real_record_screen as rs  # noqa: E402
import inference as inf  # noqa: E402
from sociopatterns_loader import load_sociopatterns, split_by_window  # noqa: E402

name = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 100
if name == "hs-window":
    d = load_sociopatterns(os.path.join(rs.ROOT, "data", "HighSchool2013_proximity_net.csv"), sessionize_gap=60.0)
    blk = split_by_window(d, 20160.0, min_contacts=5)[0]; sets, N, T, f = rg.latent(blk.contacts, imp=1)
else:
    sets, N, T, f = rg.fit_events(name)
g = inf.goodness_of_fit(sets[0], N, T, f, n_bootstrap=B, seed=0, verbose=True)
out = {k: float(v) for k, v in g.items()}
json.dump(out, open(os.path.join(ROOT, "results", "gof_boot_" + re.sub(r"\W+", "_", name).strip("_") + ".json"), "w"), indent=1)
print(out)

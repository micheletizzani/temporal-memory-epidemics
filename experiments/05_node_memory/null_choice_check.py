"""The choice of null (main text, Sec. II G): for a memoryless heavy-tailed renewal train, the inter-event-order shuffle leaves
n_hat unchanged (R ~ 1), whereas reassigning the events to uniformly random times (which destroys the node's rate structure, not
its memory) collapses it (R ~ 0). Writes results/null_choice_check.json."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, sys
import numpy as np
from plane_validation import gen_train, fit_nnode_single, shuffle_isi  # noqa: E402
rows = []
for rep in range(60):
    rng = np.random.default_rng(100 + rep); T = 3000.0
    ts = gen_train(T, 0.5, 0.0, 0.5, 0.0, rng)          # AR(1) with rho = 0: lognormal renewal, B ~ 0.4, M ~ 0
    n = fit_nnode_single(ts, T)
    if not np.isfinite(n) or n < 0.05:
        continue
    nu = fit_nnode_single(np.sort(rng.uniform(0, T, len(ts))), T); ni = fit_nnode_single(shuffle_isi(ts, rng), T)
    rows.append(dict(n=n, R_uniform=nu / n, R_isi=ni / n))
ru = np.array([r["R_uniform"] for r in rows]); ri = np.array([r["R_isi"] for r in rows])
out = dict(reps=len(rows), n_mean=float(np.mean([r["n"] for r in rows])), R_uniform_median=float(np.median(ru)),
           R_uniform_q05_q95=[float(np.percentile(ru, 5)), float(np.percentile(ru, 95))], R_isi_median=float(np.median(ri)),
           R_isi_q05_q95=[float(np.percentile(ri, 5)), float(np.percentile(ri, 95))])
print(out); json.dump(out, open(os.path.join(ROOT, "results", "null_choice_check.json"), "w"), indent=1)

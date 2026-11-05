"""Refractory (asynchronous-irregular) spiking network: B, rank M, fit and goodness-of-fit diagnostics (first 60 000 spikes,
observed initiator = presynaptic neuron, fit_b=True as record_fits.py). Writes results/snn_refractory_gof.json."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, sys
import real_record_screen as rs, record_gof as rg  # noqa: E402
import inference as inf  # noqa: E402
from memory_stats import rank_memory_events  # noqa: E402
ev = rs.directed(os.path.join(rs.DATA, "snn_refractory_network.txt"), 0, 1, 2, 60000); t0 = ev[0][0]
w = [(t - t0, i, j) for t, i, j in sorted(ev)]; N = 1 + max(max(i, j) for _, i, j in w); T = w[-1][0] + 1.0
f = inf.fit_observed_initiator(w, N, T, fit_b=True); d, _ = rg.diagnostics(w, N, T, f)
res = dict(n_node=f.n_node, M_rank=rank_memory_events(w), B=rs.burstiness_memory(w)[0], **d)
print({k: round(v, 3) for k, v in res.items()}); json.dump(res, open(os.path.join(ROOT, "results", "snn_refractory_gof.json"), "w"), indent=1)

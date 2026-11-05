"""Bias of the latent-initiator (sequential-imputation) E-step, measured on directed data where the initiator is known
(SM Table S4). The same first E0 events are fitted with the exact observed-initiator likelihood and, after discarding the
initiator, with the latent-initiator MC-EM (n_imputations=6, n_em=8, as record_fits.py). Writes results/estep_bias_directed.json."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, sys
import real_record_screen as rs  # noqa: E402
import inference as inf  # noqa: E402
E0 = 5000
out = {}
for name in ("SNN (bursting)", "Email (EU institution)"):
    ev = sorted(rs.RECORDS[name][0]())[:E0]; t0 = ev[0][0]; w = [(t - t0, i, j) for t, i, j in ev]
    N = 1 + max(max(i, j) for _t, i, j in w); T = w[-1][0] + 1.0
    obs = inf.fit_observed_initiator(w, N, T, fit_b=True)
    con = [(t, min(i, j), max(i, j)) for t, i, j in w]
    lat = inf.fit_latent_initiator(con, N, T, n_imputations=6, n_em=8, fit_b=True, seed=0)
    out[name] = dict(events=E0, nodes=int(N), n_obs=obs.n_node, n_lat=lat.n_node, kappa_obs=obs.kappa, kappa_lat=lat.kappa,
                     bias=lat.n_node / obs.n_node - 1, obs_bounds=list(obs.at_bounds), lat_bounds=list(lat.at_bounds))
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "estep_bias_directed.json"), "w"), indent=1)

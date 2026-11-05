"""Goodness of fit of every record from the SAME fit as record_fits.py (Fig. 3d, Fig. 4, Fig. 6, S2).

For each record: refit exactly as record_fits.fit_record (latent-initiator MC-EM for the proximity records, observed
initiator otherwise), average the residual diagnostics over N_IMP label imputations (proximity) and report
  KS_t      Kolmogorov-Smirnov statistic of the time-rescaled timing residuals against Exp(1)
  atom      fraction of residuals exactly zero (simultaneous events of one initiator)
  tied      fraction of events sharing time and initiator with an earlier event
  mean_tau, var_tau
  acf_z     within-node lag-one serial correlation of the residuals, as a z-score
  KS_detied, acf_z_detied   the same after removing the zero residuals
  KS_sel    KS statistic of the selection probability integral transform
and caches tau of the first imputation to results/tau_<key>.npy (read by qq_combine.py and
identity_from_tau.py).  `--hs-window` runs the high-school 20 160 s window of the main text (10 imputations).
Writes results/record_gof.json.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, sys, time
import numpy as np
import real_record_screen as rs  # noqa: E402
import record_fits as rf  # noqa: E402
import inference as inf  # noqa: E402
from sociopatterns_loader import load_sociopatterns, split_by_window  # noqa: E402

RESULTS = os.path.join(rs.ROOT, "results")
OUT = os.path.join(ROOT, "results", "record_gof.json")
KEY = {"Hospital ward": "hospital", "High school": "school", "Conference (SFHH)": "sfhh", "Baboons": "baboons",
       "Email (EU institution)": "email", "CollegeMsg (Direct Msgs)": "collegemsg", "Stack Overflow": "stackoverflow",
       "Wikibooks": "wikibooks", "Wikibooks bots": "wikibooks_bots", "Microservices": "microservices",
       "SNN (bursting)": "snn_bursty", "Cortex": "cortex_spikes"}
N_IMP = 5


def diagnostics(events, N, T, fit, seed=0):
    tau = inf.rescaled_times(events, N, T, fit.n_node, fit.gamma_n, fit.activities)
    zero = tau <= 1e-9
    seen, tied = set(), 0
    for t, i, _j in events:
        tied += (t, i) in seen; seen.add((t, i))
    seqs, seqs_d = {}, {}
    for (t, i, _j), x in zip(events, tau):
        seqs.setdefault(i, []).append(x)
        if x > 1e-9:
            seqs_d.setdefault(i, []).append(x)

    def z(sq):
        a, b = [], []
        for s in sq.values():
            if len(s) >= 3:
                a += s[:-1]; b += s[1:]
        if len(a) < 30 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1] * np.sqrt(len(a)))
    u_sel = inf.selection_pit(events, N, fit.kappa, fit.gamma_e, fit.attractiveness, rng=np.random.default_rng(seed))
    return dict(KS_t=inf._ks_uniform(1 - np.exp(-tau)), atom=float(zero.mean()), tied=tied / len(events),
                mean_tau=float(tau.mean()), var_tau=float(tau.var()), acf_z=z(seqs),
                KS_detied=inf._ks_uniform(1 - np.exp(-tau[~zero])), acf_z_detied=z(seqs_d),
                KS_sel=inf._ks_uniform(u_sel)), tau


def fit_events(name):
    """Fit as record_fits and return (list of event-sets (one per imputation), N, T, fit, latent?)."""
    if name in rf.PROXIMITY:
        c = rf.contacts_day0(name); t0 = c[0][0]; c = [(t - t0, i, j) for t, i, j in c]
        return latent(c)
    if name == "Cortex":                    # 30 random well-isolated units, all spikes of the spontaneous block (seed 1)
        import numpy as _np
        z = _np.load(os.path.join(rs.ROOT, "data", "allen_spontaneous_spikes.npz"))
        pick = _np.random.default_rng(1).choice(int(z["unit"].max()) + 1, 30, replace=False)
        rem = {int(u): k for k, u in enumerate(pick)}; m = _np.isin(z["unit"], pick)
        ev = [(float(t), rem[int(u)], (rem[int(u)] + 1) % 30) for t, u in zip(z["times"][m], z["unit"][m])]
    else:
        ev = rs.RECORDS[name][0]()
    t0 = ev[0][0]
    w = [(t - t0, i, j) for t, i, j in sorted(ev)]
    N = 1 + max(max(i, j) for _t, i, j in w); T = w[-1][0] + 1.0
    f = inf.fit_observed_initiator(w, N, T, fit_b=True)
    return [w], N, T, f


def latent(c, imp=N_IMP, n_imputations=6, n_em=8):
    N = 1 + max(max(i, j) for _t, i, j in c); T = c[-1][0] + 1.0
    f = inf.fit_latent_initiator(c, N, T, n_imputations=n_imputations, n_em=n_em, fit_b=True, seed=0)
    ordered = [(t, i, j) if i < j else (t, j, i) for (t, i, j) in c]
    theta = [f.n_node, f.gamma_n, f.kappa, f.gamma_e]; rng = np.random.default_rng(7); sets = []
    for _ in range(imp):
        side = inf._impute_initiators(ordered, N, T, theta, rng)
        sets.append([(t, (i if s == 0 else j), (j if s == 0 else i)) for (t, i, j), s in zip(ordered, side)])
    return sets, N, T, f


def summarise(rows):
    return {k: [float(np.nanmean([r[k] for r in rows])), float(np.nanstd([r[k] for r in rows]))] for k in rows[0]}


if __name__ == "__main__":
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if "--hs-window" in sys.argv:
        d = load_sociopatterns(os.path.join(rs.ROOT, "data", "HighSchool2013_proximity_net.csv"), sessionize_gap=60.0)
        blk = split_by_window(d, 20160.0, min_contacts=5)[0]
        sets, N, T, f = latent(blk.contacts, imp=10)
        rows = [diagnostics(s, N, T, f, k)[0] for k, s in enumerate(sets)]
        out["High school window"] = dict(contacts=len(blk.contacts), N=int(N), hours=T / 3600, n_node=f.n_node,
                                         kappa=f.kappa, gamma_n=f.gamma_n, gamma_e=f.gamma_e, at_bounds=list(f.at_bounds),
                                         **summarise(rows))
        print("High school window", out["High school window"], flush=True)
        json.dump(out, open(OUT, "w"), indent=1); sys.exit(0)
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(KEY)
    for name in names:
        t0 = time.time(); sets, N, T, f = fit_events(name)
        res = [diagnostics(s, N, T, f, k) for k, s in enumerate(sets)]
        np.save(os.path.join(RESULTS, f"tau_{KEY[name]}.npy"), res[0][1])
        out[name] = dict(events=len(sets[0]), N=int(N), n_node=f.n_node, kappa=f.kappa, at_bounds=list(f.at_bounds),
                         **summarise([r[0] for r in res]))
        o = out[name]
        print(f"{name:26s} E={o['events']:6d} n={f.n_node:.3f} KS={o['KS_t'][0]:.3f} atom={o['atom'][0]:.3f} tied={o['tied'][0]:.3f} "
              f"acf_z={o['acf_z'][0]:+.1f} KS_detied={o['KS_detied'][0]:.3f} acf_z_det={o['acf_z_detied'][0]:+.1f} "
              f"KSsel={o['KS_sel'][0]:.3f} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)

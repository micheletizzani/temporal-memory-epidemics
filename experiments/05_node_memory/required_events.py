# %%
"""How many events are needed to tell scale-free self-excitation from renewal?

Backs the paper's claim that the limit is a DIVERGENT DATA REQUIREMENT rather than an
impossibility. If that extrapolation does not in fact exceed our dataset sizes, the claim must
be softened -- this script is written to be able to falsify it.

THE TEST
--------
Statistic: Delta = n_hat(series) - n_hat(order-shuffled series). The shuffle preserves the
inter-event marginal exactly and destroys temporal order, so Delta isolates the order-carried
component of the fitted branching ratio.

Matched null. For every replicate we build BOTH hypotheses from the same marginal:

    ts       ~ power-law (Omori) Hawkes, tail exponent theta        [alternative]
    ts_sh    = shuffle(ts)      -- a renewal draw with the IDENTICAL marginal   [null]

    Delta_alt  = n_hat(ts)    - n_hat(ts_sh)      effect under genuine memory
    Delta_null = n_hat(ts_sh) - n_hat(ts_sh2)     effect under no order, same marginal

Using ts_sh as the null series is what makes this a matched comparison: the null and the
alternative share an inter-event distribution by construction, so any separation is due to
temporal order alone and not to a marginal difference.

SAMPLE SIZE
-----------
Delta_null has mean 0 and sampling spread sigma(E) ~ c E^{-1/2}. For a two-sided test at
alpha=0.05 with power 0.8 we need |delta| / sigma(N) >= z_{a/2} + z_b = 2.80, hence

    N_req = ( 2.80 * c / delta )^2 ,      c = sigma(E) * sqrt(E)

We measure sigma at several E and FIT the exponent instead of assuming it. If the fitted slope
is not close to -1/2 the extrapolation is not trustworthy and the script says so rather than
reporting a number.

Run cell-by-cell ('# %%') or `python required_events.py`.
Writes results/required_events.json
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import math
import os
import sys

import numpy as np

import kernel_phase as kp                                        # noqa
from kernel_families import powerlaw_mixture                     # noqa
from order_information_fraction import n_exp, _prep, shuffle_isi  # noqa  reuse audited fitter

RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

Z_POWER = 1.959964 + 0.841621          # two-sided alpha=0.05, power=0.80  -> 2.8016
N_TRUE, MU = 0.5, 1.0
THETAS = [0.2, 0.5, 1.0]               # Omori tail exponents (small = heavy)
E_GRID = [400, 800, 1600, 3200]        # event counts at which sigma is measured
REPS = 24
# stationary rate is ~MU/(1-N_TRUE) = 2 events/unit, so this comfortably exceeds max(E_GRID)
# without simulating tens of thousands of events we would only discard.
T_SIM = 2500.0                         # horizon; series are truncated to exactly E events

# event counts of the records analysed in the paper (for the comparison at the end)
DATASET_E = {"High school (sessionised)": 2_200, "Hospital ward": 3_200,
             "Conference SFHH": 3_500, "Email (EU core)": 25_000,
             "Stack Overflow (core)": 110_000, "Wikibooks (core)": 410_000,
             "Alibaba microservices": 200_000}


# %%
def _nhat(ts):
    """n_hat for a single-node event-time series, horizon = last event time."""
    import order_information_fraction as oif
    ts = np.sort(np.asarray(ts, float))
    T_old = oif.T
    try:
        oif.T = float(ts[-1])
        return n_exp(*oif._prep(ts))
    finally:
        oif.T = T_old


# %%
def deltas_at(theta, E, reps, seed0=0):
    """Return (delta_alt samples, delta_null samples) at tail theta and event count E."""
    ws, gm = powerlaw_mixture(theta)
    alt, null = [], []
    for s in range(reps):
        rng = np.random.default_rng(seed0 + 7919 * s + int(1000 * theta))
        ts = kp.hawkes_mix_sim(MU, N_TRUE, ws, gm, T_SIM, rng)
        if len(ts) < E:
            continue
        ts = np.sort(np.asarray(ts, float))[:E]           # truncate to exactly E events
        ts_sh = shuffle_isi(ts, rng)
        ts_sh2 = shuffle_isi(ts_sh, rng)
        n_r, n_s, n_s2 = _nhat(ts), _nhat(ts_sh), _nhat(ts_sh2)
        alt.append(n_r - n_s)                              # genuine order present
        null.append(n_s - n_s2)                            # same marginal, no order
    return np.array(alt), np.array(null)


# %%
def analyse(theta, reps=REPS):
    rows = []
    for E in E_GRID:
        a, nl = deltas_at(theta, E, reps)
        if len(a) < 5:
            print(f"  theta={theta} E={E}: too few replicates ({len(a)})")
            continue
        rows.append(dict(E=E, delta=float(np.mean(a)), delta_sd=float(np.std(a, ddof=1)),
                         sigma_null=float(np.std(nl, ddof=1)),
                         null_mean=float(np.mean(nl)), n=len(a)))
        print(f"  theta={theta:.2f} E={E:5d}  delta={rows[-1]['delta']:+.4f}  "
              f"sigma_null={rows[-1]['sigma_null']:.4f}  null_mean={rows[-1]['null_mean']:+.4f}")
    if len(rows) < 3:
        return dict(theta=theta, rows=rows, status="insufficient")

    # fit sigma_null ~ c * E^slope ; slope must be near -1/2 for the extrapolation to hold
    lE = np.log(np.array([r["E"] for r in rows]))
    lS = np.log(np.array([r["sigma_null"] for r in rows]))
    slope, inter = np.polyfit(lE, lS, 1)
    c_half = float(np.mean([r["sigma_null"] * math.sqrt(r["E"]) for r in rows]))

    # effect size: pooled over E (delta should be roughly E-independent); report its drift
    dels = np.array([r["delta"] for r in rows])
    delta = float(np.mean(dels))
    delta_se = float(np.std(dels, ddof=1) / math.sqrt(len(dels)))
    trust = abs(slope + 0.5) < 0.15

    def nreq(d):
        return float("inf") if d <= 0 else (Z_POWER * c_half / d) ** 2

    out = dict(theta=theta, rows=rows, slope=float(slope), c=c_half, delta=delta,
               delta_se=delta_se, slope_trustworthy=bool(trust),
               N_req=nreq(delta),
               N_req_lo=nreq(delta + 1.96 * delta_se),     # larger effect -> fewer events
               N_req_hi=nreq(max(delta - 1.96 * delta_se, 0.0)),
               status="ok" if trust else "slope_off")
    print(f"  -> slope={slope:+.3f} ({'OK' if trust else 'NOT -1/2: extrapolation unreliable'}), "
          f"delta={delta:+.4f}+/-{delta_se:.4f}, N_req={out['N_req']:.3g} "
          f"[{out['N_req_lo']:.3g}, {out['N_req_hi']:.3g}]")
    return out


# %%
if __name__ == "__main__":
    res = {"config": dict(n_true=N_TRUE, mu=MU, thetas=THETAS, E_grid=E_GRID, reps=REPS,
                          z_power=Z_POWER), "per_theta": []}
    for th in THETAS:
        res["per_theta"].append(analyse(th))

    print("\n=== required events vs the records analysed ===")
    finite = [r for r in res["per_theta"] if r.get("status") == "ok" and np.isfinite(r["N_req"])]
    if finite:
        worst = max(r["N_req"] for r in finite)
        best = min(r["N_req"] for r in finite)
        print(f"N_req across tested tails: {best:.3g} -- {worst:.3g} events")
        for name, E in sorted(DATASET_E.items(), key=lambda kv: kv[1]):
            verdict = "INSUFFICIENT" if E < best else ("sufficient for some tails"
                                                       if E < worst else "SUFFICIENT for all")
            print(f"  {name:28s} E={E:>8,d}  {verdict}")
        print("\nREAD: the paper's claim holds only if the dataset counts sit below N_req.")
        print("      If any record exceeds N_req, SOFTEN the abstract -- do not keep the sentence.")
    else:
        print("No trustworthy extrapolation obtained; the abstract sentence is NOT supported.")

    with open(os.path.join(RES, "required_events.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("\nwrote", os.path.join(RES, "required_events.json"))

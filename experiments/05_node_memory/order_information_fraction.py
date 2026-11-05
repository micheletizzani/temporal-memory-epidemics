# %%
"""Order-carried identifying signal vanishes as the self-excitation kernel tail gets heavier.

Turns the scaling argument in SM Sec. "Kernel families" into a numerical demonstration with
the correctly-specified estimator included.

For a node Hawkes process with branching ratio n and normalised kernel phi, the information
that identifies n splits into a part fixed by the one-dimensional inter-event (ISI) law --
which an order-shuffle of the ISIs preserves exactly -- and a part carried by the joint
ordering, which the shuffle destroys. To leading order the shuffle ratio

    R = n_hat(order-shuffled) / n_hat(real)

satisfies R ~ 1 - I_ord/I, so the ORDER-CARRIED FRACTION is Q = 1 - R. We sweep the Omori
tail exponent theta (phi(u) ~ u^{-(1+theta)}; SMALL theta = HEAVY tail) plus a true-EXPONENTIAL
anchor (order genuinely resolvable), and measure Q with:
  - the single-EXPONENTIAL estimator, and
  - the MATCHED SCALE-FREE estimator (sum-of-exp mixture) at a few theta, so the collapse
    cannot be blamed on kernel misspecification.

Expectation: Q > 0 for the exponential-kernel process (order resolvable); Q -> 0 as the Omori
tail heavies, for BOTH estimators -- i.e. the order-carried information vanishes intrinsically.

Run cell-by-cell ('# %%') or `python order_information_fraction.py`.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

import kernel_phase as kp                                     # Hawkes simulator            # noqa
from kernel_families import powerlaw_mixture                 # Omori (power-law) kernel     # noqa
from sum_exp_kernel import multi_replay, nll_multi, fit_mixture  # timing likelihood        # noqa

RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

GEST = np.geomspace(1e-3, 1e1, 10)     # estimation grid for the sum-of-exponentials kernel
N_TRUE, MU, T = 0.5, 1.0, 2500.0       # true branching ratio, baseline, horizon (matches b1)
THETAS = [0.2, 0.35, 0.5, 0.7, 1.0, 1.5]     # Omori tail: small = heavy tail
SEEDS = 12
THETAS_MIX = [0.2, 0.5, 1.0]           # matched scale-free estimator at 3 representative tails
SEEDS_MIX = 6
MINEV = 200


def _prep(ts):
    ev = [(float(t), 0, 1) for t in ts]
    S, C, counts = multi_replay(ev, [0] * len(ev), 2, GEST, T)
    return S, C, counts


def n_exp(S, C, counts):
    """Best single-rate exponential branching ratio."""
    ev0 = [0] * S.shape[0]; best = (-1e18, 0.0)
    for r in range(len(GEST)):
        w = np.zeros(len(GEST)); w[r] = 1.0
        o = minimize(lambda x: nll_multi(1 / (1 + math.exp(-x[0])), w, S, C, counts, ev0, 2, T)[0],
                     [0.0], method="Nelder-Mead", options=dict(maxiter=300, xatol=1e-3, fatol=1e-3))
        n = 1 / (1 + math.exp(-o.x[0])); ll = -o.fun
        if ll > best[0]:
            best = (ll, n)
    return best[1]


def n_mix(S, C, counts):
    n, _w, _ll = fit_mixture(S, C, counts, [0] * S.shape[0], 2, T, len(GEST))
    return n


def shuffle_isi(ts, rng):
    ts = np.sort(np.asarray(ts, float))
    d = np.diff(np.concatenate([[0.0], ts])); rng.shuffle(d)
    return np.cumsum(d)


def _summ(vals):
    return (float(np.median(vals)), float(np.percentile(vals, 5)),
            float(np.percentile(vals, 95)), len(vals)) if vals else (float("nan"),) * 3 + (0,)


# %%
def Q_for_kernel(ws, gm, estimator, seeds):
    """Order-carried fraction Q = 1 - n_hat(shuffled)/n_hat(real) over seeds."""
    Qs = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        ts = kp.hawkes_mix_sim(MU, N_TRUE, ws, gm, T, rng)
        if len(ts) < MINEV:
            continue
        n_r = estimator(*_prep(ts))
        n_s = estimator(*_prep(shuffle_isi(ts, rng)))
        if n_r > 1e-3:
            Qs.append(1.0 - n_s / n_r)
    m, lo, hi, k = _summ(Qs)
    return dict(Q=m, Q_lo=lo, Q_hi=hi, n=k)


# %%
if __name__ == "__main__":
    res = {"meta": dict(n_true=N_TRUE, mu=MU, T=T, gest=list(GEST)), "exp": {}, "mix": {}}

    # anchor: a true EXPONENTIAL kernel (order genuinely resolvable)
    res["exp"]["anchor_exp"] = Q_for_kernel([1.0], [1.0], n_exp, SEEDS)
    print(f"[exp] EXPONENTIAL anchor  Q={res['exp']['anchor_exp']['Q']:+.3f} "
          f"[{res['exp']['anchor_exp']['Q_lo']:+.3f},{res['exp']['anchor_exp']['Q_hi']:+.3f}]")

    for th in THETAS:                      # exponential estimator across the Omori family
        ws, gm = powerlaw_mixture(th)
        res["exp"][f"{th}"] = Q_for_kernel(ws, gm, n_exp, SEEDS)
        d = res["exp"][f"{th}"]
        print(f"[exp] theta={th:.2f}  Q={d['Q']:+.3f} [{d['Q_lo']:+.3f},{d['Q_hi']:+.3f}] (reps={d['n']})")

    for th in THETAS_MIX:                   # matched scale-free estimator (no misspecification)
        ws, gm = powerlaw_mixture(th)
        res["mix"][f"{th}"] = Q_for_kernel(ws, gm, n_mix, SEEDS_MIX)
        d = res["mix"][f"{th}"]
        print(f"[mix] theta={th:.2f}  Q={d['Q']:+.3f} [{d['Q_lo']:+.3f},{d['Q_hi']:+.3f}] (reps={d['n']})")

    with open(os.path.join(RES, "order_information_fraction.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", os.path.join(RES, "order_information_fraction.json"))

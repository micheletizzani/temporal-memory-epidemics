#!/usr/bin/env python3
"""Residual analysis: does the model actually describe the data?

Everything else in this experiment establishes that the estimator inverts its own
generative process. This script is the only one that asks whether the model class is an
adequate description of a real contact sequence, and its outcome determines whether the
empirical estimates carry mechanistic weight or are merely the parameters of a fitted
object.

Two residual checks, run on both blocks of the likelihood:

  KS on rescaled times    -- marginal distribution of the compensated inter-event
                             intervals against Exp(1)
  lag-1 serial correlation -- within-node independence of those intervals

Both are needed. Measured power of each, on synthetic data where the truth is known:

  misspecification                     KS            lag-1 acf
  --------------------------------     ----------    ---------
  none (exponential truth)             p = 0.85      z = -0.1     both pass, correctly
  power-law kernel, exponential fit    p = 0.24      z = -0.6     BOTH MISS IT
  circadian baseline, constant-a fit   p = 0.003     z = -5.1     both catch it

The middle row is the important one and is reported here rather than buried: neither
diagnostic detects a moderate power-law kernel fitted by an exponential at ~1800 events.
A pass therefore does NOT license the claim that the kernel shape is right. It licenses
the weaker claim that the model is not grossly misspecified in the ways these tests can
see.

Because proximity data has no initiator label, the residuals are computed on imputed
labellings and averaged over draws; this adds an approximation on top of the diagnostic
and is reported as a spread across imputations.

Usage
-----
    python scripts/run_goodness_of_fit.py --synthetic
    python scripts/run_goodness_of_fit.py --data data/HighSchool2013_proximity_net.csv --by-day
    python scripts/run_goodness_of_fit.py --data data/SFHH_tij.dat --bootstrap 20
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path

import argparse
import os
import sys
import time

import numpy as np


from sociopatterns_loader import (  # noqa: E402
    load_sociopatterns, restrict_to_active_nodes, split_by_day, split_by_window)
from generative_models import (  # noqa: E402
    MechanismParams, TemporalNetworkModel, simulate)
from inference import (  # noqa: E402
    _impute_initiators, fit_latent_initiator, fit_observed_initiator, goodness_of_fit)

DEFAULT_DATA = os.path.join(ROOT,
                            "data", "HighSchool2013_proximity_net.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true",
                   help="run the power/calibration checks on synthetic data and exit")
    p.add_argument("--data", type=str, default=DEFAULT_DATA)
    p.add_argument("--time-col", type=int, default=0)
    p.add_argument("--i-col", type=int, default=1)
    p.add_argument("--j-col", type=int, default=2)
    p.add_argument("--gap", type=float, default=60.0,
                   help="sessionisation gap (s). 60 suits the 20 s SocioPatterns grid. "
                        "Use 0 for email and other event-based data, where there is no "
                        "sampling grid to undo.")
    p.add_argument("--min-contacts", type=int, default=5)
    p.add_argument("--window", type=float, default=None,
                   help="window length in seconds; nodes are re-filtered WITHIN each "
                        "window. Use for email (weeks/months), where daily windows are "
                        "far too sparse: email-Eu gave 929 contacts against N=876.")
    p.add_argument("--by-day", action="store_true",
                   help="fit and test each day separately (strongly recommended: a "
                        "circadian baseline is the misspecification these tests are "
                        "most sensitive to)")
    p.add_argument("--imputations", type=int, default=5,
                   help="label imputations over which residuals are averaged")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="parametric-bootstrap replicates for calibrated p-values "
                        "(slow; the nominal p is anti-conservative without it)")
    p.add_argument("--em-sweeps", type=int, default=4)
    p.add_argument("--max-blocks", type=int, default=3,
                   help="how many windows to fit")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def synthetic_checks() -> None:
    """Calibration and power of the diagnostics, on data whose truth we control."""
    import math

    print("=" * 78)
    print("Diagnostic calibration and power (synthetic)")
    print("=" * 78)

    rng = np.random.default_rng(1)
    N, T = 40, 1200.0
    act = 0.02 + 0.03 * rng.random(N)
    print("\nA. correctly specified -- both checks must pass")
    ev = simulate(TemporalNetworkModel(N, act, MechanismParams(0.5, 0.5, 20.0, 0.05)),
                  T, rng=np.random.default_rng(9))
    goodness_of_fit(ev, N, T, fit_observed_initiator(ev, N, T))

    print("\nB. power-law node kernel, exponential fit -- KNOWN BLIND SPOT")
    N2, T2 = 30, 800.0
    act2 = np.full(N2, 0.04)
    ev2 = _sim_powerlaw(N2, act2, 0.5, 0.5, T2, np.random.default_rng(5))
    goodness_of_fit(ev2, N2, T2, fit_observed_initiator(ev2, N2, T2))
    print("  ^ this misspecification is real and both diagnostics miss it. Do not read")
    print("    a pass as evidence that the kernel shape is correct.")

    print("\nC. circadian baseline, constant-activity fit -- must be caught")
    N3, T3 = 30, 1200.0
    act3 = np.full(N3, 0.06)
    ev3 = _sim_circadian(N3, act3, 60.0, T3, np.random.default_rng(4))
    goodness_of_fit(ev3, N3, T3, fit_observed_initiator(ev3, N3, T3))
    print("\n" + "=" * 78)


def _sim_powerlaw(N, act, n_node, g, T, rng):
    import math
    hist = [[] for _ in range(N)]
    ev, t = [], 0.0

    def phi(tau):
        return n_node * g / (1.0 + g * tau) ** 2

    while t < T:
        lam = np.array([act[i] + sum(phi(t - tk) for tk in hist[i]) for i in range(N)])
        L = lam.sum()
        if L <= 0:
            break
        t += -math.log(rng.random()) / L
        if t >= T:
            break
        lam = np.array([act[i] + sum(phi(t - tk) for tk in hist[i]) for i in range(N)])
        if rng.random() * L >= lam.sum():
            continue
        i = int(rng.choice(N, p=lam / lam.sum()))
        j = int(rng.choice([x for x in range(N) if x != i]))
        hist[i].append(t)
        ev.append((t, i, j))
    return ev


def _sim_circadian(N, act, period, T, rng, amp=0.9):
    import math
    ev, t = [], 0.0
    A = act.sum() * (1 + amp)
    while t < T:
        t += -math.log(rng.random()) / A
        if t >= T:
            break
        if rng.random() * (1 + amp) >= 1.0 + amp * math.cos(2 * math.pi * t / period):
            continue
        i = int(rng.choice(N, p=act / act.sum()))
        j = int(rng.choice([x for x in range(N) if x != i]))
        ev.append((t, i, j))
    return ev


def test_block(label: str, N: int, contacts, T: float, args) -> None:
    print(f"\n{'-' * 78}\n{label}\n{'-' * 78}")
    t0 = time.time()
    fit = fit_latent_initiator(contacts, N, T, n_imputations=3,
                               n_em=args.em_sweeps, fit_b=True, seed=args.seed)
    print(f"  fit: n_node={fit.n_node:.3f}  gamma_n={fit.gamma_n:.3g}  "
          f"kappa={fit.kappa:.2f}  gamma_e={fit.gamma_e:.3g}  ({time.time() - t0:.0f}s)")
    if fit.at_bounds:
        print("  *** AT A BOUND: " + "; ".join(fit.at_bounds))
        print("  *** These are not estimates. The residual diagnostics below are still")
        print("  *** valid (the zero atom does not depend on them), but no mechanism")
        print("  *** number from this block may be reported.")

    ordered = [(t, i, j) if i < j else (t, j, i) for (t, i, j) in contacts]
    theta = [fit.n_node, fit.gamma_n, fit.kappa, fit.gamma_e]
    rng = np.random.default_rng(args.seed + 7)

    ks_t, ks_s, acf = [], [], []
    for m in range(args.imputations):
        side = _impute_initiators(ordered, N, T, theta, rng)
        events = [(t, (i if s == 0 else j), (j if s == 0 else i))
                  for (t, i, j), s in zip(ordered, side)]
        g = goodness_of_fit(events, N, T, fit,
                            n_bootstrap=(args.bootstrap if m == 0 else 0),
                            seed=args.seed + m, verbose=(m == 0))
        ks_t.append(g["ks_timing"])
        ks_s.append(g["ks_selection"])
        acf.append(g["acf1_z"])

    print(f"\n  across {args.imputations} label imputations:")
    print(f"    KS timing     {np.mean(ks_t):.4f} +/- {np.std(ks_t):.4f}")
    print(f"    KS selection  {np.mean(ks_s):.4f} +/- {np.std(ks_s):.4f}")
    print(f"    lag-1 acf z   {np.mean(acf):+.1f} +/- {np.std(acf):.1f}")

    verdict = []
    if np.mean(ks_t) > 0.05:
        verdict.append("timing marginal is off")
    if abs(np.mean(acf)) > 3.0:
        verdict.append("timing residuals serially correlated")
    if np.mean(ks_s) > 0.05:
        verdict.append("partner-choice model is off")
    print("  VERDICT: " + ("; ".join(verdict) if verdict
                           else "no detectable misspecification "
                                "(within the limited power documented above)"))


def main() -> None:
    args = parse_args()
    if args.synthetic:
        synthetic_checks()
        return

    if not os.path.exists(args.data):
        raise SystemExit(f"data file not found: {args.data}")
    print("=" * 78)
    print(f"Goodness of fit: {os.path.basename(args.data)}")
    print("=" * 78)

    data = load_sociopatterns(args.data, time_col=args.time_col, i_col=args.i_col, j_col=args.j_col,
                              sessionize_gap=(None if args.gap <= 0 else args.gap))
    data = restrict_to_active_nodes(data, min_contacts=args.min_contacts)
    print(data.describe())

    if args.window:
        blocks = split_by_window(data, args.window, min_contacts=args.min_contacts)
        print(f"\nsplit into {len(blocks)} windows of {args.window:.0f} s, "
              "nodes re-filtered within each")
        for d, blk in enumerate(blocks[:args.max_blocks]):
            test_block(f"window {d}: {len(blk.contacts)} contacts, N={blk.N}, "
                       f"{blk.T / 3600:.1f} h", blk.N, blk.contacts, blk.T, args)
    elif args.by_day:
        for d, day in enumerate(split_by_day(data)):
            if len(day.contacts) < 200:
                print(f"\nday {d}: {len(day.contacts)} contacts, skipped")
                continue
            test_block(f"day {d}: {len(day.contacts)} contacts over "
                       f"{day.T / 3600:.1f} h", day.N, day.contacts, day.T, args)
    else:
        test_block(f"whole record: {len(data.contacts)} contacts",
                   data.N, data.contacts, data.T, args)
        print("\nNOTE: fitting across overnight gaps introduces exactly the")
        print("non-stationary baseline these diagnostics are most sensitive to.")
        print("Compare against --by-day before concluding the model is wrong.")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()

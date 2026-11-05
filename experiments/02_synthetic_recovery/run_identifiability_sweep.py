#!/usr/bin/env python3
"""Identifiability phase diagram over (N, T, n_node, kappa).

This is the experiment the project is actually about. A point estimate of a mechanism
parameter says nothing about whether the mechanism is recoverable; what matters is the
sampling behaviour of the estimator as a function of how much data there is and where
in parameter space the truth sits. For every cell of the design this script runs
`--reps` independent realisations and records, per parameter:

    bias          mean(estimate) - truth
    rmse          root mean squared error
    sd            across-replicate standard deviation of the estimate
    mean_se       mean asymptotic standard error (should track `sd` if the asymptotics
                  have kicked in; a large mismatch is itself a finding)
    coverage      fraction of replicates whose +/-1.96 se interval contains the truth
    power         fraction of replicates rejecting the corresponding boundary null at 5%

`coverage` is the headline diagnostic. Cells where it collapses below the nominal 0.95
are the degenerate regime: the estimator is confidently wrong. `power` for the kappa=0
and n_node=0 nulls answers the practical question "with this much data, could I even
detect the mechanism if it were there".

Output is a tidy CSV plus, optionally, heatmaps. The CSV is the deliverable; the plots
are convenience.

Runtime
-------
The observed-initiator fit is fast (~0.3 s for 1500 events). The latent-initiator fit
is ~20x slower and is off by default; enable with --latent for a reduced design.

    python scripts/run_identifiability_sweep.py --quick
    python scripts/run_identifiability_sweep.py --reps 20 --out results/sweep.csv
    python scripts/run_identifiability_sweep.py --latent --reps 5
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path

import argparse
import csv
import itertools
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np


from generative_models import (  # noqa: E402
    MechanismParams, TemporalNetworkModel, simulate, strip_initiators)
from inference import (  # noqa: E402
    _split_events, fit_latent_initiator, fit_observed_initiator,
    likelihood_ratio_test, selection_nll, timing_nll)

PARAMS = ["n_node", "kappa"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, nargs="+", default=[25, 50],
                   help="node counts to sweep")
    p.add_argument("--T", type=float, nargs="+", default=[150.0, 600.0, 2400.0],
                   help="observation windows to sweep")
    p.add_argument("--n-node", type=float, nargs="+", default=[0.0, 0.15, 0.3, 0.45, 0.6],
                   help="true node branching ratios")
    p.add_argument("--kappa", type=float, nargs="+", default=[0.0, 5.0, 10.0, 15.0, 20.0],
                   help="true tie reinforcements")
    p.add_argument("--gamma-n", type=float, default=0.5)
    p.add_argument("--gamma-e", type=float, default=0.05)
    p.add_argument("--reps", type=int, default=10, help="replicates per cell")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latent", action="store_true",
                   help="also run the latent-initiator estimator (much slower)")
    p.add_argument("--prior", action="store_true",
                   help="use hierarchical Gamma shrinkage prior for profile activities")
    p.add_argument("--prior-shape", type=float, default=4.0)
    p.add_argument("--quick", action="store_true",
                   help="small design for a smoke test (~1 minute)")
    p.add_argument("--out", type=str, default="results/identifiability_sweep.csv")
    p.add_argument("--plot", action="store_true", help="also write heatmaps")
    return p.parse_args()


def run_cell(N: int, T: float, truth: MechanismParams, reps: int, seed0: int,
             latent: bool, use_prior: bool = False, prior_shape: float = 4.0) -> Dict[str, float]:
    est = {k: [] for k in PARAMS}
    ses = {k: [] for k in PARAMS}
    rej = {k: [] for k in PARAMS}
    est_lat = {k: [] for k in PARAMS}
    corr_lat: List[float] = []
    n_events: List[int] = []
    n_failed = 0

    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        activities = 0.02 + 0.03 * rng.random(N)
        model = TemporalNetworkModel(N, activities, truth)
        events = simulate(model, T, rng=np.random.default_rng(seed0 + 100000 + r))
        n_events.append(len(events))
        if len(events) < 30:
            n_failed += 1
            continue
        try:
            res = fit_observed_initiator(events, N, T, use_prior=use_prior, prior_shape=prior_shape)
        except (ValueError, RuntimeError):
            n_failed += 1
            continue

        est["n_node"].append(res.n_node)
        est["kappa"].append(res.kappa)
        ses["n_node"].append(res.se["n_node"])
        ses["kappa"].append(res.se["kappa"])

        contacts, initiator, side = _split_events(events)
        p_k = likelihood_ratio_test(
            res.nll_selection, selection_nll(0.0, res.gamma_e, contacts, side, N), 1)
        p_n = likelihood_ratio_test(
            res.nll_timing, timing_nll(0.0, res.gamma_n, contacts, initiator, N, T, use_prior=use_prior, prior_shape=prior_shape), 1)
        rej["kappa"].append(1.0 if (np.isfinite(p_k) and p_k < 0.05) else 0.0)
        rej["n_node"].append(1.0 if (np.isfinite(p_n) and p_n < 0.05) else 0.0)

        if latent:
            try:
                rl = fit_latent_initiator(strip_initiators(events), N, T,
                                          n_imputations=3, n_em=5, seed=seed0 + r,
                                          use_prior=use_prior, prior_shape=prior_shape)
                est_lat["n_node"].append(rl.n_node)
                est_lat["kappa"].append(rl.kappa)
                if np.isfinite(rl.corr_n_kappa):
                    corr_lat.append(rl.corr_n_kappa)
            except (ValueError, RuntimeError):
                pass

    row: Dict[str, float] = {
        "N": N, "T": T, "true_n_node": truth.n_node, "true_kappa": truth.kappa,
        "gamma_n": truth.gamma_n, "gamma_e": truth.gamma_e,
        "reps": reps, "n_ok": len(est["n_node"]), "n_failed": n_failed,
        "mean_events": float(np.mean(n_events)) if n_events else 0.0,
    }
    for k in PARAMS:
        tv = truth.n_node if k == "n_node" else truth.kappa
        e = np.array(est[k], float)
        s = np.array(ses[k], float)
        if len(e) == 0:
            row.update({f"{k}_bias": np.nan, f"{k}_rmse": np.nan, f"{k}_sd": np.nan,
                        f"{k}_mean_se": np.nan, f"{k}_coverage": np.nan,
                        f"{k}_power": np.nan})
            continue
        ok = np.isfinite(s) & (s > 0)
        cover = (np.abs(e[ok] - tv) < 1.96 * s[ok]).mean() if ok.any() else np.nan
        row.update({
            f"{k}_bias": float(e.mean() - tv),
            f"{k}_rmse": float(np.sqrt(np.mean((e - tv) ** 2))),
            f"{k}_sd": float(e.std(ddof=1)) if len(e) > 1 else np.nan,
            f"{k}_mean_se": float(np.nanmean(s)) if ok.any() else np.nan,
            f"{k}_coverage": float(cover),
            f"{k}_power": float(np.mean(rej[k])) if rej[k] else np.nan,
        })
        if latent:
            el = np.array(est_lat[k], float)
            row[f"{k}_bias_latent"] = float(el.mean() - tv) if len(el) else np.nan
            row[f"{k}_rmse_latent"] = (float(np.sqrt(np.mean((el - tv) ** 2)))
                                       if len(el) else np.nan)
    if latent:
        row["corr_n_kappa_latent"] = float(np.mean(corr_lat)) if corr_lat else np.nan
    return row


def main() -> None:
    args = parse_args()
    if args.quick:
        args.N, args.T = [25], [150.0, 600.0]
        args.n_node, args.kappa = [0.0, 0.6], [0.0, 20.0]
        args.reps = 4

    design = list(itertools.product(args.N, args.T, args.n_node, args.kappa))
    print(f"design: {len(design)} cells x {args.reps} reps "
          f"= {len(design) * args.reps} fits"
          + (" (+ latent)" if args.latent else ""))

    rows: List[Dict[str, float]] = []
    t_start = time.time()
    for c, (N, T, n, k) in enumerate(design):
        truth = MechanismParams(n, args.gamma_n, k, args.gamma_e)
        seed0 = args.seed + 1000 * c
        t0 = time.time()
        row = run_cell(N, T, truth, args.reps, seed0, args.latent,
                       use_prior=args.prior, prior_shape=args.prior_shape)
        rows.append(row)
        print(f"[{c + 1:3d}/{len(design)}] N={N:3d} T={T:7.0f} n={n:.2f} k={k:5.1f} "
              f"| E~{row['mean_events']:7.0f} "
              f"| n_bias={row['n_node_bias']:+.3f} cov={row['n_node_coverage']:.2f} "
              f"| k_bias={row['kappa_bias']:+.2f} cov={row['kappa_coverage']:.2f} "
              f"| {time.time() - t0:.1f}s")

    out = args.out
    if not os.path.isabs(out):
        out = os.path.join(ROOT, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = sorted({key for r in rows for key in r})
    fields = (["N", "T", "true_n_node", "true_kappa"]
              + [f for f in fields if f not in ("N", "T", "true_n_node", "true_kappa")])
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out}  ({time.time() - t_start:.0f}s total)")

    print("\ncells where the estimator is confidently wrong (coverage < 0.80):")
    bad = [r for r in rows
           if (np.isfinite(r.get("n_node_coverage", np.nan))
               and r["n_node_coverage"] < 0.80)
           or (np.isfinite(r.get("kappa_coverage", np.nan))
               and r["kappa_coverage"] < 0.80)]
    if not bad:
        print("  none")
    for r in bad:
        print(f"  N={r['N']:.0f} T={r['T']:.0f} n={r['true_n_node']:.2f} "
              f"k={r['true_kappa']:.1f}: cov(n)={r['n_node_coverage']:.2f} "
              f"cov(k)={r['kappa_coverage']:.2f} (mean {r['mean_events']:.0f} events)")

    if args.plot:
        _plot(rows, os.path.dirname(out))


def _plot(rows: List[Dict[str, float]], outdir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available; skipping plots)")
        return

    import sys as _sys
    pass
    from palette import SEQ  # noqa: E402

    Ts = sorted({r["T"] for r in rows})
    ns = sorted({r["true_n_node"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7))
    metrics = ["n_node_coverage", "kappa_coverage"]
    labels = [r"$n_{\rm node}$ coverage", r"$\kappa$ coverage"]

    for ax, metric, title in zip(axes, metrics, labels):
        ax.axhline(0.95, color="k", linestyle="--", lw=1, zorder=0, label="Nominal (0.95)")
        for ti, T in enumerate(Ts):
            y_vals = []
            for n in ns:
                covs = [r[metric] for r in rows if r["T"] == T and r["true_n_node"] == n
                        and np.isfinite(r.get(metric, np.nan))]
                y_vals.append(np.mean(covs) if covs else np.nan)
            ax.plot(ns, y_vals, marker="o", color=SEQ[ti % len(SEQ)], lw=2,
                    markeredgecolor="#333", markeredgewidth=0.4, label=f"T = {T:.0f}")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(r"true $n_{\rm node}$")
        ax.set_ylabel("Coverage")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    # one shared legend below both panels
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(h), frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = os.path.join(outdir, "coverage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()

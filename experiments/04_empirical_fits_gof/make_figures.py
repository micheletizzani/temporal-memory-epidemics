#!/usr/bin/env python3
"""Regenerate every paper figure from the fitted models, and emit the LaTeX macros.

Figures
-------
  qq.png       QQ plot of rescaled times against Exp(1), one panel per dataset.
               This is the figure that shows a reader WHY the model is rejected
               instead of asserting a KS number. The zero atom, the tail, and the
               bulk deviation are all visible and separable by eye.
  plane.png    (n_node, kappa) per day, with the goodness-of-fit panel beside it.
               The two panels must stay together: reporting the separation without
               the rejection would misrepresent what the parameters mean.

Also writes `paper_macros.tex`, which can be pasted into the macro block of
generated/main.tex so no numeral is ever typed into prose by hand.

Usage
-----
    python scripts/make_figures.py --datasets hospital school sfhh --outdir results
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path

import argparse
import json
import os
import sys
import time

import numpy as np


from sociopatterns_loader import (  # noqa: E402
    load_sociopatterns, restrict_to_active_nodes, split_by_window)
import inference as inf  # noqa: E402


DATASETS = {
    "school": dict(path="HighSchool2013_proximity_net.csv", cols=(0, 1, 2), gap=60.0,
                   window=86400.0, label="High school"),
    "sfhh": dict(path="SFHH_tij.dat", cols=(0, 1, 2), gap=60.0,
                 window=86400.0, label="Conference (SFHH)"),
    "hospital": dict(path="hospital_lyon_contacts.dat", cols=(0, 1, 2), gap=60.0,
                     window=86400.0, label="Hospital ward"),
    "email": dict(path="email-Eu-core-temporal.txt", cols=(2, 0, 1), gap=0.0,
                  window=2592000.0, label="Email (EU institution)"),
    # SNN spike train (t i j): directed, initiator observed, so it is fit with the
    # exact observed-initiator MLE rather than by imputing labels.
    "snn": dict(path=os.path.join(ROOT, "data",
                                  "snn_bursty_network.txt"),
                cols=(0, 1, 2), directed=True, max_events=60000,
                label="SNN (bursting)"),
}


def _fit_directed(cfg, args):
    """Fit a directed dataset (SNN) with observed initiators; return the QQ payload."""
    path = cfg["path"] if os.path.isabs(cfg["path"]) else os.path.join(ROOT, "data",
                                                                       cfg["path"])
    if not os.path.exists(path):
        print(f"  missing {path}, skipped")
        return None
    d = np.loadtxt(path)
    if cfg.get("max_events") and len(d) > cfg["max_events"]:
        d = d[:cfg["max_events"]]
    tc, ic, jc = cfg["cols"]
    t, i, j = d[:, tc].astype(float), d[:, ic].astype(int), d[:, jc].astype(int)
    order = np.argsort(t, kind="stable")
    t, i, j = t[order], i[order], j[order]
    ids = sorted(set(i) | set(j)); remap = {o: k for k, o in enumerate(ids)}
    ev = [(float(t[k]), remap[int(i[k])], remap[int(j[k])]) for k in range(len(t))]
    N, T = len(ids), float(t.max())
    t0 = time.time()
    fit = inf.fit_observed_initiator(ev, N, T, fit_b=True)
    tau = inf.rescaled_times(ev, N, T, fit.n_node, fit.gamma_n, fit.activities)
    print(f"  [snn] {len(ev)} events, N={N}, n_node={fit.n_node:.3f} "
          f"kappa={fit.kappa:.1f} ({time.time()-t0:.0f}s)"
          + ("  *** AT A BOUND" if fit.at_bounds else ""))
    return dict(label=cfg["label"], tau=tau, fit=fit, n_contacts=len(ev),
                N=N, at_bounds=fit.at_bounds)


def fit_one(key: str, block_index: int, args):
    """Fit one window of one dataset and return the rescaled residuals."""
    cfg = DATASETS[key]
    if cfg.get("directed"):
        return _fit_directed(cfg, args)
    path = os.path.join(ROOT, "data", cfg["path"])
    if not os.path.exists(path):
        print(f"  [{key}] missing {path}, skipped")
        return None

    tcol, icol, jcol = cfg["cols"]
    data = load_sociopatterns(path, time_col=tcol, i_col=icol, j_col=jcol,
                              sessionize_gap=(None if cfg["gap"] <= 0 else cfg["gap"]))
    blocks = split_by_window(data, cfg["window"], min_contacts=args.min_contacts)
    if block_index >= len(blocks):
        return None
    blk = blocks[block_index]

    t0 = time.time()
    fit = inf.fit_latent_initiator(blk.contacts, blk.N, blk.T,
                                   n_imputations=args.imputations,
                                   n_em=args.em_sweeps, fit_b=True, seed=args.seed)
    ordered = [(t, i, j) if i < j else (t, j, i) for (t, i, j) in blk.contacts]
    side = inf._impute_initiators(ordered, blk.N, blk.T,
                                  [fit.n_node, fit.gamma_n, fit.kappa, fit.gamma_e],
                                  np.random.default_rng(args.seed + 7))
    ev = [(t, (i if s == 0 else j), (j if s == 0 else i))
          for (t, i, j), s in zip(ordered, side)]
    tau = inf.rescaled_times(ev, blk.N, blk.T, fit.n_node, fit.gamma_n, fit.activities)

    print(f"  [{key}] window {block_index}: {len(blk.contacts)} contacts, N={blk.N}, "
          f"n_node={fit.n_node:.3f} kappa={fit.kappa:.1f} "
          f"({time.time() - t0:.0f}s)"
          + ("  *** AT A BOUND" if fit.at_bounds else ""))
    return dict(label=cfg["label"], tau=tau, fit=fit, n_contacts=len(blk.contacts),
                N=blk.N, at_bounds=fit.at_bounds)


def make_qq(results, outdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    if n == 4:
        fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.8), squeeze=False)
    elif n <= 3:
        fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.4), squeeze=False)
    else:  # 5-6 panels: two rows
        ncol = (n + 1) // 2
        fig, axes = plt.subplots(2, ncol, figsize=(3.3 * ncol, 6.8), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]
    for ax in flat_axes[n:]:          # hide unused panels
        ax.axis("off")

    for ax, r in zip(flat_axes, results):
        tau = np.sort(r["tau"])
        m = len(tau)
        # theoretical Exp(1) quantiles at the plotting positions
        q = -np.log(1.0 - (np.arange(1, m + 1) - 0.5) / m)
        ax.plot(q, tau, ".", ms=2, alpha=0.5)
        hi = max(q.max(), np.quantile(tau, 0.999))
        ax.plot([0, hi], [0, hi], "k--", lw=1)
        zero = (tau <= 1e-9).mean()
        ax.set_title(f"{r['label']}\n"
                     f"zero atom = {zero:.1%}, KS = {inf._ks_uniform(1 - np.exp(-tau)):.3f}",
                     fontsize=8)
        ax.set_xlabel("Exp(1) quantile")
        ax.set_ylabel(r"rescaled time $\tau$")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.grid(alpha=0.25)
    fig.suptitle("Time-rescaling residuals: points below the diagonal at the origin are "
                 "the zero atom", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(outdir, "qq.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  wrote {path}")


def write_macros(results, outdir: str) -> None:
    lines = ["% Auto-generated by scripts/make_figures.py -- do not edit by hand.", ""]
    key = {"High school": "Hs", "Conference (SFHH)": "Sf", "Hospital ward": "Ho",
           "Email (EU institution)": "Em"}
    for r in results:
        k = key.get(r["label"], "X")
        f = r["fit"]
        tau = r["tau"]
        lines += [
            f"\\newcommand{{\\gof{k}Contacts}}{{{r['n_contacts']}}}",
            f"\\newcommand{{\\gof{k}N}}{{{r['N']}}}",
            f"\\newcommand{{\\gof{k}KS}}{{{inf._ks_uniform(1 - np.exp(-tau)):.3f}}}",
            f"\\newcommand{{\\gof{k}ZeroAtom}}{{{(tau <= 1e-9).mean() * 100:.1f}\\%}}",
            f"\\newcommand{{\\gof{k}MeanTau}}{{{tau.mean():.3f}}}",
            f"\\newcommand{{\\gof{k}VarTau}}{{{tau.var():.3f}}}",
            f"\\newcommand{{\\fit{k}Nnode}}{{{f.n_node:.3f}}}",
            f"\\newcommand{{\\fit{k}Kappa}}{{{f.kappa:.1f}}}",
            f"\\newcommand{{\\fit{k}AtBounds}}{{{'yes' if f.at_bounds else 'no'}}}",
            "",
        ]
    path = os.path.join(outdir, "paper_macros.tex")
    open(path, "w").write("\n".join(lines))
    print(f"  wrote {path}  ({len(results)} datasets)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["hospital", "school", "sfhh", "email"],
                    choices=sorted(DATASETS))
    ap.add_argument("--block", type=int, default=0, help="which window to plot")
    ap.add_argument("--min-contacts", type=int, default=5)
    ap.add_argument("--imputations", type=int, default=2)
    ap.add_argument("--em-sweeps", type=int, default=2)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print("fitting:")
    results = [r for r in (fit_one(k, args.block, args) for k in args.datasets)
               if r is not None]
    if not results:
        raise SystemExit("no datasets fitted")

    print("figures:")
    make_qq(results, outdir)
    write_macros(results, outdir)
    json.dump({r["label"]: {"ks": float(inf._ks_uniform(1 - np.exp(-r["tau"]))),
                            "zero_atom": float((r["tau"] <= 1e-9).mean()),
                            "n_node": r["fit"].n_node, "kappa": r["fit"].kappa,
                            "at_bounds": r["at_bounds"]}
               for r in results},
              open(os.path.join(outdir, "figure_summary.json"), "w"), indent=1)
    print("done.")


if __name__ == "__main__":
    main()

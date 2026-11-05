#!/usr/bin/env python3
"""Fit one dataset and cache its time-rescaling residuals tau to results/tau_<key>.npy.

Split out so each fit finishes inside a single short shell call; the 5-panel QQ figure is
then assembled from the cached taus by qq_combine.py.

Usage: python cache_tau.py <key> <imputations> <em_sweeps>
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import sys

import numpy as np

from make_figures import fit_one, DATASETS     # noqa: E402


class Args:
    min_contacts = 5
    seed = 0


def main():
    key = sys.argv[1]
    a = Args()
    a.imputations = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    a.em_sweeps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    r = fit_one(key, 0, a)
    if r is None:
        raise SystemExit(f"no fit for {key}")
    resdir = os.path.join(ROOT, "results")
    os.makedirs(resdir, exist_ok=True)
    tau = np.asarray(r["tau"], float)
    np.save(os.path.join(resdir, f"tau_{key}.npy"), tau)
    import inference as inf
    ks = inf._ks_uniform(1.0 - np.exp(-tau))
    np.save(os.path.join(resdir, f"tau_{key}_meta.npy"),
            np.array([ks, (tau <= 1e-9).mean(), r["N"], r["n_contacts"]]))
    print(f"{key}: n={r['n_contacts']} N={r['N']} KS={ks:.3f} "
          f"zero_atom={(tau<=1e-9).mean():.3f} label={r['label']}")


if __name__ == "__main__":
    main()

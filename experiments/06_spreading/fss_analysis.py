#!/usr/bin/env python3
"""
fss_analysis.py  —  Finite-size scaling analysis for epidemic_poles results.

Consumes poles_<model>_N<N>_<kind>.json produced by epidemic_poles.jl and
delivers three analyses:

  1.  chi/N COLLAPSE
      Intensive susceptibility chi/N = Var(O)/<O>. If all N overlay:
      the growth of chi is the explicit N-prefactor, not a divergence.

  2.  BINDER / MOMENT-RATIO CROSSINGS
      Binder cumulant U4 = 1 - <O^4> / (3<O^2>^2) and ratio <O^2>/<O>^2.
      Crossings across N locate beta_c without a prefactor assumption.

  3.  MODEL-SPECIFIC TRANSITION PROBE:
      - SIS: Quasi-stationary susceptibility chi_QS = N * Var(rho|survival) / <rho|survival>.
        Power-law fit of peak vs N: exponent alpha > 1 signals divergence.
      - SIR: Macroscopic outbreak probability P_macro (fraction of runs with final size > 5/N).

Usage:
  python fss_analysis.py sis node [--dir results] [--text] [--out file.pdf]
  python fss_analysis.py sis all
  python fss_analysis.py all all
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
#!/usr/bin/env python3
"""
fss_analysis.py  —  Finite-size scaling analysis for epidemic_poles results.

Consumes poles_<model>_N<N>_<kind>.json produced by epidemic_poles.jl and
delivers three analyses:

  1.  chi/N COLLAPSE
  2.  BINDER / MOMENT-RATIO CROSSINGS
  3.  MODEL-SPECIFIC TRANSITION PROBE
"""

# %% Setup and Imports
import json
import sys
import os
import glob
import argparse
import numpy as np
import warnings

# Detect if we are running in an interactive environment (e.g., Jupyter, VS Code)
is_interactive = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules

try:
    import matplotlib
    # Only force "Agg" if running strictly from the command line
    if not is_interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not found - text-only output")

RESULTS_DIR = os.path.join(ROOT, "results")


# %% I/O Functions
def load(model, N, kind, results_dir):
    path = os.path.join(results_dir, f"poles_{model}_N{N}_{kind}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def available_Ns(model, kind, results_dir):
    pattern = os.path.join(results_dir, f"poles_{model}_N*_{kind}.json")
    Ns = []
    for p in glob.glob(pattern):
        try:
            n = int(os.path.basename(p).split("_N")[1].split("_")[0])
            Ns.append(n)
        except Exception:
            pass
    return sorted(Ns)


# %% Observable Extractors
def chi_over_N(d):
    """Returns (betas, chi/N) or None."""
    betas = np.array(d.get("betas", []))
    if not len(betas):
        return None
    N = d["N"]
    if "chi" in d:
        chi = np.array(d["chi"])
    elif "chi_sir" in d:
        chi = np.array(d["chi_sir"])
    elif "var" in d and "mean" in d:
        m1 = np.array(d["mean"])
        vr = np.array(d["var"])
        with np.errstate(invalid="ignore"):
            chi = np.where(m1 > 1e-9, N * vr / m1, 0.0)
    else:
        return None
    return betas, chi / N

def binder_u4(d):
    betas = np.array(d.get("betas", []))
    if "binder" not in d or not len(betas):
        return None
    return betas, np.array(d["binder"])

def qs_chi(d):
    betas = np.array(d.get("betas", []))
    if "qs_chi" not in d or not len(betas):
        return None
    return betas, np.array(d["qs_chi"])

def p_macro(d):
    betas = np.array(d.get("betas", []))
    if "p_macro" not in d or not len(betas):
        return None
    return betas, np.array(d["p_macro"])

def order_param(d):
    betas = np.array(d.get("betas", []))
    if not len(betas):
        return None
    key = "rho" if d.get("model") == "sis" else "final_size"
    if key not in d:
        key = "mean"
    if key not in d:
        return None
    return betas, np.array(d[key])


# %% Binder Crossing Finder
def find_crossings(betas_list, binder_list):
    crossings = []
    for i in range(len(binder_list) - 1):
        b1, u1 = betas_list[i], binder_list[i]
        b2, u2 = betas_list[i+1], binder_list[i+1]
        b_common = np.union1d(b1, b2)
        u1i = np.interp(b_common, b1, u1)
        u2i = np.interp(b_common, b2, u2)
        diff = u1i - u2i
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        for sc in sign_changes:
            d_lo, d_hi = diff[sc], diff[sc+1]
            b_lo, b_hi = b_common[sc], b_common[sc+1]
            if abs(d_hi - d_lo) > 1e-12:
                beta_x = b_lo - d_lo * (b_hi - b_lo) / (d_hi - d_lo)
                crossings.append(float(beta_x))
    return crossings


# %% Text Report Generator
def text_report(model, kind, results_dir):
    Ns = available_Ns(model, kind, results_dir)
    if not Ns:
        print(f"[!] No files for {model}/{kind} in {results_dir}")
        return

    line = "=" * 62
    print(f"\n{line}")
    print(f"  FSS  model={model}  pole={kind}  N={Ns}")
    print(line)

    # --- chi/N collapse ---
    print("\n  chi/N  (N-independent -> crossover; peaked -> transition)")
    rows = {}
    ref_betas = None
    for N in Ns:
        d = load(model, N, kind, results_dir)
        if d is None:
            rows[N] = None
            continue
        result = chi_over_N(d)
        rows[N] = result
        if result is not None and ref_betas is None:
            ref_betas = result[0]

    if ref_betas is not None:
        hdr = f"    {'beta':>6}  " + "  ".join(f"N={N:>6}" for N in Ns)
        print(hdr)
        for bi, beta in enumerate(ref_betas):
            row = f"    {beta:>6.3f}  "
            for N in Ns:
                result = rows[N]
                if result is None:
                    row += "     N/A  "
                elif bi < len(result[1]):
                    row += f"  {result[1][bi]:>8.5f}"
                else:
                    row += "         -"
            print(row)

    # --- Binder ---
    print("\n  Binder U4 crossings -> beta_c")
    betas_list, binder_list, Ns_b = [], [], []
    for N in Ns:
        d = load(model, N, kind, results_dir)
        if d is None:
            continue
        result = binder_u4(d)
        if result is None:
            continue
        betas_list.append(result[0])
        binder_list.append(result[1])
        Ns_b.append(N)
    if binder_list:
        crossings = find_crossings(betas_list, binder_list)
        if crossings:
            print(f"  Crossing estimates: beta_c ~ {[f'{c:.4f}' for c in crossings]}")
        else:
            print("  No crossings in available beta range.")
        print(f"  (Ns with Binder: {Ns_b})")
    else:
        print("  [Binder data not found]")

    # --- Model-specific Probe ---
    if model == "sis":
        print("\n  QS susceptibility peak (chi_QS ~ N^alpha; alpha>1 -> divergence)")
        qs_peaks = []
        for N in Ns:
            d = load(model, N, kind, results_dir)
            if d is None:
                continue
            result = qs_chi(d)
            if result is None:
                continue
            betas, vals = result
            pk = int(np.argmax(vals))
            qs_peaks.append((N, float(betas[pk]), float(vals[pk])))

        if qs_peaks:
            print(f"    {'N':>8}  {'beta_peak':>10}  {'chi_QS_peak':>13}  {'chi_QS/N':>10}")
            for N, bp, pk in qs_peaks:
                print(f"    {N:>8}  {bp:>10.4f}  {pk:>13.2f}  {pk/N:>10.6f}")
            if len(qs_peaks) >= 3:
                Nf = np.array([p[0] for p in qs_peaks], float)
                pf = np.array([p[2] for p in qs_peaks])
                alpha, _ = np.polyfit(np.log(Nf), np.log(pf), 1)
                verdict = ("crossover (chi/N~const)" if abs(alpha - 1.0) < 0.15 else
                           f"consistent with divergence (alpha={alpha:.2f})" if alpha > 1.1 else
                           f"inconclusive (alpha={alpha:.2f})")
                print(f"  Power-law fit alpha = {alpha:.3f}: {verdict}")
        else:
            print("  [QS data not found]")
    else: # SIR
        print("\n  Macroscopic Outbreak Probability P_macro (final_size > 5/N)")
        ref_b = None
        p_rows = {}
        for N in Ns:
            d = load("sir", N, kind, results_dir)
            if d is None:
                p_rows[N] = None
                continue
            res = p_macro(d)
            p_rows[N] = res
            if res is not None and ref_b is None:
                ref_b = res[0]
        if ref_b is not None:
            hdr = f"    {'beta':>6}  " + "  ".join(f"N={N:>6}" for N in Ns)
            print(hdr)
            for bi, enumerate_b in enumerate(ref_b):
                row = f"    {enumerate_b:>6.3f}  "
                for N in Ns:
                    res = p_rows[N]
                    if res is None:
                        row += "     N/A  "
                    elif bi < len(res[1]):
                        row += f"  {res[1][bi]:>8.4f}"
                    else:
                        row += "         -"
                print(row)


# %% Plotting Functions
def plot_fss(model, kind, results_dir, outfile):
    Ns = available_Ns(model, kind, results_dir)
    if not Ns:
        return
    colors = cm.viridis(np.linspace(0.1, 0.9, len(Ns)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.suptitle(f"FSS Analysis: {model.upper()} Model ({kind.capitalize()} Pole)", fontsize=13, fontweight="bold", y=0.98)
    ax1, ax2, ax3 = axes

    has_binder = has_p3 = False
    betas_list_b, binder_list_b = [], []

    for ni, (N, col) in enumerate(zip(Ns, colors)):
        d = load(model, N, kind, results_dir)
        if d is None:
            continue
        lbl = f"N={N}"

        r = chi_over_N(d)
        if r:
            ax1.plot(r[0], r[1], "o-", color=col, ms=4, lw=1.3, label=lbl)

        r = binder_u4(d)
        if r:
            ax2.plot(r[0], r[1], "s--", color=col, ms=4, lw=1.3, label=lbl)
            betas_list_b.append(r[0]); binder_list_b.append(r[1])
            has_binder = True

        r = qs_chi(d) if model == "sis" else p_macro(d)
        if r:
            ax3.plot(r[0], r[1], "^:", color=col, ms=4, lw=1.3, label=lbl)
            has_p3 = True

    # Subplot 1
    ax1.set_xlabel(r"$\beta$", fontsize=10)
    ax1.set_ylabel(r"$\chi / N$", fontsize=10)
    ax1.set_title(r"$\chi / N$ Collapse", fontsize=11, fontweight="bold", pad=8)
    ax1.text(0.5, -0.16, "(flat = crossover; peak = transition)", transform=ax1.transAxes,
             ha="center", va="top", fontsize=8.5, color="#333333", style="italic")

    # Subplot 2
    ax2.set_xlabel(r"$\beta$", fontsize=10)
    ax2.set_ylabel(r"Binder $U_4$", fontsize=10)
    ax2.set_title(r"Binder Cumulant $U_4$", fontsize=11, fontweight="bold", pad=8)
    if has_binder:
        crossings = find_crossings(betas_list_b, binder_list_b)
        ax2.axhline(2/3, color="k", lw=0.7, ls="--", alpha=0.4)
        for cx in crossings:
            ax2.axvline(cx, color="red", lw=1, ls=":", alpha=0.8)
        cx_str = ", ".join([f"{c:.3f}" for c in crossings[:3]]) if crossings else "none"
        ax2.text(0.5, -0.16, f"(crossings locate $\\beta_c$: {cx_str})", transform=ax2.transAxes,
                 ha="center", va="top", fontsize=8.5, color="#333333", style="italic")
    else:
        ax2.text(0.5, -0.16, "(crossings locate $\\beta_c$)", transform=ax2.transAxes,
                 ha="center", va="top", fontsize=8.5, color="#333333", style="italic")

    # Subplot 3
    ax3.set_xlabel(r"$\beta$", fontsize=10)
    if model == "sis":
        ax3.set_ylabel(r"$\chi_{\rm QS}$", fontsize=10)
        ax3.set_title(r"Quasi-Stationary $\chi_{\rm QS}$", fontsize=11, fontweight="bold", pad=8)
        ax3.text(0.5, -0.16, r"(peak $\propto N^\alpha$: $\alpha > 1 \rightarrow$ transition)", transform=ax3.transAxes,
                 ha="center", va="top", fontsize=8.5, color="#333333", style="italic")
    else:
        ax3.set_ylabel(r"$P_{\rm macro}$", fontsize=10)
        ax3.set_title(r"Macroscopic Outbreak $P_{\rm macro}$", fontsize=11, fontweight="bold", pad=8)
        ax3.text(0.5, -0.16, r"($P(R(\infty)/N > 5/N)$ threshold)", transform=ax3.transAxes,
                 ha="center", va="top", fontsize=8.5, color="#333333", style="italic")

    # Build a single central legend from unique labels across all axes
    dict_legend = {}
    for ax in [ax1, ax2, ax3]:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in dict_legend:
                dict_legend[l] = h

    if dict_legend:
        fig.legend(dict_legend.values(), dict_legend.keys(),
                   loc='lower center', bbox_to_anchor=(0.5, 0.009),
                   ncol=min(len(dict_legend), 10), fontsize=9, frameon=False)

    # Leave room at the bottom for the legend
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved -> {outfile}")
    if is_interactive: 
        plt.show() # Display inline in notebook environments
    plt.close()


def plot_sir_gap(results_dir, outfile):
    Ns = available_Ns("sir", "node", results_dir)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cn = cm.Blues(np.linspace(0.4, 0.9, max(1, len(Ns))))
    cl = cm.Reds(np.linspace(0.4, 0.9, max(1, len(Ns))))
    for N, cc, cr in zip(Ns, cn, cl):
        dn = load("sir", N, "node", results_dir)
        dl = load("sir", N, "link", results_dir)
        if dn:
            r = order_param(dn)
            if r: ax.plot(r[0], r[1], "o-", color=cc, ms=3, lw=1.3, label=f"node N={N}")
        if dl:
            r = order_param(dl)
            if r: ax.plot(r[0], r[1], "s--", color=cr, ms=3, lw=1.1, label=f"link N={N}")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel("SIR final size")
    ax.set_title("SIR gap sharpening\n(node supercritical; link/Poisson subcritical)")
    ax.grid(alpha=0.3)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.009), ncol=2, fontsize=9, frameon=False)

    # Leave room at the bottom for the legend
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved -> {outfile}")
    if is_interactive:
        plt.show()
    plt.close()


# %% Execution
def main(args_list=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=["sis", "sir", "all"])
    parser.add_argument("kind", nargs="?", default="node",
                        choices=["node", "link", "poisson", "all"])
    parser.add_argument("--dir", default=RESULTS_DIR)
    parser.add_argument("--out", default=None)
    parser.add_argument("--text", action="store_true", help="text only, no plots")
    
    # Parse explicitly provided arguments (notebook mode) or sys.argv (CLI mode)
    args = parser.parse_args(args=args_list)

    models = ["sis", "sir"] if args.model == "all" else [args.model]
    kinds  = ["node", "link", "poisson"] if args.kind == "all" else [args.kind]

    for model in models:
        for kind in kinds:
            text_report(model, kind, args.dir)
            if not args.text and HAS_MPL:
                outfile = args.out or os.path.join(args.dir, f"fss_{model}_{kind}.pdf")
                plot_fss(model, kind, args.dir, outfile)

    if "sir" in models and not args.text and HAS_MPL:
        plot_sir_gap(args.dir, os.path.join(args.dir, "sir_gap_sharpening.pdf"))


if __name__ == "__main__":
    if is_interactive:
        # Provide default arguments so the kernel doesn't crash on parse_args
        # You can adjust these string parameters directly in the cell to test different subsets
        main(["all", "all"]) 
    else:
        main()

# %%
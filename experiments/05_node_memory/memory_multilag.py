"""Multi-lag memory M_k (k = 1..10) of every record on the event set of the shuffle screen, with the headline
within-node rank coefficient of memory_stats.py.

For each record: M_k, the 2.5-97.5% band of M_k after a per-node order shuffle (200 shuffles), a node
bootstrap 5-95% interval of M_1 (200 resamples), and the number of lags outside the band with |M_k| >= 0.05.
Controls: a heavy-tailed renewal train and an AR(1)-log-interval train (phase-plane generator).
Writes results/memory_multilag.json and generated/multilag_table.tex.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, sys
import numpy as np
import real_record_screen as rs  # noqa: E402
from memory_stats import node_gaps, rank_memory  # noqa: E402
from plane_validation import gen_train  # noqa: E402

LAGS = list(range(1, 11)); NPERM, NBOOT, FLOOR = 200, 200, 0.05
OUT = os.path.join(ROOT, "results", "memory_multilag.json")
TEX = os.path.abspath(os.path.join(ROOT, "generated", "multilag_table.tex"))


def analyse(gaps, seed=0):
    rng = np.random.default_rng(seed)
    Mk = [rank_memory(gaps, k) for k in LAGS]
    keys = list(gaps)
    null = np.array([[rank_memory({i: rng.permutation(gaps[i]) for i in keys}, k) for k in LAGS] for _ in range(NPERM)])
    lo, hi = np.nanpercentile(null, 2.5, axis=0), np.nanpercentile(null, 97.5, axis=0)
    flagged = [k for k, m, b in zip(LAGS, Mk, hi) if m > b and m >= FLOOR]          # positive memory only
    boot = [rank_memory({n: gaps[keys[j]] for n, j in enumerate(rng.integers(0, len(keys), len(keys)))}, 1)
            for _ in range(NBOOT)]
    return dict(M=Mk, band_lo=lo.tolist(), band_hi=hi.tolist(), flagged=flagged,
                M1_ci=[float(np.nanpercentile(boot, 5)), float(np.nanpercentile(boot, 95))], nodes=len(keys))


if __name__ == "__main__":
    res = {}
    for name, (load, _r) in rs.RECORDS.items():
        g = node_gaps(load()); res[name] = analyse(g)
        r = res[name]; print(f"{name:26s} nodes {r['nodes']:4d} M1 {r['M'][0]:+.3f} [{r['M1_ci'][0]:+.3f},{r['M1_ci'][1]:+.3f}] "
                             f"M5 {r['M'][4]:+.3f} M10 {r['M'][9]:+.3f} lags with positive memory {r['flagged']}", flush=True)
    rng = np.random.default_rng(5)
    ctl = {}
    for tag, rho in (("renewal (lognormal)", 0.0), ("AR(1) log-interval, rho=0.6", 0.6)):
        gs = {}
        for i in range(30):
            ts = gen_train(1500.0, 1.0, rho, 0.5, 0.0, rng)
            gs[i] = np.diff(ts)
        ctl[tag] = analyse(gs); a = np.concatenate(list(gs.values()))
        ctl[tag]["B"] = float((a.std() - a.mean()) / (a.std() + a.mean()))
        print(tag, [round(x, 3) for x in ctl[tag]["M"]], "B", round(ctl[tag]["B"], 2), flush=True)
    json.dump(dict(records=res, controls=ctl), open(OUT, "w"), indent=1)

    SH = {"Hospital ward": "Hospital", "High school": "High school", "Conference (SFHH)": "Conference", "Baboons": "Baboons",
          "Email (EU institution)": "Email", "CollegeMsg (Direct Msgs)": "CollegeMsg", "Stack Overflow": "Stack Overflow",
          "Wikibooks": "Wikibooks", "Wikibooks bots": "Wikibooks bots", "Microservices": "Microservices", "SNN (bursting)": "Spiking network"}
    def fm(x):
        t = f"{x:+.2f}"; return "$0.00$" if t in ("+0.00", "-0.00") else f"${t}$"
    rows = []
    for k, v in res.items():
        m = v["M"]; fl = v["flagged"]
        rng_txt = "none" if not fl else (f"1--{max(fl)}" if fl == list(range(1, max(fl) + 1)) else ",".join(map(str, fl)))
        rows.append(f"    {SH[k]} & {v['nodes']} & {fm(m[0])} [{fm(v['M1_ci'][0])[1:-1].join('$$')}, {fm(v['M1_ci'][1])[1:-1].join('$$')}] & {fm(m[1])} & {fm(m[4])} & {fm(m[9])} & {rng_txt} \\\\")
    for k, v in ctl.items():
        m = v["M"]
        rows.append(f"    \\emph{{control: {k}}} & --- & {fm(m[0])} & {fm(m[1])} & {fm(m[4])} & {fm(m[9])} & {'none' if not v['flagged'] else '1--' + str(max(v['flagged']))} \\\\")
    tab = ("\\begin{table}[h]\n  \\centering\n  \\small\n  \\setlength{\\tabcolsep}{4pt}\n  \\begin{tabular}{l r l r r r l}\n    \\toprule\n"
           "    record & nodes & $M_1$ [5--95\\% over nodes] & $M_2$ & $M_5$ & $M_{10}$ & lags with positive memory \\\\\n    \\midrule\n"
           + "\n".join(rows) + "\n    \\bottomrule\n  \\end{tabular}\n  \\caption{Multi-lag memory $M_k$ (within-node rank coefficient) of every record on the event set of the shuffle screen (proximity: day~0 participation trains; directed: initiator trains), with the $5$--$95\\%$ interval of $M_1$ over nodes (bootstrap of nodes). Last column: the lags $k\\le10$ at which $M_k$ exceeds the $97.5\\%$ point of its per-node-shuffle null and is at least $0.05$. The two controls are a heavy-tailed memoryless renewal train and an autoregressive log-interval train with injected memory (Sec.~\\ref{sm:multilag}). The email log and the spiking network have no memory at any lag; the spiking network's $M_1$ is negative (regular, refractory-like firing).}\n  \\label{tab:multilag}\n\\end{table}\n")
    open(TEX, "w").write(tab); print("wrote", TEX)

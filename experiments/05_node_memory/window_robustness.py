"""Window dependence of the shuffle ratio.  real_record_screen draws B_SUB = 4 random windows of 4000 events. Here 12 windows
(different start points) are drawn for the long proximity records, raw participation trains and (high school) trains with
coincident contacts collapsed; each window reports its duration T, n_hat, the per-window R (mean of 2 shuffles).
Writes results/window_robustness.json."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import json, os, sys
import numpy as np
import real_record_screen as rs  # noqa: E402
from screen_neural import per_node_shuffle  # noqa: E402

NW, NS, SUB = 12, 2, rs.SUB


def collapse(ev):
    seen, out = set(), []
    for t, i, j in ev:
        if (t, i) not in seen:
            seen.add((t, i)); out.append((t, i, j))
    return out


def windows(ev, seed=1):
    E = len(ev); rng = np.random.default_rng(seed); out = []
    for b in range(NW):
        s = int(rng.integers(0, E - SUB)); w = ev[s:s + SUB]; t0 = w[0][0]
        w = [(t - t0, i, j) for t, i, j in w]
        N = 1 + max(max(i, j) for _t, i, j in w); T = w[-1][0] + 1e-9
        ne = rs.n_exp(w, N, T)
        Rs = [rs.n_exp(per_node_shuffle(list(w), np.random.default_rng(1000 * b + k)), N, T) / max(ne, 1e-3) for k in range(NS)]
        out.append(dict(T=T, n_hat=ne, R=float(np.mean(Rs))))
        print("   ", {k: round(v, 3) for k, v in out[-1].items()}, flush=True)
    return out


if __name__ == "__main__":
    outp = os.path.join(ROOT, "results", "window_robustness.json"); res = {}
    for name, coll in (("High school", False), ("High school", True), ("Conference (SFHH)", False)):
        ev = sorted(rs.RECORDS[name][0]())
        tag = name + (" (collapsed)" if coll else "")
        print(tag, flush=True)
        res[tag] = windows(collapse(ev) if coll else ev)
        R = np.array([w["R"] for w in res[tag]])
        print(tag, "R mean %.3f median %.3f 5-95 [%.3f, %.3f]" % (R.mean(), np.median(R), *np.percentile(R, [5, 95])), flush=True)
        json.dump(res, open(outp, "w"), indent=1)

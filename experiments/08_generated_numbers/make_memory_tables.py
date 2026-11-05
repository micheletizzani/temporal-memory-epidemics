"""Writes generated/memory_tables.tex (two SM tables) from the JSON results of
memory_artifact_check.py, memory_modulation_check.py, memory_tie_check.py and record_matched_surrogate.py."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import glob, json, os
R = os.path.join(ROOT, "results")
OUT = os.path.abspath(os.path.join(ROOT, "generated", "memory_tables.tex"))
art = json.load(open(os.path.join(R, "memory_artifact_check.json")))
mod = json.load(open(os.path.join(R, "memory_modulation_check.json")))
tie = json.load(open(os.path.join(R, "memory_tie_check.json")))
SHORT = {"Hospital ward": "Hospital", "High school": "High school", "Conference (SFHH)": "Conference",
         "Baboons": "Baboons", "Email (EU institution)": "Email", "CollegeMsg (Direct Msgs)": "CollegeMsg",
         "Stack Overflow": "Stack Overflow", "Wikibooks": "Wikibooks", "Wikibooks bots": "Wikibooks bots",
         "Microservices": "Microservices", "SNN (bursting)": "Spiking network"}
def f(x):
    t = f"{x:+.2f}"
    return "$0.00$" if t in ("+0.00", "-0.00") else f"${t}$"


def rng(h):
    a, b = f"{min(h):.2f}", f"{max(h):.2f}"
    return f"${a}$" if a == b else f"${a}$--${b}$"
rows = []
for k, s in SHORT.items():
    a, m, t = art[k], mod[k], tie[k]
    rows.append(f"    {s} & {f(a['M_pool'])} & {f(a['M_pool_shuf'])} & {f(a['M_within'])} & {f(a['M_rank'])} & "
                f"{f(a['M_rank_shuf'])} & {f(m['rank_100'])} & {f(t['rank_collapsed'])} \\\\")
tab1 = r"""\begin{table}[h]
  \centering
  \small
  \setlength{\tabcolsep}{4pt}
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{l r r r r r r r}
    \toprule
    record & pooled $M$ & after shuffle & within-node $M$ & within-node rank $M$ & after shuffle & profile removed & repeats collapsed \\
    \midrule
""" + "\n".join(rows) + r"""
    \bottomrule
  \end{tabular}}
  \caption{Lag-one memory coefficients under four definitions, on the event set of the shuffle screen
  (proximity: day~0 participation trains; directed: initiator trains). \emph{Pooled $M$}: Pearson
  correlation of consecutive intervals pooled over nodes; it contains a
  between-node scale component that a per-node shuffle keeps (\emph{after shuffle}). \emph{Within-node $M$}: the same
  after dividing each node's intervals by its mean. \emph{Within-node rank $M$}: the memory coefficient of the main text, the Pearson correlation of
  within-node interval ranks, robust to heavy tails; \emph{after shuffle}: its value after the per-node order
  shuffle (mean of six). \emph{Profile removed}: rank $M$ after mapping time through the cumulative global event rate
  (100 bins), which removes any activity profile shared by all nodes. \emph{Repeats collapsed}: rank $M$ after
  keeping only events whose partner differs from the node's previous partner. The pooled $M$ of the email log and of
  the spiking network is entirely a between-node effect (it does not fall after the shuffle, and the within-node
  coefficients are zero or negative); the other records have order memory by the rank coefficient, which is not
  explained by a common activity profile or by repeated ties. For orientation, the positive control
  (exponential node memory of $30$~s injected into the high-school activity pattern) gives a rank coefficient of
  $0.01$, $0.08$, $0.17$, $0.28$, $0.37$ at $\nnode=0$, $0.2$, $0.4$, $0.6$, $0.8$ (script
  \texttt{poscon\_memory\_rank.py}). Scripts: \texttt{memory\_artifact\_check.py},
  \texttt{memory\_modulation\_check.py}, \texttt{memory\_tie\_check.py}.}
  \label{tab:memdefs}
\end{table}
"""
mrows = []
for p in sorted(glob.glob(os.path.join(R, "matched_*.json"))):
    if "matched_control" in p:
        continue
    d = json.load(open(p)); r = d["real"]
    bound = " (rate at grid bound)" if r["gamma_hat"] in (1e-3, 10.0) or abs(r["gamma_hat"] - 1e-3) < 1e-9 or abs(r["gamma_hat"] - 10) < 1e-6 else ""
    h = d["hawkes"]["R"]
    mrows.append(f"    {SHORT.get(d['record'], d['record'])} & ${r['R']:.2f}$ & ${d['local']['R']:.2f}$ & ${r['n_hat']:.2f}${'$^\\ddagger$' if bound else ''} & "
                 f"{rng(h)} & ${d['null']['R_mean']:.2f}$ & {f(d['M_within_real'])} & {f(d['hawkes']['M_within'])} \\\\")
tab2 = r"""\begin{table}[h]
  \centering
  \small
  \setlength{\tabcolsep}{4pt}
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{l r r r r r r r}
    \toprule
    record & measured $R$ & local-shuffle $R$ & screen $\hat n$ & matched Hawkes $R$ & Poisson $R$ & within-node $M$, record & within-node $M$, matched Hawkes \\
    \midrule
""" + "\n".join(mrows) + r"""
    \bottomrule
  \end{tabular}}
  \caption{The shuffle ratio of each record compared with controls run through the identical screen. \emph{Measured $R$}
  is the screen of the main text. \emph{Local-shuffle $R$} shuffles intervals only within blocks of $20$ consecutive
  intervals per node, which keeps slow rate changes. \emph{Screen $\hat n$} is the single-exponential
  branching ratio of the screen's own estimator; $\ddagger$: its fitted rate sits at an end of the rate grid (the
  slowest, $10^{-3}$\,s$^{-1}$, or the fastest, $10$\,s$^{-1}$, far below the $20$\,s resolution), so $g$ is not identified under
  this estimator. \emph{Matched Hawkes $R$}: range over three surrogates of exponential self-excitation at the screen's fitted
  $(\hat n,\hat\gamma)$, with the record's own per-node event counts, duration and time resolution. \emph{Poisson
  $R$}: the same without self-excitation. The last two columns compare the within-node Pearson memory coefficient of the
  record and of the matched Hawkes surrogate. Script: \texttt{record\_matched\_surrogate.py}.}
  \label{tab:matched_R}
\end{table}
"""
open(OUT, "w").write(tab1 + "\n" + tab2); print("wrote", OUT)

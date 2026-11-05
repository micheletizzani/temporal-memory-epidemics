"""Loader for SocioPatterns-style contact data.

SocioPatterns proximity files record a *sampled contact*, not an event: each line says
"these two people were within range during this 20-second window". A dyad in continuous
conversation therefore emits one line every 20 s, and a naive point-process fit reads
that regular sampling as extreme self-excitation. Sessionising -- collapsing runs of
consecutive samples on the same dyad into a single contact event with a start time --
is not cosmetic cleaning; without it the tie-memory parameter is measuring the sampling
grid rather than social behaviour.

The loader therefore returns both representations and makes the caller choose:

    load_sociopatterns(...)                       -> raw samples
    load_sociopatterns(..., sessionize_gap=60.0)  -> one event per conversation

`sessionize_gap` is the maximum gap (in the file's own time units, seconds for
SocioPatterns) between consecutive samples of the same dyad that still counts as the
same conversation. For the 20 s SocioPatterns grid, 60 s (two missed frames) is a
common choice; report whatever you use and check sensitivity.

The high-school file is `HighSchool2013_proximity_net.csv`, whitespace-separated with
columns: time, node_i, node_j, class_i, class_j.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

Contact = Tuple[float, int, int]


@dataclass
class ContactData:
    N: int
    contacts: List[Contact]          # (t, i, j) with i < j, sorted by t, t starting at 0
    node_map: Dict[str, int]         # original id -> 0-based index
    T: float
    n_raw_samples: int
    metadata: Dict[str, str]

    def describe(self) -> str:
        dyads = {(i, j) for _t, i, j in self.contacts}
        pairs = 0.5 * self.N * (self.N - 1)
        return (
            f"N = {self.N} nodes, {len(self.contacts)} contacts "
            f"(from {self.n_raw_samples} raw samples)\n"
            f"T = {self.T:.0f} time units ({self.T / 3600:.1f} h)\n"
            f"distinct dyads = {len(dyads)} ({len(dyads) / pairs:.1%} of all pairs)\n"
            f"repeat fraction = {1 - len(dyads) / max(len(self.contacts), 1):.3f}"
        )


def load_sociopatterns(filepath: str,
                       delimiter: Optional[str] = None,
                       time_col: int = 0, i_col: int = 1, j_col: int = 2,
                       sessionize_gap: Optional[float] = None,
                       max_time: Optional[float] = None) -> ContactData:
    """Read a contact file into a ContactData record.

    Parameters
    ----------
    delimiter:
        None splits on any whitespace (the SocioPatterns default). Pass ',' for CSV.
    sessionize_gap:
        If given, consecutive samples of the same dyad separated by less than this are
        merged into one contact, timed at the start of the run. Strongly recommended for
        SocioPatterns; see the module docstring.
    max_time:
        Optional hard cut on the time axis, applied AFTER shifting to start at 0. Use
        this to restrict to a sub-window deliberately. It is not a silent performance
        truncation: the returned T reflects the cut, so the likelihood's compensator
        stays consistent.

    Raises
    ------
    ValueError
        On an unreadable file, a malformed line, self-loops, or non-monotone time after
        sorting. Failing loudly beats silently dropping rows.
    """
    rows: List[Tuple[float, str, str]] = []
    n_bad = 0
    with open(filepath, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delimiter) if delimiter else line.split()
            if len(parts) <= max(time_col, i_col, j_col):
                n_bad += 1
                if n_bad <= 5:
                    print(f"  [loader] skipping short line {lineno}: {line[:60]!r}")
                continue
            try:
                t = float(parts[time_col])
            except ValueError:
                n_bad += 1
                continue
            rows.append((t, parts[i_col], parts[j_col]))

    if n_bad:
        print(f"  [loader] skipped {n_bad} malformed lines")
    if not rows:
        raise ValueError(f"no usable rows read from {filepath}")

    ids = sorted({a for _t, a, _b in rows} | {b for _t, _a, b in rows})
    node_map = {nid: k for k, nid in enumerate(ids)}
    N = len(ids)

    samples: List[Contact] = []
    n_self = 0
    for t, a, b in rows:
        i, j = node_map[a], node_map[b]
        if i == j:
            n_self += 1
            continue
        samples.append((t, min(i, j), max(i, j)))
    if n_self:
        print(f"  [loader] dropped {n_self} self-loops")
    samples.sort(key=lambda r: (r[0], r[1], r[2]))
    n_raw = len(samples)

    t0 = samples[0][0]
    samples = [(t - t0, i, j) for t, i, j in samples]

    if sessionize_gap is not None:
        contacts = _sessionize(samples, sessionize_gap)
    else:
        contacts = samples

    if max_time is not None:
        contacts = [c for c in contacts if c[0] <= max_time]
        if not contacts:
            raise ValueError(f"max_time={max_time} removed every contact")

    T = float(contacts[-1][0]) if max_time is None else float(max_time)
    if T <= 0.0:
        raise ValueError("observation window has zero length")

    meta = {"path": filepath,
            "sessionize_gap": str(sessionize_gap),
            "max_time": str(max_time)}
    return ContactData(N=N, contacts=contacts, node_map=node_map, T=T,
                       n_raw_samples=n_raw, metadata=meta)


def _sessionize(samples: Sequence[Contact], gap: float) -> List[Contact]:
    """Collapse runs of samples on the same dyad separated by less than `gap`."""
    if gap <= 0.0:
        raise ValueError(f"sessionize_gap must be positive, got {gap}")
    last_seen: Dict[Tuple[int, int], float] = {}
    out: List[Contact] = []
    for t, i, j in samples:
        key = (i, j)
        prev = last_seen.get(key)
        if prev is None or (t - prev) > gap:
            out.append((t, i, j))
        last_seen[key] = t
    out.sort(key=lambda r: r[0])
    return out


def restrict_to_active_nodes(data: ContactData, min_contacts: int = 5) -> ContactData:
    """Drop nodes with fewer than `min_contacts` contacts and reindex.

    Nodes that appear in almost no contacts contribute essentially nothing to the
    likelihood but do enter the partner-choice normaliser, so keeping a long tail of
    near-inactive nodes shifts kappa. Removing them is a modelling decision, so it is an
    explicit call rather than something the loader does silently.
    """
    counts: Dict[int, int] = {}
    for _t, i, j in data.contacts:
        counts[i] = counts.get(i, 0) + 1
        counts[j] = counts.get(j, 0) + 1
    keep = sorted(k for k, c in counts.items() if c >= min_contacts)
    if len(keep) < 2:
        raise ValueError(f"min_contacts={min_contacts} leaves fewer than 2 nodes")
    remap = {old: new for new, old in enumerate(keep)}
    contacts = [(t, remap[i], remap[j]) for t, i, j in data.contacts
                if i in remap and j in remap]
    if len(contacts) < 10:
        raise ValueError(f"min_contacts={min_contacts} leaves only {len(contacts)} contacts")
    inv = {v: k for k, v in data.node_map.items()}
    node_map = {inv[old]: new for old, new in remap.items()}
    return ContactData(N=len(keep), contacts=contacts, node_map=node_map,
                       T=data.T, n_raw_samples=data.n_raw_samples,
                       metadata={**data.metadata, "min_contacts": str(min_contacts)})


def split_by_window(data: ContactData, window: float,
                    min_contacts: int = 0) -> List[ContactData]:
    """Split into windows of `window` time units, each re-zeroed AND re-filtered.

    `min_contacts > 0` re-applies the activity filter *within each window*. This is not
    cosmetic. Filtering globally and then splitting leaves every window carrying the
    whole record's node set, so nodes with no events in that window still enter the
    partner-choice normaliser Z_i and inflate it. On email-Eu that meant fitting 929
    contacts against N = 876 nodes -- 2.1 events per node -- and kappa came out at 89
    and 141 on consecutive days, which is what an inflated denominator does. Filter
    per window.
    """
    if window <= 0.0:
        raise ValueError(f"window must be positive, got {window}")
    buckets: Dict[int, List[Contact]] = {}
    for t, i, j in data.contacts:
        buckets.setdefault(int(t // window), []).append((t, i, j))
    out: List[ContactData] = []
    for b in sorted(buckets):
        block = buckets[b]
        t0 = block[0][0]
        shifted = [(t - t0, i, j) for t, i, j in block]
        span = shifted[-1][0]
        if span <= 0 or len(shifted) < 10:
            continue
        wd = ContactData(N=data.N, contacts=shifted, node_map=data.node_map,
                         T=span, n_raw_samples=len(block),
                         metadata={**data.metadata, "window": str(b)})
        if min_contacts > 0:
            try:
                wd = restrict_to_active_nodes(wd, min_contacts=min_contacts)
            except ValueError:
                continue
        out.append(wd)
    if not out:
        raise ValueError("no window contained enough contacts to fit")
    return out


def split_by_day(data: ContactData, day_seconds: float = 86400.0) -> List[ContactData]:
    """Split into per-day records, each re-zeroed in time.

    Human contact data has a hard daily cycle: fitting a memory kernel across an
    overnight gap makes the kernel try to explain the gap rather than the memory.
    Fitting days separately and pooling the estimates avoids that.
    """
    if day_seconds <= 0.0:
        raise ValueError(f"day_seconds must be positive, got {day_seconds}")
    buckets: Dict[int, List[Contact]] = {}
    for t, i, j in data.contacts:
        buckets.setdefault(int(t // day_seconds), []).append((t, i, j))
    out: List[ContactData] = []
    for d in sorted(buckets):
        block = buckets[d]
        t0 = block[0][0]
        shifted = [(t - t0, i, j) for t, i, j in block]
        T = shifted[-1][0]
        if T <= 0 or len(shifted) < 10:
            continue
        out.append(ContactData(N=data.N, contacts=shifted, node_map=data.node_map,
                               T=T, n_raw_samples=len(block),
                               metadata={**data.metadata, "day": str(d)}))
    if not out:
        raise ValueError("no day contained enough contacts to fit")
    return out

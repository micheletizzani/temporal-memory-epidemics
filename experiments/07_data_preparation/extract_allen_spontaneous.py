"""Extract spike trains of well-isolated units in the longest spontaneous block of the cached
Allen Brain Observatory Neuropixels session (no stimulus, so the trains are stationary).

Replaces, for the node-timing statistics, the coincidence-constructed events of
download_allen_brain.py, whose first 6e4 events cover only ~5 s (~1.6 spikes per unit per 4000-
event window). Needs h5py only (run with the 01_identifiability_AvsB venv).
Writes data/allen_spontaneous_spikes.npz:
  times[k], unit[k] (0-indexed, time-sorted), unit_ids, block=(start, stop).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import h5py
import numpy as np

NWB = os.path.join(ROOT, "allen_cache", "session_715093703", "session_715093703.nwb")
OUT = os.path.join(ROOT, "data", "allen_spontaneous_spikes.npz")
MIN_SPIKES = 200

with h5py.File(NWB, "r") as f:
    sp = f["intervals"]["spontaneous_presentations"]
    starts, stops = sp["start_time"][:], sp["stop_time"][:]
    k = int(np.argmax(stops - starts)); a, b = float(starts[k]), float(stops[k])
    u = f["units"]
    idx = u["spike_times_index"][:]; st = u["spike_times"]
    good = u["quality"][:] == b"good"; ids = u["id"][:]
    T, U, kept = [], [], []
    lo = 0
    for n, hi in enumerate(idx):
        if good[n]:
            s = st[lo:hi]; s = s[(s >= a) & (s < b)]
            if len(s) >= MIN_SPIKES:
                T.append(s - a); U.append(np.full(len(s), len(kept))); kept.append(int(ids[n]))
        lo = hi
t = np.concatenate(T); un = np.concatenate(U); o = np.argsort(t, kind="stable")
np.savez_compressed(OUT, times=t[o], unit=un[o], unit_ids=np.array(kept), block=np.array([a, b]))
print(f"block [{a:.1f}, {b:.1f}] s: {len(kept)} good units with >= {MIN_SPIKES} spikes, {len(t)} spikes")

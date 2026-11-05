"""
To run this script, you must first install the Allen SDK:
    pip install allensdk
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import os
import numpy as np

try:
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
except ImportError:
    print("Please install allensdk to run this script: pip install allensdk")
    exit(1)

def download_neuropixels_sample(out_file="allen_brain_network.txt"):
    # Set up cache directory
    manifest_path = os.path.join("allen_cache", "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    
    print("Fetching session list...")
    sessions = cache.get_session_table()
    # Grab the first session
    session_id = sessions.index.values[0]
    
    print(f"Downloading session {session_id} (this may take a while and ~2GB of space)...")
    session = cache.get_session_data(session_id)
    
    print("Extracting spike times...")
    spikes = session.spike_times
    
    # We need a temporal network (t, i, j). Neuropixels only gives (t, i) natively.
    # To get (t, i, j), we would need to compute functional connectivity (e.g. cross-correlograms)
    # For this sample, we construct a dummy connectivity network just to format the data
    # linking neurons that fire sequentially within 5ms.
    
    print("Building naive functional connectivity network from sequential spikes...")
    # Sort all spikes chronologically
    all_spikes = []
    for unit_id, unit_spikes in spikes.items():
        # Subsample to keep it fast
        for t in unit_spikes[:1000]:
            all_spikes.append((t, unit_id))
            
    all_spikes.sort(key=lambda x: x[0])
    
    events = []
    # Create (t, i, j) if i spikes and j spikes within 5ms
    for idx in range(len(all_spikes) - 1):
        t_i, unit_i = all_spikes[idx]
        
        # Check next few spikes
        lookahead = 1
        while idx + lookahead < len(all_spikes):
            t_j, unit_j = all_spikes[idx + lookahead]
            if t_j - t_i > 0.005: # 5ms window
                break
            if unit_i != unit_j:
                events.append((t_i, unit_i, unit_j))
            lookahead += 1
            
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, 'w') as f:
        for ev in events:
            f.write(f"{ev[0]:.4f} {ev[1]} {ev[2]}\n")
            
    print(f"Saved {len(events)} inferred functional events to {out_file}")

if __name__ == "__main__":
    download_neuropixels_sample(out_file="data/allen_brain_network.txt")

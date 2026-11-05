import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from paths import ROOT  # repository root; also puts experiments/* on sys.path
import numpy as np
import os
import argparse

def simulate_izhikevich(N_e=40, N_i=10, p_connect=0.2, T_ms=10000, out_file="snn_refractory_network.txt"):
    """
    Simulates a small Izhikevich spiking neural network and logs events as a directed 
    temporal network (t, i, j) where i is the presynaptic neuron and j is the postsynaptic target.
    This generates extremely bursty dynamics with strict timescale separation.
    """
    N = N_e + N_i
    
    # Izhikevich parameters
    re = np.random.rand(N_e)
    ri = np.random.rand(N_i)
    
    a = np.concatenate((0.02 * np.ones(N_e), 0.02 + 0.08 * ri))
    b = np.concatenate((0.2 * np.ones(N_e), 0.25 - 0.05 * ri))
    c = np.concatenate((-65 + 15 * re**2, -65 * np.ones(N_i)))
    d = np.concatenate((8 - 6 * re**2, 2 * np.ones(N_i)))
    
    # Structural connectivity matrix
    S = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and np.random.rand() < p_connect:
                # Excitatory weights are positive, inhibitory are much stronger to prevent bursts
                weight = 0.5 * np.random.rand() if i < N_e else -5.0 * np.random.rand()
                S[j, i] = weight # S[j, i] is weight from i to j
    
    v = -65 * np.ones(N)
    u = b * v
    
    events = []
    
    print(f"Simulating {N} neurons for {T_ms} ms in AI (Asynchronous Irregular) regime...")
    for t in range(T_ms):
        # Background thalamic input (strong independent noise to break synchronization)
        # We need enough tonic drive to keep them firing, plus noise
        I_base = np.concatenate((10 * np.ones(N_e), 5 * np.ones(N_i)))
        I_noise = np.concatenate((10 * np.random.randn(N_e), 5 * np.random.randn(N_i)))
        I = I_base + I_noise
        
        # Identify spiking neurons
        fired = np.where(v >= 30)[0]
        
        for i in fired:
            # For each firing neuron, record its spikes transmitted to structurally connected targets
            # Find postsynaptic targets (j where S[j, i] != 0)
            targets = np.where(S[:, i] != 0)[0]
            for j in targets:
                # Append (t, presynaptic, postsynaptic)
                events.append((t, i, j))
                
        # Reset firing neurons
        if len(fired) > 0:
            v[fired] = c[fired]
            u[fired] += d[fired]
            # Add synaptic inputs to targets
            I += np.sum(S[:, fired], axis=1)
            
        # Update voltages (Euler method, half-step for stability)
        v = v + 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
        v = v + 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
        u = u + a * (b * v - u)
        
    print(f"Simulation complete. Generated {len(events)} directed transmission events.")
    
    # Write out temporal network
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, 'w') as f:
        for ev in events:
            f.write(f"{ev[0]} {ev[1]} {ev[2]}\n")
            
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=50, help="Total number of neurons")
    parser.add_argument("--T", type=int, default=100000, help="Simulation time in ms")
    parser.add_argument("--out", type=str, default="data/snn_refractory_network.txt")
    args = parser.parse_args()
    
    N_e = int(args.N * 0.8)
    N_i = args.N - N_e
    
    simulate_izhikevich(N_e=N_e, N_i=N_i, T_ms=args.T, out_file=args.out)

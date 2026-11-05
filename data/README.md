# Data

The raw records are public but not redistributed here. Put each file in this folder under the name given below. Figures and numbers can be rebuilt from `results/` without them.

| Record (paper name) | File(s) expected here | Source |
|---|---|---|
| High school | `HighSchool2013_proximity_net.csv` | SocioPatterns, <http://www.sociopatterns.org/datasets/> |
| Conference (SFHH) | `SFHH_tij.dat` | SocioPatterns |
| Hospital ward | `hospital_lyon_contacts.dat` | SocioPatterns |
| Baboons | `baboons_proximity_data.txt` | SocioPatterns, <http://www.sociopatterns.org/datasets/baboons-interactions/> |
| Email | `email-Eu-core-temporal.txt` | SNAP, <https://snap.stanford.edu/data/>; analyses use the first 10^4 messages |
| CollegeMsg | `CollegeMsg.txt` | SNAP |
| Stack Overflow | `so_core_events.txt` | SNAP `sx-stackoverflow`, see below |
| Wikibooks | `wikibooks_core_events.txt` | KONECT `edit-enwikibooks`, <http://konect.cc/networks/>, obtained through the Network Repository, <http://networkrepository.com/>; see below |
| Wikibooks bots | `wikimedia/wikibooks_bots_events.txt` | Wikimedia dump, <https://dumps.wikimedia.org/enwikibooks/>, then `experiments/07_data_preparation/build_wikibooks_bots.py` |
| Microservices | `alibaba_microservices_sample.txt` | Alibaba trace `MSCallGraph_0`, <https://github.com/alibaba/clusterdata>; see below |
| Spiking network | `snn_bursty_network.txt`, `snn_refractory_network.txt` | Simulated; see below |
| Cortex | `allen_spontaneous_spikes.npz` | Allen Brain Observatory, <https://portal.brain-map.org/>, through `download_allen_brain.py` and `extract_allen_spontaneous.py` |

## Active cores

```bash
# Wikibooks: editors with >= 80 edits and pages with >= 40
python experiments/07_data_preparation/build_core_subgraphs.py wikibooks data/edit-enwikibooks.edges

# Stack Overflow: 400 random users with >= 1000 non-self interactions, then both endpoints >= 40 events
python experiments/07_data_preparation/preprocess_directed.py data/sx-stackoverflow.txt.gz \
    --icol 0 --jcol 1 --tcol 2 --min-events 1000 --cap-nodes 400 --tag stackoverflow
python experiments/07_data_preparation/build_core_subgraphs.py stackoverflow data/stackoverflow_events.txt
```

The Wikibooks command reproduces the analysed record. The first Stack Overflow command applies the same rule but draws a different set of 400 users from the one analysed, so it gives a comparable record, not the same one.

## Records that cannot be regenerated exactly

- **Microservices.** A 2×10^5-call sample (about 3%) of `MSCallGraph_0`. The sampling script and seed were not kept. Every sampled call was traced back to the raw trace.
- **Spiking network, refractory regime.** `simulate_snn.py` (100 s at 1 ms) has this regime's parameters but no random seed, so it gives a statistically equivalent network: spike count and memory coefficient match (M = −0.285 against −0.277).
- **Spiking network, bursting regime.** The parameter set of this run was not kept.

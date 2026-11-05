# Data

The raw records are public but not redistributed here. Put each file in this folder under the name given below. Figures and numbers can be rebuilt from `results/` without them.

| Record (paper name) | File(s) expected here | Source |
|---|---|---|
| High school | `HighSchool2013_proximity_net.csv` | SocioPatterns, <http://www.sociopatterns.org/datasets/> |
| Conference (SFHH) | `SFHH_tij.dat` | SocioPatterns |
| Hospital ward | `hospital_lyon_contacts.dat` | SocioPatterns |
| Baboons | `baboons_proximity_data.txt` | Zenodo, <https://doi.org/10.5281/zenodo.3820287> |
| Email | `email-Eu-core-temporal.txt` | SNAP, <https://snap.stanford.edu/data/> |
| CollegeMsg | `CollegeMsg.txt` | SNAP |
| Stack Overflow | `so_core_events.txt` | SNAP `sx-stackoverflow`, then `experiments/07_data_preparation/preprocess_directed.py` |
| Wikibooks | `wikibooks_core_events.txt` | KONECT `edit-enwikibooks`, <http://konect.cc/networks/>, active core built in the same way |
| Wikibooks bots | `wikimedia/wikibooks_bots_events.txt` | Wikimedia dump, <https://dumps.wikimedia.org/enwikibooks/>, then `experiments/07_data_preparation/build_wikibooks_bots.py` |
| Microservices | `alibaba_microservices_sample.txt` | Alibaba cluster trace, <https://github.com/alibaba/clusterdata>, a 2×10^5-call sample of `MSCallGraph_0` |
| Spiking network | `snn_bursty_network.txt`, `snn_refractory_network.txt` | `experiments/07_data_preparation/simulate_snn.py` |
| Cortex | `allen_spontaneous_spikes.npz` | Allen Brain Observatory, <https://portal.brain-map.org/>, through `download_allen_brain.py` and `extract_allen_spontaneous.py` |

Stack Overflow core (Sec. IV E of the paper: 400 users drawn at random among those with at least 10 interactions, keeping all their outgoing interactions):

```bash
python experiments/07_data_preparation/preprocess_directed.py data/sx-stackoverflow.txt.gz \
    --icol 0 --jcol 1 --tcol 2 --min-events 10 --cap-nodes 400 --tag so_core
```

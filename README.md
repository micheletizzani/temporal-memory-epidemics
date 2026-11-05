# Fundamental limits to identifying node and tie memory in temporal networks

Code for the paper

> M. Tizzani, *Fundamental limits to identifying node and tie memory in temporal networks: marginal artefacts and spreading dynamics*, Physical Review Research (accepted, 2026). DOI: [10.1103/n24h-kkc6](https://doi.org/10.1103/n24h-kkc6). Preprint: [arXiv:2609.10399](https://doi.org/10.48550/arXiv.2609.10399).

The model places memory on nodes (self-excitation, branching ratio `n_node`) or on ties (reinforcement, `kappa`). The code fits both from contact records, tests the fits, measures what the fitted `n_node` reads, and simulates spreading on the results.

## Quick start

```bash
pip install -r requirements.txt
python demo.py                 # about 10 s, no data: reproduces Eqs. (1)-(2) of the paper
pytest tests                   # unit tests of the likelihood and estimators
bash reproduce_figures.sh      # rebuilds paper figures from the saved results in results/
```

Run every script from the repository root. Python 3.9 or later; Julia 1.10 for the scripts in `julia/` (`julia --project=. -e 'using Pkg; Pkg.instantiate()'`).

## Layout

| Folder | Contents |
|---|---|
| `core/` | The model (`generative_models.py`), the estimators (`inference.py`), the data loader, the memory coefficient (`memory_stats.py`) and the figure style |
| `experiments/` | One folder per part of the paper; see the map below |
| `julia/` | Julia port of the model, the large sweeps, the label smoother and the at-scale epidemics |
| `results/` | Saved output of the analyses, used to rebuild figures and numbers without rerunning |
| `figures/` | The figures as published |
| `generated/` | LaTeX macros and tables generated from `results/` |
| `data/` | Empty on GitHub; `data/README.md` says where each record comes from |
| `tests/` | Unit tests and the Python/Julia parity fixture |

## Map from the paper to the code

`S` numbers refer to the Supplemental Material. Paths are relative to `experiments/` unless they start with `julia/`.

### Main text

| Paper section | Result | Scripts |
|---|---|---|
| Sec. II A | Recovery with observed initiators, Eq. (1) | `02_synthetic_recovery/run_synthetic_recovery.py` |
| Sec. II B | Cost of the missing label, Eq. (2) | `02_synthetic_recovery/run_synthetic_recovery.py`, `03_latent_initiator/mwg_synthetic_driver.py` |
| Sec. II C | Coverage sweep and hierarchical prior | `02_synthetic_recovery/run_identifiability_sweep.py`, `julia/run_identifiability_sweep.jl`, `02_synthetic_recovery/coverage_shrinkage.py` |
| Sec. II D | Empirical estimates | `04_empirical_fits_gof/record_fits.py`, `record_perday.py` |
| Sec. II F | Goodness of fit, repeat test | `04_empirical_fits_gof/record_gof.py`, `gof_bootstrap.py`, `group_gof.py`, `collegemsg_selection_diagnostics.py` |
| Sec. II F | Inflation of `kappa` under a latent initiator | `03_latent_initiator/confound_2x2.py` |
| Sec. II G | Shuffle ratio and memory coefficient of the records | `05_node_memory/real_record_screen.py`, `core/memory_stats.py` |
| Sec. II G | Calibration on controlled memory | `05_node_memory/phase_scaling.py`, `phase_combine.py`, `fss_numbers.py` |
| Sec. II G | Phase space and crossover fit | `05_node_memory/hawkes_phase_space.py`, `plane_validation.py`, `record_phase_position.py` |
| Sec. II G | Controls matched to each record | `05_node_memory/record_matched_surrogate.py` |
| Sec. II G | Positive control | `05_node_memory/semi_synthetic.py`, `poscon_shuffle_R.py`, `poscon_memory_rank.py` |
| Sec. II H | Pure poles | `06_spreading/epidemic_divergence.py` |
| Sec. II H | Matched surrogates | `06_spreading/matched_control_epidemics.py` |
| Sec. II H | Forecast from the fitted model | `06_spreading/fitted_vs_empirical.py` |
| Sec. II H | At scale | `julia/epidemic_poles.jl`, `06_spreading/at_scale_numbers.py`, `fss_analysis.py` |
| Sec. III | Sequential imputation bias | `03_latent_initiator/estep_bias_directed.py`, `mcmc_estep.py` |
| Sec. III | Sum-of-exponentials kernel | `04_empirical_fits_gof/sum_exp_kernel.py` |
| Sec. IV A–D | Model, likelihood, estimators | `core/generative_models.py`, `core/inference.py` |
| Sec. IV D | Python/Julia parity | `tests/parity_fixture.json`, `julia/check_parity.jl` |
| Sec. IV E | Datasets | `07_data_preparation/`, `data/README.md` |

### Figures

| Figure | File in `figures/` | Built by | From |
|---|---|---|---|
| Fig. 1 | `figure1_theory` | `01_model_and_separation/build_fig_theory.py` | schematic |
| Fig. 2 | `confound_orderparam` | `03_latent_initiator/build_fig_confound.py` | `confound_theory.py` |
| Fig. 3 | `fig_identifiability` | `02_synthetic_recovery/fig_identifiability.py` | sweep, `coverage_shrinkage.py`, `profile_envelope.py`, `record_fits.py` |
| Fig. 4 | `fig1_identity` | `04_empirical_fits_gof/identity_from_tau.py` | `record_gof.py`, `cache_tau*.py` |
| Fig. 5 | `fig_finitesize` | `05_node_memory/fig_finitesize.py` | `phase_scaling.py`, `real_record_screen.py`, positive control |
| Fig. 6 | `plane_with_snn` | `05_node_memory/plane_validation.py` | `hawkes_phase_space.py`, `record_phase_position.py` |
| Fig. 7 | `epidemic_poles` | `06_spreading/build_fig_epidemics.py` | `epidemic_divergence.py`, `julia/epidemic_poles.jl` |
| Fig. 8 | `dynamics` | `01_model_and_separation/dynamics_figure.py` | simulation in the script |
| Fig. S1 | `sm_distributions` | `04_empirical_fits_gof/make_distributions.py` | raw data |
| Fig. S2 | `qq` | `04_empirical_fits_gof/qq_combine.py` | `record_gof.py`, `cache_tau*.py` |
| Fig. S3 | `semi_synthetic` | `05_node_memory/semi_synthetic.py` | positive control |
| Fig. S4 | `plane_with_snn` | `05_node_memory/plane_validation.py` | as Fig. 6 |
| Fig. S5 | `social_contagion` | `06_spreading/social_contagion.py` | simulation in the script |
| Fig. S6 | `mark_confound` | `01_model_and_separation/build_fig_mark.py` | `mark_confound.py` |
| Fig. S7 | `ridge_epidemic` | `06_spreading/ridge_epidemic.py` | simulation in the script |
| Fig. S8 | `fitted_vs_empirical`, `fitted_vs_empirical_baboons` | `06_spreading/fitted_vs_empirical.py` | run once per record |
| Fig. S9 | `order_information` | `05_node_memory/build_fig_order_information.py` | `order_information_fraction.py` |
| Fig. S10 | `kernel_families` | `05_node_memory/build_fig_kernel_families.py` | `kernel_families.py` |
| Fig. S11 | `dynamics_poles` | `06_spreading/build_fig_dynamics.py` | `julia/epidemic_poles.jl` |
| Fig. S12 | `fss_sis_node` | `06_spreading/build_fig_fss.py` | `julia/epidemic_poles.jl` |
| Fig. S13 | `confound2x2_real` | `03_latent_initiator/build_fig_confound2x2.py` | `confound_2x2.py` |

### Supplemental Material

| Section | Topic | Scripts |
|---|---|---|
| S1 | Datasets, per-node statistics | `04_empirical_fits_gof/make_distributions.py`, `07_data_preparation/` |
| S2 | Goodness of fit, repeat test | `04_empirical_fits_gof/record_gof.py`, `gof_bootstrap.py`, `collegemsg_selection_diagnostics.py` |
| S3 | Robustness: E-step, smoother, group encoding, boundary null, prior, multi-lag memory, bipartite check | `03_latent_initiator/estep_bias_directed.py`, `mwg_full_driver.py`, `julia/mwg_full.jl`, `04_empirical_fits_gof/group_gof.py`, `group_collapse_check.py`, `02_synthetic_recovery/boundary_null.py`, `coverage_shrinkage.py`, `05_node_memory/memory_multilag.py`, `bipartite_memory_check.py` |
| S4 | Sum-of-exponentials kernel | `04_empirical_fits_gof/sum_exp_kernel.py` |
| S5 | Positive control | `05_node_memory/semi_synthetic.py`, `poscon_shuffle_R.py`, `poscon_memory_rank.py` |
| S6 | Finite-size scaling of `R(M)` | `05_node_memory/phase_scaling.py`, `fss_numbers.py` |
| S7 | Cortex, spiking network, phase space | `05_node_memory/screen_allen_spontaneous.py`, `screen_allen_local_shuffle.py`, `cortex_rank_memory.py`, `screen_neural.py`, `hawkes_phase_space.py` |
| S8 | Group memory and social contagion | `06_spreading/social_contagion.py` |
| S9 | Proofs; numerical test of weight attribution | `01_model_and_separation/mark_confound.py` |
| S10 | Spreading cost, matched controls | `06_spreading/ridge_epidemic.py`, `matched_control_epidemics.py` |
| S11 | Forecast error, recovery rate | `06_spreading/fitted_vs_empirical.py`, `julia/sweep_mu_sensitivity.jl` |
| S12 | Kernel families, scale-free estimator | `05_node_memory/kernel_families.py`, `kernel_phase.py`, `scalefree_screen_hardened.py`, `order_information_fraction.py`, `required_events.py` |
| S13 | SIS and SIR at scale | `julia/epidemic_poles.jl`, `06_spreading/at_scale_numbers.py`, `fss_analysis.py` |
| S14 | Confound as missing information | `03_latent_initiator/confound_theory.py`, `confound_2x2.py` |
| S15 | Memory coefficients, matched controls | `05_node_memory/memory_artifact_check.py`, `memory_modulation_check.py`, `memory_tie_check.py`, `record_matched_surrogate.py` |

### Tables

| Table | Produced by |
|---|---|
| Table I | Collects the spreading experiments of Sec. II H; see the scripts above |
| Table S1 | `07_data_preparation/`, `data/README.md` |
| Table S2 | `04_empirical_fits_gof/record_gof.py`, `gof_bootstrap.py`; written by `08_generated_numbers/make_text_macros.py` |
| Table S3 | `04_empirical_fits_gof/collegemsg_selection_diagnostics.py` |
| Table S4 | `03_latent_initiator/estep_bias_directed.py` |
| Tables S5, S6 | `02_synthetic_recovery/run_identifiability_sweep.py`, `coverage_shrinkage.py` |
| Table S7 | `05_node_memory/memory_multilag.py` |
| Table S8 | `04_empirical_fits_gof/sum_exp_kernel.py` |
| Table S9 | `06_spreading/matched_control_epidemics.py` |
| Table S10 | `06_spreading/fitted_vs_empirical.py`; written by `make_text_macros.py` |
| Table S11 | `05_node_memory/required_events.py` |
| Table S12 | `05_node_memory/scalefree_screen_hardened.py` |
| Table S13 | `memory_artifact_check.py`, `memory_modulation_check.py`, `memory_tie_check.py`; written by `make_memory_tables.py` |
| Table S14 | `05_node_memory/record_matched_surrogate.py`; written by `make_memory_tables.py` |

## Numbers in the text

`08_generated_numbers/make_text_macros.py` and `make_memory_tables.py` read `results/` and write the LaTeX macros and tables in `generated/`. `record_macros.py` produces the values of the per-record fits.

## Rerunning the analyses

Scripts that fit records need the raw data in `data/`; see `data/README.md`. Fits take from seconds to hours. A few take arguments, for example `record_matched_surrogate.py "Hospital ward"` and `gof_bootstrap.py "Hospital ward" 100`; the parameters of the synthetic realisation in Sec. II A are the script's defaults with

```bash
python experiments/02_synthetic_recovery/run_synthetic_recovery.py --N 40 --T 600 --n-node 0.5 --gamma-n 0.5 --kappa 20 --gamma-e 0.05 --seed 7
```

After rerunning `real_record_screen.py`, run `real_record_screen.py --refresh-memory` and `cortex_rank_memory.py` so that the record uses the within-node rank coefficient of the paper. Rerunning writes to `results/`, `figures/` and `generated/`.

## Citation

See `CITATION.cff`. Code is released under the MIT License. The data belong to their original authors and are not redistributed here.

#!/usr/bin/env bash
# Rebuild the paper's figures from results/, without raw data. Run from the repository root.
set -euo pipefail
E_=experiments
python $E_/01_model_and_separation/build_fig_theory.py          # Fig. 1
python $E_/03_latent_initiator/build_fig_confound.py            # Fig. 2
python $E_/02_synthetic_recovery/fig_identifiability.py         # Fig. 3
python $E_/04_empirical_fits_gof/identity_from_tau.py           # Fig. 4
python $E_/05_node_memory/fig_finitesize.py                     # Fig. 5
python $E_/05_node_memory/plane_validation.py                   # Figs. 6, S4
python $E_/06_spreading/build_fig_epidemics.py                  # Fig. 7
python $E_/01_model_and_separation/dynamics_figure.py           # Fig. 8
python $E_/04_empirical_fits_gof/qq_combine.py                  # Fig. S2
python $E_/06_spreading/fitted_vs_empirical.py --plot-only      # Fig. S8
python $E_/01_model_and_separation/build_fig_mark.py            # Fig. S6
python $E_/05_node_memory/build_fig_order_information.py        # Fig. S9
python $E_/05_node_memory/build_fig_kernel_families.py          # Fig. S10
python $E_/06_spreading/build_fig_dynamics.py                   # Fig. S11
python $E_/06_spreading/build_fig_fss.py                        # Fig. S12
python $E_/03_latent_initiator/build_fig_confound2x2.py         # Fig. S13
# Numbers and tables in generated/
python $E_/08_generated_numbers/make_text_macros.py
python $E_/08_generated_numbers/make_memory_tables.py

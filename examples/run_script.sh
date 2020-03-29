#!/usr/bin/env bash

python3 0_generate_graph.py $1
python3 1_scan_k.py $1
python3 2_compute_passive_modes.py $1
python3 3_plot_modes.py $1
python3 4_compute_trajectories.py $1
python3 5_compute_lasing_thresholds.py $1
python3 6_plot_threshold_modes.py $1
python3 7_mode_competition.py $1

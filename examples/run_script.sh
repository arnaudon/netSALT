#!/usr/bin/env bash
rm -rf $1

echo Step 0
python3 0_generate_graph.py $1
echo Step 1
python3 1_scan_k.py $1
echo Step 2
python3 2_compute_passive_modes.py $1
echo Step 3
python3 3_plot_modes.py $1
echo Step 4
python3 4_compute_trajectories.py $1
echo Step 5
python3 5_compute_lasing_thresholds.py $1
echo Step 6
python3 6_plot_threshold_modes.py $1
echo Step 7
python3 7_mode_competition_matrix.py $1
echo Step 8
python3 8_mode_competition.py $1
echo Step 9
python3 9_all_plots.py $1

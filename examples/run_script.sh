#!/usr/bin/env bash
python3 /home/dsaxena/NAQ-graphs/examples/0_generate_graph.py line_PRA
python3 /home/dsaxena/NAQ-graphs/examples/1_scan_k.py line_PRA
python3 /home/dsaxena/NAQ-graphs/examples/2_compute_passive_modes.py line_PRA
python3 /home/dsaxena/NAQ-graphs/examples/3_plot_modes.py line_PRA
python3 /home/dsaxena/NAQ-graphs/examples/4_compute_trajectories.py line_PRA
python3 /home/dsaxena/NAQ-graphs/examples/5_compute_lasing_thresholds.py line_PRA
python3 /home/dsaxena/NAQ-graphs/examples/6_plot_threshold_modes.py line_PRA
python3 /home/dsaxena/NAQ-graphs/examples/7_mode_competition.py line_PRA
bash
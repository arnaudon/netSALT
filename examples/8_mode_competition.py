import os
import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

import naq_graphs as naq
from graph_generator import generate_graph
from naq_graphs import plotting

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

os.chdir(graph_tpe)

graph = naq.load_graph()
naq.update_parameters(graph, params)

modes_df = naq.load_modes()

mode_competition_matrix = naq.load_mode_competition_matrix()

D0_max = 8.0 * np.min(modes_df["lasing_thresholds"])
D0_min = 0.8 * np.min(modes_df["lasing_thresholds"])
n_points = 5000
pump_intensities = np.linspace(D0_min, D0_max, n_points)

modes_df = naq.compute_modal_intensities(
    modes_df, pump_intensities, mode_competition_matrix
)

naq.save_modes(modes_df)

plotting.plot_ll_curve(graph, modes_df, with_legend=False)

plotting.plot_stem_spectra(graph, modes_df, -1)

plt.show()

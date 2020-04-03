import os
import pickle as pickle
import sys

import matplotlib.pyplot as plt
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

positions = [graph.nodes[u]["position"] for u in graph]

modes_df = naq.load_modes()

modes_df = naq.find_threshold_lasing_modes(modes_df, graph)

naq.save_modes(modes_df)

qualities = pickle.load(open("scan.pkl", "rb"))

plotting.plot_scan(graph, qualities, modes_df)
plotting.plot_pump_traj(modes_df)

plt.scatter(
    np.real(modes_df["threshold_lasing_modes"].to_numpy()),
    -np.imag(modes_df["threshold_lasing_modes"].to_numpy()),
    c="m",
)

plt.savefig("mode_trajectories.png")
plt.show()

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

passive_modes = naq.load_modes()

modes_trajectories, modes_trajectories_approx = pickle.load(
    open("trajectories.pkl", "rb")
)

threshold_lasing_modes, lasing_thresholds = naq.find_threshold_lasing_modes(
    passive_modes, graph, n_workers=params["n_workers"], threshold=1e-5,
)

naq.save_modes(threshold_lasing_modes, lasing_thresholds, filename="threshold_modes")

print("threshold modes", threshold_lasing_modes)
print("non interacting thresholds", lasing_thresholds)

qualities = pickle.load(open("scan.pkl", "rb"))
plotting.plot_scan(graph, qualities, passive_modes)
plotting.plot_pump_traj(passive_modes, modes_trajectories, modes_trajectories_approx)
plt.scatter(
    np.array(threshold_lasing_modes)[:, 0],
    np.array(threshold_lasing_modes)[:, 1],
    c="m",
)
plt.savefig("mode_trajectories.png")
plt.show()

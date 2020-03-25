import os
import sys

import pickle as pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt

from graph_generator import generate_graph

from naq_graphs import set_dielectric_constant, set_dispersion_relation
from naq_graphs.dispersion_relations import dispersion_relation_pump
from naq_graphs import (
    create_naq_graph,
    find_threshold_lasing_modes,
    pump_trajectories,
    load_modes,
    save_modes,
)
from naq_graphs.plotting import plot_pump_traj, plot_scan
from naq_graphs.io import load_graph

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

os.chdir(graph_tpe)

graph, params = load_graph()
positions = [graph.nodes[u]["position"] for u in graph]

modes = load_modes()

#set pump profile for PRA example
if graph_tpe == 'line_PRA' and params["dielectric_params"]["method"] == "custom":
    pump_edges = round(len(graph.edges())/2)
    nopump_edges = len(graph.edges())-pump_edges
    params["pump"] = np.append(np.ones(pump_edges),np.zeros(nopump_edges))
    params["pump"][0] = 0 #first edge is outside
else:
    #params["pump"] = np.ones(len(graph.edges())) # uniform pump on ALL edges 
    params["pump"] = np.zeros(len(graph.edges())) # uniform pump on inner edges 
    for i, (u,v) in enumerate(graph.edges()): 
        if graph[u][v]["inner"]:
            params["pump"][i] = 1


D0s = np.linspace(0, params["D0_max"], params["D0_steps"])

modes_trajectories, modes_trajectories_approx = pickle.load(open("trajectories.pkl", "rb"))

threshold_lasing_modes, lasing_thresholds = find_threshold_lasing_modes(
    modes,
    graph,
    params,
    D0_max=D0s[-1],
    D0_steps=D0s[1] - D0s[0],
    n_workers=params['n_workers'],
    threshold=1e-5,
)

save_modes(threshold_lasing_modes, lasing_thresholds, filename="threshold_modes")

print("threshold modes", threshold_lasing_modes)
print("non interacting thresholds", lasing_thresholds)

ks, alphas, qualities = pickle.load(open("scan.pkl", "rb"))
plot_scan(ks, alphas, qualities, modes)
plot_pump_traj(modes, modes_trajectories, modes_trajectories_approx)
plt.scatter(
    np.array(threshold_lasing_modes)[:, 0],
    np.array(threshold_lasing_modes)[:, 1],
    c="m",
)
plt.savefig("mode_trajectories.png")
plt.show()

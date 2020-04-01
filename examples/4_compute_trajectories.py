import os
import sys

import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt

import naq_graphs as naq

from naq_graphs import plotting

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

os.chdir(graph_tpe)

graph, params = naq.load_graph()

positions = [graph.nodes[u]["position"] for u in graph]

passive_modes = naq.load_modes()

# set pump profile for PRA example
if graph_tpe == "line_PRA" and params["dielectric_params"]["method"] == "custom":
    pump_edges = round(len(graph.edges()) / 2)
    nopump_edges = len(graph.edges()) - pump_edges
    params["pump"] = np.append(np.ones(pump_edges), np.zeros(nopump_edges))
    params["pump"][0] = 0  # first edge is outside
else:
    params["pump"] = np.zeros(len(graph.edges()))  # uniform pump on inner edges
    for i, (u, v) in enumerate(graph.edges()):
        if graph[u][v]["inner"]:
            params["pump"][i] = 1

D0s = np.linspace(0, params["D0_max"], params["D0_steps"])
modes_trajectories, modes_trajectories_approx = naq.pump_trajectories(
    passive_modes, graph, params, D0s, return_approx=True
)

naq.save_modes(modes_trajectories, modes_trajectories_approx, filename="trajectories")

ks, alphas, qualities = pickle.load(open("scan.pkl", "rb"))
plotting.plot_scan(ks, alphas, qualities, passive_modes)
plotting.plot_pump_traj(passive_modes, modes_trajectories, modes_trajectories_approx)
plt.savefig("mode_trajectories.png")
plt.show()

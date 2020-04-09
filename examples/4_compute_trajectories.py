import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

import naq_graphs as naq
from naq_graphs import plotting

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

os.chdir(graph_tpe)

graph = naq.load_graph()
modes_df = naq.load_modes()

if graph_tpe == "line_PRA" and params["dielectric_params"]["method"] == "custom":
    # set pump profile for PRA example
    pump_edges = round(len(graph.edges()) / 2)
    nopump_edges = len(graph.edges()) - pump_edges
    params["pump"] = np.append(np.ones(pump_edges), np.zeros(nopump_edges))
    params["pump"][0] = 0  # first edge is outside
else:
    params["pump"] = np.zeros(len(graph.edges()))  # uniform pump on inner edges
    for i, (u, v) in enumerate(graph.edges()):
        if graph[u][v]["inner"]:
            params["pump"][i] = 1

naq.update_parameters(graph, params)
naq.save_graph(graph)

modes_df = naq.pump_trajectories(modes_df, graph, return_approx=True)
naq.save_modes(modes_df)

qualities = naq.load_qualities()

ax = plotting.plot_scan(graph, qualities, modes_df, filename="scan_with_trajectories")
ax.set_ylim(graph.graph['params']['alpha_max'], -np.max(np.imag(modes_df['mode_trajectories'].to_numpy())))
plt.show()

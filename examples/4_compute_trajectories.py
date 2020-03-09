import os
import sys

import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt

from graph_generator import generate_graph

from naq_graphs import set_dielectric_constant, set_dispersion_relation
from naq_graphs.dispersion_relations import dispersion_relation_pump
from naq_graphs import create_naq_graph, pump_trajectories, load_modes, save_modes
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
    params["pump"] = np.ones(len(graph.edges())) #for uniform pump on all edges (inner and outer) 

D0s = np.linspace(0, params["D0_max"], params["D0_steps"])
modes_trajectories, modes_trajectories_approx = pump_trajectories(
    modes, graph, params, D0s, n_workers=1, return_approx=True
)

save_modes(modes_trajectories,modes_trajectories_approx, filename="trajectories")

ks, alphas, qualities = pickle.load(open("scan.pkl", "rb"))
plot_scan(ks, alphas, qualities, modes)
plot_pump_traj(modes, modes_trajectories, modes_trajectories_approx)
plt.savefig("mode_trajectories.png")
plt.show()

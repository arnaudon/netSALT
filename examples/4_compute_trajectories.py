import os as os
import sys as sys

import numpy as np
import yaml as yaml
import pickle as pickle
import matplotlib.pyplot as plt
import networkx as nx

from graph_generator import generate_graph

from naq_graphs import set_dielectric_constant, set_dispersion_relation
from naq_graphs.dispersion_relations import dispersion_relation_pump
from naq_graphs import create_naq_graph, pump_trajectories, load_modes
from naq_graphs.plotting import plot_pump_traj, plot_scan

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

graph, positions = generate_graph(tpe=graph_tpe, params=params)

os.chdir(graph_tpe)

create_naq_graph(graph, params, positions=positions)

positions = [graph.nodes[u]["position"] for u in graph]

set_dielectric_constant(graph, params)
set_dispersion_relation(graph, dispersion_relation_pump, params)

modes = load_modes()
params["pump"] = np.ones(len(graph.edges()))

D0s = np.linspace(0, params["D0_max"], params["D0_steps"])
modes_trajectories, modes_trajectories_approx = pump_trajectories(
    modes, graph, params, D0s, n_workers=4, return_approx=True
)

ks, alphas, qualities = pickle.load(open("scan.pkl", "rb"))
plot_scan(ks, alphas, qualities, modes)
plot_pump_traj(modes, modes_trajectories, modes_trajectories_approx)
plt.savefig("mode_trajectories.png")
plt.show()

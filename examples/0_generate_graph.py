import os as os
import pickle as pickle
import sys as sys

import matplotlib.pyplot as plt
import numpy as np
import yaml as yaml

import netsalt
from graph_generator import generate_graph, generate_index
from netsalt import plotting

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

graph, positions = generate_graph(tpe=graph_tpe, params=params)
graph.graph["name"] = graph_tpe

if not os.path.isdir(graph_tpe):
    os.mkdir(graph_tpe)
os.chdir(graph_tpe)

netsalt.create_quantum_graph(graph, params, positions=positions)
netsalt.set_total_length(graph, params["innerL"], inner=True)

custom_index = generate_index(graph_tpe, graph, params)
netsalt.set_dielectric_constant(graph, params, custom_values=custom_index)

netsalt.set_dispersion_relation(graph, netsalt.physics.dispersion_relation_pump, params)

graph = netsalt.oversample_graph(graph, params)
netsalt.update_parameters(graph, params)
netsalt.save_graph(graph)

plotting.plot_quantum_graph(
    graph,
    edge_colors=params["dielectric_constant"],
    node_size=0.1, #1.
    color_map="plasma", #"Pastel1"
    cbar_min=1,
    cbar_max=np.max(np.abs(params["dielectric_constant"])),
)

plt.savefig("original_graph.svg")
#plt.show()

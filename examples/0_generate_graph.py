import os as os
import pickle as pickle
import sys as sys

import matplotlib.pyplot as plt
import numpy as np
import yaml as yaml

import naq_graphs as naq
from graph_generator import generate_graph, generate_index
from naq_graphs import plotting

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

naq.create_naq_graph(graph, params, positions=positions)
naq.set_total_length(graph, params["innerL"], inner=True)

custom_index = generate_index(graph_tpe, graph, params)
naq.set_dielectric_constant(graph, params, custom_values=custom_index)

naq.set_dispersion_relation(
    graph, naq.dispersion_relations.dispersion_relation_pump, params
)

naq.update_parameters(graph, params)
naq.save_graph(graph)

plotting.plot_naq_graph(graph, edge_colors=params["dielectric_constant"], node_size=0.1)

plt.savefig("original_graph.svg")
plt.show()

import os as os
import pickle as pickle
import sys as sys

import matplotlib.pyplot as plt
import numpy as np
import yaml as yaml

import naq_graphs as naq
from graph_generator import generate_graph
from naq_graphs import plotting

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

graph, positions = generate_graph(tpe=graph_tpe, params=params)

if not os.path.isdir(graph_tpe):
    os.mkdir(graph_tpe)
os.chdir(graph_tpe)

naq.create_naq_graph(graph, params, positions=positions)

naq.set_total_length(graph, params["innerL"], inner=True)

if graph_tpe == "line_PRA" and params["dielectric_params"]["method"] == "custom":
    custom_index = []  # line PRA example
    for u, v in graph.edges:
        custom_index.append(3.0 ** 2)
    custom_index[0] = 1.0 ** 2
    custom_index[-1] = 1.0 ** 2

    count_inedges = len(graph.edges) - 2.0
    print("Number of inner edges", count_inedges)
    if count_inedges % 4 == 0:
        for i in range(round(count_inedges / 4)):
            custom_index[i + 1] = 1.5 ** 2
    else:
        print("Change number of inner edges to be multiple of 4")
    naq.set_dielectric_constant(graph, params, custom_values=custom_index)

elif graph_tpe == "line_semi":
    custom_index = []  # line OSA example or Esterhazy PRA 2014
    for u, v in graph.edges:
        custom_index.append(params["dielectric_params"]["inner_value"])
    custom_index[0] = 100.0 ** 2
    custom_index[-1] = 1.0 ** 2
    naq.set_dielectric_constant(graph, params, custom_values=custom_index)

else:
    naq.set_dielectric_constant(graph, params)  # for "uniform" and all other graphs

naq.set_dispersion_relation(
    graph, naq.dispersion_relations.dispersion_relation_pump, params
)


naq.update_parameters(graph, params)
naq.save_graph(graph)

plotting.plot_naq_graph(graph, edge_colors=params["dielectric_constant"], node_size=0.1)

plt.savefig("original_graph.svg")
plt.show()

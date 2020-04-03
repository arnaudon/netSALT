import os
import sys

import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt
import networkx as nx

from graph_generator import generate_graph

import naq_graphs as naq

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

os.chdir(graph_tpe)

graph, params = naq.load_graph()

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
    custom_index = []  # line OSA example
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


# set pump profile for PRA example
if graph_tpe == "line_PRA" and params["dielectric_params"]["method"] == "custom":
    pump_edges = round(len(graph.edges()) / 2)
    nopump_edges = len(graph.edges()) - pump_edges
    params["pump"] = np.append(np.ones(pump_edges), np.zeros(nopump_edges))
    params["pump"][0] = 0  # first edge is outside
else:
    # params["pump"] = np.ones(len(graph.edges())) # uniform pump on ALL edges
    params["pump"] = np.zeros(len(graph.edges()))  # uniform pump on inner edges
    for i, (u, v) in enumerate(graph.edges()):
        if graph[u][v]["inner"]:
            params["pump"][i] = 1

threshold_modes, lasing_thresholds = naq.load_modes(filename="threshold_modes")

mode_competition_matrix = naq.compute_mode_competition_matrix(
    graph, params, threshold_modes, lasing_thresholds
)
pickle.dump(mode_competition_matrix, open("mode_competition_matrix.pkl", "wb"))

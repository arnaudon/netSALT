import os
import sys

import numpy as np
import yaml
import matplotlib.pyplot as plt
import networkx as nx

from graph_generator import generate_graph

from naq_graphs import set_dielectric_constant, set_dispersion_relation
from naq_graphs.dispersion_relations import dispersion_relation_pump
from naq_graphs import (
    create_naq_graph,
    load_modes,
    oversample_graph,
)
from naq_graphs import threshold_mode_on_nodes, mean_mode_on_edges
from naq_graphs.io import load_graph


if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

os.chdir(graph_tpe)

graph, params = load_graph()

graph = oversample_graph(graph, params)
positions = [graph.nodes[u]["position"] for u in graph]

modes, lasing_thresholds = load_modes(filename="threshold_modes")

if not os.path.isdir("threshold_modes"):
    os.mkdir("threshold_modes")

for i, mode in enumerate(modes):
    params["D0"] = lasing_thresholds[i]

    node_solution = threshold_mode_on_nodes(mode, graph)
    #edge_solution = mean_mode_on_edges(mode, graph) #this fn calls mode_on_nodes and so does not work for pumped modes

    plt.figure(figsize=(6, 4))
    nodes = nx.draw_networkx_nodes(
        graph,
        pos=positions,
        node_color=abs(node_solution) ** 2,
        node_size=5,
        cmap=plt.get_cmap("Blues"),
    )
    plt.colorbar(nodes, label=r"$|E|^2$ (a.u)")
    #edges_k = nx.draw_networkx_edges(
    #    graph,
    #    pos=positions,
    #    edge_color=edge_solution,
    #    width=5,
    #    edge_cmap=plt.get_cmap("Blues"),
    #)

    plt.title("k="+str(np.around(mode[0],3)-1j*np.around(mode[1],3)))
    
    plt.savefig("threshold_modes/mode_" + str(i) + ".png")
    plt.close()

    if graph_tpe == "line_PRA":
        position_x = [graph.nodes[u]["position"][0] for u in graph]
        E_sorted = node_solution[np.argsort(position_x)]
        node_positions = np.sort(position_x-position_x[1])
        
        plt.figure(figsize=(6, 4))
        plt.plot(node_positions[1:-1],abs(E_sorted[1:-1])**2) #only plot over inner edges
        plt.title("k="+str(np.around(mode[0],3)-1j*np.around(mode[1],3)))
        plt.savefig("threshold_modes/profile_mode_" + str(i) + ".svg")
        plt.close()

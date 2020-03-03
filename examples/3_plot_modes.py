import os
import sys

import yaml
import matplotlib.pyplot as plt
import networkx as nx

from graph_generator import generate_graph

from naq_graphs import set_dielectric_constant, set_dispersion_relation
from naq_graphs.dispersion_relations import dispersion_relation_dielectric
from naq_graphs import create_naq_graph, oversample_graph, load_modes
from naq_graphs import mode_on_nodes, mean_mode_on_edges
from naq_graphs.io import load_graph

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

os.chdir(graph_tpe)

graph, params = load_graph()

graph = oversample_graph(graph, edgesize=params["plot_edgesize"])
positions = [graph.nodes[u]["position"] for u in graph]

set_dielectric_constant(graph, params)
set_dispersion_relation(graph, dispersion_relation_dielectric, params)

modes = load_modes()

if not os.path.isdir("modes"):
    os.mkdir("modes")

for i, mode in enumerate(modes):
    node_solution = mode_on_nodes(mode, graph)
    edge_solution = mean_mode_on_edges(mode, graph)

    plt.figure(figsize=(6, 4))
    nodes = nx.draw_networkx_nodes(
        graph,
        pos=positions,
        node_color=abs(node_solution) ** 2,
        node_size=5,
        cmap=plt.get_cmap("Blues"),
    )
    plt.colorbar(nodes, label=r"$|E|^2$ (a.u)")
    edges_k = nx.draw_networkx_edges(
        graph,
        pos=positions,
        edge_color=edge_solution,
        width=5,
        edge_cmap=plt.get_cmap("Blues"),
    )

    plt.savefig("modes/mode_" + str(i) + ".png")
    plt.close()

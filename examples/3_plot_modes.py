import os
import sys

import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import naq_graphs as naq

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

os.chdir(graph_tpe)

graph, params = naq.load_graph()

#graph = naq.oversample_graph(graph, params)
positions = [graph.nodes[u]["position"] for u in graph]

modes = naq.load_modes()

if not os.path.isdir("modes"):
    os.mkdir("modes")

for i, mode in tqdm(enumerate(modes), total=len(modes)):
    node_solution = naq.mode_on_nodes(mode, graph)
    edge_solution = naq.mean_mode_on_edges(mode, graph)

    plt.figure(figsize=(6, 4))
    nodes = nx.draw_networkx_nodes(
        graph,
        pos=positions,
        node_color=abs(node_solution) ** 2,
        node_size=2,
        cmap=plt.get_cmap("Blues"),
    )
    plt.colorbar(nodes, label=r"$|E|^2$ (a.u)")
    edges_k = nx.draw_networkx_edges(
        graph,
        pos=positions,
        edge_color=edge_solution,
        width=2,
        edge_cmap=plt.get_cmap("Blues"),
    )
    plt.title("k=" + str(np.around(modes[i, 0], 3) - 1j * np.around(modes[i, 1], 3)))

    plt.savefig("modes/mode_" + str(i) + ".png")
    plt.close()

    if graph_tpe == "line_PRA" or graph_tpe == "line_semi":
        position_x = [graph.nodes[u]["position"][0] for u in graph]
        E_sorted = node_solution[np.argsort(position_x)]
        node_positions = np.sort(position_x - position_x[1])

        plt.figure()
        plt.plot(
            node_positions[1:-1], abs(E_sorted[1:-1]) ** 2
        )  # only plot over inner edges
        plt.title(
            "k=" + str(np.around(modes[i, 0], 3) - 1j * np.around(modes[i, 1], 3))
        )
        plt.savefig("modes/profile_mode_" + str(i) + ".svg")

        naq.save_modes(node_positions, E_sorted, filename="modes/passivemode_" + str(i))

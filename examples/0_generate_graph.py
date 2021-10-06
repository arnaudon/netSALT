import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

import netsalt
from graph_generator import generate_graph, generate_index
from netsalt import plotting

if __name__ == "__main__":

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

    netsalt.set_dispersion_relation(graph, netsalt.physics.dispersion_relation_pump)

    #graph = netsalt.oversample_graph(graph, params)
    netsalt.update_parameters(graph, params)
    netsalt.save_graph(graph)

    # graph properties
    print("graph properties:")
    deg = nx.degree_histogram(graph)
    print("degree distribution", deg)
    c = nx.cycle_basis(graph)
    print("length cycle basis", len(c))

    print("number of nodes", len(graph.nodes()))
    print("number of edges", len(graph.edges()))
    print("number of inner edges", sum(graph.graph["params"]["inner"]))

    lengths = [graph[u][v]["length"] for u, v in graph.edges if graph[u][v]["inner"]]
    print("min edge length", np.min(lengths))
    print("max edge length", np.max(lengths))
    print("mean edge length", np.mean(lengths))

    # modify colormap
    cmap = get_cmap("Pastel1_r")
    # colors = cmap.colors  # list of colors
    # print(colors)
    newcolors = cmap(np.take(np.linspace(0, 1, 9), [0, 4, 2, 3, 1, 8, 6, 7, 5]))
    newcmp = ListedColormap(newcolors)

    plotting.plot_quantum_graph(
        graph,
        edge_colors=params["dielectric_constant"],
        node_size=5,  # 0.1
        color_map=newcmp,  # "Pastel1", #"plasma"
        cbar_min=1,
        cbar_max=np.max(np.abs(params["dielectric_constant"])),
    )

    plt.savefig("original_graph.svg")
    plt.show()

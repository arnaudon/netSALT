import os
import sys

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


graph = naq.oversample_graph(graph, params)
positions = [graph.nodes[u]["position"] for u in graph]

threshold_modes, lasing_thresholds = naq.load_modes(filename="threshold_modes")

if not os.path.isdir("threshold_modes"):
    os.mkdir("threshold_modes")

for i, threshold_mode in enumerate(threshold_modes):
    params["D0"] = lasing_thresholds[i]

    node_solution = naq.mode_on_nodes(threshold_mode, graph)
    edge_solution = naq.mean_mode_on_edges(threshold_mode, graph)

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

    plt.title(
        "k="
        + str(np.around(threshold_mode[0], 3) - 1j * np.around(threshold_mode[1], 3))
    )

    plt.savefig("threshold_modes/mode_" + str(i) + ".png")
    plt.close()

    if graph_tpe == "line_PRA" or graph_tpe == "line_semi":
        position_x = [graph.nodes[u]["position"][0] for u in graph]
        E_sorted = node_solution[np.argsort(position_x)]
        node_positions = np.sort(position_x - position_x[1])

        plt.figure(figsize=(6, 4))
        plt.plot(
            node_positions[1:-1], abs(E_sorted[1:-1]) ** 2
        )  # only plot over inner edges
        plt.title(
            "k="
            + str(
                np.around(threshold_mode[0], 3) - 1j * np.around(threshold_mode[1], 3)
            )
        )
        plt.savefig("threshold_modes/profile_mode_" + str(i) + ".svg")
        plt.close()

        naq.save_modes(
            node_positions, E_sorted, filename="threshold_modes/thresholdmode_" + str(i)
        )

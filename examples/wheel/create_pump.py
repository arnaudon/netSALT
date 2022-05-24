"""
README: This code defines a custom pump profile and saves it to file. 
        To use the custom pump profile, modify luigi file section:
            [CreatePumpProfile]
	    mode = custom
	    custom_pump_path = pump_profile.yaml
"""

import yaml
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from netsalt.io import load_graph

def plot_graph(graph, positions, ax=None):

    """Plot the graph with edge labels"""

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()
    else:
        fig = None

    nx.draw_networkx(
        graph, 
        pos = positions, 
        node_size=20, 
        font_size=10, 
        with_labels=True
    )  

    edge_labels=dict([((u,v),i) for i, (u,v) in enumerate(graph.edges())])
    
    nx.draw_networkx_edge_labels(
        graph, 
        pos = positions, 
        edge_labels=edge_labels
    )
    
    plt.show()


if __name__ == "__main__":

    graph = load_graph("graph.gpickle")
    positions = [graph.nodes[u]["position"] for u in graph]
    plot_graph(graph, positions)


    """Define edges which are not to be pumped {by default all inner edges are pumped}"""

    edge_list = [(0, 1), (0, 2), (0, 3), (0, 6), (4, 5)]

    pump = np.ones(len(graph.edges))
    for i, edge in enumerate(graph.edges):
        if edge in edge_list:
            pump[i] = 0

    yaml.dump(pump.tolist(), open("pump_profile.yaml", "w"))

    print('pump on edges:', pump)


    """Plot the pump profile on top of the graph structure."""

    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    pumped_edges = [e for e, _pump in zip(graph.edges, pump) if _pump > 0.0]

    nx.draw_networkx_edges(
        graph,
        pos=positions,
        edgelist=pumped_edges,
        edge_color="0.8",
        width=10,
    )

    plot_graph(graph, positions, ax)


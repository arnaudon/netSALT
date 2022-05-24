import numpy as np
import networkx as nx

from netsalt.io import save_graph

if __name__ == "__main__":
    n = 7
    graph = nx.wheel_graph(n)

    for u in graph.nodes:
        if u == 0:
            graph.nodes[u]["position"] = [0, 0]
        else:
            graph.nodes[u]["position"] = [np.cos(u / (n - 1) * 2 * np.pi), np.sin(u / (n - 1) * 2 * np.pi)]

    graph = nx.convert_node_labels_to_integers(graph)
    save_graph(graph, "graph.gpickle")

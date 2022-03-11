import numpy as np
import networkx as nx

from netsalt.io import save_graph

if __name__ == "__main__":
    n = 12
    graph = nx.cycle_graph(n)
    for u in graph.nodes:
        graph.nodes[u]["position"] = [np.cos(u / n * 2 * np.pi), np.sin(u / n * 2 * np.pi)]
    graph = nx.convert_node_labels_to_integers(graph)
    save_graph(graph, "graph.gpickle")

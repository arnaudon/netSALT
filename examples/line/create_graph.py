import numpy as np
import networkx as nx

from netsalt.io import save_graph

if __name__ == "__main__":
    n = 4
    graph = nx.grid_2d_graph(n, 1, periodic=False)
    graph = nx.convert_node_labels_to_integers(graph)
    pos = np.array([[i / (len(graph) - 1), 0] for i in range(len(graph))])
    for n, _pos in zip(graph.nodes, pos):
        graph.nodes[n]['position'] = _pos
        graph.nodes[n]['node_loss'] = 0.0  # add here node-dependent loss fraction
    save_graph(graph, "graph.gpickle")
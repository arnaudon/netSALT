import numpy as np
import networkx as nx

from netsalt.io import save_graph


def add_leads(graph, lead_prob=1):
    if lead_prob == 0:
        return graph

    from scipy.spatial import ConvexHull

    pos = np.array([graph.nodes[u]["position"] for u in graph.nodes])
    hull = ConvexHull(pos)
    k = 0
    n_tot = len(graph)
    hull_nodes = set(hull.simplices.flatten())
    print(len(hull_nodes), "hull nodes")

    for n in hull_nodes:
        p = np.random.rand()
        if p < lead_prob:
            graph.add_node(n_tot + k)
            graph.add_edge(n, n_tot + k)
            graph.nodes[n_tot + k]["position"] = pos[n] + 1  # * 1.4
            k += 1

    return graph


if __name__ == "__main__":
    n = 2
    m = 2
    graph = nx.grid_2d_graph(n, m, periodic=False)
    for u in graph.nodes:
        graph.nodes[u]["position"] = np.array(u, dtype=np.float)
    graph = nx.convert_node_labels_to_integers(graph)

    graph = add_leads(graph, lead_prob=0)
    save_graph(graph, "graph.gpickle")

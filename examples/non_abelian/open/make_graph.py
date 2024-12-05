import networkx as nx
import numpy as np


def make_graph(with_leads=False):
    n = 30
    graph = nx.Graph()
    graph = nx.cycle_graph(n)

    graph.add_edge(2, 8)
    graph.add_edge(27, 16)
    graph.add_edge(16, 10)
    x = np.linspace(0, 2 * np.pi * (1 - 1.0 / (len(graph) - 1)), len(graph))
    pos = np.array([np.cos(x), np.sin(x)]).T
    pos = list(pos)
    if with_leads:
        graph.add_edge(0, n)
        graph.add_edge(14, n + 1)
        graph.add_edge(16, n + 2)
        pos.append(np.array([1.4, 0]))
        pos.append(np.array([-1.4, 0.3]))
        pos.append(np.array([-1.4, -0.3]))

    return graph, pos

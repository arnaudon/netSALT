import numpy as np
import matplotlib.pyplot as plt
from netsalt.modes import get_edge_transfer
from netsalt.quantum_graph import create_quantum_graph


from netsalt.physics import set_dispersion_relation
from netsalt.physics import dispersion_relation_resistance
import networkx as nx
from netsalt.modes import estimate_boundary_flow


def make_graph2():
    n = 30
    graph = nx.Graph()
    graph = nx.cycle_graph(n)

    graph.add_edge(0, 8)
    graph.add_edge(1, 4)
    graph.add_edge(2, 23)
    graph.add_edge(5, 20)
    graph.add_edge(8, 15)
    graph.add_edge(2, 10)
    graph.add_edge(3, 11)
    graph.add_edge(4, 12)
    x = np.linspace(0, 2 * np.pi * (1 - 1.0 / (len(graph) - 1)), len(graph))
    pos = np.array([np.cos(x), np.sin(x)]).T
    pos = list(pos)

    graph.add_edge(0, n)
    graph.add_edge(3, n + 1)
    graph.add_edge(15, n + 2)
    graph.add_edge(17, n + 3)
    pos.append(np.array([1.4, 0]))
    pos.append(np.array([-1.4, 0.0]))
    pos.append(np.array([-1.4, -0.5]))
    pos.append(np.array([1.4, 0]))

    return graph, pos


if __name__ == "__main__":

    params = {"open_model": "open", "c": 1.0, "R": 1000.0}
    np.random.seed(42)

    graph, pos = make_graph2()

    create_quantum_graph(graph, params=params, positions=pos)
    set_dispersion_relation(graph, dispersion_relation_resistance)

    n_deg = np.array([len(graph[u]) for u in graph.nodes])
    e_deg = np.array([len(graph[v]) for u, v in graph.edges])
    n_ids = list(np.argwhere(n_deg == 1).flatten())
    e_ids = list(np.argwhere(e_deg == 1).flatten())
    e_ids += list(2 * np.argwhere(e_deg == 1).flatten() + 1)
    input_flow = np.zeros(2 * len(graph.edges))
    input_flow[e_ids[0]] = 1.0
    input_flow[e_ids[3]] = 1.0

    out_flow, _k = estimate_boundary_flow(graph, input_flow)

    res_edges = []
    ks = np.logspace(-5, 2, 1000)
    for k in ks:
        r_edge = get_edge_transfer(k, graph, input_flow)[e_ids]
        res_edges.append(r_edge)

    plt.figure()
    ax = plt.gca()
    f = lambda x: np.abs(x)
    for i in range(len(np.array(res_edges)[0])):
        ax.semilogx(ks, f(np.array(res_edges)[:, i]), label=i)
    ax.axvline(_k)
    ax.set_ylim(0, 2)
    plt.legend(loc="upper right")

    plt.show()

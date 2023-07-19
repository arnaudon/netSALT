import numpy as np
from scipy.sparse import linalg
from tqdm import tqdm
import matplotlib.pyplot as plt
from netsalt.modes import get_node_transfer, get_edge_transfer
from netsalt.quantum_graph import (
    get_total_inner_length,
    create_quantum_graph,
    construct_weight_matrix,
    construct_incidence_matrix,
    laplacian_quality,
    mode_quality,
    construct_laplacian,
)

from netsalt.modes import mode_on_nodes

from netsalt.physics import dispersion_relation_linear, set_dispersion_relation
from netsalt.physics import dispersion_relation_resistance
from netsalt.modes import scan_frequencies
from netsalt.plotting import plot_scan
import networkx as nx


def make_graph():
    n = 30
    graph = nx.Graph()
    graph = nx.cycle_graph(n)

    # graph.add_edge(0, 8)
    # graph.add_edge(1, 4)
    # graph.add_edge(2, 23)
    # graph.add_edge(5, 20)
    # graph.add_edge(8, 15)
    x = np.linspace(0, 2 * np.pi * (1 - 1.0 / (len(graph) - 1)), len(graph))
    pos = np.array([np.cos(x), np.sin(x)]).T
    pos = list(pos)

    graph.add_edge(0, n)
    graph.add_edge(15, n + 1)
    # graph.add_edge(16, n + 2)
    # graph.add_edge(3, n + 3)
    pos.append(np.array([1.4, 0]))
    pos.append(np.array([-1.4, 0.0]))
    # pos.append(np.array([-1.4, -0.5]))
    # pos.append(np.array([1.4, 0]))
    print(len(graph.edges))

    return graph, pos


if __name__ == "__main__":
    graph, pos = make_graph()
    params = {
        "open_model": "open",
        "n_workers": 7,
        "k_n": 10000,
        "k_min": 10.0001,
        "k_max": 15.0,
        "alpha_n": 20,
        "alpha_min": 0.00,
        "alpha_max": 0.2,
    }
    np.random.seed(42)
    a = 3 + 0.0 * np.random.uniform(0.0, 1.0, len(graph.edges))

    e_deg = np.array([len(graph[v]) for u, v in graph.edges])
    # a[e_deg == 1] = 1.0
    params["c"] = 1.0  # np.sqrt(a)
    params["R"] = 0.0  # 0 / a

    nx.draw(graph, pos=pos)
    nx.draw_networkx_labels(graph, pos=pos)
    create_quantum_graph(graph, params=params, positions=pos)
    set_dispersion_relation(graph, dispersion_relation_resistance)
    n_deg = np.array([len(graph[u]) for u in graph.nodes])
    n_ids = list(np.argwhere(n_deg == 1).flatten())
    e_ids = list(np.argwhere(e_deg == 1).flatten())
    e_ids += list(2 * np.argwhere(e_deg == 1).flatten() + 1)
    input_flow = np.zeros(len(graph.nodes))
    input_flow[n_ids[0]] = 1.0

    resonance = 2 * np.pi / get_total_inner_length(graph)
    res_nodes = []
    res_edges = []
    ks = np.linspace(params["k_min"], params["k_max"], params["k_n"])
    for k in tqdm(ks):
        r_node = get_node_transfer(k, graph, input_flow)[n_ids]
        r_edge = get_edge_transfer(k, graph, input_flow)[e_ids]
        res_nodes.append(r_node)
        res_edges.append(r_edge)

    plt.figure()
    ax = plt.gca()
    f = lambda x: np.abs(x)
    plt.axvline(resonance, c="k")
    plt.axvline(resonance * 2, c="k")
    plt.axvline(resonance * 3, c="k")

    ax.plot(ks, f(np.array(res_nodes)[:, 0]), label="0")
    ax.plot(ks, f(np.array(res_nodes)[:, 1]), label="1")
    ax.set_xlim(params["k_min"], params["k_max"])
    plt.legend(loc="upper right")

    plt.figure()
    ax = plt.gca()
    f = lambda x: np.abs(x)
    plt.axvline(resonance, c="k")
    plt.axvline(resonance * 2, c="k")
    plt.axvline(resonance * 3, c="k")

    ax.plot(ks, f(np.array(res_edges)[:, 0]), label="0")
    ax.plot(ks, f(np.array(res_edges)[:, 1]), label="1")
    ax.plot(ks, f(np.array(res_edges)[:, 2]), label="2")
    ax.plot(ks, f(np.array(res_edges)[:, 3]), label="3")
    ax.set_xlim(params["k_min"], params["k_max"])
    plt.legend(loc="upper right")

    plt.show()

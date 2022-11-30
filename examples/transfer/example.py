import numpy as np
from scipy.sparse import linalg
from tqdm import tqdm
import matplotlib.pyplot as plt
from netsalt.transfer import get_edge_transfer_matrix, get_static_boundary_flow
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

    #graph.add_edge(0, 8)
    #graph.add_edge(1, 4)
    #graph.add_edge(2, 23)
    #graph.add_edge(5, 20)
    #graph.add_edge(8, 15)
    x = np.linspace(0, 2 * np.pi * (1 - 1.0 / (len(graph) - 1)), len(graph))
    pos = np.array([np.cos(x), np.sin(x)]).T
    pos = list(pos)

    graph.add_edge(0, n)
    graph.add_edge(14, n + 1)
    graph.add_edge(16, n + 2)
    # graph.add_edge(3, n + 3)
    pos.append(np.array([1.4, 0]))
    pos.append(np.array([-1.4, 0.0]))
    pos.append(np.array([-1.4, -0.5]))
    # pos.append(np.array([1.4, 0]))
    print(len(graph.edges))

    return graph, pos


if __name__ == "__main__":
    graph, pos = make_graph()
    params = {
        "open_model": "open",
        "n_workers": 7,
        "k_n": 2000,
        "k_min": 0.00001,
        "k_max": 5.2,
        "alpha_n": 20,
        "alpha_min": 0.00,
        "alpha_max": 0.2,
    }
    np.random.seed(42)
    a = 3 + 0.0 * np.random.uniform(0.0, 1.0, len(graph.edges))

    e_deg = np.array([len(graph[v]) for u, v in graph.edges])
    #a[e_deg == 1] = 1.0
    params["c"] = np.sqrt(a)
    params["R"] = 1.0 / a

    nx.draw(graph, pos=pos)
    nx.draw_networkx_labels(graph, pos=pos)
    create_quantum_graph(graph, params=params, positions=pos)
    set_dispersion_relation(graph, dispersion_relation_resistance)
    # n_deg = np.array([v for u, v in graph.edges])
    # _ids = np.argwhere(e_deg == 1).flatten()
    print(e_deg)
    ids = list(2 * np.argwhere(e_deg == 1).flatten())
    ids += list(2 * np.argwhere(e_deg == 1).flatten() + 1)
    print(ids)
    input_flow = np.zeros(2 * len(graph.edges))
    input_flow[ids[3]] = 1.0

    res = []
    ks = np.linspace(params["k_min"], params["k_max"], params["k_n"])
    for k in ks:
        r = get_edge_transfer_matrix(k, graph, input_flow)[ids]
        res.append(r)
    out_flow, _k = get_static_boundary_flow(graph, input_flow)#, k_frac=0.00001)
    print(out_flow)
    B = nx.incidence_matrix(graph, oriented=True).toarray()
    W = np.diag(1./(graph.graph["params"]["R"] * graph.graph["lengths"]))
    L = B.dot(W).dot(B.T)
    p = linalg.spsolve(L, out_flow)
    print(p)
    #print(L.dot(f), f.sum())
    #print(out_flow, f)

    plt.figure()
    #pp=nx.draw(graph, pos=pos, node_color=p)
    nx.draw_networkx_nodes(graph, pos=pos, node_color=p, node_size=50)
    f= abs(B.T.dot(p))
    pp=nx.draw_networkx_edges(graph, pos=pos, edge_color=f, width=5)
    plt.colorbar(pp)

    r = get_edge_transfer_matrix(_k, graph, input_flow)
    L = construct_laplacian(_k, graph)
    K = np.append(graph.graph["ks"], graph.graph["ks"])
    BT, B = construct_incidence_matrix(graph)
    _input_flow = BT.dot(K * input_flow)
    r = np.abs(linalg.spsolve(L, _input_flow))

    plt.figure()
    nx.draw_networkx_nodes(graph, pos=pos, node_size=50, node_color=r)
    #pp=nx.draw_networkx_edges(graph, pos=pos, edge_color=ff, width=5)
    plt.colorbar(pp)
    plt.figure()
    plt.plot(r, p, '+')
    plt.show()

    plt.figure()
    ax = plt.gca()
    f = lambda x: np.abs(x)
    ax.plot(ks, f(np.array(res)[:, 3]), label="input")
    ax.plot(ks, f(np.array(res)[:, 0]), label="output")
    # ax.plot(ks,- np.abs(np.array(res)[:, 1]) - np.real(np.array(res)[:, 2]), label="sum")
    # ax.plot(
    #    ks,
    #    np.real(np.array(res)[:, 1]) + np.real(np.array(res)[:, 2]) + np.real(np.array(res)[:, 0]),
    #    label="sum2",
    # )
    ax.plot(ks, f(np.array(res)[:, 4]), label="input 2")
    ax.plot(ks, f(np.array(res)[:, 1]), label="output 2")
    ax.plot(ks, f(np.array(res)[:, 5]), label="input 3")
    ax.plot(ks, f(np.array(res)[:, 2]), label="output 3")
    ax.axvline(_k)
    # ax2 = ax.twinx()
    # qualities = scan_frequencies(graph)
    # plot_scan(graph, qualities, ax=ax2)
    # ax.set_ylim(-0.1, 1.1)
    plt.legend(loc="upper right")

    plt.show()

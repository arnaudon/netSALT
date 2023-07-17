import numpy as np
from scipy.sparse import linalg
from tqdm import tqdm
from copy import copy
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

    graph.add_edge(0, 8)
    graph.add_edge(1, 4)
    graph.add_edge(2, 23)
    graph.add_edge(5, 20)
    graph.add_edge(8, 15)
    x = np.linspace(0, 2 * np.pi * (1 - 1.0 / (len(graph) - 1)), len(graph))
    pos = np.array([np.cos(x), np.sin(x)]).T
    pos = list(pos)

    # graph.add_edge(0, n)
    # graph.add_edge(14, n + 1)
    # graph.add_edge(16, n + 2)
    # graph.add_edge(3, n + 3)
    # pos.append(np.array([1.4, 0]))
    # pos.append(np.array([-1.4, 0.0]))
    # pos.append(np.array([-1.4, -0.5]))
    # pos.append(np.array([1.4, 0]))

    return graph, pos


if __name__ == "__main__":
    np.random.seed(42)
    graph, pos = make_graph()
    graph_open, pos = make_graph()
    graph_reversed, pos = make_graph()
    params = {
        "open_model": "directed",
        "n_workers": 7,
        "k_n": 500,
        "k_min": 10.0,
        "k_max": 15.0,
        "alpha_n": 100,
        "alpha_min": -0.8,
        "alpha_max": 0.8,
    }
    params["c"] = 1.0

    nx.draw(graph, pos=pos)
    nx.draw_networkx_labels(graph, pos=pos)

    create_quantum_graph(graph, params=copy(params), positions=pos)
    set_dispersion_relation(graph, dispersion_relation_linear)

    params["open_model"] = "open"
    create_quantum_graph(graph_open, params=copy(params), positions=pos)
    set_dispersion_relation(graph_open, dispersion_relation_linear)

    params["open_model"] = "directed_reversed"
    create_quantum_graph(graph_reversed, params=copy(params), positions=pos)
    set_dispersion_relation(graph_reversed, dispersion_relation_linear)

    qualities = scan_frequencies(graph)
    plot_scan(graph, qualities)

    qualities = scan_frequencies(graph_reversed)
    plot_scan(graph, qualities)
    plt.show()

    k = 1.0
    ks = np.linspace(10.0, 15, 1000)
    qs = []
    qs_reversed = []
    qs_open = []
    for k in tqdm(ks):
        imk = 0.05
        qs.append(mode_quality([k, imk], graph))
        qs_reversed.append(mode_quality([k, imk], graph_reversed))
        qs_open.append(mode_quality([k, imk], graph_open))
    plt.figure()
    plt.plot(ks, qs, label="directed")
    plt.plot(ks, qs_reversed, label="directed reversed")
    plt.plot(ks, qs_open, label="open")
    plt.legend()
    plt.yscale("log")
    plt.show()


def lkj():
    # print(L.dot(f), f.sum())
    # print(out_flow, f)

    plt.figure()
    # pp=nx.draw(graph, pos=pos, node_color=p)
    nx.draw_networkx_nodes(graph, pos=pos, node_color=p, node_size=50)
    f = abs(B.T.dot(p))
    pp = nx.draw_networkx_edges(graph, pos=pos, edge_color=f, width=5)
    plt.colorbar(pp)

    r = get_edge_transfer_matrix(_k, graph, input_flow)
    L = construct_laplacian(_k, graph)
    K = np.append(graph.graph["ks"], graph.graph["ks"])
    BT, B = construct_incidence_matrix(graph)
    _input_flow = BT.dot(K * input_flow)
    r = np.abs(linalg.spsolve(L, _input_flow))

    plt.figure()
    nx.draw_networkx_nodes(graph, pos=pos, node_size=50, node_color=r)
    # pp=nx.draw_networkx_edges(graph, pos=pos, edge_color=ff, width=5)
    plt.colorbar(pp)
    plt.figure()
    plt.plot(r, p, "+")
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

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from netsalt.quantum_graph import create_quantum_graph, laplacian_quality, mode_quality
from netsalt.modes import mode_on_nodes

from netsalt.modes import scan_frequencies
from netsalt.plotting import plot_scan
from netsalt.physics import dispersion_relation_linear, set_dispersion_relation
import networkx as nx


from netsalt.non_abelian import construct_so3_laplacian, so3_mode_on_nodes, scan_frequencies_so3


def make_graph(n):
    graph = nx.cycle_graph(n)
    graph.add_edge(0, 8)
    graph.add_edge(0, 20)
    graph.add_edge(10, 15)
    x = np.linspace(0, 2 * np.pi * (1 - 1.0 / (len(graph) - 1)), len(graph))
    pos = np.array([np.cos(x), np.sin(x)]).T
    pos = list(pos)
    graph.add_edge(1, n)
    graph.add_edge(15, n + 1)
    pos += [[1.2, 0]]
    pos += [[-1.2, 0]]

    return graph, pos


if __name__ == "__main__":

    params = {
        "open_model": "open",
        "c": 1.0,
        "k_min": 10.0,
        "k_max": 12.0,
        "k_n": 200,
        "alpha_min": 0.0,
        "alpha_max": 0.4,
        "alpha_n": 50,
        "n_workers": 7,
    }
    n = 30
    graph, pos = make_graph(n)
    graph_u1, pos = make_graph(n)

    nx.draw(graph, pos=pos)

    create_quantum_graph(graph, params=params, positions=pos)
    create_quantum_graph(graph_u1, params=params, positions=pos)

    set_dispersion_relation(graph_u1, dispersion_relation_linear)

    qualities_u1 = scan_frequencies(graph_u1)
    plot_scan(graph_u1, qualities_u1)
    plt.suptitle('u1')

    qualities = scan_frequencies_so3(graph)
    plot_scan(graph, qualities)
    plt.suptitle("so3")
    plt.show()


def lkj():
    ks = np.linspace(10.0, 15, 500)
    qs = []
    qs_u1 = []
    for k in tqdm(ks):
        kim = 0.05
        L = construct_so3_laplacian(k + 1j * kim, graph)
        qs.append(laplacian_quality(L))
        qs_u1.append(mode_quality([k, kim], graph_u1))

    plt.figure()
    plt.plot(ks, qs_u1, "+-r", label="u1")
    plt.plot(ks, qs, "-", label="so3")
    plt.legend(loc="best")
    plt.yscale("log")
    plt.show()

    k = ks[np.argmin(qs)]
    L = construct_so3_laplacian(k, graph)
    mode = so3_mode_on_nodes(L)

    k_u1 = k  # ks[np.argmin(qs_u1)]
    mode_u1 = mode_on_nodes([k_u1, 0], graph_u1)

    plt.figure()
    x = np.abs(mode[::3])
    y = np.abs(mode[1::3])
    z = np.abs(mode[2::3])
    plt.plot(x, label="x")
    plt.plot(y, label="y")
    plt.plot(z, label="z")
    n = np.sqrt(x**2 + y**2 + z**2)

    plt.plot(n, "+-", label="norm")
    plt.plot(np.abs(mode_u1), label="u1")
    plt.legend()

    plt.figure()
    plt.plot(ks, qs_u1, "+-r")
    plt.plot(ks, qs, "-")
    plt.axvline(k)
    plt.axvline(k_u1, c="r")
    plt.yscale("log")

    plt.figure()
    plt.plot(np.sqrt(x**2 + y**2 + z**2) - np.abs(mode_u1))
    plt.show()

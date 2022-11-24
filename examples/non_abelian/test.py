import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from netsalt.quantum_graph import create_quantum_graph, laplacian_quality, mode_quality
from netsalt.modes import mode_on_nodes

from netsalt.physics import dispersion_relation_linear, set_dispersion_relation
import networkx as nx


from netsalt.non_abelian import construct_so3_laplacian, so3_mode_on_nodes
def make_graph():
    graph = nx.cycle_graph(n)
    graph.add_edge(0, 8)
    graph.add_edge(0, 20)
    graph.add_edge(10, 15)
    x = np.linspace(0, 2*np.pi*(1-1. / (len(graph) - 1)), len(graph))
    pos = np.array([np.cos(x), np.sin(x)]).T
    pos = list(pos)


    return graph, pos

if __name__ == "__main__":

    params = {"open_model": "open", "c": 1.0}
    n = 30
    graph, pos = make_graph()
    graph_u1, pos = make_graph()

    nx.draw(graph, pos=pos)

    create_quantum_graph(graph, params=params, positions=pos)
    create_quantum_graph(graph_u1, params=params, positions=pos)
    set_dispersion_relation(graph_u1, dispersion_relation_linear)

    ks = np.linspace(1.53, 1.58, 2000)
    qs = []
    qs_u1 = []
    for k in tqdm(ks):
        k += 0.0966j
        L = construct_so3_laplacian(k, graph, abelian_scale=20.0)
        qs.append(laplacian_quality(L))
        qs_u1.append(mode_quality([k, 0], graph_u1))

    plt.figure()
    plt.plot(ks, qs_u1, "+-r")
    plt.plot(ks, qs, "-")
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

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy as sc
from netsalt.quantum_graph import create_quantum_graph, laplacian_quality, mode_quality
from netsalt.modes import mode_on_nodes

from netsalt.physics import dispersion_relation_linear, set_dispersion_relation
import networkx as nx
from scipy import sparse, linalg


def hat_inv(xi_vec):
    """Convert vector Lie algebra to matrix Lie algebra element."""
    xi = np.zeros((3, 3))
    xi[1, 2] = -xi_vec[0]
    xi[2, 1] = xi_vec[0]
    xi[0, 2] = xi_vec[1]
    xi[2, 0] = -xi_vec[1]
    xi[0, 1] = -xi_vec[2]
    xi[1, 0] = xi_vec[2]
    return xi


def hat(xi):
    """Convert matrix Lie algebra to vector Lie algebra element."""
    xi_vec = np.zeros(3)
    xi_vec[0] = xi[2, 1]
    xi_vec[1] = xi[0, 2]
    xi_vec[2] = xi[1, 0]
    return xi_vec


def proj_perp(chi):
    return np.eye(3) - proj_paral(chi)


def proj_paral(chi_mat, center_scale=1.0):
    chi_vec = hat(chi_mat)
    return center_scale * np.outer(chi_vec, chi_vec) / np.linalg.norm(chi_vec) ** 2


def norm(chi_mat):
    chi_vec = hat(chi_mat)
    return np.linalg.norm(chi_vec)


def Ad(chi_mat):
    return linalg.expm(chi_mat)


def set_so3_wavenumber(graph, wavenumber):
    x = np.array([0.2, 0.2, 1.0])
    chi_vec = wavenumber * x / np.linalg.norm(x)
    chi_mat = hat_inv(chi_vec)
    graph.graph["ks"] = len(graph.edges) * [chi_mat]

    x2 = np.array([1.0, 0.2, 0.2])
    chi_vec2 = wavenumber * x2 / np.linalg.norm(x2)
    chi_mat2 = hat_inv(chi_vec2)
    graph.graph["ks"][:10] = 10 * [chi_mat2]
    graph.graph["ks"][20:30] = 10 * [chi_mat2]


def construct_so3_incidence_matrix(graph):
    dim = 3

    def _ext(i):
        return slice(dim * i, dim * (i + 1))

    Bout = sparse.lil_matrix((len(graph.edges) * 2 * dim, len(graph) * dim), dtype=np.complex128)
    BT = sparse.lil_matrix((len(graph) * dim, len(graph.edges) * 2 * dim), dtype=np.complex128)
    for ei, (u, v) in enumerate(graph.edges):

        one = np.eye(dim)
        expl = Ad(graph.graph["lengths"][ei] * graph.graph["ks"][ei])
        expl = 0.0j + expl.dot(proj_perp(graph.graph["ks"][ei]))
        expl += proj_paral(graph.graph["ks"][ei]) * np.exp(
            1.0j * graph.graph["lengths"][ei] * norm(graph.graph["ks"][ei])
        )
        out = True if (len(graph[u]) == 1 or len(graph[v]) == 1) else False

        Bout[_ext(2 * ei), _ext(u)] = -one
        Bout[_ext(2 * ei), _ext(v)] = 0 if out else expl
        Bout[_ext(2 * ei + 1), _ext(u)] = 0 if out else expl
        Bout[_ext(2 * ei + 1), _ext(v)] = -one

        BT[_ext(u), _ext(2 * ei)] = -one
        BT[_ext(v), _ext(2 * ei)] = expl
        BT[_ext(u), _ext(2 * ei + 1)] = expl
        BT[_ext(v), _ext(2 * ei + 1)] = -one

    return BT, Bout


def construct_so3_weight_matrix(graph, with_k=True):
    dim = 3

    def _ext(i):
        return slice(dim * i, dim * (i + 1))

    Winv = sparse.lil_matrix(
        (len(graph.edges) * 2 * dim, len(graph.edges) * 2 * dim), dtype=np.complex128
    )
    for ei, (u, v) in enumerate(graph.edges):
        chi = graph.graph["ks"][ei]
        length = graph.graph["lengths"][ei]

        w_perp = Ad(2.0 * length * chi).dot(proj_perp(chi))
        w_paral = np.exp(2.0j * length * norm(chi)) * proj_paral(chi)

        w = w_perp + w_paral - np.eye(3)

        winv = linalg.inv(w)

        if with_k:
            winv = (chi.dot(proj_perp(chi)) + 1.0j * norm(chi) * proj_paral(chi)).dot(winv)

        Winv[_ext(2 * ei), _ext(2 * ei)] = winv
        Winv[_ext(2 * ei + 1), _ext(2 * ei + 1)] = winv
    return Winv


def construct_so3_laplacian(wavenumber, graph):
    """Construct quantum laplacian from a graph.

    The quantum laplacian is L(k) = B^T(k) W^{-1}(k) B(k), with quantum incidence and weight matrix.

    Args:
        wavenumber (complex): wavenumber
        graph (graph): quantum graph
    """
    set_so3_wavenumber(graph, wavenumber)
    BT, B = construct_so3_incidence_matrix(graph)
    Winv = construct_so3_weight_matrix(graph)
    return BT.dot(Winv).dot(B)


def so3_mode_on_nodes(laplacian):
    v0 = np.random.random(laplacian.shape[0])
    min_eigenvalue, node_solution = sc.sparse.linalg.eigs(
        laplacian, k=1, sigma=0, v0=v0, which="LM"
    )
    quality_thresh = graph.graph["params"].get("quality_threshold", 1e2)
    if abs(min_eigenvalue[0]) > quality_thresh:
        raise Exception(
            "Not a mode, as quality is too high: "
            + str(abs(min_eigenvalue[0]))
            + " > "
            + str(quality_thresh)
        )

    return node_solution[:, 0]


if __name__ == "__main__":

    params = {"open_model": "open", "c": 1.0}
    n = 70

    graph = nx.cycle_graph(n)
    # graph.add_edge(0, n)
    # graph.add_edge(10, n + 1)
    # graph.add_edge(20, n + 2)

    graph.add_edge(0, 2)
    graph.add_edge(0, 5)
    graph.add_edge(0, 7)
    graph.add_edge(0, 20)
    graph.add_edge(0, 50)
    graph.add_edge(10, 12)
    graph.add_edge(20, 12)

    graph_u1 = nx.cycle_graph(n)
    # graph_u1.add_edge(0, n)
    # graph_u1.add_edge(10, n + 1)
    # graph_u1.add_edge(20, n + 2)

    graph_u1.add_edge(0, 2)
    graph_u1.add_edge(0, 5)
    graph_u1.add_edge(0, 7)
    graph_u1.add_edge(0, 20)
    graph_u1.add_edge(0, 50)
    graph_u1.add_edge(10, 12)
    graph_u1.add_edge(20, 12)

    x = np.linspace(0, 2 * np.pi / (len(graph) - 1), len(graph))
    pos = 100.1 * np.array([np.cos(x), np.sin(x)]).T
    pos = list(pos)
    pos.append([0, 0])
    create_quantum_graph(graph, params=params, positions=pos)
    create_quantum_graph(graph_u1, params=params, positions=pos)
    set_dispersion_relation(graph_u1, dispersion_relation_linear)

    ks = np.linspace(20.0, 25.0, 500)
    qs = []
    qs_u1 = []
    for k in tqdm(ks):
        L = construct_so3_laplacian(k, graph)
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

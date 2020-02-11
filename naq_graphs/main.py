"""main functions of NAQ graphs"""
import multiprocessing

import numpy as np
import scipy as sc

from .modes import (
    find_rough_modes_from_scan,
    clean_duplicate_modes,
    mode_quality,
    refine_mode_brownian_ratchet,
    construct_laplacian,
)
from .utils import _to_complex
from .graph_construction import construct_incidence_matrix, construct_weight_matrix


class WorkerModes:
    """worker to refine modes"""

    def __init__(self, graph, params):
        self.graph = graph
        self.params = params

    def __call__(self, mode):
        return refine_mode_brownian_ratchet(mode, self.graph, self.params)


class WorkerScan:
    """worker to scan complex frequency"""

    def __init__(self, graph):
        self.graph = graph

    def __call__(self, freq):
        return mode_quality(_to_complex(freq), self.graph)


def scan_frequencies(graph, params, n_workers=1):
    """scan a range of complex frequencies and return mode qualities"""
    ks = np.linspace(params["k_min"], params["k_max"], params["k_n"])
    alphas = np.linspace(params["alpha_min"], params["alpha_max"], params["alpha_n"])
    freqs = [[k, a] for k in ks for a in alphas]

    worker_scan = WorkerScan(graph)
    pool = multiprocessing.Pool(n_workers)
    qualities_list = pool.map(worker_scan, freqs)
    pool.close()

    id_k = [k_i for k_i in range(len(ks)) for a_i in range(len(alphas))]
    id_a = [a_i for k_i in range(len(ks)) for a_i in range(len(alphas))]
    qualities = sc.sparse.coo_matrix(
        (qualities_list, (id_k, id_a)), shape=(params["k_n"], params["alpha_n"])
    ).toarray()

    return ks, alphas, qualities


def find_modes(ks, alphas, qualities, graph, params, n_workers=1):
    """find the modes from a scan"""
    rough_modes = find_rough_modes_from_scan(
        ks, alphas, qualities, min_distance=2, threshold_abs=1.0
    )

    worker_modes = WorkerModes(graph, params)
    pool = multiprocessing.Pool(n_workers)
    refined_modes = pool.map(worker_modes, rough_modes)
    pool.close()

    if len(refined_modes) == 0:
        raise Exception("No modes found!")

    modes = clean_duplicate_modes(refined_modes, ks[1] - ks[0], alphas[1] - alphas[0])
    return modes[np.argsort(modes[:, 1])]


def mode_on_nodes(mode, graph, eigenvalue_max=1e-2):
    """compute the mode solution on the nodes of the graph"""
    laplacian = construct_laplacian(_to_complex(mode), graph)

    min_eigenvalue, node_solution = sc.sparse.linalg.eigs(laplacian, k=1, sigma=0)

    if abs(min_eigenvalue) > eigenvalue_max:
        raise Exception(
            "Not a mode, as quality is too high: "
            + str(np.round(abs(min_eigenvalue[0]), 5))
            + " > "
            + str(eigenvalue_max)
        )

    return node_solution[:, 0]


def flux_on_edges(mode, graph, eigenvalue_max=1e-2):
    """compute the flux on each edge (in both directions)"""

    node_solution = mode_on_nodes(mode, graph, eigenvalue_max=eigenvalue_max)

    BT, _ = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=False)

    return Winv.dot(BT.T).dot(node_solution)


def mean_mode_on_edges(mode, graph, eigenvalue_max=1e-2):
    """Compute the average |E|^2 on each edge"""
    edge_flux = flux_on_edges(mode, graph, eigenvalue_max=eigenvalue_max)

    mean_edge_solution = np.zeros(len(graph.edges))
    for ei, e in enumerate(list(graph.edges)):
        (u, v) = e[:2]

        z = np.zeros([2, 2], dtype=np.complex)

        l = graph[u][v]["length"]
        k = graph[u][v]["k"]
        z[0, 0] = (np.exp(1.0j * l * (k - np.conj(k))) - 1.0) / (
            1.0j * l * (k - np.conj(k))
        )
        z[0, 1] = (np.exp(1.0j * l * k) - np.exp(-1.0j * l * np.conj(k))) / (
            1.0j * l * (k + np.conj(k))
        )

        z[1, 0] = z[0, 1]
        z[1, 1] = z[0, 0]

        mean_edge_solution[ei] = np.real(
            np.conj(edge_flux[2 * ei : 2 * ei + 2]).T.dot(
                z.dot(edge_flux[2 * ei : 2 * ei + 2])
            )
        )

    return mean_edge_solution

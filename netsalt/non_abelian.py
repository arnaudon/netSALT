"""Module for non-abelian quantum graphs."""
import numpy as np
import scipy as sc
from tqdm import tqdm

import multiprocessing
from scipy import sparse, linalg
from netsalt.quantum_graph import laplacian_quality
from netsalt.utils import get_scan_grid, to_complex


def hat_inv(xi_vec):
    """Convert vector Lie algebra to matrix Lie algebra element."""
    xi = np.zeros((3, 3), dtype=np.complex128)
    xi[1, 2] = -xi_vec[0]
    xi[2, 1] = xi_vec[0]
    xi[0, 2] = xi_vec[1]
    xi[2, 0] = -xi_vec[1]
    xi[0, 1] = -xi_vec[2]
    xi[1, 0] = xi_vec[2]
    return xi


def hat(xi):
    """Convert matrix Lie algebra to vector Lie algebra element."""
    xi_vec = np.zeros(3, dtype=np.complex128)
    xi_vec[0] = xi[2, 1]
    xi_vec[1] = xi[0, 2]
    xi_vec[2] = xi[1, 0]
    return xi_vec


def proj_perp(chi):
    return np.eye(3) - proj_paral(chi)


def proj_paral(chi_mat):
    chi_vec = hat(chi_mat)
    return np.outer(chi_vec, chi_vec) / np.linalg.norm(chi_vec) ** 2


def norm(chi_mat):
    chi_vec = hat(chi_mat)
    return np.linalg.norm(chi_vec)


def Ad(chi_mat):
    return linalg.expm(chi_mat)


def set_so3_wavenumber(graph, wavenumber, chis=None):
    if chis is None:
        x = np.array([0.0, 0.0, 1.0])
        chi_vec = wavenumber * x / np.linalg.norm(x)
        chi_mat = hat_inv(chi_vec)
        graph.graph["ks"] = len(graph.edges) * [chi_mat]

        x2 = np.array([1.0, 0.0, 0.0])
        chi_vec2 = wavenumber * x2 / np.linalg.norm(x2)
        chi_mat2 = hat_inv(chi_vec2)
        graph.graph["ks"][:10] = 10 * [chi_mat2]
    else:
        graph.graph["ks"] = chis


def construct_so3_incidence_matrix(graph, abelian_scale=1.0):
    dim = 3

    def _ext(i):
        return slice(dim * i, dim * (i + 1))

    Bout = sparse.lil_matrix((len(graph.edges) * 2 * dim, len(graph) * dim), dtype=np.complex128)
    BT = sparse.lil_matrix((len(graph) * dim, len(graph.edges) * 2 * dim), dtype=np.complex128)
    for ei, (u, v) in enumerate(graph.edges):

        one = np.eye(dim)
        expl = Ad(graph.graph["lengths"][ei] * graph.graph["ks"][ei])
        expl = np.array(expl.dot(proj_perp(graph.graph["ks"][ei])), dtype=np.complex128)
        expl += (
            abelian_scale
            * proj_paral(graph.graph["ks"][ei])
            * np.exp(1.0j * graph.graph["lengths"][ei] * norm(graph.graph["ks"][ei]))
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


def construct_so3_weight_matrix(graph, with_k=True, abelian_scale=1.0):
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
        w_paral = abelian_scale * np.exp(2.0j * length * norm(chi)) * proj_paral(chi)
        w = w_perp + w_paral - np.eye(3)

        winv = linalg.inv(w)

        if with_k:
            winv = (chi.dot(proj_perp(chi)) + 1.0j * norm(chi) * proj_paral(chi)).dot(winv)

        Winv[_ext(2 * ei), _ext(2 * ei)] = winv
        Winv[_ext(2 * ei + 1), _ext(2 * ei + 1)] = winv
    return Winv


def construct_so3_laplacian(wavenumber, graph, abelian_scale=1.0, chis=None):
    """Construct quantum laplacian from a graph."""
    set_so3_wavenumber(graph, wavenumber, chis=chis)
    BT, B = construct_so3_incidence_matrix(graph, abelian_scale=abelian_scale)
    Winv = construct_so3_weight_matrix(graph, abelian_scale=abelian_scale)
    return BT.dot(Winv).dot(B)


def so3_mode_on_nodes(laplacian, quality_thresh=1e2):
    v0 = np.random.random(laplacian.shape[0])
    min_eigenvalue, node_solution = sc.sparse.linalg.eigs(
        laplacian, k=1, sigma=0, v0=v0, which="LM"
    )
    if abs(min_eigenvalue[0]) > quality_thresh:
        raise Exception(
            "Not a mode, as quality is too high: "
            + str(abs(min_eigenvalue[0]))
            + " > "
            + str(quality_thresh)
        )

    return node_solution[:, 0]


class WorkerScan:
    """Worker to scan complex frequency."""

    def __init__(self, graph, quality_method="eigenvalue"):
        self.graph = graph
        self.quality_method = quality_method
        np.random.seed(42)

    def __call__(self, freq):
        L = construct_so3_laplacian(to_complex(freq), self.graph, abelian_scale=1.0)
        return laplacian_quality(L)


def scan_frequencies_so3(graph, quality_method="eigenvalue"):
    """Scan a range of complex frequencies and return mode qualities."""
    ks, alphas = get_scan_grid(graph)
    freqs = [[k, a] for k in ks for a in alphas]

    worker_scan = WorkerScan(graph, quality_method=quality_method)
    chunksize = max(1, int(0.1 * len(freqs) / graph.graph["params"]["n_workers"]))
    with multiprocessing.Pool(graph.graph["params"]["n_workers"]) as pool:
        qualities_list = list(
            tqdm(
                pool.imap(worker_scan, freqs, chunksize=chunksize),
                total=len(freqs),
            )
        )

    id_k = [k_i for k_i in range(len(ks)) for a_i in range(len(alphas))]
    id_a = [a_i for k_i in range(len(ks)) for a_i in range(len(alphas))]
    qualities = sc.sparse.coo_matrix(
        (qualities_list, (id_k, id_a)),
        shape=(graph.graph["params"]["k_n"], graph.graph["params"]["alpha_n"]),
    ).toarray()

    return qualities

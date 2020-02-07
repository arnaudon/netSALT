"""main functions of NAQ graphs"""
import multiprocessing

import numpy as np
import scipy as sc

from .graph_construction import construct_laplacian

class WorkerScan:
    """worker to scan complex frequency"""

    def __init__(self, graph, dispersion_relation):
        self.graph = graph
        self.dispersion_relation = dispersion_relation

    def __call__(self, freq):
        return mode_quality(
            construct_laplacian(
                freq[0] - 1.0j * freq[1], self.graph, self.dispersion_relation
            )
        )


def scan_frequencies(graph, dispersion_relation, params, n_workers=1):
    """scan a range of complex frequencies and return mode qualities"""
    ks = np.linspace(params["k_min"], params["k_max"], params["k_n"])
    alphas = np.linspace(params["alpha_min"], params["alpha_max"], params["alpha_n"])
    freqs = [[k, a] for k in ks for a in alphas]

    worker_scan = WorkerScan(graph, dispersion_relation)
    pool = multiprocessing.Pool(n_workers)
    qualities_list = pool.map(worker_scan, freqs)

    id_k = [k_i for k_i in range(len(ks)) for a_i in range(len(alphas))]
    id_a = [a_i for k_i in range(len(ks)) for a_i in range(len(alphas))]
    qualities = sc.sparse.coo_matrix(
        (qualities_list, (id_k, id_a)), shape=(params["k_n"], params["alpha_n"])
    ).toarray()

    return ks, alphas, qualities


def mode_quality(laplacian, method="eigenvalue"):
    """return the quality of a mode encoded in the naq laplacian,
       either with smallest eigenvalue, or smallest singular value"""
    if method == "eigenvalue":
        return abs(
            sc.sparse.linalg.eigs(laplacian, k=1, sigma=0, return_eigenvectors=False)
        )[0]
    if method == "singularvalue":
        return sc.sparse.linalg.svds(
            laplacian, k=1, which="SM", return_singular_vectors=False
        )[0]

    return 0.



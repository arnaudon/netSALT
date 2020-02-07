"""main functions of NAQ graphs"""
import multiprocessing

import numpy as np
import scipy as sc

from .modes import (
    find_rough_modes_from_scan,
    clean_duplicate_modes,
    mode_quality,
    refine_mode_brownian_ratchet,
)
from .utils import _to_complex


class WorkerModes:
    """worker to refine modes"""

    def __init__(self, graph, dispersion_relation, params):
        self.graph = graph
        self.dispersion_relation = dispersion_relation
        self.params = params

    def __call__(self, mode):
        return refine_mode_brownian_ratchet(
            mode, self.graph, self.dispersion_relation, self.params
        )


class WorkerScan:
    """worker to scan complex frequency"""

    def __init__(self, graph, dispersion_relation):
        self.graph = graph
        self.dispersion_relation = dispersion_relation

    def __call__(self, freq):
        return mode_quality(_to_complex(freq), self.graph, self.dispersion_relation)


def scan_frequencies(graph, dispersion_relation, params, n_workers=1):
    """scan a range of complex frequencies and return mode qualities"""
    ks = np.linspace(params["k_min"], params["k_max"], params["k_n"])
    alphas = np.linspace(params["alpha_min"], params["alpha_max"], params["alpha_n"])
    freqs = [[k, a] for k in ks for a in alphas]

    worker_scan = WorkerScan(graph, dispersion_relation)
    pool = multiprocessing.Pool(n_workers)
    qualities_list = pool.map(worker_scan, freqs)
    pool.close()

    id_k = [k_i for k_i in range(len(ks)) for a_i in range(len(alphas))]
    id_a = [a_i for k_i in range(len(ks)) for a_i in range(len(alphas))]
    qualities = sc.sparse.coo_matrix(
        (qualities_list, (id_k, id_a)), shape=(params["k_n"], params["alpha_n"])
    ).toarray()

    return ks, alphas, qualities


def find_modes(ks, alphas, qualities, graph, dispersion_relation, params, n_workers=1):
    """find the modes from a scan"""
    rough_modes = find_rough_modes_from_scan(
        ks, alphas, qualities, min_distance=2, threshold_abs=1.
    )

    worker_modes = WorkerModes(graph, dispersion_relation, params)
    pool = multiprocessing.Pool(n_workers)
    refined_modes = pool.map(worker_modes, rough_modes)
    pool.close()

    return clean_duplicate_modes(refined_modes, ks[1] - ks[0], alphas[1] - alphas[0])

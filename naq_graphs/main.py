"""main functions of NAQ graphs"""
import multiprocessing

import numpy as np
import scipy as sc
from tqdm import tqdm

from .modes import (
    find_rough_modes_from_scan,
    clean_duplicate_modes,
    mode_quality,
    refine_mode_brownian_ratchet,
    pump_linear,
    lasing_threshold_linear,
)
from .utils import _to_complex


class WorkerModes:
    """worker to refine modes"""

    def __init__(self, modes, graph, params, D0s=None):
        self.graph = graph
        self.params = params
        self.modes = modes
        self.D0s = D0s  # to allow for different D0s for each mode

    def __call__(self, mode_id):
        if self.D0s is not None:
            self.params["D0"] = self.D0s[mode_id]
        return refine_mode_brownian_ratchet(
            self.modes[mode_id], self.graph, self.params
        )


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
    qualities_list = list(
        tqdm(
            pool.imap(worker_scan, freqs, chunksize=int(0.1 * len(freqs) / n_workers)),
            total=len(freqs),
        )
    )
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

    worker_modes = WorkerModes(rough_modes, graph, params)
    pool = multiprocessing.Pool(n_workers)
    refined_modes = pool.map(worker_modes, range(len(rough_modes)))
    pool.close()

    if len(refined_modes) == 0:
        raise Exception("No modes found!")

    modes = clean_duplicate_modes(refined_modes, ks[1] - ks[0], alphas[1] - alphas[0])
    return modes[np.argsort(modes[:, 1])]


def pump_trajectories(
    modes, graph, params, D0s, n_workers=1, return_approx=False
):  # pylint: disable=too-many-locals
    """For a sequence of D0s, find the mode positions of the modes modes."""
    pool = multiprocessing.Pool(n_workers)

    if return_approx:
        new_modes_approx_all = []

    new_modes = [modes.copy()]
    for d in tqdm(range(len(D0s) - 1)):
        new_modes_approx = new_modes[-1].copy()
        for m in range(len(modes)):
            new_modes_approx[m] = pump_linear(
                new_modes[-1][m], graph, params, D0s[d], D0s[d + 1]
            )

        if return_approx:
            new_modes_approx_all.append(new_modes_approx)

        params["D0"] = D0s[d + 1]
        worker_modes = WorkerModes(new_modes_approx, graph, params)
        new_modes_tmp = np.array(pool.map(worker_modes, range(len(new_modes_approx))))

        for i, mode in enumerate(new_modes_tmp):
            if mode is None:
                print("Mode not be updated, consider changing the search parameters.")
                new_modes_tmp[i] = new_modes[-1][i]
        new_modes.append(new_modes_tmp)

    pool.close()

    if return_approx:
        return np.array(new_modes), np.array(new_modes_approx_all)
    return np.array(new_modes)


def find_threshold_lasing_modes(  # pylint: disable=too-many-locals
    modes, graph, params, D0_max, D0_steps, threshold=1e-2, n_workers=1
):
    """find the threshold lasing modes and associated lasing thresholds"""
    pool = multiprocessing.Pool(n_workers)
    stepsize = params["search_stepsize"]

    new_modes = modes.copy()
    threshold_lasing_modes = []

    lasing_thresholds = []
    D0s = np.zeros(len(modes))

    while len(new_modes) > 0:

        print(len(new_modes), "modes left to find")

        new_D0s = np.zeros(len(new_modes))
        new_modes_approx = []
        for i, new_mode in enumerate(new_modes):
            new_D0s[i] = D0s[i] + lasing_threshold_linear(
                new_mode, graph, params, D0s[i]
            )

            new_D0s[i] = min(D0_steps + D0s[i], new_D0s[i])

            new_modes_approx.append(
                pump_linear(new_mode, graph, params, D0s[i], new_D0s[i])
            )

        # this is a trick to reduce the stepsizes as we are near the solution
        params["search_stepsize"] = (
            stepsize * np.mean(abs(new_D0s - D0s)) / np.mean(new_D0s)
        )

        worker_modes = WorkerModes(new_modes_approx, graph, params, D0s=new_D0s)
        new_modes_tmp = np.array(pool.map(worker_modes, range(len(new_modes_approx))))

        selected_modes = []
        selected_D0s = []
        for i, mode in enumerate(new_modes_tmp):
            if mode is None:
                print(
                    "A mode could not be updated, consider changing the search parameters"
                )
                selected_modes.append(new_modes[i])
                selected_D0s.append(D0s[i])

            elif abs(mode[1]) < threshold:
                threshold_lasing_modes.append(mode)
                lasing_thresholds.append(new_D0s[i])

            elif new_D0s[i] < D0_max:
                selected_modes.append(mode)
                selected_D0s.append(new_D0s[i])

        new_modes = selected_modes.copy()
        D0s = selected_D0s.copy()

    pool.close()

    return threshold_lasing_modes, lasing_thresholds

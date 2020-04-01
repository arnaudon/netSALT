"""Main functions of NAQ graphs"""
import multiprocessing

import numpy as np
import scipy as sc
from tqdm import tqdm

from . import modes
from .utils import _to_complex


class WorkerModes:
    """Worker to find modes."""

    def __init__(self, estimated_modes, graph, params, D0s=None, search_radii=None):
        "Init function of the worker." ""
        self.graph = graph
        self.params = params
        self.estimated_modes = estimated_modes
        self.D0s = D0s
        self.search_radii = search_radii

    def set_search_radii(self, mode):
        """This fixes a local search region set by search radii."""
        if self.search_radii is not None:
            self.params["k_min"] = mode[0] - self.search_radii[0]
            self.params["k_max"] = mode[0] + self.search_radii[0]
            self.params["alpha_min"] = mode[1] - self.search_radii[1]
            self.params["alpha_max"] = mode[1] + self.search_radii[1]
            # the 0.1 is hardcoded, and seems to be a good value
            self.params["search_stepsize"] = 0.1 * np.linalg.norm(self.search_radii)

    def __call__(self, mode_id):
        """Call function of the worker."""
        if self.D0s is not None:
            self.params["D0"] = self.D0s[mode_id]
        mode = self.estimated_modes[mode_id]
        self.set_search_radii(mode)
        return modes.refine_mode_brownian_ratchet(mode, self.graph, self.params)


class WorkerScan:
    """Worker to scan complex frequency."""

    def __init__(self, graph):
        self.graph = graph

    def __call__(self, freq):
        return modes.mode_quality(_to_complex(freq), self.graph)


def scan_frequencies(graph, params, n_workers=1):
    """Scan a range of complex frequencies and return mode qualities."""
    ks = np.linspace(params["k_min"], params["k_max"], params["k_n"])
    alphas = np.linspace(params["alpha_min"], params["alpha_max"], params["alpha_n"])
    freqs = [[k, a] for k in ks for a in alphas]

    worker_scan = WorkerScan(graph)
    pool = multiprocessing.Pool(n_workers)
    qualities_list = list(
        tqdm(pool.imap(worker_scan, freqs, chunksize=10), total=len(freqs),)
    )
    pool.close()

    id_k = [k_i for k_i in range(len(ks)) for a_i in range(len(alphas))]
    id_a = [a_i for k_i in range(len(ks)) for a_i in range(len(alphas))]
    qualities = sc.sparse.coo_matrix(
        (qualities_list, (id_k, id_a)), shape=(params["k_n"], params["alpha_n"])
    ).toarray()

    return ks, alphas, qualities


def find_modes(ks, alphas, qualities, graph, params, n_workers=1):
    """Find the modes from a scan."""
    estimated_modes = modes.find_rough_modes_from_scan(
        ks, alphas, qualities, min_distance=2, threshold_abs=1.0
    )
    print("Found", len(estimated_modes), "mode candidates.")
    search_radii = [1 * (ks[1] - ks[0]), 1 * (alphas[1] - alphas[0])]
    worker_modes = WorkerModes(
        estimated_modes, graph, params, search_radii=search_radii
    )
    pool = multiprocessing.Pool(n_workers)
    refined_modes = list(
        tqdm(
            pool.imap(worker_modes, range(len(estimated_modes))),
            total=len(estimated_modes),
        )
    )
    pool.close()

    if len(refined_modes) == 0:
        raise Exception("No modes found!")

    true_modes = modes.clean_duplicate_modes(
        refined_modes, ks[1] - ks[0], alphas[1] - alphas[0]
    )
    print("Found", len(true_modes), "after refinements.")
    return true_modes[np.argsort(true_modes[:, 1])]


def pump_trajectories(  # pylint: disable=too-many-locals
    passive_modes, graph, params, D0s, n_workers=1, return_approx=False
):
    """For a sequence of D0s, find the mode positions of the modes modes."""
    pool = multiprocessing.Pool(n_workers)

    if return_approx:
        new_modes_approx_all = []

    new_modes = [passive_modes.copy()]
    for d in tqdm(range(len(D0s) - 1)):
        new_modes_approx = new_modes[-1].copy()
        for m in range(len(passive_modes)):
            new_modes_approx[m] = modes.pump_linear(
                new_modes[-1][m], graph, params, D0s[d], D0s[d + 1]
            )

        if return_approx:
            new_modes_approx_all.append(new_modes_approx)

        params["D0"] = D0s[d + 1]
        params["alpha_min"] = -1e10  # to allow for going to upper plane
        worker_modes = WorkerModes(new_modes_approx, graph, params)
        new_modes_tmp = np.array(
            list(pool.imap(worker_modes, range(len(new_modes_approx))))
        )

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
    passive_modes, graph, params, D0_max, D0_steps, threshold=1e-2, n_workers=1
):
    """Find the threshold lasing modes and associated lasing thresholds."""
    pool = multiprocessing.Pool(n_workers)
    stepsize = params["search_stepsize"]

    new_modes = passive_modes.copy()
    threshold_lasing_modes = []

    lasing_thresholds = []
    D0s = np.zeros(len(passive_modes))

    while len(new_modes) > 0:
        print(len(new_modes), "modes left to find")

        new_D0s = np.zeros(len(new_modes))
        new_modes_approx = []
        for i, new_mode in enumerate(new_modes):
            new_D0s[i] = D0s[i] + modes.lasing_threshold_linear(
                new_mode, graph, params, D0s[i]
            )

            new_D0s[i] = min(D0_steps + D0s[i], new_D0s[i])

            new_modes_approx.append(
                modes.pump_linear(new_mode, graph, params, D0s[i], new_D0s[i])
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

"""Main functions of NAQ graphs"""
import multiprocessing

import numpy as np
import pandas as pd
import scipy as sc
from tqdm import tqdm

from . import modes
from .utils import from_complex, get_scan_grid, to_complex


class WorkerModes:
    """Worker to find modes."""

    def __init__(self, estimated_modes, graph, D0s=None, search_radii=None):
        """Init function of the worker."""
        self.graph = graph
        self.params = graph.graph["params"]
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
        return modes.mode_quality(to_complex(freq), self.graph)


def scan_frequencies(graph):
    """Scan a range of complex frequencies and return mode qualities."""
    ks, alphas = get_scan_grid(graph)
    freqs = [[k, a] for k in ks for a in alphas]

    worker_scan = WorkerScan(graph)
    pool = multiprocessing.Pool(graph.graph["params"]["n_workers"])
    qualities_list = list(
        tqdm(pool.imap(worker_scan, freqs, chunksize=10), total=len(freqs),)
    )
    pool.close()

    id_k = [k_i for k_i in range(len(ks)) for a_i in range(len(alphas))]
    id_a = [a_i for k_i in range(len(ks)) for a_i in range(len(alphas))]
    qualities = sc.sparse.coo_matrix(
        (qualities_list, (id_k, id_a)),
        shape=(graph.graph["params"]["k_n"], graph.graph["params"]["alpha_n"]),
    ).toarray()

    return qualities


def find_modes(graph, qualities):
    """Find the modes from a scan."""
    ks, alphas = get_scan_grid(graph)
    estimated_modes = modes.find_rough_modes_from_scan(
        ks, alphas, qualities, min_distance=2, threshold_abs=1.0
    )
    print("Found", len(estimated_modes), "mode candidates.")
    search_radii = [1 * (ks[1] - ks[0]), 1 * (alphas[1] - alphas[0])]
    worker_modes = WorkerModes(estimated_modes, graph, search_radii=search_radii)
    pool = multiprocessing.Pool(graph.graph["params"]["n_workers"])
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

    modes_df = _init_dataframe()
    modes_df["passive"] = [to_complex(true_mode) for true_mode in true_modes]
    return modes_df


def _init_dataframe():
    """Initialize multicolumn dataframe."""
    indexes = pd.MultiIndex(
        levels=[[], []], codes=[[], []], names=["data", "D0"], dtype=np.float
    )
    return pd.DataFrame(columns=indexes)


def pump_trajectories(modes_df, graph, return_approx=False):
    """For a sequence of D0s, find the mode positions of the modes modes."""

    D0s = np.linspace(
        graph.graph["params"]["D0_min"],
        graph.graph["params"]["D0_max"],
        graph.graph["params"]["D0_steps"],
    )

    pool = multiprocessing.Pool(graph.graph["params"]["n_workers"])
    n_modes = len(modes_df)

    pumped_modes = [[from_complex(mode) for mode in modes_df["passive"]]]
    pumped_modes_approx = pumped_modes.copy()
    for d in tqdm(range(len(D0s) - 1)):
        pumped_modes_approx.append(pumped_modes[-1].copy())
        for m in range(n_modes):
            pumped_modes_approx[-1][m] = modes.pump_linear(
                pumped_modes[-1][m], graph, D0s[d], D0s[d + 1]
            )

        worker_modes = WorkerModes(
            pumped_modes_approx[-1], graph, D0s=n_modes * [D0s[d + 1]]
        )
        pumped_modes.append(pool.map(worker_modes, range(n_modes)))
        for i, mode in enumerate(pumped_modes[-1]):
            if mode is None:
                print("Mode not be updated, consider changing the search parameters.")
                pumped_modes[-1][i] = pumped_modes[-2][i]

    pool.close()

    for D0, pumped_mode in zip(D0s, pumped_modes):
        modes_df["mode_trajectories", D0] = [to_complex(mode) for mode in pumped_mode]

    if return_approx:
        for D0, pumped_mode_approx in zip(D0s, pumped_modes_approx):
            modes_df["mode_trajectories_approx", D0] = [
                to_complex(mode) for mode in pumped_mode_approx
            ]

    return modes_df


def find_threshold_lasing_modes(modes_df, graph, threshold=1e-5):
    """Find the threshold lasing modes and associated lasing thresholds."""
    pool = multiprocessing.Pool(graph.graph["params"]["n_workers"])
    stepsize = graph.graph["params"]["search_stepsize"]

    D0_steps = (
        graph.graph["params"]["D0_max"] - graph.graph["params"]["D0_min"]
    ) / graph.graph["params"]["D0_steps"]

    new_modes = modes_df["passive"].to_numpy()

    threshold_lasing_modes = []
    lasing_thresholds = []
    D0s = np.zeros(len(modes_df))
    while len(new_modes) > 0:
        print(len(new_modes), "modes left to find")

        new_D0s = np.zeros(len(new_modes))
        new_modes_approx = []
        for i, new_mode in enumerate(new_modes):
            new_D0s[i] = D0s[i] + modes.lasing_threshold_linear(new_mode, graph, D0s[i])
            new_D0s[i] = min(new_D0s[i], D0_steps + D0s[i])

            new_modes_approx.append(
                modes.pump_linear(new_mode, graph, D0s[i], new_D0s[i])
            )

        # this is a trick to reduce the stepsizes as we are near the solution
        graph.graph["params"]["search_stepsize"] = (
            stepsize * np.mean(abs(new_D0s - D0s)) / np.mean(new_D0s)
        )

        worker_modes = WorkerModes(new_modes_approx, graph, D0s=new_D0s)
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

            elif new_D0s[i] < graph.graph["params"]["D0_max"]:
                selected_modes.append(mode)
                selected_D0s.append(new_D0s[i])

        new_modes = selected_modes.copy()
        D0s = selected_D0s.copy()

    pool.close()
    modes_df["threshold_lasing_modes"] = [
        to_complex(mode) for mode in threshold_lasing_modes
    ]
    modes_df["lasing_thresholds"] = lasing_thresholds
    return modes_df

"""functions related to modes search, etc..."""

import numpy as np
import scipy as sc

from skimage.feature import peak_local_max

from .graph_construction import construct_laplacian
from .utils import _to_complex


def laplacian_quality(laplacian, method="eigenvalue"):
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

    return 0.0


def mode_quality(mode, graph, dispersion_relation):
    """quality of a mode, small means good quality"""
    laplacian = construct_laplacian(_to_complex(mode), graph, dispersion_relation)
    return laplacian_quality(laplacian)


def find_rough_modes_from_scan(
    ks, alphas, qualities, min_distance=2, threshold_abs=10
):
    """use scipy.ndimage algorithms to detect minima in the scan"""
    data = 1.0 / (1e-10 + qualities)
    rough_mode_ids = peak_local_max(data, min_distance=min_distance, threshold_abs=threshold_abs)
    return [[ks[rough_mode_id[0]], alphas[rough_mode_id[1]]] for rough_mode_id in rough_mode_ids]


def refine_mode_brownian_ratchet(
    initial_mode,
    graph,
    dispersion_relation,
    params,
    disp=False,
    save_mode_trajectories=False,
):
    """Accurately find a mode from an initial guess, using brownian ratchet algorithm"""
    current_mode = initial_mode.copy()
    if save_mode_trajectories:
        mode_trajectories = [current_mode.copy()]

    initial_quality = mode_quality(current_mode, graph, dispersion_relation)
    current_quality = initial_quality

    search_stepsize = params["search_stepsize"]
    tries_counter = 0
    step_counter = 0
    while (
        current_quality > params["quality_threshold"]
        and step_counter < params["max_steps"]
    ):
        new_mode = (
            current_mode
            + search_stepsize
            * current_quality
            / initial_quality
            * np.random.uniform(-1, 1, 2)
        )
        new_quality = mode_quality(new_mode, graph, dispersion_relation)
        if disp:
            print(
                "New quality:",
                new_quality,
                "Step size:",
                search_stepsize,
                "Current mode: ",
                current_mode,
                "New mode:",
                new_mode,
            )

        # if the quality improves, update the mode
        if new_quality < current_quality:
            current_quality = new_quality
            current_mode = new_mode
            tries_counter = 0
            if save_mode_trajectories:
                mode_trajectories.append(current_mode.copy())
        else:
            tries_counter += 1

        # if no improvements after some iterations, multiply the steps by reduction factor
        if tries_counter > params["max_tries_reduction"]:
            search_stepsize *= params["reduction_factor"]
            tries_counter = 0

    if current_quality < params["quality_threshold"]:
        if save_mode_trajectories:
            return np.array(mode_trajectories)
        return current_mode


def clean_duplicate_modes(modes, k_size, alpha_size):
    """clean duplicate modes"""
    duplicate_mode_ids = []
    for mode_id_0, mode_0 in enumerate(modes):
        for mode_id_1, mode_1 in enumerate(modes[mode_id_0 + 1 :]):
            if (
                mode_id_1 + mode_id_0 + 1 not in duplicate_mode_ids
                and abs(mode_0[0] - mode_1[0]) < k_size
                and abs(mode_0[1] - mode_1[1]) < alpha_size
            ):
                duplicate_mode_ids.append(mode_id_0)
                break

    for ids in duplicate_mode_ids:
        del modes[ids]

    return modes

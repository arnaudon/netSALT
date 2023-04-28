"""Search algorighms for mode detections."""
import logging

import numpy as np
from skimage.feature import peak_local_max

from .quantum_graph import mode_quality

L = logging.getLogger(__name__)


def find_rough_modes_from_scan(ks, alphas, qualities, min_distance=2, threshold_abs=10):
    """Use scipy.ndimage algorithms to detect minima in the scan.

    Args:
        ks (list): list of real part of wavenumbers from the scan
        alphas (list): list of imaginary part of wavenumbers from the scan
        qualities (array): matrix with mode qualities for each pair of k/alpha
        min_distance (int): smallest distance between peaks (for skimage.feature.peak_local_max)
        threshold_abs (float): minimal intensity between peaks (for skimage.feature.peak_local_max)
    """
    data = 1.0 / (1e-10 + qualities)
    rough_mode_ids = peak_local_max(data, min_distance=min_distance, threshold_abs=threshold_abs)
    return [[ks[rough_mode_id[0]], alphas[rough_mode_id[1]]] for rough_mode_id in rough_mode_ids]


def refine_mode_brownian_ratchet(
    initial_mode,
    graph,
    params,
    disp=False,
    save_mode_trajectories=False,
    seed=42,
    quality_method="eigenvalue",
):
    """Accurately find a mode from an initial guess, using brownian ratchet algorithm.

    This algorithm is quite complex, but generally randomly propose a move to a new mode location,
    and accept if the quality decreases.

    TODO: get rid of params, or better handling of them, and complete the doc on the small details

    Args:
        initial_mode (complex): initial gues for a mode
        graph (graph): quantum graph
        params (dict): includes search_stepsize, max_steps, quality_threhsold, max_tries_reduction
            reduction_factor
        disp (bool): to print some state of the search for debuing
        save_mode_trajectories (bool): true to save intermediate modes
        seed (int): seed for rng
    """
    np.random.seed(seed)

    current_mode = initial_mode.copy()
    if save_mode_trajectories:
        mode_trajectories = [current_mode.copy()]

    initial_quality = mode_quality(current_mode, graph, quality_method=quality_method)
    current_quality = initial_quality

    search_stepsize = params.get("search_stepsize", 0.01)
    tries_counter = 0
    step_counter = 0
    while current_quality > params.get("quality_threshold", 1e-4) and step_counter < params.get(
        "max_steps", 10000
    ):
        new_mode = (
            current_mode
            + search_stepsize * current_quality / initial_quality * np.random.uniform(-1, 1, 2)
        )

        new_quality = mode_quality(new_mode, graph, quality_method=quality_method)

        if disp:
            L.debug(
                "New quality: %s, Step size: %s, Current mode: %s, New mode: %s, step %s",
                new_quality,
                search_stepsize,
                current_mode,
                new_mode,
                step_counter,
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
        if tries_counter > params.get("max_tries_reduction", 50):
            search_stepsize *= params.get("reduction_factor", 0.8)
            tries_counter = 0

        if search_stepsize < 1e-10:
            disp = True
            L.info("Warning: mode search stepsize under 1e-10 for mode: %s", current_mode)
            L.info("We retry from a larger one, but consider fine tuning search parameters.")
            search_stepsize = 1e-8
        step_counter += 1

    if current_quality < params.get("quality_threshold", 1e-4):
        if save_mode_trajectories:
            return np.array(mode_trajectories)
        return current_mode

    L.info("Maximum number of tries attained and no mode found, we retry from scratch!")
    params["search_stepsize"] = params.get("search_stepsize", 0.1) * 5

    return refine_mode_brownian_ratchet(
        initial_mode,
        graph,
        params,
        disp=disp,
        save_mode_trajectories=save_mode_trajectories,
        quality_method=quality_method,
    )


def clean_duplicate_modes(all_modes, k_size, alpha_size):
    """Remave duplicate modes from a list of modes with a threshold in real/imag(k).

    Args:
        all_modes (list): list of modes
        k_size (float): minimal distance in real(k)
        alpha_size (float): minimal distance in imag(k)
    """
    duplicate_mode_ids = []
    for mode_id_0, mode_0 in enumerate(all_modes):
        for mode_id_1, mode_1 in enumerate(all_modes[mode_id_0 + 1 :]):
            if (
                mode_id_1 + mode_id_0 + 1 not in duplicate_mode_ids
                and abs(mode_0[0] - mode_1[0]) < k_size
                and abs(mode_0[1] - mode_1[1]) < alpha_size
            ):
                duplicate_mode_ids.append(mode_id_0)
                break

    return np.delete(np.array(all_modes), duplicate_mode_ids, axis=0)

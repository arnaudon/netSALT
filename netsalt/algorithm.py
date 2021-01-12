"""All algorithms."""
import logging
import numpy as np
from skimage.feature import peak_local_max

from .quantum_graph import mode_quality

L = logging.getLogger(__name__)


def find_rough_modes_from_scan(ks, alphas, qualities, min_distance=2, threshold_abs=10):
    """Use scipy.ndimage algorithms to detect minima in the scan."""
    data = 1.0 / (1e-10 + qualities)
    rough_mode_ids = peak_local_max(data, min_distance=min_distance, threshold_abs=threshold_abs)
    return [[ks[rough_mode_id[0]], alphas[rough_mode_id[1]]] for rough_mode_id in rough_mode_ids]


def refine_mode_brownian_ratchet(
    initial_mode,
    graph,
    params,
    disp=False,
    save_mode_trajectories=False,
):
    """Accurately find a mode from an initial guess, using brownian ratchet algorithm."""
    current_mode = initial_mode.copy()
    if save_mode_trajectories:
        mode_trajectories = [current_mode.copy()]

    initial_quality = mode_quality(current_mode, graph)
    current_quality = initial_quality

    search_stepsize = params["search_stepsize"]
    tries_counter = 0
    step_counter = 0
    while current_quality > params["quality_threshold"] and step_counter < params["max_steps"]:
        new_mode = (
            current_mode
            + search_stepsize * current_quality / initial_quality * np.random.uniform(-1, 1, 2)
        )

        new_quality = mode_quality(new_mode, graph)

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
        if tries_counter > params["max_tries_reduction"]:
            search_stepsize *= params["reduction_factor"]
            tries_counter = 0
        if search_stepsize < 1e-10:
            disp = True
            L.info("Warning: mode search stepsize under 1e-10 for mode: %s", current_mode)
            L.info("We retry from a larger one, but consider fine tuning search parameters.")
            search_stepsize = 1e-8
        step_counter += 1
    if current_quality < params["quality_threshold"]:
        if save_mode_trajectories:
            return np.array(mode_trajectories)
        return current_mode
    L.info("Maximum number of tries attained and no mode found, we retry from scratch!")
    params["search_stepsize"] *= 5
    return refine_mode_brownian_ratchet(
        initial_mode,
        graph,
        params,
        disp=disp,
        save_mode_trajectories=save_mode_trajectories,
    )


def clean_duplicate_modes(all_modes, k_size, alpha_size):
    """Clean duplicate modes."""
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

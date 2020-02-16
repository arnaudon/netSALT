"""functions related to modes search, etc..."""

import numpy as np
import scipy as sc

from skimage.feature import peak_local_max

from .graph_construction import (
    construct_laplacian,
    construct_incidence_matrix,
    construct_weight_matrix,
)
from .utils import _to_complex, _from_complex
from .dispersion_relations import _gamma


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


def mode_quality(mode, graph):
    """quality of a mode, small means good quality"""
    laplacian = construct_laplacian(_to_complex(mode), graph)
    return laplacian_quality(laplacian)


def find_rough_modes_from_scan(ks, alphas, qualities, min_distance=2, threshold_abs=10):
    """use scipy.ndimage algorithms to detect minima in the scan"""
    data = 1.0 / (1e-10 + qualities)
    rough_mode_ids = peak_local_max(
        data, min_distance=min_distance, threshold_abs=threshold_abs
    )
    return [
        [ks[rough_mode_id[0]], alphas[rough_mode_id[1]]]
        for rough_mode_id in rough_mode_ids
    ]


def refine_mode_brownian_ratchet(
    initial_mode, graph, params, disp=False, save_mode_trajectories=False,
):
    """Accurately find a mode from an initial guess, using brownian ratchet algorithm"""
    current_mode = initial_mode.copy()
    if save_mode_trajectories:
        mode_trajectories = [current_mode.copy()]

    initial_quality = mode_quality(current_mode, graph)
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
        new_quality = mode_quality(new_mode, graph)
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

    return np.array(modes)


def compute_z_matrix(graph):
    """Construct the matrix Z used for computing the pump overlapping factor"""

    Z = sc.sparse.lil_matrix(
        (2 * len(graph.edges()), 2 * len(graph.edges())), dtype=np.complex
    )

    for ei, e in enumerate(list(graph.edges())):
        (u, v) = e[:2]

        l = graph[u][v]["length"]
        k = graph[u][v]["k"]

        Z[2 * ei, 2 * ei] = (np.exp(2.0j * l * k) - 1.0) / (2.0j * k)
        Z[2 * ei, 2 * ei + 1] = l * np.exp(1.0j * l * k)

        Z[2 * ei + 1, 2 * ei] = Z[2 * ei, 2 * ei + 1]
        Z[2 * ei + 1, 2 * ei + 1] = Z[2 * ei, 2 * ei]

    return Z.asformat("csc")


def _convert_edges(vector, n_edges):
    edge_vector = np.zeros(2 * n_edges, dtype=np.complex)
    edge_vector[::2] = vector
    edge_vector[1::2] = vector
    return edge_vector


def compute_overlapping_factor(mode, graph, params):  # pylint: disable=too-many-locals
    """compute the overlappin factor of a mode with the pump"""

    dielectric_constant = sc.sparse.diags(
        _convert_edges(params["dielectric_constant"], len(graph.edges))
    )
    in_mask = sc.sparse.diags(
        _convert_edges(1 * np.array(params["inner"]), len(graph.edges))
    )
    pump_mask = sc.sparse.diags(_convert_edges(params["pump"], len(graph.edges))).dot(
        in_mask
    )

    node_solution = mode_on_nodes(mode, graph)

    z_matrix = compute_z_matrix(graph)
    BT, Bout = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=False)

    edge_norm = Winv.dot(z_matrix).dot(Winv)
    pump_matrix = BT.dot(edge_norm).dot(pump_mask).dot(Bout)
    pump_norm = node_solution.T.dot(pump_matrix.dot(node_solution))

    edge_norm_dielectric = dielectric_constant.dot(edge_norm)
    in_matrix = BT.dot(edge_norm_dielectric).dot(in_mask).dot(Bout)
    in_norm = node_solution.T.dot(in_matrix.dot(node_solution))

    return pump_norm / in_norm


def pump_linear(mode, graph, params, D0_0, D0_1):
    """find the linear approximation of the new wavenumber,
    for an original pump mode with D0_0, to a new pump D0_1"""
    params["D0"] = D0_0
    overlapping_factor = compute_overlapping_factor(mode, graph, params)
    freq = _to_complex(mode)

    return _from_complex(
        freq
        * np.sqrt(
            (1.0 + _gamma(freq, params) * overlapping_factor * D0_0)
            / (1.0 + _gamma(freq, params) * overlapping_factor * D0_1)
        )
    )


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

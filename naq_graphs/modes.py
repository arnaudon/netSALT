"""Functions related to modes."""
import multiprocessing
from functools import partial

import numpy as np
import scipy as sc
from skimage.feature import peak_local_max
from tqdm import tqdm

from .dispersion_relations import gamma
from .graph_construction import (
    construct_incidence_matrix,
    construct_laplacian,
    construct_weight_matrix,
)
from .utils import from_complex, to_complex


def laplacian_quality(laplacian, method="eigenvalue"):
    """Return the quality of a mode encoded in the naq laplacian."""
    if method == "eigenvalue":
        try:
            return abs(
                sc.sparse.linalg.eigs(
                    laplacian, k=1, sigma=0, return_eigenvectors=False, which="LM"
                )
            )[0]
        except sc.sparse.linalg.ArpackNoConvergence:
            # If eigenvalue solver did not converge, set to 1.0,
            return 1.0
        except RuntimeError:
            print("Runtime error, things may be bad!")
            return abs(
                sc.sparse.linalg.eigs(
                    laplacian + 1e-5 * sc.sparse.eye(laplacian.shape[0]),
                    k=1,
                    sigma=0,
                    return_eigenvectors=False,
                    which="LM",
                )
            )[0]

            return 1.0e-20
    if method == "singularvalue":
        return sc.sparse.linalg.svds(
            laplacian,
            k=1,
            which="SM",
            return_singular_vectors=False,  # , v0=np.ones(laplacian.shape[0])
        )[0]
    return 1.0


def mode_quality(mode, graph):
    """Quality of a mode, small means good quality."""
    laplacian = construct_laplacian(to_complex(mode), graph)
    return laplacian_quality(laplacian)


def find_rough_modes_from_scan(ks, alphas, qualities, min_distance=2, threshold_abs=10):
    """Use scipy.ndimage algorithms to detect minima in the scan."""
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
    """Accurately find a mode from an initial guess, using brownian ratchet algorithm."""
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

        new_mode[0] = np.clip(new_mode[0], 0.8 * params["k_min"], 1.2 * params["k_max"])
        new_mode[1] = np.clip(new_mode[1], None, 1.2 * params["alpha_max"])

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
                "step",
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
            print("Warning: mode search stepsize under 1e-10 for mode:", current_mode)
            print(
                "We retry from a larger one, but consider fine tuning search parameters."
            )
            search_stepsize = 1e-8
        step_counter += 1
    if current_quality < params["quality_threshold"]:
        if save_mode_trajectories:
            return np.array(mode_trajectories)
        return current_mode
    print("WARNING: Maximum number of tries attained and no mode found!")


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


def compute_z_matrix(graph):
    """Construct the matrix Z used for computing the pump overlapping factor."""
    data_diag = (np.exp(2.0j * graph.graph["lengths"] * graph.graph["ks"]) - 1.0) / (
        2.0j * graph.graph["ks"]
    )
    data_off_diag = graph.graph["lengths"] * np.exp(
        1.0j * graph.graph["lengths"] * graph.graph["ks"]
    )
    data = np.dstack([data_diag, data_diag, data_off_diag, data_off_diag]).flatten()

    m = len(graph.edges)
    edge_ids = np.arange(m)
    row = np.dstack(
        [2 * edge_ids, 2 * edge_ids + 1, 2 * edge_ids, 2 * edge_ids + 1]
    ).flatten()
    col = np.dstack(
        [2 * edge_ids, 2 * edge_ids + 1, 2 * edge_ids + 1, 2 * edge_ids]
    ).flatten()
    return sc.sparse.csc_matrix((data, (col, row)), shape=(2 * m, 2 * m))


def _convert_edges(vector):
    edge_vector = np.zeros(2 * len(vector), dtype=np.complex)
    edge_vector[::2] = vector
    edge_vector[1::2] = vector
    return edge_vector


def _get_dielectric_constant_matrix(params):
    """Return sparse diagonal matrix of dielectric constants."""
    return sc.sparse.diags(_convert_edges(params["dielectric_constant"]))


def _get_mask_matrices(params):
    """Return sparse diagonal matrices of pump and inner edge masks."""
    in_mask = sc.sparse.diags(_convert_edges(np.array(params["inner"])))
    pump_mask = sc.sparse.diags(_convert_edges(params["pump"])).dot(in_mask)
    return in_mask, pump_mask


def _graph_norm(BT, Bout, Winv, z_matrix, node_solution, mask):
    """Compute the norm of the node solution on the graph."""

    weight_matrix = Winv.dot(z_matrix).dot(Winv)
    inner_matrix = BT.dot(weight_matrix).dot(mask).dot(Bout)
    norm = node_solution.T.dot(inner_matrix.dot(node_solution))
    return norm


def compute_overlapping_single_edges(passive_mode, graph):
    """Compute the overlappin factor of a mode with the pump."""
    dielectric_constant = _get_dielectric_constant_matrix(graph.graph["params"])
    in_mask, pump_mask = _get_mask_matrices(graph.graph["params"])
    inner_dielectric_constants = dielectric_constant.dot(in_mask)

    node_solution = mode_on_nodes(passive_mode, graph)

    z_matrix = compute_z_matrix(graph)

    BT, Bout = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=False)

    inner_norm = np.real(
        _graph_norm(BT, Bout, Winv, z_matrix, node_solution, inner_dielectric_constants)
    )

    pump_norm = np.zeros(len(graph.edges))
    for pump_edge, inner in enumerate(graph.graph["params"]["inner"]):
        if inner:
            mask = np.zeros(len(graph.edges))
            mask[pump_edge] = 1.0
            pump_mask = sc.sparse.diags(_convert_edges(mask))
            pump_norm[pump_edge] = np.real(
                _graph_norm(BT, Bout, Winv, z_matrix, node_solution, pump_mask)
            )

    return pump_norm / inner_norm


def compute_overlapping_factor(passive_mode, graph):
    """Compute the overlappin factor of a mode with the pump."""
    dielectric_constant = _get_dielectric_constant_matrix(graph.graph["params"])
    in_mask, pump_mask = _get_mask_matrices(graph.graph["params"])
    inner_dielectric_constants = dielectric_constant.dot(in_mask)

    node_solution = mode_on_nodes(passive_mode, graph)

    z_matrix = compute_z_matrix(graph)

    BT, Bout = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=False)

    pump_norm = _graph_norm(BT, Bout, Winv, z_matrix, node_solution, pump_mask)
    inner_norm = _graph_norm(
        BT, Bout, Winv, z_matrix, node_solution, inner_dielectric_constants
    )

    return pump_norm / inner_norm


def lasing_threshold_linear(mode, graph, D0):
    """Find the linear approximation of the new wavenumber."""
    graph.graph["params"]["D0"] = D0
    overlapping_factor = -compute_overlapping_factor(mode, graph)
    return 1.0 / (
        q_value(mode)
        * np.imag(gamma(to_complex(mode), graph.graph["params"]))
        * np.real(overlapping_factor)
    )


def q_value(mode):
    """Compute the q_value of a mode."""
    mode = from_complex(mode)
    return 0.5 * mode[0] / mode[1]


def pump_linear(mode_0, graph, D0_0, D0_1):
    """Find the linear approximation of the new wavenumber."""
    graph.graph["params"]["D0"] = D0_0
    overlapping_factor = compute_overlapping_factor(mode_0, graph)
    freq = to_complex(mode_0)
    gamma_overlap = gamma(freq, graph.graph["params"]) * overlapping_factor
    return from_complex(
        freq * np.sqrt((1.0 + gamma_overlap * D0_0) / (1.0 + gamma_overlap * D0_1))
    )


def mode_on_nodes(mode, graph):
    """Compute the mode solution on the nodes of the graph."""
    laplacian = construct_laplacian(to_complex(mode), graph)
    min_eigenvalue, node_solution = sc.sparse.linalg.eigs(
        laplacian, k=1, sigma=0, v0=np.ones(len(graph)), which="LM"
    )

    if abs(min_eigenvalue[0]) > graph.graph["params"]["quality_threshold"]:
        raise Exception(
            "Not a mode, as quality is too high: "
            + str(abs(min_eigenvalue[0]))
            + " > "
            + str(graph.graph["params"]["quality_threshold"])
            + ", mode: "
            + str(mode)
        )

    return node_solution[:, 0]


def flux_on_edges(mode, graph):
    """Compute the flux on each edge (in both directions)."""

    node_solution = mode_on_nodes(mode, graph)

    BT, _ = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=False)

    return Winv.dot(BT.T).dot(node_solution)


def mean_mode_on_edges(mode, graph):
    """Compute the average |E|^2 on each edge."""
    edge_flux = flux_on_edges(mode, graph)

    mean_edge_solution = np.zeros(len(graph.edges))
    for ei in range(len(graph.edges)):
        k = graph.graph["ks"][ei]
        l = graph.graph["lengths"][ei]
        z = np.zeros([2, 2], dtype=np.complex)

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


def _precomputations_mode_competition(graph, pump_mask, mode_threshold):
    """precompute some quantities for a mode for mode competitiion matrix"""
    mode, threshold = mode_threshold

    graph.graph["params"]["D0"] = threshold
    node_solution = mode_on_nodes(mode, graph)

    z_matrix = compute_z_matrix(graph)
    BT, Bout = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=False)
    pump_norm = _graph_norm(BT, Bout, Winv, z_matrix, node_solution, pump_mask)

    edge_flux = flux_on_edges(mode, graph) / np.sqrt(pump_norm)
    k_mu = graph.graph["ks"]
    gam = gamma(to_complex(mode), graph.graph["params"])

    return k_mu, edge_flux, gam


def _compute_mode_competition_element(lengths, params, data):
    """Computes a single element of the mode competition matrix."""
    mu_data, nu_data, gamma_nu = data
    k_mus, edge_flux_mu = mu_data
    k_nus, edge_flux_nu = nu_data

    matrix_element = 0
    for ei in range(len(lengths)):
        if params["pump"][ei] > 0.0 and params["inner"][ei]:
            k_mu = k_mus[ei]
            k_nu = k_nus[ei]
            length = lengths[ei]

            inner_matrix = np.zeros([4, 4], dtype=np.complex128)

            # A terms
            ik_tmp = 1.0j * (k_nu - np.conj(k_nu) + 2.0 * k_mu)
            inner_matrix[0, 0] = inner_matrix[3, 3] = (
                np.exp(ik_tmp * length) - 1.0
            ) / ik_tmp

            # B terms
            ik_tmp = 1.0j * (k_nu - np.conj(k_nu) - 2.0 * k_mu)
            inner_matrix[0, 3] = inner_matrix[3, 0] = (
                np.exp(2.0j * k_mu * length) * (np.exp(ik_tmp * length) - 1.0) / ik_tmp
            )

            # C terms
            ik_tmp = 1.0j * (k_nu + np.conj(k_nu) + 2.0 * k_mu)
            inner_matrix[1, 0] = inner_matrix[2, 3] = (
                np.exp(1.0j * (k_nu + 2.0 * k_mu) * length)
                - np.exp(-1.0j * np.conj(k_nu) * length)
            ) / ik_tmp

            # D terms
            ik_tmp = 1.0j * (k_nu + np.conj(k_nu) - 2.0 * k_mu)
            inner_matrix[1, 3] = inner_matrix[2, 0] = (
                np.exp(1.0j * k_nu * length)
                - np.exp(1.0j * (2.0 * k_mu - np.conj(k_nu)) * length)
            ) / ik_tmp

            # E terms
            ik_tmp = 1.0j * (k_nu - np.conj(k_nu))
            inner_matrix[0, 1] = inner_matrix[0, 2] = inner_matrix[3, 1] = inner_matrix[
                3, 2
            ] = (
                np.exp(1.0j * k_mu * length) * (np.exp(ik_tmp * length) - 1.0) / ik_tmp
            )

            # F terms
            ik_tmp = 1.0j * (k_nu + np.conj(k_nu))
            inner_matrix[1, 1] = inner_matrix[1, 2] = inner_matrix[2, 1] = inner_matrix[
                2, 2
            ] = (
                np.exp(1.0j * k_mu * length)
                * (
                    np.exp(1.0j * k_nu * length)
                    - np.exp(-1.0j * np.conj(k_nu) * length)
                )
                / ik_tmp
            )

            # left vector
            flux_nu_plus = edge_flux_nu[2 * ei]
            flux_nu_minus = edge_flux_nu[2 * ei + 1]
            left_vector = np.array(
                [
                    abs(flux_nu_plus) ** 2,
                    flux_nu_plus * np.conj(flux_nu_minus),
                    np.conj(flux_nu_plus) * flux_nu_minus,
                    abs(flux_nu_minus) ** 2,
                ]
            )

            # right vector
            flux_mu_plus = edge_flux_mu[2 * ei]
            flux_mu_minus = edge_flux_mu[2 * ei + 1]
            right_vector = np.array(
                [
                    flux_mu_plus ** 2,
                    flux_mu_plus * flux_mu_minus,
                    flux_mu_plus * flux_mu_minus,
                    flux_mu_minus ** 2,
                ]
            )

            matrix_element += left_vector.dot(inner_matrix.dot(right_vector))

    return -matrix_element * np.imag(gamma_nu)


def compute_mode_competition_matrix(graph, modes_df):
    """Compute the mode competition matrix, or T matrix."""
    threshold_modes = modes_df["threshold_lasing_modes"].to_numpy()
    lasing_thresholds = modes_df["lasing_thresholds"].to_numpy()

    threshold_modes = threshold_modes[lasing_thresholds < np.inf]
    lasing_thresholds = lasing_thresholds[lasing_thresholds < np.inf]

    pool = multiprocessing.Pool(graph.graph["params"]["n_workers"])

    precomp = partial(
        _precomputations_mode_competition,
        graph,
        _get_mask_matrices(graph.graph["params"])[1],
    )

    precomp_results = list(
        tqdm(
            pool.imap(precomp, zip(threshold_modes, lasing_thresholds)),
            total=len(lasing_thresholds),
        )
    )

    lengths = graph.graph["lengths"]

    input_data = []
    for mu in range(len(threshold_modes)):
        for nu in range(len(threshold_modes)):
            input_data.append(
                [
                    precomp_results[mu][:2],
                    precomp_results[nu][:2],
                    precomp_results[nu][2],
                ]
            )

    output_data = list(
        tqdm(
            pool.imap(
                partial(
                    _compute_mode_competition_element, lengths, graph.graph["params"]
                ),
                input_data,
            ),
            total=len(input_data),
        )
    )

    mode_competition_matrix = np.zeros(
        [len(threshold_modes), len(threshold_modes)], dtype=np.complex128
    )
    index = 0
    for mu in range(len(threshold_modes)):
        for nu in range(len(threshold_modes)):
            mode_competition_matrix[mu, nu] = output_data[index]
            index += 1

    pool.close()

    mode_competition_matrix_full = np.zeros(
        [
            len(modes_df["threshold_lasing_modes"]),
            len(modes_df["threshold_lasing_modes"]),
        ]
    )
    mode_competition_matrix_full[
        np.ix_(lasing_thresholds < np.inf, lasing_thresholds < np.inf)
    ] = np.real(mode_competition_matrix)

    return mode_competition_matrix_full


def _find_next_lasing_mode(
    pump_intensity,
    threshold_modes,
    lasing_thresholds,
    lasing_mode_ids,
    mode_competition_matrix,
):
    """Find next interacting lasing mode."""
    interacting_lasing_thresholds = np.ones(len(threshold_modes)) * 1e10
    for mu in range(len(threshold_modes)):
        if mu not in lasing_mode_ids:
            sub_mode_comp_matrix_mu = mode_competition_matrix[
                np.ix_(lasing_mode_ids + [mu,], lasing_mode_ids)
            ]
            sub_mode_comp_matrix_inv = np.linalg.pinv(
                mode_competition_matrix[np.ix_(lasing_mode_ids, lasing_mode_ids)]
            )
            sub_mode_comp_matrix_mu_inv = sub_mode_comp_matrix_mu[-1, :].dot(
                sub_mode_comp_matrix_inv
            )

            factor = (1.0 - sub_mode_comp_matrix_mu_inv.sum()) / (
                1.0
                - lasing_thresholds[mu]
                * sub_mode_comp_matrix_mu_inv.dot(
                    1.0 / lasing_thresholds[lasing_mode_ids]
                )
            )
            if lasing_thresholds[mu] * factor > pump_intensity:
                interacting_lasing_thresholds[mu] = lasing_thresholds[mu] * factor

    next_lasing_mode_id = np.argmin(interacting_lasing_thresholds)
    next_lasing_threshold = interacting_lasing_thresholds[next_lasing_mode_id]
    return next_lasing_mode_id, next_lasing_threshold


def compute_modal_intensities(modes_df, pump_intensities, mode_competition_matrix):
    """Compute the modal intensities of the modes up to D0, with D0_steps."""
    threshold_modes = modes_df["threshold_lasing_modes"]
    lasing_thresholds = modes_df["lasing_thresholds"]

    next_lasing_mode_id = np.argmin(lasing_thresholds)
    next_lasing_threshold = lasing_thresholds[next_lasing_mode_id]

    modal_intensities = np.zeros([len(threshold_modes), len(pump_intensities)])

    lasing_mode_ids = []
    interacting_lasing_thresholds = np.inf * np.ones(len(modes_df))
    interacting_lasing_thresholds[next_lasing_mode_id] = next_lasing_threshold
    for i, pump_intensity in tqdm(
        enumerate(pump_intensities), total=len(pump_intensities)
    ):
        while pump_intensity > next_lasing_threshold:
            lasing_mode_ids.append(next_lasing_mode_id)
            next_lasing_mode_id, next_lasing_threshold = _find_next_lasing_mode(
                pump_intensity,
                threshold_modes,
                lasing_thresholds,
                lasing_mode_ids,
                mode_competition_matrix,
            )
            interacting_lasing_thresholds[next_lasing_mode_id] = next_lasing_threshold

        if len(lasing_mode_ids) > 0:
            mode_competition_matrix_inv = np.linalg.pinv(
                mode_competition_matrix[np.ix_(lasing_mode_ids, lasing_mode_ids)]
            )
            modal_intensities[
                lasing_mode_ids, i
            ] = pump_intensity * mode_competition_matrix_inv.dot(
                1.0 / lasing_thresholds[lasing_mode_ids]
            ) - mode_competition_matrix_inv.sum(
                1
            )

    modes_df["interacting_lasing_thresholds"] = interacting_lasing_thresholds

    if "modal_intensities" in modes_df:
        del modes_df["modal_intensities"]

    modal_intensities = np.array(modal_intensities).T
    for pump_intensity, modal_intensity in zip(pump_intensities, modal_intensities):
        modes_df["modal_intensities", pump_intensity] = modal_intensity
    print(
        len(np.where(modal_intensities[-1] > 0)[0]),
        "lasing modes out of",
        len(modal_intensities[-1]),
    )

    return modes_df

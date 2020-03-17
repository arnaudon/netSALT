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


def _convert_edges(vector):
    edge_vector = np.zeros(2 * len(vector), dtype=np.complex)
    edge_vector[::2] = vector
    edge_vector[1::2] = vector
    return edge_vector


def _get_dielectric_constant_matrix(params):
    """return sparse diagonal matrix of dielectric constants"""
    return sc.sparse.diags(_convert_edges(params["dielectric_constant"]))


def _get_mask_matrices(params):
    """return sparse diagonal matrices of pump and inner edge masks"""
    in_mask = sc.sparse.diags(_convert_edges(np.array(params["inner"])))
    pump_mask = sc.sparse.diags(_convert_edges(params["pump"])).dot(in_mask)
    return in_mask, pump_mask


def _graph_norm(BT, Bout, Winv, z_matrix, node_solution, mask):
    """compute the norm of the node solution on the graph"""

    weight_matrix = Winv.dot(z_matrix).dot(Winv)
    inner_matrix = BT.dot(weight_matrix).dot(mask).dot(Bout)
    norm = node_solution.T.dot(inner_matrix.dot(node_solution))
    return norm


def compute_overlapping_factor(mode, graph, params):  # pylint: disable=too-many-locals
    """compute the overlappin factor of a mode with the pump"""
    dielectric_constant = _get_dielectric_constant_matrix(params)
    in_mask, pump_mask = _get_mask_matrices(params)
    inner_dielectric_constants = dielectric_constant.dot(in_mask)

    node_solution = mode_on_nodes(mode, graph)

    z_matrix = compute_z_matrix(graph)

    BT, Bout = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=False)

    pump_norm = _graph_norm(BT, Bout, Winv, z_matrix, node_solution, pump_mask)
    inner_norm = _graph_norm(
        BT, Bout, Winv, z_matrix, node_solution, inner_dielectric_constants
    )

    return pump_norm / inner_norm


def lasing_threshold_linear(mode, graph, params, D0):
    """find the linear approximation of the new wavenumber,
    for an original pump mode with D0_0, to a new pump D0_1"""
    params["D0"] = D0
    overlapping_factor = -compute_overlapping_factor(mode, graph, params)
    return 1.0 / (
        q_value(mode)
        * np.imag(_gamma(_to_complex(mode), params))
        * np.real(overlapping_factor)
    )


def q_value(mode):
    """compute the q_value of a mode"""
    mode = _from_complex(mode)
    return 0.5 * mode[0] / mode[1]


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


def compute_mode_competition_matrix(graph, params, threshold_modes, lasing_thresholds):
    """compute the mode competition matrix, or T matrix"""

    in_mask, pump_mask = _get_mask_matrices(params)

    edge_fluxes = []
    pump_norms = []
    k_mus = []
    deltas = []
    lambdas = []
    gammas = []
    for mode, threshold in zip(threshold_modes, lasing_thresholds):
        params["D0"] = threshold

        # save flux on edges
        edge_fluxes.append(flux_on_edges(mode, graph))

        # save denominator term
        node_solution = mode_on_nodes(mode, graph)
        z_matrix = compute_z_matrix(graph)
        BT, Bout = construct_incidence_matrix(graph)
        Winv = construct_weight_matrix(graph, with_k=False)
        pump_norms.append(
            _graph_norm(BT, Bout, Winv, z_matrix, node_solution, pump_mask)
        )

        # save wavenumbers
        k_mu = np.array([graph[u][v]["k"] for u, v in graph.edges])
        k_mus.append(k_mu)
        deltas.append(k_mu - np.conj(k_mu))
        lambdas.append(k_mu + np.conj(k_mu))

        gammas.append(_gamma(_to_complex(mode), params))

    T = np.zeros([len(threshold_modes), len(threshold_modes)], dtype="complex64")
    lengths = np.array([graph[u][v]["length"] for u, v in graph.edges])

    for mu in range(len(threshold_modes)):
        for nu in range(len(threshold_modes)):
            for ei, e in enumerate(graph.edges):
                if ei in params["pump"] and ei in params["inner"]:
                    flux_mu_plus = edge_fluxes[mu][2 * ei]
                    flux_mu_minus = edge_fluxes[mu][2 * ei + 1]
                    flux_nu_plus = edge_fluxes[nu][2 * ei]
                    flux_nu_minus = edge_fluxes[nu][2 * ei + 1]

                    k_mu = k_mus[mu][ei]
                    k_nu = k_mus[nu][ei]
                    _delta = deltas[nu][ei]
                    _lambda = lambdas[nu][ei]

                    length = lengths[ei]

                    exp_term_1 = (
                        np.exp(1.0j * (2.0 * k_mu + _delta) * length) - 1.0
                    ) / (1.0j * (2.0 * k_mu + _delta))

                    T[mu, nu] += exp_term_1 * (
                        abs(flux_nu_plus) ** 2 * flux_mu_plus ** 2
                        + abs(flux_nu_minus) ** 2 * flux_mu_minus ** 2
                    )

                    exp_term_2 = (
                        np.exp(2.0j * k_mu * length) - np.exp(1.0j * _delta * length)
                    ) / (1.0j * (2 * k_mu - _delta))

                    T[mu, nu] += exp_term_2 * (
                        abs(flux_nu_plus) ** 2 * flux_mu_minus ** 2
                        + abs(flux_nu_minus) ** 2 * flux_mu_plus ** 2
                    )

                    exp_term_3 = (
                        np.exp(1.0j * k_mu * length)
                        * (np.exp(1.0j * _delta * length) - 1.0)
                        / (1.0j * _delta)
                    )

                    T[mu, nu] += (
                        2
                        * exp_term_3
                        * (
                            abs(flux_nu_plus) ** 2 * flux_mu_plus * flux_mu_minus
                            + abs(flux_nu_minus) ** 2 * flux_mu_plus * flux_mu_minus
                        )
                    )

                    exp_term_4 = (
                        np.exp(1.0j * (2.0 * k_mu + k_nu) * length)
                        - np.exp(-1.0j * np.conj(k_nu) * length)
                    ) / (1.0j * (2 * k_nu + _lambda))

                    T[mu, nu] += exp_term_4 * (
                        flux_nu_plus * np.conj(flux_nu_minus) * flux_mu_plus ** 2
                        + np.conj(flux_nu_plus) * flux_nu_minus * flux_mu_minus ** 2
                    )

                    exp_term_5 = (
                        np.exp(1.0j * (2.0 * k_mu - np.conj(k_nu)) * length)
                        - np.exp(1.0j * k_nu * length)
                    ) / (1.0j * (2 * k_nu - _lambda))

                    T[mu, nu] += exp_term_5 * (
                        flux_nu_plus * np.conj(flux_nu_minus) * flux_mu_minus ** 2
                        + np.conj(flux_nu_plus) * flux_nu_minus * flux_mu_plus ** 2
                    )

                    exp_term_6 = (
                        np.exp(1.0j * k_mu * length)
                        * (
                            np.exp(1.0j * k_nu * length)
                            - np.exp(-1.0j * np.conj(k_nu) * length)
                        )
                        / (1.0j * _lambda)
                    )

                    T[mu, nu] += (
                        2
                        * exp_term_6
                        * (
                            flux_nu_plus
                            * np.conj(flux_nu_minus)
                            * flux_mu_minus
                            * flux_mu_plus
                            + np.conj(flux_nu_plus)
                            * flux_nu_minus
                            * flux_mu_minus
                            * flux_mu_plus
                        )
                    )
            T[mu, nu] /= pump_norms[mu]
            T[mu, nu] *= -np.imag(gammas[nu])

    return np.real(T)


def _find_next_lasing_mode(
    threshold_modes, lasing_thresholds, lasing_mode_ids, mode_competition_matrix
):
    """find next interacting lasing mode"""
    interacting_lasing_thresholds = np.ones(len(threshold_modes)) * 1e10
    for mu in range(len(threshold_modes)):
        if mu not in lasing_mode_ids:
            sub_mode_comp_matrix_mu = mode_competition_matrix[
                np.ix_(lasing_mode_ids + [mu,], lasing_mode_ids)
            ]
            sub_mode_comp_matrix_inv = np.linalg.inv(
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
            interacting_lasing_thresholds[mu] = lasing_thresholds[mu] * factor

    next_lasing_mode_id = np.argmin(interacting_lasing_thresholds)
    next_lasing_threshold = interacting_lasing_thresholds[next_lasing_mode_id]
    return next_lasing_mode_id, next_lasing_threshold


def compute_modal_intensities(
    graph, params, threshold_modes, lasing_thresholds, pump_intensities
):
    """compute the modal intensities of the modes up to D0, with D0_steps"""

    lasing_thresholds = np.array(lasing_thresholds)

    mode_competition_matrix = compute_mode_competition_matrix(
        graph, params, threshold_modes, lasing_thresholds
    )

    next_lasing_mode_id = np.argmin(lasing_thresholds)
    next_lasing_threshold = lasing_thresholds[next_lasing_mode_id]

    modal_intensities = np.zeros([len(threshold_modes), len(pump_intensities)])

    lasing_mode_ids = []
    for i, pump_intensity in enumerate(pump_intensities):

        while pump_intensity > next_lasing_threshold:
            lasing_mode_ids.append(next_lasing_mode_id)
            next_lasing_mode_id, next_lasing_threshold = _find_next_lasing_mode(
                threshold_modes,
                lasing_thresholds,
                lasing_mode_ids,
                mode_competition_matrix,
            )
        if len(lasing_mode_ids) > 0:
            mode_competition_matrix_inv = np.linalg.inv(
                mode_competition_matrix[np.ix_(lasing_mode_ids, lasing_mode_ids)]
            )
            modal_intensities[
                lasing_mode_ids, i
            ] = pump_intensity * mode_competition_matrix_inv.dot(
                1.0 / lasing_thresholds[lasing_mode_ids]
            ) - mode_competition_matrix_inv.sum(
                1
            )

    return modal_intensities

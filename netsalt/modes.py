"""Functions related to modes."""
import multiprocessing
import warnings
from functools import partial
import logging
import os

import numpy as np
import pandas as pd
import scipy as sc
from scipy import optimize
from tqdm import tqdm

from .algorithm import (
    clean_duplicate_modes,
    find_rough_modes_from_scan,
    refine_mode_brownian_ratchet,
)
from .physics import gamma, q_value
from .quantum_graph import (
    construct_incidence_matrix,
    construct_laplacian,
    construct_weight_matrix,
    mode_quality,
)
from .utils import from_complex, get_scan_grid, to_complex

warnings.filterwarnings("ignore")
warnings.filterwarnings("error", category=np.ComplexWarning)

L = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

# pylint: disable=too-many-locals


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
        return refine_mode_brownian_ratchet(mode, self.graph, self.params)


class WorkerScan:
    """Worker to scan complex frequency."""

    def __init__(self, graph):
        self.graph = graph

    def __call__(self, freq):
        return mode_quality(to_complex(freq), self.graph)


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


def _init_dataframe():
    """Initialize multicolumn dataframe."""
    indexes = pd.MultiIndex(
        levels=[[], []], codes=[[], []], names=["data", "D0"], dtype=np.float
    )
    return pd.DataFrame(columns=indexes)


def find_modes(graph, qualities):
    """Find the modes from a scan."""
    ks, alphas = get_scan_grid(graph)
    estimated_modes = find_rough_modes_from_scan(
        ks, alphas, qualities, min_distance=2, threshold_abs=1.0
    )
    L.info("Found %s mode candidates.", len(estimated_modes))
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

    refined_modes = [
        refined_mode for refined_mode in refined_modes if refined_mode is not None
    ]

    true_modes = clean_duplicate_modes(
        refined_modes, ks[1] - ks[0], alphas[1] - alphas[0]
    )
    L.info("Found %s after refinements.", len(true_modes))

    modes_sorted = true_modes[np.argsort(true_modes[:, 1])]
    if graph.graph["params"]["n_modes_max"]:
        L.info(
            "...but we will use the top %s modes only",
            graph.graph["params"]["n_modes_max"],
        )
        modes_sorted = modes_sorted[: graph.graph["params"]["n_modes_max"]]

    modes_df = _init_dataframe()
    modes_df["passive"] = [to_complex(mode_sorted) for mode_sorted in modes_sorted]
    return modes_df


def _convert_edges(vector):
    """Convert single edge values to double edges."""
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


def compute_overlapping_single_edges(passive_mode, graph):
    """Compute the overlappin factor of a mode with the pump."""
    dielectric_constant = _get_dielectric_constant_matrix(graph.graph["params"])
    in_mask, _ = _get_mask_matrices(graph.graph["params"])
    inner_dielectric_constants = dielectric_constant.dot(in_mask)

    node_solution = mode_on_nodes(passive_mode, graph)

    z_matrix = compute_z_matrix(graph)

    BT, Bout = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=False)

    inner_norm = _graph_norm(
        BT, Bout, Winv, z_matrix, node_solution, inner_dielectric_constants
    )

    pump_norm = np.zeros(len(graph.edges), dtype=np.complex)
    for pump_edge, inner in enumerate(graph.graph["params"]["inner"]):
        if inner:
            mask = np.zeros(len(graph.edges))
            mask[pump_edge] = 1.0
            pump_mask = sc.sparse.diags(_convert_edges(mask))
            pump_norm[pump_edge] = _graph_norm(
                BT, Bout, Winv, z_matrix, node_solution, pump_mask
            )

    return np.real(pump_norm / inner_norm)


def compute_overlapping_factor(passive_mode, graph):
    """Compute the overlapping factor of a mode with the pump."""
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
    r"""Compute the average :math:`|E|^2` on each edge."""
    edge_flux = flux_on_edges(mode, graph)

    mean_edge_solution = np.zeros(len(graph.edges))
    for ei in range(len(graph.edges)):
        k = graph.graph["ks"][ei]
        length = graph.graph["lengths"][ei]
        z = np.zeros([2, 2], dtype=np.complex)

        z[0, 0] = (np.exp(1.0j * length * (k - np.conj(k))) - 1.0) / (
            1.0j * length * (k - np.conj(k))
        )
        z[0, 1] = (np.exp(1.0j * length * k) - np.exp(-1.0j * length * np.conj(k))) / (
            1.0j * length * (k + np.conj(k))
        )

        z[1, 0] = z[0, 1]
        z[1, 1] = z[0, 0]

        mean_edge_solution[ei] = np.real(
            np.conj(edge_flux[2 * ei : 2 * ei + 2]).T.dot(
                z.dot(edge_flux[2 * ei : 2 * ei + 2])
            )
        )

    return mean_edge_solution


def mean_mode_E4_on_edges(mode, graph):
    r"""Compute the average :math:`|E|^4` on each edge."""
    edge_flux = flux_on_edges(mode, graph)

    meanE4_edge_solution = np.zeros(len(graph.edges))
    for ei in range(len(graph.edges)):
        k = graph.graph["ks"][ei]
        length = graph.graph["lengths"][ei]
        z = np.zeros([4, 4], dtype=np.complex)

        z[0, 0] = (np.exp(2.0j * length * (k - np.conj(k))) - 1.0) / (
            2.0j * length * (k - np.conj(k))
        )
        z[1, 1] = (np.exp(2.0j * length * k) - np.exp(-2.0j * length * np.conj(k))) / (
            2.0j * length * (k + np.conj(k))
        )
        z[0, 1] = (
            (np.exp(1.0j * length * (k - np.conj(k))))
            * (np.exp(1.0j * length * k) - np.exp(-1.0j * length * k))
            / (2.0j * length * k)
        )
        z[0, 3] = np.exp(1.0j * length * (k - np.conj(k)))

        z[2, 2] = z[1, 1]
        z[3, 3] = z[0, 0]
        z[3, 0] = z[0, 3]
        z[1, 2] = z[0, 3]
        z[2, 1] = z[0, 3]
        z[1, 0] = z[0, 1]
        z[2, 3] = z[0, 1]
        z[3, 2] = z[0, 1]
        z[0, 2] = np.conj(z[0, 1])
        z[2, 0] = z[0, 2]
        z[1, 3] = z[0, 2]
        z[3, 1] = z[0, 2]

        fluxvec = np.outer(
            np.conj(edge_flux[2 * ei : 2 * ei + 2]), edge_flux[2 * ei : 2 * ei + 2]
        ).flatten()
        meanE4_edge_solution[ei] = np.real(fluxvec.T.dot(z.dot(fluxvec)))

    return meanE4_edge_solution


def compute_mode_IPR(graph, modes_df, index, df_entry="passive"):
    """
    Compute the IPR of a mode
    """
    mode = modes_df[df_entry][index]

    mode_E4_mean = mean_mode_E4_on_edges(mode, graph)
    mode_E2_mean = mean_mode_on_edges(mode, graph)

    edge_length = np.zeros(len(graph.edges))
    integral_E2 = 0
    integral_E4 = 0
    for ei, inner in enumerate(graph.graph["params"]["inner"]):
        if inner:
            edge_length[ei] = graph.graph["lengths"][ei]
            integral_E2 += mode_E2_mean[ei] * edge_length[ei]
            integral_E4 += mode_E4_mean[ei] * edge_length[ei]

    tot_length = np.sum(edge_length)  # total inner length
    IPR = tot_length * integral_E4 / integral_E2 ** 2

    return IPR


def compute_IPRs(graph, modes_df, df_entry="passive"):
    """Compute IPR of all modes on the graph."""

    IPRs = []
    for index in tqdm(modes_df.index, total=len(modes_df)):
        IPR = compute_mode_IPR(graph, modes_df, index, df_entry)
        IPRs.append(IPR)

    if "IPR" in modes_df:
        del modes_df["IPR"]

    modes_df["IPR"] = IPRs

    return modes_df


def gamma_q_value(graph, modes_df, index, df_entry="passive"):
    """Compute gamma * Q factor for a given mode."""
    mode = modes_df[df_entry][index]
    gamma_q = (
        -q_value(mode)
        * np.imag(gamma(to_complex(mode), graph.graph["params"]))
    )
    return gamma_q


def compute_gamma_q_values(graph, modes_df, df_entry="passive"):
    """Compute gamma * Q factor for all modes on the graph."""

    gammaQs = []
    for index in tqdm(modes_df.index, total=len(modes_df)):
        gammaQ = gamma_q_value(graph, modes_df, index, df_entry)
        gammaQs.append(gammaQ)

    return gammaQs


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
    for ei, length in enumerate(lengths):
        if params["pump"][ei] > 0.0 and params["inner"][ei]:
            k_mu = k_mus[ei]
            k_nu = k_nus[ei]

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
    lasing_thresholds_all = modes_df["lasing_thresholds"].to_numpy()

    threshold_modes = threshold_modes[lasing_thresholds_all < np.inf]
    lasing_thresholds = lasing_thresholds_all[lasing_thresholds_all < np.inf]

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
        np.ix_(lasing_thresholds_all < np.inf, lasing_thresholds_all < np.inf)
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
    interacting_lasing_thresholds = np.ones(len(threshold_modes)) * np.inf
    for mu in range(len(threshold_modes)):
        if mu not in lasing_mode_ids:
            sub_mode_comp_matrix_mu = mode_competition_matrix[
                np.ix_(lasing_mode_ids + [mu], lasing_mode_ids)
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


def compute_modal_intensities(modes_df, max_pump_intensity, mode_competition_matrix):
    """Compute the modal intensities of the modes up to D0, with D0_steps."""
    threshold_modes = modes_df["threshold_lasing_modes"]
    lasing_thresholds = modes_df["lasing_thresholds"]

    next_lasing_mode_id = np.argmin(lasing_thresholds)
    next_lasing_threshold = lasing_thresholds[next_lasing_mode_id]
    L.debug("First lasing mode id: %s", next_lasing_mode_id)

    modal_intensities = pd.DataFrame(index=range(len(threshold_modes)))

    lasing_mode_ids = [next_lasing_mode_id]
    interacting_lasing_thresholds = np.inf * np.ones(len(modes_df))
    interacting_lasing_thresholds[next_lasing_mode_id] = next_lasing_threshold
    modal_intensities.loc[next_lasing_mode_id, next_lasing_threshold] = 0

    pump_intensity = next_lasing_threshold
    L.debug("Max pump intensity %s", max_pump_intensity)
    while pump_intensity <= max_pump_intensity:
        L.debug("Current pump intensity %s", pump_intensity)
        # 1) compute the current mode intensities
        mode_competition_matrix_inv = np.linalg.pinv(
            mode_competition_matrix[np.ix_(lasing_mode_ids, lasing_mode_ids)]
        )
        slopes = mode_competition_matrix_inv.dot(
            1.0 / lasing_thresholds[lasing_mode_ids]
        )
        shifts = mode_competition_matrix_inv.sum(1)

        modal_intensities.loc[lasing_mode_ids, pump_intensity] = (
            slopes * pump_intensity - shifts
        )

        if pump_intensity >= max_pump_intensity:
            # we stop here if we hit the max intensity, to get the final point
            L.debug("Max pump intensity reached.")
            break

        # 2) search for next lasing mode
        next_lasing_mode_id, next_lasing_threshold = _find_next_lasing_mode(
            pump_intensity,
            threshold_modes,
            lasing_thresholds,
            lasing_mode_ids,
            mode_competition_matrix,
        )
        L.debug("Next lasing threshold %s", next_lasing_threshold)

        # 3) deal with vanishing modes before next lasing mode
        vanishing_mode_id = None
        if any(slopes < -1e-10):
            vanishing_intensities = shifts / slopes
            vanishing_intensities[slopes > -1e-10] = np.inf

            if np.min(vanishing_intensities) < next_lasing_threshold:
                vanishing_mode_id = lasing_mode_ids[np.argmin(vanishing_intensities)]

        # 4) prepare for the next step
        if vanishing_mode_id is None:
            if next_lasing_threshold < max_pump_intensity:
                interacting_lasing_thresholds[
                    next_lasing_mode_id
                ] = next_lasing_threshold
                pump_intensity = next_lasing_threshold

                L.debug("New lasing mode id: %s", next_lasing_mode_id)
                lasing_mode_ids.append(next_lasing_mode_id)
            else:
                pump_intensity = max_pump_intensity

        elif np.min(vanishing_intensities) + 1e-10 > 0:
            L.debug("Vanishing mode id: %s", vanishing_mode_id)

            pump_intensity = np.min(vanishing_intensities) + 1e-10
            modal_intensities.loc[vanishing_mode_id, pump_intensity] = 0
            del lasing_mode_ids[
                np.where(np.array(lasing_mode_ids) == vanishing_mode_id)[0][0]
            ]

    modes_df["interacting_lasing_thresholds"] = interacting_lasing_thresholds

    if "modal_intensities" in modes_df:
        del modes_df["modal_intensities"]

    for pump_intensity in modal_intensities:
        modes_df["modal_intensities", pump_intensity] = modal_intensities[
            pump_intensity
        ]
    L.info(
        "%s lasing modes out of %s",
        len(np.where(modal_intensities.to_numpy()[:, -1] > 0)[0]),
        len(modal_intensities.index),
    )

    return modes_df


def pump_trajectories(modes_df, graph, return_approx=False):
    """For a sequence of D0s, find the mode positions of the modes modes."""

    D0s = np.linspace(
        0, graph.graph["params"]["D0_max"], graph.graph["params"]["D0_steps"],
    )

    pool = multiprocessing.Pool(graph.graph["params"]["n_workers"])
    n_modes = len(modes_df)

    pumped_modes = [[from_complex(mode) for mode in modes_df["passive"]]]
    pumped_modes_approx = pumped_modes.copy()
    for d in range(len(D0s) - 1):
        L.info(
            "Step %s / %s, computing for D0= %s",
            str(d + 1),
            str(len(D0s) - 1),
            str(D0s[d + 1]),
        )
        pumped_modes_approx.append(pumped_modes[-1].copy())
        for m in range(n_modes):
            pumped_modes_approx[-1][m] = pump_linear(
                pumped_modes[-1][m], graph, D0s[d], D0s[d + 1]
            )

        worker_modes = WorkerModes(
            pumped_modes_approx[-1], graph, D0s=n_modes * [D0s[d + 1]]
        )
        pumped_modes.append(
            list(tqdm(pool.imap(worker_modes, range(n_modes)), total=n_modes))
        )
        for i, mode in enumerate(pumped_modes[-1]):
            if mode is None:
                L.info("Mode not be updated, consider changing the search parameters.")
                pumped_modes[-1][i] = pumped_modes[-2][i]

    pool.close()
    if "mode_trajectories" in modes_df:
        del modes_df["mode_trajectories"]
    for D0, pumped_mode in zip(D0s, pumped_modes):
        modes_df["mode_trajectories", D0] = [to_complex(mode) for mode in pumped_mode]

    if return_approx:
        if "mode_trajectories_approx" in modes_df:
            del modes_df["mode_trajectories_approx"]
        for D0, pumped_mode_approx in zip(D0s, pumped_modes_approx):
            modes_df["mode_trajectories_approx", D0] = [
                to_complex(mode) for mode in pumped_mode_approx
            ]

    return modes_df


def _get_new_D0(arg, graph=None, D0_steps=0.1):
    """Internal function for multiprocessing."""
    mode_id, new_mode, D0 = arg
    increment = lasing_threshold_linear(new_mode, graph, D0)
    if increment > -D0_steps:
        new_D0 = abs(D0 + increment)
        new_D0 = min(new_D0, D0_steps + D0)
    else:
        L.debug("Intensity increment is negative, we set step to half max step.")
        new_D0 = D0 + 0.5 * D0_steps

    L.debug("Mode %s at intensity %s", mode_id, new_D0)
    new_modes_approx = pump_linear(new_mode, graph, D0, new_D0)
    return mode_id, new_D0, new_modes_approx


def find_threshold_lasing_modes(modes_df, graph):
    """Find the threshold lasing modes and associated lasing thresholds."""
    pool = multiprocessing.Pool(graph.graph["params"]["n_workers"])
    stepsize = graph.graph["params"]["search_stepsize"]

    D0_steps = graph.graph["params"]["D0_max"] / graph.graph["params"]["D0_steps"]

    new_modes = modes_df["passive"].to_numpy()

    threshold_lasing_modes = np.zeros([len(modes_df), 2])
    lasing_thresholds = np.inf * np.ones(len(modes_df))
    D0s = np.zeros(len(modes_df))
    current_modes = np.arange(len(modes_df))
    while len(current_modes) > 0:
        L.info("%s modes left to find", len(current_modes))

        new_D0s = np.zeros(len(modes_df))
        new_modes_approx = np.empty([len(new_modes), 2])
        args = (
            (mode_id, new_modes[mode_id], D0s[mode_id]) for mode_id in current_modes
        )
        for mode_id, new_D0, new_mode_approx in pool.imap(
            partial(_get_new_D0, graph=graph, D0_steps=D0_steps), args
        ):
            new_D0s[mode_id] = new_D0
            new_modes_approx[mode_id] = new_mode_approx

        # this is a trick to reduce the stepsizes as we are near the solution
        graph.graph["params"]["search_stepsize"] = (
            stepsize * np.mean(abs(new_D0s[new_D0s > 0] - D0s[new_D0s > 0])) / D0_steps
        )

        L.debug("Current search_stepsize: %s", graph.graph["params"]["search_stepsize"])
        worker_modes = WorkerModes(new_modes_approx, graph, D0s=new_D0s)
        new_modes_tmp = np.zeros([len(modes_df), 2])
        new_modes_tmp[current_modes] = list(
            tqdm(pool.imap(worker_modes, current_modes), total=len(current_modes))
        )

        to_delete = []
        for i, mode_index in enumerate(current_modes):
            if new_modes_tmp[mode_index] is None:
                L.info(
                    "A mode could not be updated, consider modifying the search parameters."
                )
                new_modes_tmp[mode_index] = new_modes[mode_index]
            elif abs(new_modes_tmp[mode_index][1]) < 1e-6:
                to_delete.append(i)
                threshold_lasing_modes[mode_index] = new_modes_tmp[mode_index]
                lasing_thresholds[mode_index] = new_D0s[mode_index]

            elif new_D0s[mode_index] > graph.graph["params"]["D0_max"]:
                to_delete.append(i)

        current_modes = np.delete(current_modes, to_delete)
        D0s = new_D0s.copy()
        new_modes = new_modes_tmp.copy()

    pool.close()

    modes_df["threshold_lasing_modes"] = [
        to_complex(mode) for mode in threshold_lasing_modes
    ]
    modes_df["lasing_thresholds"] = lasing_thresholds
    return modes_df


def lasing_threshold_linear(mode, graph, D0):
    """Find the linear approximation of the new wavenumber."""
    graph.graph["params"]["D0"] = D0
    return 1.0 / (
        q_value(mode)
        * -np.imag(gamma(to_complex(mode), graph.graph["params"]))
        * np.real(compute_overlapping_factor(mode, graph))
    )


def _overlap_matrix_element(graph, mode):
    """Compute the overlapp between a mode and each inner edges of the graph."""
    return list(
        -q_value(mode)
        * compute_overlapping_single_edges(mode, graph)
        * np.imag(gamma(to_complex(mode), graph.graph["params"]))
    )


def pump_cost(
    pump,
    modes_to_optimise,
    pump_overlapps,
    pump_min_size,
    mode="diff",
    n_modes=20,  # 10
):
    """Cost function to minimize."""
    pump = np.round(pump, 0)
    if pump.sum() < pump_min_size:
        return 1e10

    pump_with_opt_modes = pump_overlapps[modes_to_optimise][:, pump == 1].sum(axis=1)
    pump_without_opt_modes = sorted(
        pump_overlapps[~modes_to_optimise][:, pump == 1].sum(axis=1), reverse=True
    )

    if mode == "diff":
        return np.mean(pump_without_opt_modes[:n_modes]) - np.min(pump_with_opt_modes)
    if mode == "diff2":
        return np.max(pump_without_opt_modes) - np.min(pump_with_opt_modes)
    if mode == "ratio":
        return np.mean(pump_without_opt_modes[:n_modes]) / np.min(pump_with_opt_modes)
    raise Exception("Optimisation mode not understood")


def _optimise(seed, costf=None, bounds=None, disp=None, maxiter=100, popsize=5):
    """Wrapper of differnetial evolution algorithm to launch multiple seeds."""
    return optimize.differential_evolution(
        func=costf,
        bounds=bounds,
        maxiter=maxiter,
        disp=disp,
        popsize=popsize,
        workers=1,
        seed=seed,
        strategy="randtobest1bin",
    )


def compute_pump_overlapping_matrix(graph, modes_df):
    """Compute the matrix of pump overlapp with each edge."""
    with multiprocessing.Pool(graph.graph["params"]["n_workers"]) as pool:
        overlapp_iter = pool.imap(
            partial(_overlap_matrix_element, graph), modes_df["passive"]
        )
        pump_overlapps = np.empty([len(modes_df["passive"]), len(graph.edges)])
        for mode_id, overlapp in tqdm(
            enumerate(overlapp_iter), total=len(pump_overlapps)
        ):
            pump_overlapps[mode_id] = overlapp
    return pump_overlapps


def optimize_pump(
    modes_df,
    graph,
    lasing_modes_id,
    pump_min_frac=0.5,
    maxiter=500,
    popsize=5,
    seed=1,
    n_seeds=24,
    disp=True,
):
    """Optimise the pump for lasing a set of modes.

    Args:
        modes_df (dataframe): modes dataframe
        graph (networkx): quantum raph
        lasing_modes_id (list): list of modes to optimise the pump for lasing first
        pump_min_frac (float): minimum fraction of edges in the pump
        maxiter (int): maximum number of iterations (for scipy.optimize.differential_evolution)
        popsize (int): size of population (for scipy.optimize.differential_evolution)
        seed (int): seed for random number generator
        n_seeds (int): number of run with different seends in parallel
        disp (bool): if True, display the optimisation iterations

    Returns:
        optimal_pump, pump_overlapps, costs: best pump, overlapping matrix, all costs from seeds
    """
    if "pump" not in graph.graph["params"]:
        graph.graph["params"]["pump"] = np.ones(len(graph.edges))

    pump_overlapps = compute_pump_overlapping_matrix(graph, modes_df)

    mode_mask = np.array(len(pump_overlapps) * [False])
    mode_mask[lasing_modes_id] = True
    pump_min_size = int(
        pump_min_frac * len(np.where(graph.graph["params"]["inner"])[0])
    )

    _costf = partial(
        pump_cost,
        modes_to_optimise=mode_mask,
        pump_min_size=pump_min_size,
        pump_overlapps=pump_overlapps,
    )

    np.random.seed(seed)
    with multiprocessing.Pool(graph.graph["params"]["n_workers"]) as pool:
        results = list(
            tqdm(
                pool.imap(
                    partial(
                        _optimise,
                        costf=_costf,
                        bounds=len(graph.edges) * [(0, 1)],
                        disp=disp,
                        maxiter=maxiter,
                        popsize=popsize,
                    ),
                    np.random.randint(0, 1000000, n_seeds),
                ),
                total=n_seeds,
            )
        )

    costs = [result.fun for result in results]
    optimal_pump = np.round(results[np.argmin(costs)].x, 0)
    final_cost = _costf(optimal_pump)
    L.info("Final cost is: %s", final_cost)
    if final_cost > 0:
        L.info("This pump may not provide single lasing!")

    return optimal_pump, pump_overlapps, costs, final_cost

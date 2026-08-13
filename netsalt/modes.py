"""Mode search and lasing physics.

A mode is a wavenumber ``k`` at which the quantum Laplacian ``L(k)`` is
singular (``det L(k) = 0``). This module drives the whole flow: locating
passive modes (``scan_frequencies`` / ``find_passive_modes``), raising the
pump and tracking each mode to its lasing threshold where ``alpha = -Im(k)``
reaches 0 (``pump_trajectories``, ``find_threshold_lasing_modes``), and
solving the above-threshold mode competition for the steady-state modal
intensities (``compute_mode_competition_matrix``,
``compute_modal_intensities``). See the ``theory`` page in the docs for the
physical model.
"""

import contextlib
import logging
import multiprocessing
import warnings
from functools import partial
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy as sc
from tqdm import tqdm

try:
    from numpy.exceptions import ComplexWarning
except ImportError:  # NumPy < 1.25
    from numpy import ComplexWarning

from .algorithm import (
    clean_duplicate_modes,
    find_rough_modes_from_scan,
    refine_mode,
)
from .physics import dispersion_relation_pump_saturated, gamma, q_value
from .quantum_graph import (
    DENSE_EIG_MAX,
    construct_incidence_matrix,
    construct_laplacian,
    construct_weight_matrix,
    graph_with_params,
    graph_with_pump,
    mode_quality,
    oversample_graph,
    set_wavenumber,
)
from .utils import from_complex, get_scan_grid, to_complex

L = logging.getLogger(__name__)


@contextlib.contextmanager
def _scoped_warning_filters():
    """Suppress noisy warnings and promote ComplexWarning to error, scoped to a block.

    Module-level ``warnings.filterwarnings`` calls leak into the global warning
    state of every consumer of this library, so filters are applied per-call
    instead.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", category=ComplexWarning)
        yield


class WorkerModes:
    """Worker to find modes.

    Note on state: the per-mode pump (``D0``), search window, and search
    stepsize are applied to a throwaway copy of the graph and its ``params``
    inside :meth:`__call__`, so the shared ``graph.graph["params"]`` is never
    mutated in place. The refiner reads ``D0`` back off that local copy's
    params when it rebuilds the laplacian, which keeps each mode's computation
    self-contained.
    """

    def __init__(
        self,
        estimated_modes,
        graph,
        D0s=None,
        search_radii=None,
        search_stepsize=None,
        seed=42,
        quality_method="eigenvalue",
    ):
        """Init function of the worker."""
        self.graph = graph
        self.estimated_modes = estimated_modes
        self.D0s = D0s
        self.search_radii = search_radii
        self.search_stepsize = search_stepsize
        self.seed = seed
        self.quality_method = quality_method

    def _search_radii_updates(self, mode):
        """Per-mode local search window centred on the initial guess.

        Returned as a dict for the caller to apply to a *local* params copy —
        deliberately not mutating shared state.
        """
        return {
            "k_min": mode[0] - self.search_radii[0],
            "k_max": mode[0] + self.search_radii[0],
            "alpha_min": mode[1] - self.search_radii[1],
            "alpha_max": mode[1] + self.search_radii[1],
            # the 0.1 factor is hardcoded and seems to be a good value
            "search_stepsize": 0.1 * np.linalg.norm(self.search_radii),
        }

    def __call__(self, mode_id):
        """Call function of the worker."""
        mode = self.estimated_modes[mode_id]
        graph = self.graph
        # Apply the per-mode pump / search window / stepsize to a throwaway
        # graph + params copy so the shared graph.graph["params"] is never
        # mutated in place. One graph copy per mode candidate, which is cheap
        # next to the eigenvalue refinement that follows.
        overrides = {}
        if self.D0s is not None:
            overrides["D0"] = self.D0s[mode_id]
        if self.search_radii is not None:
            overrides.update(self._search_radii_updates(mode))
        if self.search_stepsize is not None:
            overrides["search_stepsize"] = self.search_stepsize
        if overrides:
            graph = graph_with_params(graph, **overrides)
        params = graph.graph["params"]
        # Derive a per-mode seed so each call has an independent RNG stream
        # rather than sharing ``self.seed`` across every mode in the pool.
        rng = np.random.default_rng([self.seed, mode_id])
        return refine_mode(
            mode,
            graph,
            params,
            quality_method=self.quality_method,
            rng=rng,
        )


class WorkerScan:
    """Worker to scan complex frequency."""

    def __init__(self, graph, quality_method="eigenvalue", seed=42):
        self.graph = graph
        self.quality_method = quality_method
        self.rng = np.random.default_rng(seed)

    def __call__(self, freq):
        return mode_quality(
            to_complex(freq), self.graph, quality_method=self.quality_method, rng=self.rng
        )


def scan_frequencies(graph, quality_method="eigenvalue"):
    """Scan a range of complex frequencies and return mode qualities."""
    ks, alphas = get_scan_grid(graph)
    freqs = [[k, a] for k in ks for a in alphas]

    worker_scan = WorkerScan(graph, quality_method=quality_method)
    chunksize = max(1, int(0.1 * len(freqs) / graph.graph["params"]["n_workers"]))
    with (
        _scoped_warning_filters(),
        multiprocessing.Pool(graph.graph["params"]["n_workers"]) as pool,
    ):
        qualities_list = list(
            tqdm(
                pool.imap(worker_scan, freqs, chunksize=chunksize),
                total=len(freqs),
            )
        )

    id_k = [k_i for k_i in range(len(ks)) for a_i in range(len(alphas))]
    id_a = [a_i for k_i in range(len(ks)) for a_i in range(len(alphas))]
    qualities = sc.sparse.coo_matrix(
        (qualities_list, (id_k, id_a)),
        shape=(graph.graph["params"]["k_n"], graph.graph["params"]["alpha_n"]),
    ).toarray()

    return qualities


def _init_dataframe():
    """Initialize multicolumn dataframe."""
    indexes = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["data", "D0"])
    return pd.DataFrame(columns=indexes)


def find_modes(graph, qualities, quality_method="eigenvalue", min_distance=2, threshold_abs=1.0):
    """Find the modes from a scan."""
    ks, alphas = get_scan_grid(graph)
    estimated_modes = find_rough_modes_from_scan(
        ks, alphas, qualities, min_distance=min_distance, threshold_abs=threshold_abs
    )
    L.info("Found %s mode candidates.", len(estimated_modes))
    search_radii = [1 * (ks[1] - ks[0]), 1 * (alphas[1] - alphas[0])]
    worker_modes = WorkerModes(
        estimated_modes, graph, search_radii=search_radii, quality_method=quality_method
    )
    with (
        _scoped_warning_filters(),
        multiprocessing.Pool(graph.graph["params"]["n_workers"]) as pool,
    ):
        refined_modes = list(
            tqdm(
                pool.imap(worker_modes, range(len(estimated_modes))),
                total=len(estimated_modes),
            )
        )

    if len(refined_modes) == 0:
        raise ValueError("No mode found!")

    refined_modes = [refined_mode for refined_mode in refined_modes if refined_mode is not None]

    true_modes = clean_duplicate_modes(refined_modes, ks[1] - ks[0], alphas[1] - alphas[0])
    L.info("Found %s after refinements.", len(true_modes))

    # sort by decreasing Q*\Gamma value
    _gammas = gamma(to_complex(true_modes.T), graph.graph["params"])
    q_factors = -1 * np.imag(_gammas) * true_modes[:, 0] / (2 * true_modes[:, 1])
    modes_sorted = true_modes[np.argsort(q_factors)[::-1]]
    q_factors = np.sort(q_factors)[::-1]
    if "n_modes_max" in graph.graph["params"] and graph.graph["params"]["n_modes_max"]:
        L.info(
            "...but we will use the top %s modes only",
            graph.graph["params"]["n_modes_max"],
        )
        modes_sorted = modes_sorted[: graph.graph["params"]["n_modes_max"]]
        q_factors = q_factors[: graph.graph["params"]["n_modes_max"]]

    modes_df = _init_dataframe()
    modes_df["passive"] = [to_complex(mode_sorted) for mode_sorted in modes_sorted]
    modes_df["q_factor"] = q_factors
    return modes_df


def find_passive_modes(graph, qualities=None, method=None, **kwargs):
    """Find all passive modes in the scan rectangle.

    Dispatches based on ``method`` (or ``params["mode_search_method"]``
    if ``method`` is None). Two paths are available:

    * ``"contour"`` — Beyn's contour integration via
      :func:`netsalt.find_modes_contour`. No refinement step needed;
      modes come back at quality ``1e-8`` or better. ~75× faster than
      the grid path on production buffon.
    * ``"grid"`` — legacy: :func:`scan_frequencies` (already done, pass
      via ``qualities``) + :func:`find_modes` with its peak-detection and
      per-mode refinement.

    Default selection mirrors the call signature so the legacy
    ``find_passive_modes(graph, qualities)`` keeps working:

    * ``method`` argument explicit → that wins.
    * else ``params["mode_search_method"]`` set → that wins.
    * else if ``qualities`` was supplied → ``"grid"`` (legacy default).
    * else → ``"contour"``.

    A loud ``UserWarning`` fires when ``method="contour"`` is chosen with
    a non-None ``qualities`` argument, since the qualities field is
    ignored by Beyn and the caller likely expected it to be used.

    Args:
        graph: a fully-configured netsalt quantum graph.
        qualities: the grid-scan quality field, required for
            ``method="grid"`` and ignored for ``"contour"``.
        method: ``"contour"``, ``"grid"``, or None to pick from
            ``graph.graph["params"]["mode_search_method"]`` (falling back
            to ``"grid"`` when ``qualities`` is provided, ``"contour"``
            otherwise).
        **kwargs: forwarded to the chosen implementation (e.g. ``n_k``,
            ``n_alpha``, ``n_quad``, ``probe_dim`` for contour; ``min_distance``,
            ``threshold_abs``, ``quality_method`` for grid).

    Returns:
        A modes dataframe with a ``passive`` column of complex ``k`` and
        a ``q_factor`` column.
    """
    if method is None:
        method = graph.graph["params"].get("mode_search_method")
    if method is None:
        method = "grid" if qualities is not None else "contour"

    if method == "contour" and qualities is not None:
        warnings.warn(
            "find_passive_modes(method='contour') ignores the supplied qualities field; "
            "pass method='grid' if you want the legacy peak-detection path on that scan.",
            stacklevel=2,
        )

    if method == "contour":
        from .contour import default_contour_n_k, find_modes_contour

        # Reasonable defaults; callers can override via kwargs.
        contour_defaults = {
            "n_k": kwargs.pop("n_k", None),
            "n_alpha": kwargs.pop("n_alpha", 2),
            "n_quad": kwargs.pop("n_quad", 80),
            "probe_dim": kwargs.pop("probe_dim", None),
        }
        if contour_defaults["n_k"] is None:
            contour_defaults["n_k"] = default_contour_n_k(
                graph, probe_dim=contour_defaults["probe_dim"]
            )
        modes = find_modes_contour(graph, **contour_defaults, **kwargs)
        # Build modes_df in the same shape find_modes returns.
        modes_df = _init_dataframe()
        modes_df["passive"] = [to_complex(m) for m in modes]
        # q_factor = -Im(gamma) * Re(k) / (2 * alpha) — the same formula
        # find_modes uses, applied to our contour output.
        if len(modes):
            _g = gamma(to_complex(modes.T), graph.graph["params"])
            modes_df["q_factor"] = -np.imag(_g) * modes[:, 0] / (2 * modes[:, 1])
        return modes_df

    if method == "grid":
        if qualities is None:
            raise ValueError(
                "method='grid' requires the qualities grid; call scan_frequencies first."
            )
        return find_modes(graph, qualities, **kwargs)

    raise ValueError(f"Unknown mode_search_method {method!r}; expected 'contour' or 'grid'.")


def _convert_edges(vector):
    """Convert single edge values to double edges."""
    edge_vector = np.zeros(2 * len(vector), dtype=np.complex128)
    edge_vector[::2] = vector
    edge_vector[1::2] = vector
    return edge_vector


def _get_dielectric_constant_matrix(params):
    """Return sparse diagonal matrix of dielectric constants."""
    return sc.sparse.diags(_convert_edges(params["dielectric_constant"]))


def _get_mask_matrices(params):
    """Return sparse diagonal matrices of pump and inner edge masks."""
    in_mask = sc.sparse.diags(_convert_edges(np.asarray(params["inner"])))
    pump_mask = sc.sparse.diags(_convert_edges(np.asarray(params["pump"]))).dot(in_mask)
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
    row = np.dstack([2 * edge_ids, 2 * edge_ids + 1, 2 * edge_ids, 2 * edge_ids + 1]).flatten()
    col = np.dstack([2 * edge_ids, 2 * edge_ids + 1, 2 * edge_ids + 1, 2 * edge_ids]).flatten()
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

    inner_norm = _graph_norm(BT, Bout, Winv, z_matrix, node_solution, inner_dielectric_constants)

    pump_norm = np.zeros(len(graph.edges), dtype=np.complex128)
    for pump_edge, inner in enumerate(graph.graph["params"]["inner"]):
        if inner:
            mask = np.zeros(len(graph.edges))
            mask[pump_edge] = 1.0
            pump_mask = sc.sparse.diags(_convert_edges(mask))
            pump_norm[pump_edge] = _graph_norm(BT, Bout, Winv, z_matrix, node_solution, pump_mask)

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
    inner_norm = _graph_norm(BT, Bout, Winv, z_matrix, node_solution, inner_dielectric_constants)

    return pump_norm / inner_norm


def pump_linear(mode_0, graph, D0_0, D0_1, overlapping_factor=None):
    """Find the linear approximation of the new wavenumber.

    ``overlapping_factor`` is the mode's overlap with the pump evaluated at
    ``D0_0``. It is an optional argument only so callers that already have it
    can avoid recomputing it — :func:`compute_overlapping_factor` costs an
    eigensolve plus several sparse products, and :func:`_get_new_D0` used to
    pay for it twice with identical arguments.
    """
    graph = graph_with_pump(graph, D0_0)
    if overlapping_factor is None:
        overlapping_factor = compute_overlapping_factor(mode_0, graph)
    freq = to_complex(mode_0)
    gamma_overlap = gamma(freq, graph.graph["params"]) * overlapping_factor
    return from_complex(freq * np.sqrt((1.0 + gamma_overlap * D0_0) / (1.0 + gamma_overlap * D0_1)))


def mode_on_nodes(mode, graph, check_quality=True):
    """Compute the mode solution on the nodes of the graph.

    ``check_quality`` (default ``True``) raises if the near-null eigenvalue is
    above ``quality_threshold`` -- i.e. ``mode`` is not actually a mode of
    ``graph``. The self-consistent / full-SALT solvers evaluate a mode's profile
    on a graph pumped *above* that mode's threshold (where the linear operator is
    no longer singular at the threshold frequency); they pass
    ``check_quality=False`` to take the smallest-eigenvalue field anyway. The
    result still reduces continuously to the true mode as the pump returns to
    threshold.
    """
    laplacian = construct_laplacian(to_complex(mode), graph)
    # Dense fast path for small graphs (see DENSE_EIG_MAX): a direct eigensolve is
    # several times faster than ARPACK shift-invert at small N and returns the same
    # nearest-zero eigenpair (smallest-magnitude eigenvalue and its eigenvector).
    # Fall back to ARPACK if the operator is non-finite (a probed ``k`` overflowed),
    # since ``np.linalg.eig`` would raise.
    dense = laplacian.toarray() if laplacian.shape[0] <= DENSE_EIG_MAX else None
    if dense is not None and np.isfinite(dense).all():
        eigenvalues, eigenvectors = np.linalg.eig(dense)
        idx = int(np.argmin(np.abs(eigenvalues)))
        min_eigenvalue = eigenvalues[idx]
        node_solution = eigenvectors[:, idx]
    else:
        min_eigenvalue_arr, node_solution_arr = sc.sparse.linalg.eigs(
            laplacian, k=1, sigma=0, v0=np.ones(len(graph)), which="LM"
        )
        min_eigenvalue = min_eigenvalue_arr[0]
        node_solution = node_solution_arr[:, 0]

    quality_thresh = graph.graph["params"].get("quality_threshold", 1e-4)
    if check_quality and abs(min_eigenvalue) > quality_thresh:
        raise ValueError(
            "Not a mode, as quality is too high: "
            + str(abs(min_eigenvalue))
            + " > "
            + str(quality_thresh)
            + ", mode: "
            + str(mode)
        )

    return node_solution


def flux_on_edges(mode, graph, check_quality=True):
    """Compute the flux on each edge (in both directions)."""

    node_solution = mode_on_nodes(mode, graph, check_quality=check_quality)

    _, B = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=False)

    return Winv.dot(B).dot(node_solution)


def mean_mode_on_edges(mode, graph, check_quality=True):
    r"""Compute the average :math:`Real(E^2)` on each edge."""
    edge_flux = flux_on_edges(mode, graph, check_quality=check_quality)
    return _mean_intensity_from_flux(edge_flux, graph)


def _mean_intensity_from_flux(edge_flux, graph):
    """Per-edge average ``|E|^2`` from a precomputed edge flux.

    Split out of :func:`mean_mode_on_edges` so callers that already hold the flux
    (and ``graph.graph['ks']`` for the same mode) avoid a second eigen-solve.
    """
    mean_edge_solution = np.zeros(len(graph.edges))
    for ei in range(len(graph.edges)):
        k = 1.0j * graph.graph["ks"][ei]
        length = graph.graph["lengths"][ei]
        z = np.zeros([2, 2], dtype=np.complex128)

        if abs(np.real(k)) > 0:  # in case we deal with closed graph, we have 0 / 0
            z[0, 0] = (np.exp(length * (k + np.conj(k))) - 1.0) / (length * (k + np.conj(k)))
        else:
            z[0, 0] = 1.0
            z[1, 1] = 1.0
        z[0, 1] = (np.exp(length * k) - np.exp(length * np.conj(k))) / (length * (k - np.conj(k)))
        z[1, 0] = z[0, 1]
        z[1, 1] = z[0, 0]

        mean_edge_solution[ei] = np.abs(
            edge_flux[2 * ei : 2 * ei + 2].T.dot(z.dot(np.conj(edge_flux[2 * ei : 2 * ei + 2])))
        )

    return mean_edge_solution


def mean_mode_E4_on_edges(mode, graph):
    r"""Compute the average :math:`|E|^4` on each edge."""
    edge_flux = flux_on_edges(mode, graph)

    meanE4_edge_solution = np.zeros(len(graph.edges))
    for ei in range(len(graph.edges)):
        k = graph.graph["ks"][ei]
        length = graph.graph["lengths"][ei]
        z = np.zeros([4, 4], dtype=np.complex128)

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
    IPR = tot_length * integral_E4 / integral_E2**2

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
    return -q_value(mode) * np.imag(gamma(to_complex(mode), graph.graph["params"]))


def compute_gamma_q_values(graph, modes_df, df_entry="passive"):
    """Compute gamma * Q factor for all modes on the graph."""
    return [
        gamma_q_value(graph, modes_df, index, df_entry)
        for index in tqdm(modes_df.index, total=len(modes_df))
    ]


def _precomputations_mode_competition(graph, pump_mask, mode_threshold, check_quality=True):
    """precompute some quantities for a mode for mode competition matrix"""
    mode, threshold = mode_threshold

    graph = graph_with_pump(graph, threshold)
    node_solution = mode_on_nodes(mode, graph, check_quality=check_quality)

    z_matrix = compute_z_matrix(graph)
    BT, Bout = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=False)
    pump_norm = _graph_norm(BT, Bout, Winv, z_matrix, node_solution, pump_mask)

    edge_flux = flux_on_edges(mode, graph, check_quality=check_quality) / np.sqrt(pump_norm)
    k_mu = graph.graph["ks"]
    gam = gamma(to_complex(mode), graph.graph["params"])

    return k_mu, edge_flux, gam


def _compute_mode_competition_element_reference(lengths, params, data, with_gamma=True):
    """Scalar reference implementation of a mode-competition matrix element.

    This is the original per-edge Python loop. It is kept **only** as the test
    oracle for the vectorised kernel below (see
    ``tests/test_unit.py::TestModeCompetitionVectorisation``); nothing in the
    library calls it.
    """
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
            inner_matrix[0, 0] = inner_matrix[3, 3] = (np.exp(ik_tmp * length) - 1.0) / ik_tmp

            # B terms
            ik_tmp = 1.0j * (k_nu - np.conj(k_nu) - 2.0 * k_mu)
            inner_matrix[0, 3] = inner_matrix[3, 0] = (
                np.exp(2.0j * k_mu * length) * (np.exp(ik_tmp * length) - 1.0) / ik_tmp
            )

            # C terms
            ik_tmp = 1.0j * (k_nu + np.conj(k_nu) + 2.0 * k_mu)
            inner_matrix[1, 0] = inner_matrix[2, 3] = (
                np.exp(1.0j * (k_nu + 2.0 * k_mu) * length) - np.exp(-1.0j * np.conj(k_nu) * length)
            ) / ik_tmp

            # D terms
            ik_tmp = 1.0j * (k_nu + np.conj(k_nu) - 2.0 * k_mu)
            inner_matrix[1, 3] = inner_matrix[2, 0] = (
                np.exp(1.0j * k_nu * length) - np.exp(1.0j * (2.0 * k_mu - np.conj(k_nu)) * length)
            ) / ik_tmp

            # E terms
            ik_tmp = 1.0j * (k_nu - np.conj(k_nu))
            inner_matrix[0, 1] = inner_matrix[0, 2] = inner_matrix[3, 1] = inner_matrix[3, 2] = (
                np.exp(1.0j * k_mu * length) * (np.exp(ik_tmp * length) - 1.0) / ik_tmp
            )

            # F terms
            ik_tmp = 1.0j * (k_nu + np.conj(k_nu))
            inner_matrix[1, 1] = inner_matrix[1, 2] = inner_matrix[2, 1] = inner_matrix[2, 2] = (
                np.exp(1.0j * k_mu * length)
                * (np.exp(1.0j * k_nu * length) - np.exp(-1.0j * np.conj(k_nu) * length))
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
                    flux_mu_plus**2,
                    flux_mu_plus * flux_mu_minus,
                    flux_mu_plus * flux_mu_minus,
                    flux_mu_minus**2,
                ]
            )

            matrix_element += left_vector.dot(inner_matrix.dot(right_vector))

    if with_gamma:
        return -matrix_element * np.imag(gamma_nu)
    return matrix_element


#: Peak working-set budget, in bytes, for the batched mode-competition kernel.
#:
#: The contraction over ``(mu, nu, edge)`` is blocked over ``mu`` so that the
#: temporaries stay inside this budget. The kernel keeps at most ~7 complex128
#: arrays of shape ``(mu_chunk, n_modes, n_edges)`` alive at once; the divisor
#: below is deliberately conservative (12) to leave headroom for NumPy's own
#: scratch buffers. 512 MiB was chosen because it is comfortably below a typical
#: compute-node per-core allowance while still giving large chunks at research
#: scale (M=400 modes, E=2500 edges gives a chunk of 2 rows, i.e. 2M elements of
#: vectorised work per block -- far more than enough to amortise loop overhead).
MODE_COMPETITION_MEMORY_BUDGET = 512 * 1024**2

_COMPETITION_TEMPORARIES = 12


def _competition_chunk_size(n_modes, n_edges, budget=None):
    """Number of ``mu`` rows to process per block under the memory budget."""
    budget = MODE_COMPETITION_MEMORY_BUDGET if budget is None else budget
    per_row = _COMPETITION_TEMPORARIES * 16 * max(int(n_modes), 1) * max(int(n_edges), 1)
    return max(1, int(budget // max(per_row, 1)))


def _competition_edge_mask(params):
    """Boolean mask of the edges that contribute to the competition matrix.

    Mirrors ``params["pump"][ei] > 0.0 and params["inner"][ei]`` from the scalar
    reference implementation.
    """
    pump = np.asarray(params["pump"], dtype=float)
    inner = np.asarray(params["inner"]).astype(bool)
    return (pump > 0.0) & inner


def _split_fluxes(fluxes, mask):
    """Split a stack of ``2E`` edge fluxes into the ``+`` / ``-`` halves on masked edges."""
    fluxes = np.atleast_2d(np.asarray(fluxes, dtype=np.complex128))
    return fluxes[:, 0::2][:, mask], fluxes[:, 1::2][:, mask]


class _CompetitionLeftTerms(NamedTuple):
    """Mode-nu (left-vector) per-edge factors, all of shape ``(n_modes, n_edges)``."""

    s: np.ndarray  # k_nu - conj(k_nu)
    bpc: np.ndarray  # k_nu + conj(k_nu)
    x: np.ndarray  # exp(1j * s * length)
    p: np.ndarray  # exp(1j * k_nu * length)
    q: np.ndarray  # exp(-1j * conj(k_nu) * length)
    l0: np.ndarray  # |flux_nu_plus|^2
    l1: np.ndarray  # flux_nu_plus * conj(flux_nu_minus)
    l2: np.ndarray  # conj(flux_nu_plus) * flux_nu_minus
    l3: np.ndarray  # |flux_nu_minus|^2
    ef: np.ndarray  # collapsed E/F contribution, see below


class _CompetitionRightTerms(NamedTuple):
    """Mode-mu (right-vector) per-edge factors, all of shape ``(n_modes, n_edges)``."""

    k2: np.ndarray  # 2 * k_mu
    y: np.ndarray  # exp(2j * k_mu * length)
    r0: np.ndarray  # flux_mu_plus ** 2
    r3: np.ndarray  # flux_mu_minus ** 2
    g: np.ndarray  # 2 * exp(1j * k_mu * length) * flux_mu_plus * flux_mu_minus


def _competition_left_terms(lengths, ks, fp, fm):
    """Precompute the nu-dependent factors of the inner 4x4 matrix and left vector."""
    ks = np.asarray(ks, dtype=np.complex128)
    ks_c = np.conj(ks)

    s = ks - ks_c
    bpc = ks + ks_c
    x = np.exp(1.0j * s * lengths)
    p = np.exp(1.0j * ks * lengths)
    q = np.exp(-1.0j * ks_c * lengths)

    l0 = np.abs(fp) ** 2
    l1 = fp * np.conj(fm)
    l2 = np.conj(fp) * fm
    l3 = np.abs(fm) ** 2

    # Degenerate wavenumbers (real or purely imaginary k_nu) make these
    # denominators vanish. The scalar reference divides by zero there too,
    # producing inf / nan and a RuntimeWarning; that behaviour is deliberately
    # preserved rather than "fixed" -- see TestModeCompetitionVectorisation.
    e_nu = (x - 1.0) / (1.0j * s)
    f_nu = (p - q) / (1.0j * bpc)

    # The E and F blocks of the inner matrix both factor as
    # ``exp(1j * k_mu * length) * (nu-only term)``, and both multiply the same
    # right-vector entry (``flux_mu_plus * flux_mu_minus``, which appears twice).
    # Their whole contribution therefore collapses to a single mode-by-mode
    # matrix product ``g @ ef.T`` instead of an (mu, nu, edge) tensor.
    ef = e_nu * (l0 + l3) + f_nu * (l1 + l2)

    return _CompetitionLeftTerms(s, bpc, x, p, q, l0, l1, l2, l3, ef)


def _competition_right_terms(lengths, ks, fp, fm):
    """Precompute the mu-dependent factors of the inner 4x4 matrix and right vector."""
    ks = np.asarray(ks, dtype=np.complex128)
    return _CompetitionRightTerms(
        k2=2.0 * ks,
        y=np.exp(2.0j * ks * lengths),
        r0=fp**2,
        r3=fm**2,
        g=2.0 * np.exp(1.0j * ks * lengths) * fp * fm,
    )


def _mode_competition_contraction(left, right, chunk=None):
    """Contract the inner 4x4 matrices over all edges for every ``(mu, nu)`` pair.

    Expanding ``left_vector @ inner_matrix @ right_vector`` with the symmetries of
    ``inner_matrix`` (``A`` on the diagonal corners, ``B`` on the anti-diagonal
    corners, ``C``/``D`` on the first column and last column of the middle rows,
    ``E``/``F`` filling the middle columns) leaves six scalar terms per edge::

        A * (l0 r0 + l3 r3) + B * (l0 r3 + l3 r0)
      + C * (l1 r0 + l2 r3) + D * (l1 r3 + l2 r0)
      + E * 2 p_mu * (l0 + l3) + F * 2 p_mu * (l1 + l2)

    The four transcendentals that the scalar loop evaluates per ``(mu, nu, edge)``
    all factor into a mu-only and a nu-only exponential, so only multiplies and
    divides remain inside the blocked tensor.
    """
    n_mu, n_edges = right.y.shape
    n_nu = left.x.shape[0]
    out = np.zeros((n_mu, n_nu), dtype=np.complex128)
    if n_edges == 0 or n_mu == 0 or n_nu == 0:
        return out

    if chunk is None:
        chunk = _competition_chunk_size(n_nu, n_edges)

    x, p, q = left.x[None], left.p[None], left.q[None]
    s, bpc = left.s[None], left.bpc[None]
    l0, l1, l2, l3 = left.l0[None], left.l1[None], left.l2[None], left.l3[None]

    for start in range(0, n_mu, chunk):
        stop = min(start + chunk, n_mu)
        y = right.y[start:stop, None, :]
        k2 = right.k2[start:stop, None, :]
        r0 = right.r0[start:stop, None, :]
        r3 = right.r3[start:stop, None, :]

        # A terms
        acc = (x * y - 1.0) / (1.0j * (s + k2)) * (l0 * r0 + l3 * r3)
        # B terms: exp(2j k_mu l) * (exp(1j (s - 2 k_mu) l) - 1) == x - y
        acc += (x - y) / (1.0j * (s - k2)) * (l0 * r3 + l3 * r0)
        # C terms
        acc += (p * y - q) / (1.0j * (bpc + k2)) * (l1 * r0 + l2 * r3)
        # D terms
        acc += (p - y * q) / (1.0j * (bpc - k2)) * (l1 * r3 + l2 * r0)
        out[start:stop] = acc.sum(axis=2)

    # E and F terms, collapsed to one complex matrix product.
    out += right.g @ left.ef.T
    return out


def _compute_mode_competition_element(lengths, params, data, with_gamma=True):
    """Computes a single element of the mode competition matrix.

    Vectorised over edges: drop-in replacement for
    :func:`_compute_mode_competition_element_reference`.
    """
    mu_data, nu_data, gamma_nu = data
    k_mus, edge_flux_mu = mu_data
    k_nus, edge_flux_nu = nu_data

    mask = _competition_edge_mask(params)
    lengths = np.asarray(lengths, dtype=float)[mask]

    fp_nu, fm_nu = _split_fluxes(edge_flux_nu, mask)
    fp_mu, fm_mu = _split_fluxes(edge_flux_mu, mask)
    left = _competition_left_terms(lengths, np.atleast_2d(k_nus)[:, mask], fp_nu, fm_nu)
    right = _competition_right_terms(lengths, np.atleast_2d(k_mus)[:, mask], fp_mu, fm_mu)

    matrix_element = _mode_competition_contraction(left, right)[0, 0]
    if with_gamma:
        return -matrix_element * np.imag(gamma_nu)
    return matrix_element


def _compute_mode_competition_matrix_batched(lengths, params, precomp, with_gamma=True):
    """Full ``M x M`` mode-competition matrix from the per-mode precomputations.

    ``precomp`` is the list of ``(k_mus, edge_flux, gamma)`` tuples produced by
    :func:`_precomputations_mode_competition`.
    """
    n_modes = len(precomp)
    if n_modes == 0:
        return np.zeros((0, 0), dtype=np.complex128)

    mask = _competition_edge_mask(params)
    lengths = np.asarray(lengths, dtype=float)[mask]

    ks = np.asarray([np.asarray(entry[0]) for entry in precomp], dtype=np.complex128)[:, mask]
    fluxes = np.asarray([np.asarray(entry[1]) for entry in precomp], dtype=np.complex128)
    fp, fm = _split_fluxes(fluxes, mask)

    left = _competition_left_terms(lengths, ks, fp, fm)
    right = _competition_right_terms(lengths, ks, fp, fm)

    matrix = _mode_competition_contraction(left, right)
    if with_gamma:
        gammas = np.asarray([entry[2] for entry in precomp], dtype=np.complex128)
        matrix = -matrix * np.imag(gammas)[None, :]
    return matrix


def _mode_competition_matrix_block(
    graph, threshold_modes, pumps, with_gamma=True, check_quality=True
):
    """Build the dense competition matrix over the lasing modes only.

    ``pumps`` is the per-mode pump strength at which each mode's profile is
    evaluated. The production matrix uses each mode's own lasing threshold
    (approximation #2); the self-consistent path passes a single operating pump
    for every mode. ``_precomputations_mode_competition`` already treats its
    ``threshold`` argument as the pump (``graph_with_pump``), so no change to the
    precompute kernel is needed — only the pump values fed in. ``check_quality``
    is forwarded to ``mode_on_nodes`` (set ``False`` when evaluating profiles at
    a pump above the modes' thresholds).
    """
    precomp = partial(
        _precomputations_mode_competition,
        graph,
        _get_mask_matrices(graph.graph["params"])[1],
        check_quality=check_quality,
    )

    n_workers = graph.graph["params"]["n_workers"]
    chunksize = max(1, int(0.1 * len(pumps) / n_workers))
    with multiprocessing.Pool(n_workers) as pool:
        precomp_results = list(
            tqdm(
                pool.imap(
                    precomp,
                    zip(threshold_modes, pumps, strict=True),
                    chunksize=chunksize,
                ),
                total=len(pumps),
            )
        )

    # The M*M elements used to be fanned out over a multiprocessing pool, one
    # Python edge-loop per element. The whole tensor contraction is now a handful
    # of batched array ops (blocked over mu to respect
    # MODE_COMPETITION_MEMORY_BUDGET), which is faster in a single process than
    # the pool ever was -- so no pool here.
    return np.real(
        _compute_mode_competition_matrix_batched(
            graph.graph["lengths"],
            graph.graph["params"],
            precomp_results,
            with_gamma=with_gamma,
        )
    )


def _scatter_competition_block(block, lasing_mask, n_total):
    """Place a lasing-only competition block back into a full n_total matrix."""
    full = np.zeros([n_total, n_total])
    full[np.ix_(lasing_mask, lasing_mask)] = block
    return full


def compute_mode_competition_matrix(graph, modes_df, with_gamma=True):
    """Compute the mode competition matrix, or T matrix.

    Each mode's profile is evaluated at its own lasing threshold pump (the
    linearised, near-threshold model).
    """
    threshold_modes_all = modes_df["threshold_lasing_modes"].to_numpy()
    lasing_thresholds_all = modes_df["lasing_thresholds"].to_numpy()
    lasing_mask = lasing_thresholds_all < np.inf

    threshold_modes = threshold_modes_all[lasing_mask]
    lasing_thresholds = lasing_thresholds_all[lasing_mask]

    block = _mode_competition_matrix_block(
        graph, threshold_modes, lasing_thresholds, with_gamma=with_gamma
    )
    return _scatter_competition_block(block, lasing_mask, len(threshold_modes_all))


def _find_next_lasing_mode(
    pump_intensity,
    modes_df,
    lasing_thresholds,
    lasing_mode_ids,
    mode_competition_matrix,
):
    """Find next interacting lasing mode."""
    interacting_lasing_thresholds = np.ones(len(modes_df)) * np.inf
    for mu in modes_df.index:
        # Modes with an infinite threshold never lase, so they are not
        # candidates. Running them through the formula below gave
        # ``inf * -0.0 = nan`` -- harmless, since ``nan > pump`` is False, but it
        # emitted a RuntimeWarning per mode per event and buried real warnings.
        if mu not in lasing_mode_ids and np.isfinite(lasing_thresholds[mu]):
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
                * sub_mode_comp_matrix_mu_inv.dot(1.0 / lasing_thresholds[lasing_mode_ids])
            )
            _int_thresh = lasing_thresholds[mu] * factor
            if _int_thresh > pump_intensity and _int_thresh > lasing_thresholds[mu]:
                interacting_lasing_thresholds[mu] = _int_thresh

    next_lasing_mode_id = np.argmin(interacting_lasing_thresholds)
    next_lasing_threshold = interacting_lasing_thresholds[next_lasing_mode_id]
    return next_lasing_mode_id, next_lasing_threshold


#: Condition number of the active competition submatrix above which the split of
#: intensity between modes is reported as unresolved. The solve inverts that
#: submatrix, so its conditioning is exactly the amplification factor from
#: threshold/competition errors to per-mode intensities. 1e8 leaves ~8 digits of
#: double precision, i.e. the per-mode split is meaningless beyond it even though
#: the *total* stays well determined.
COMPETITION_CONDITION_WARN = 1e8


def competition_conditioning(mode_competition_matrix, lasing_mode_ids):
    """Condition number of the competition submatrix over the given modes.

    Near-degenerate modes -- closer than the gain linewidth -- have nearly
    parallel competition rows, so the submatrix is ill-conditioned. Both places
    the sweep inverts it are then unreliable: the intensity *split* between
    those modes (only their sum is determined) and, upstream of that, *which* of
    them lases at all, since :func:`_find_next_lasing_mode` picks the winner
    from the same inverse. The solver uses ``pinv``, which returns an answer
    regardless; this is the number that says whether to believe it.
    """
    if not len(lasing_mode_ids):
        return 1.0
    submatrix = mode_competition_matrix[np.ix_(lasing_mode_ids, lasing_mode_ids)]
    return float(np.linalg.cond(submatrix))


def _intensity_slopes_shifts(mode_competition_matrix, lasing_thresholds, lasing_mode_ids):
    """Linear modal-intensity solve for the active set.

    Returns ``(slopes, shifts)`` such that the modal intensities at pump ``D0``
    are ``slopes * D0 - shifts``.
    """
    mode_competition_matrix_inv = np.linalg.pinv(
        mode_competition_matrix[np.ix_(lasing_mode_ids, lasing_mode_ids)]
    )
    slopes = mode_competition_matrix_inv.dot(1.0 / lasing_thresholds[lasing_mode_ids])
    shifts = mode_competition_matrix_inv.sum(1)
    return slopes, shifts


def compute_modal_intensities(modes_df, max_pump_intensity, mode_competition_matrix):
    """Compute the modal intensities of the modes up to D0, with D0_steps.

    Event-driven sweep over the pump strength with the fixed (near-threshold)
    competition matrix: intensities grow piecewise-linearly between mode
    activation / vanishing events.

    The per-mode intensities come from inverting the competition submatrix over
    the currently-lasing modes. When that submatrix is ill-conditioned (a
    near-degenerate mode pair), the split between those modes is not resolvable
    and a warning fires; the worst conditioning seen over the sweep is recorded
    in ``modes_df.attrs["competition_condition_max"]``. See
    :func:`competition_conditioning`.
    """
    lasing_thresholds = np.asarray(modes_df["lasing_thresholds"]).ravel()

    # Conditioning over every candidate (finite-threshold) mode, not just the
    # active set. A near-degenerate pair is usually never *co*-active -- the
    # sweep picks one and suppresses the other -- so watching only the active
    # set misses the pathology entirely. What is unresolved there is which of
    # them won.
    candidate_ids = [int(i) for i in np.where(np.isfinite(lasing_thresholds))[0]]
    candidate_condition = competition_conditioning(mode_competition_matrix, candidate_ids)
    worst_condition = 1.0

    next_lasing_mode_id = int(np.argmin(lasing_thresholds))
    next_lasing_threshold = lasing_thresholds[next_lasing_mode_id]
    L.debug("First lasing mode id: %s", next_lasing_mode_id)

    modal_intensities = pd.DataFrame(index=range(len(modes_df)))

    lasing_mode_ids = [next_lasing_mode_id]
    interacting_lasing_thresholds = np.inf * np.ones(len(modes_df))
    interacting_lasing_thresholds[next_lasing_mode_id] = next_lasing_threshold
    modal_intensities.loc[next_lasing_mode_id, next_lasing_threshold] = 0

    pump_intensity = next_lasing_threshold
    L.debug("Max pump intensity %s", max_pump_intensity)
    # safety cap so the event loop always terminates (it needs <~2*n_modes events)
    max_events = 100 * (len(modes_df) + 1)
    event = 0
    while pump_intensity <= max_pump_intensity:
        event += 1
        if event > max_events:
            warnings.warn(
                "modal-intensity sweep hit its event cap; returning the partial L--I curve.",
                stacklevel=2,
            )
            break
        L.debug("Current pump intensity %s", pump_intensity)

        # 1) compute the current mode intensities
        worst_condition = max(
            worst_condition, competition_conditioning(mode_competition_matrix, lasing_mode_ids)
        )
        slopes, shifts = _intensity_slopes_shifts(
            mode_competition_matrix, lasing_thresholds, lasing_mode_ids
        )

        # if we hit the max intensity, we add last points and stop
        if pump_intensity >= max_pump_intensity:
            L.debug("Max pump intensity reached.")
            modal_intensities.loc[lasing_mode_ids, max_pump_intensity] = np.clip(
                slopes * max_pump_intensity - shifts, 0.0, None
            )
            break

        modal_intensities.loc[lasing_mode_ids, pump_intensity] = np.clip(
            slopes * pump_intensity - shifts, 0.0, None
        )

        # 2) search for next lasing mode
        next_lasing_mode_id, next_lasing_threshold = _find_next_lasing_mode(
            pump_intensity,
            modes_df,
            lasing_thresholds,
            lasing_mode_ids,
            mode_competition_matrix,
        )
        L.debug("Next lasing threshold %s", next_lasing_threshold)

        # 3) deal with vanishing modes before next lasing mode
        vanishing_mode_id = None
        if any(slopes < -1e-10):
            vanishing_pump_intensities = shifts / slopes
            vanishing_pump_intensities[slopes > -1e-10] = np.inf

            if np.min(vanishing_pump_intensities) < next_lasing_threshold:
                vanishing_mode_id = lasing_mode_ids[np.argmin(vanishing_pump_intensities)]

        # 4) prepare for the next step
        if vanishing_mode_id is None:
            if next_lasing_threshold < max_pump_intensity:
                interacting_lasing_thresholds[next_lasing_mode_id] = next_lasing_threshold
                pump_intensity = next_lasing_threshold

                L.debug("New lasing mode id: %s", next_lasing_mode_id)
                lasing_mode_ids.append(next_lasing_mode_id)
            else:
                pump_intensity = max_pump_intensity

        elif np.min(vanishing_pump_intensities) + 1e-10 > 0:
            L.debug("Vanishing mode id: %s", vanishing_mode_id)

            mode_id = np.where(np.array(lasing_mode_ids) == vanishing_mode_id)[0][0]
            pump_intensity = np.min(vanishing_pump_intensities) + 1e-10

            # if it vanishes after max pump, we compute the modal amp at that pump
            if pump_intensity > max_pump_intensity:
                pump_intensity = max_pump_intensity
                modal_intensities.loc[vanishing_mode_id, max_pump_intensity] = (
                    slopes[mode_id] * max_pump_intensity - shifts[mode_id]
                )
            else:
                modal_intensities.loc[vanishing_mode_id, pump_intensity] = 0
            del lasing_mode_ids[mode_id]

    modes_df["interacting_lasing_thresholds"] = interacting_lasing_thresholds

    if "modal_intensities" in modes_df:
        del modes_df["modal_intensities"]

    for pump_intensity in modal_intensities:
        # we force to be of given precision for stability
        modes_df["modal_intensities", np.around(pump_intensity, 8)] = modal_intensities[
            pump_intensity
        ]
    modes_df.attrs["competition_condition_max"] = worst_condition
    modes_df.attrs["competition_condition_candidates"] = candidate_condition
    worst = max(worst_condition, candidate_condition)
    if worst > COMPETITION_CONDITION_WARN:
        warnings.warn(
            f"The mode-competition matrix reached condition number {worst:.2e} "
            f"(candidate set {candidate_condition:.2e}, worst active set {worst_condition:.2e}), "
            "so the per-mode result is not resolved: which of the near-degenerate modes lases, "
            "and how intensity splits between co-lasing ones, are both set by differences below "
            "the numerical noise floor. Their total is still well determined. Treat the "
            "per-mode intensities as indicative only.",
            stacklevel=2,
        )
    L.info(
        "%s lasing modes out of %s",
        len(np.where(modal_intensities.to_numpy()[:, -1] > 0)[0]),
        len(modal_intensities.index),
    )
    return modes_df


def _finalise_modal_intensities(modes_df, modal_intensities, interacting_lasing_thresholds):
    """Attach an L--I sweep to ``modes_df`` (shared by the iterative solvers).

    Mirrors the tail of :func:`compute_modal_intensities`: stores the
    interacting thresholds and one ``("modal_intensities", D0)`` column per pump,
    rounded to 8 decimals. Columns are written in increasing pump order so the
    L--I curve is monotone in ``D0`` regardless of the order they were computed.
    """
    modes_df["interacting_lasing_thresholds"] = interacting_lasing_thresholds

    if "modal_intensities" in modes_df:
        del modes_df["modal_intensities"]

    pumps = sorted(modal_intensities.columns)
    for pump_intensity in pumps:
        modes_df["modal_intensities", np.around(pump_intensity, 8)] = modal_intensities[
            pump_intensity
        ]

    n_lasing = 0
    if pumps:
        last = np.nan_to_num(modal_intensities[pumps[-1]].to_numpy())
        n_lasing = int(np.sum(last > 0))
    L.info("%s lasing modes out of %s", n_lasing, len(modal_intensities.index))
    return modes_df


def _single_mode_field_intensity(graph, mode, pump_mask):
    """Normalised per-edge field intensity of ``mode`` on ``graph``.

    Returns ``mean_mode_on_edges`` (per-edge ``|E|^2``) divided by the
    pump-region norm ``∫_pump |ψ|^2`` (``_graph_norm``) -- the same normalisation
    the linear competition matrix applies to its edge fluxes
    (``_precomputations_mode_competition``). The amplitude ``a`` it scales is thus
    a modal-intensity in the ``∫_pump |Ê|^2 = 1`` convention (its own unit; see
    :func:`compute_modal_intensities_full_salt_newton`).
    """
    # Nudge alpha off exactly 0: at the lasing point the operator is singular and
    # ARPACK shift-invert (sigma=0) cannot factorise it; a tiny imaginary offset
    # lifts the singularity and the field is continuous there.
    mode = [float(mode[0]), float(mode[1]) if abs(mode[1]) > 1e-7 else 1e-7]
    node_solution = mode_on_nodes(mode, graph, check_quality=False)
    z_matrix = compute_z_matrix(graph)
    BT, Bout = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=False)
    pump_norm = _graph_norm(BT, Bout, Winv, z_matrix, node_solution, pump_mask)
    # reuse the single eigen-solve above for the per-edge intensity (avoids a
    # second mode_on_nodes inside mean_mode_on_edges)
    edge_flux = Winv.dot(Bout).dot(node_solution)
    intensity = _mean_intensity_from_flux(edge_flux, graph)
    return intensity / abs(pump_norm)


def _saturated_graph_at(graph, mode, a, D0, pump, field_intensity):
    """Throwaway copy whose ``D0_eff`` carries the spatial-hole-burning denominator.

    ``D0_eff[e] = D0·pump[e] / (1 + Γ(k)·a·|Ê(x)|^2[e])`` with the Lorentzian
    gain clamp ``Γ(k) = -Im γ(k)``. The copy uses
    :func:`dispersion_relation_pump_saturated`, so ``construct_laplacian`` builds
    the saturated operator ``L_sat(k)`` from it without touching shared state.
    """
    gain_clamp = -np.imag(gamma(to_complex(mode), graph.graph["params"]))
    denom = 1.0 + gain_clamp * a * np.asarray(field_intensity)
    g = graph_with_params(graph, D0_eff=D0 * np.asarray(pump) / denom)
    g.graph["dispersion_relation"] = dispersion_relation_pump_saturated
    return g


def _refine_local(mode, graph, tol, max_steps, seed, k_window=1.0):
    """Local complex-``k`` refine driving ``(Re λ₁, Im λ₁) → 0`` via MINPACK ``hybr``.

    The engine of *continuous mode-following*: started from the mode's previous
    position it converges to the nearby root, so each mode tracks itself across
    pump and ARPACK cannot swap one mode's eigenvalue for another's. A fixed
    ``default_rng(seed)`` makes the ARPACK start vector (hence the residual)
    deterministic; an exactly-singular factorisation (the structural ``k = 0`` DC
    point, or the lasing point itself) is read a hair off in ``k``; and a refined
    mode that jumped more than ``k_window`` is rejected (it left its branch).
    """
    mode = np.asarray(mode, dtype=float)

    def residual(x):
        try:
            lam = mode_quality(
                x, graph, quality_method="complex_eigenvalue", rng=np.random.default_rng(seed)
            )
        except RuntimeError:
            lam = mode_quality(
                [x[0] + 1e-7, x[1]],
                graph,
                quality_method="complex_eigenvalue",
                rng=np.random.default_rng(seed),
            )
        return [lam.real, lam.imag]

    result = sc.optimize.root(
        residual,
        mode,
        method="hybr",
        tol=0,
        options={"maxfev": int(max_steps), "xtol": max(tol, 1e-9)},
    )
    refined = np.asarray(result.x)
    if abs(refined[0] - mode[0]) > k_window:
        return mode
    return refined


def _d0_eff_array(graph, modes, a_arr, D0, pump, fields):
    """Per-edge saturated effective pump ``D0_eff`` for the multi-mode field."""
    pump = np.asarray(pump)
    denom = np.ones(len(pump))
    for m, a_nu, f_nu in zip(modes, a_arr, fields, strict=True):
        gain_clamp = -np.imag(gamma(to_complex(m), graph.graph["params"]))
        denom = denom + gain_clamp * a_nu * np.asarray(f_nu)
    return D0 * pump / denom


def _saturated_graph_from_d0_eff(graph, d0_eff):
    """Throwaway copy carrying a precomputed ``D0_eff`` and the saturated law."""
    g = graph_with_params(graph, D0_eff=d0_eff)
    g.graph["dispersion_relation"] = dispersion_relation_pump_saturated
    return g


def _saturated_graph_multi(graph, modes, a_arr, D0, pump, fields):
    """Throwaway copy carrying the *multi-mode* spatial-hole-burning denominator.

    ``D0_eff[e] = D0·pump[e] / (1 + Σ_ν Γ(k_ν)·a_ν·|Ê_ν(x)|^2[e])`` -- every
    lasing mode burns the shared gain. Reduces to :func:`_saturated_graph_at` for
    one mode.
    """
    return _saturated_graph_from_d0_eff(graph, _d0_eff_array(graph, modes, a_arr, D0, pump, fields))


def _lam_real_k(graph, k, seed):
    """Complex ``λ₁`` of ``graph`` at the real wavenumber ``k``.

    Evaluated only on the real axis, where the saturated lasing operator is
    (near-)singular and ARPACK-friendly; an exactly-singular factorisation is read
    a hair off in ``k``.
    """
    try:
        return mode_quality(
            [k, 0.0], graph, quality_method="complex_eigenvalue", rng=np.random.default_rng(seed)
        )
    except RuntimeError:
        return mode_quality(
            [k + 1e-7, 0.0],
            graph,
            quality_method="complex_eigenvalue",
            rng=np.random.default_rng(seed),
        )


def _salt_block_residual(graph, x, n, D0, pump, fields, seed):
    """Stacked ``[Re λ₁, Im λ₁]`` at real ``k_μ`` for the active set, fields frozen.

    Holding the hole-burning fields fixed makes each residual a *single* eigensolve
    per mode (no inner fixed point) -- a clean, noise-free function of ``(k, a)``
    whose finite-difference Jacobian the trust region can rely on.
    """
    ks = x[:n]
    a = np.clip(x[n:], 0.0, None)
    g = _saturated_graph_multi(graph, [[k, 0.0] for k in ks], a, D0, pump, fields)
    out = np.empty(2 * n)
    for i in range(n):
        lam = _lam_real_k(g, ks[i], seed)
        out[2 * i], out[2 * i + 1] = lam.real, lam.imag
    return out


def _newton_onset_unit_scale(graph, mode0, field0, threshold, t_self):
    """Per-mode factor converting the Newton amplitude to the linear-intensity unit.

    The Newton amplitude solves the *operator* clamp; the linear/SPA intensity uses
    the competition diagonal ``T_μμ``. They share the onset slope once the Newton
    amplitude is rescaled by ``s_linear / s_newton``, with ``s_linear =
    1/(T_μμ·D0_thr)`` and ``s_newton = da/dD0`` the Newton amplitude's slope *at*
    threshold. ``s_newton`` is **analytic** -- first-order perturbation of the
    lasing condition. The saturated operator perturbs the effective gain
    ``g = γ·(D0·P - D0·Γ·a·H)`` with the coherent overlap factors

    .. math:: P = \\frac{\\int_{pump} E^2}{\\int_{inner} ε E^2}, \\qquad
              H = \\frac{\\int_{pump} f\\,E^2}{\\int_{inner} ε E^2},

    where ``f`` is the per-edge hole-burning intensity ``field0`` (per-edge
    *constant* on the oversampled work graph -- exactly the profile the operator
    itself clamps with, so the first order is exact, not a within-edge
    approximation). The mode frequency responds as ``δω ∝ -δg/(1+g)``
    (:func:`pump_linear`); holding ``Im ω = 0`` (the lasing condition) gives

    .. math:: s_{newton} = \\frac{{\\rm Im}[γP/Q]}{D_0^{thr}\\,Γ\\,{\\rm Im}[γH/Q]},
              \\qquad Q = 1 + γ D_0^{thr} P .

    With the shared ``∫_pump |E|^2 = 1`` normalization this comes out ≈ 1 --
    the Newton amplitude *is* the linear modal intensity to first order -- so the
    rescale is a small consistency correction, and the near-threshold agreement
    between the two solvers is a genuine prediction rather than a calibration
    (the earlier *measured* probe at ``1.2·D0_thr`` carried a secant bias of up
    to a few % on modes whose curve already bends there).
    """
    s_linear = 1.0 / (t_self * threshold) if (t_self > 0.0 and threshold > 0.0) else 0.0
    if s_linear <= 0.0:
        return 1.0
    g = graph_with_pump(graph, threshold)
    params = g.graph["params"]
    node_solution = mode_on_nodes(mode0, g, check_quality=False)
    z_matrix = compute_z_matrix(g)
    BT, Bout = construct_incidence_matrix(g)
    Winv = construct_weight_matrix(g, with_k=False)
    inner = np.asarray(params["inner"], dtype=float)
    eps_mask = _get_dielectric_constant_matrix(params).dot(sc.sparse.diags(_convert_edges(inner)))
    pump_profile = np.asarray(params["pump"], dtype=float) * inner
    hole_profile = pump_profile * np.asarray(field0, dtype=float)
    inner_norm = _graph_norm(BT, Bout, Winv, z_matrix, node_solution, eps_mask)
    p_overlap = (
        _graph_norm(
            BT, Bout, Winv, z_matrix, node_solution, sc.sparse.diags(_convert_edges(pump_profile))
        )
        / inner_norm
    )
    h_overlap = (
        _graph_norm(
            BT, Bout, Winv, z_matrix, node_solution, sc.sparse.diags(_convert_edges(hole_profile))
        )
        / inner_norm
    )
    gam = gamma(to_complex(mode0), params)
    gain_clamp = -np.imag(gam)
    q_factor = 1.0 + gam * threshold * p_overlap
    denom = threshold * gain_clamp * np.imag(gam * h_overlap / q_factor)
    if not (np.isfinite(denom) and abs(denom) > 0.0):
        return 1.0
    s_newton = float(np.imag(gam * p_overlap / q_factor) / denom)
    if not (np.isfinite(s_newton) and s_newton > 0.0):
        return 1.0
    return float(s_linear / s_newton)


NEWTON_DENSE_EIG_MAX = (
    50  # full_salt_newton runs its (banded, oversampled) eigensolves through ARPACK
)


def _auto_oversample_size(graph, modes_df, resolution=12, node_cap=3000):
    """Sub-edge length that resolves the lasing standing wave (for hole burning).

    The operator-level hole burning samples ``|E_ν(x)|^2`` per edge; with the bare
    edges (one sample per edge) the per-edge **mean** intensity over-estimates the
    spatial overlap between modes -- it washes out the within-edge nodes/antinodes
    where the coherent competition is actually weak -- so the saturation
    **over-clamps** and suppresses co-lasing modes that the competition matrix (and
    the Ge-Chong-Stone single-pole SALT, PRA 82, 063824) correctly lases. Sampling
    a few points per wavelength fixes it. The local wavelength is
    ``λ = 2π / (n·Re k)`` with ``n = sqrt(ε)``; target ``λ_min / resolution``,
    capped so the oversampled graph stays bounded. ``resolution`` defaults to 12
    (~λ/12): a convergence study on ``line_PRA`` shows the modal *intensities*
    converge to ~1% there, while the cheaper λ/6 (used before the ARPACK
    eigensolve scaling) got the lasing count right but was ~15% under-resolved.
    ARPACK makes λ/12 essentially free.
    """
    cand = np.where(np.asarray(modes_df["lasing_thresholds"]).ravel() < np.inf)[0]
    tms = modes_df["threshold_lasing_modes"].to_numpy()
    k_max = max((abs(from_complex(tms[i])[0]) for i in cand), default=0.0)
    if k_max <= 0.0:
        return None
    eps = [abs(graph[u][v].get("dielectric_constant", 1.0) or 1.0) for u, v in graph.edges]
    n_max = float(np.sqrt(max(eps) if eps else 1.0))
    target = 2.0 * np.pi / (n_max * k_max) / resolution
    lengths = np.array([graph[u][v]["length"] for u, v in graph.edges], dtype=float)
    est_nodes = float(np.sum(np.maximum(lengths / max(target, 1e-12), 1.0)))
    if est_nodes > node_cap:
        # Keep the oversampled graph (and its eigensolves) bounded. On a large
        # graph this *reduces* the effective resolution below ``resolution`` --
        # raise ``node_cap`` (the full_salt_newton ``oversample_node_cap`` knob)
        # to recover within-edge accuracy at higher cost.
        target *= est_nodes / node_cap
    return float(target)


#: Residual the fixed-set solve drives towards before declaring convergence.
#: This is the method's own target, not the bar a result is judged against --
#: see the ``accept_tol`` / ``solve_tol`` split in
#: :func:`_full_salt_newton_impl`.
SALT_RESIDUAL_TARGET = 1e-6

#: Amplitude below which a mode counts as not lasing. Modes that fall under it
#: are dropped from the active set rather than carried with a ~ 0, which would
#: otherwise hand the coupled least-squares an equation it cannot satisfy.
SALT_LASING_AMPLITUDE = 1e-4

#: Net-gain margin for admitting a candidate to the active set. Gain clamping
#: pins an above-threshold mode's alpha at ~0-, so a loose cutoff adds the mode
#: several pump steps late and snaps its L--I curve.
SALT_GAIN_MARGIN = -1e-6


class SaltSolution(NamedTuple):
    """Result of a fixed-active-set SALT solve.

    ``residuals`` is the acceptance test and is what should be believed rather
    than any internal convergence flag: it is ``|lambda_1(L_sat, k_mu)|``, i.e.
    how singular the shared saturated operator actually is at each lasing mode's
    real frequency. ``converged`` reports whether the iteration reached its own
    stopping rule; a solve can be un-converged and still have a tiny residual
    (it ran out of iterations after arriving), or converged with a large one (it
    stalled). Look at ``residuals``.
    """

    ks: np.ndarray  # real lasing frequencies
    amplitudes: np.ndarray  # a_mu >= 0, in the saturation's own unit
    fields: list  # per-edge |E_mu|^2 entering the hole-burning denominator
    residuals: np.ndarray  # |lambda_1(L_sat, k_mu)| per mode
    converged: bool
    iterations: int


def salt_residuals(graph, ks, amplitudes, fields, D0, pump, seed=42):
    r"""Residual of the SALT condition for a candidate solution.

    Builds the shared saturated operator

    .. math:: L_{sat}(k;\,\{a_\nu\}), \qquad
              D_0^{eff}(x) = \frac{D_0\,\mathrm{pump}(x)}
                                  {1 + \sum_\nu \Gamma_\nu a_\nu |E_\nu(x)|^2}

    and reports ``|lambda_1|`` at every mode's real ``k``. A genuine SALT
    solution has the operator singular at each of them simultaneously, so these
    numbers are zero to solver tolerance.

    This is deliberately independent of *how* the candidate was obtained, so it
    is the acceptance test for any solver -- including a hand-constructed guess,
    or the ``linear`` solver's prediction checked against the real operator.

    Args:
        graph: the (oversampled) work graph the solve was run on.
        ks: real lasing frequencies, one per mode.
        amplitudes: modal amplitudes ``a_mu >= 0``.
        fields: per-edge ``|E_mu|^2`` profiles entering the hole burning.
        D0: pump strength.
        pump: per-edge pump profile.
        seed: fixes the ARPACK start vector so the residual is deterministic.

    Returns:
        ``np.ndarray`` of ``|lambda_1|``, one per mode.
    """
    ks = np.asarray(ks, dtype=float)
    graph_sat = _saturated_graph_multi(
        graph, [[k, 0.0] for k in ks], np.asarray(amplitudes, dtype=float), D0, pump, fields
    )
    return np.array([abs(_lam_real_k(graph_sat, float(k), seed)) for k in ks])


def solve_salt_fixed_set(
    graph,
    modes,
    amplitudes,
    fields,
    D0,
    pump,
    pump_mask,
    *,
    seed=42,
    max_steps=30,
    outer=25,
    damping=0.7,
    k_window_cap=None,
    residual_tol=1e-6,
):
    """Solve the SALT equations for a *given* set of lasing modes.

    This is the whole nonlinear solve with none of the active-set machinery: the
    caller states which modes lase and gets back their frequencies, amplitudes
    and residuals. It applies no corrections of any kind -- if the answer is bad
    that shows up in ``residuals`` rather than being patched.

    Use it directly when you already know the candidate set (typically from the
    ``linear`` solver) and want the above-threshold correction for those modes.
    :func:`compute_modal_intensities_full_salt_newton` is the continuation
    wrapper that also *discovers* the set.

    Method: block iteration. The saturated background fields are frozen while a
    bounded trust-region least-squares solves every mode's
    ``(k_mu real, a_mu >= 0)`` against :func:`_salt_block_residual`; the fields
    are then refreshed and the step repeated. Freezing makes each residual a
    single clean eigensolve per mode, so the finite-difference Jacobian is
    noise-free.

    ``k`` is confined to a window below the inter-mode spacing. That is a
    locality constraint on a local solver -- the same role the search box plays
    in :func:`~netsalt.algorithm.refine_mode_root` -- and without it the trust
    region can zero a mode's residual by drifting its ``k`` onto a neighbouring
    root at ``a = 0`` instead of raising its amplitude to lase.

    Returns:
        :class:`SaltSolution`.
    """
    n = len(modes)
    if n == 0:
        return SaltSolution(np.empty(0), np.empty(0), [], np.empty(0), True, 0)

    k0 = np.array([float(m[0]) for m in modes])
    a = np.clip(np.asarray(amplitudes, dtype=float), 1e-3, None)
    fields = [np.asarray(f, dtype=float) for f in fields]

    if n > 1:
        gaps = np.abs(k0[:, None] - k0[None, :])
        gaps[np.diag_indices(n)] = np.inf
        window = float(np.clip(0.2 * gaps.min(), 1e-6, 0.1))
    else:
        window = 0.1
    if k_window_cap is not None:
        # A sparse active set on a dense spectrum must not wander across
        # neighbouring roots, which the active set's own spacing cannot see.
        window = min(window, float(k_window_cap))
    a_max = max(1.0e3 * max(float(np.max(a)), 1.0e-3), 1.0e3)
    lo = np.concatenate([k0 - window, np.zeros(n)])
    hi = np.concatenate([k0 + window, np.full(n, a_max)])

    ks = k0.copy()
    converged = False
    iterations = 0
    for _step in range(1, outer + 1):
        iterations = _step
        x0 = np.clip(np.concatenate([ks, a]), lo, hi)
        try:
            result = sc.optimize.least_squares(
                lambda x, _f=fields: _salt_block_residual(graph, x, n, D0, pump, _f, seed),
                x0,
                bounds=(lo, hi),
                xtol=1e-6,
                ftol=1e-6,
                gtol=1e-6,
                max_nfev=int(max_steps),
            )
            ks, a = result.x[:n], np.clip(result.x[n:], 0.0, None)
        except (RuntimeError, ValueError, sc.sparse.linalg.ArpackError):
            break

        graph_sat = _saturated_graph_multi(graph, [[k, 0.0] for k in ks], a, D0, pump, fields)
        refreshed = [
            _single_mode_field_intensity(graph_sat, [ks[i], 0.0], pump_mask) for i in range(n)
        ]
        change = sum(np.linalg.norm(refreshed[i] - fields[i]) for i in range(n))
        fields = [(1.0 - damping) * fields[i] + damping * refreshed[i] for i in range(n)]

        # The SALT condition itself is the stopping rule: the saturated operator
        # singular at every *lasing* mode's real k. The field change is only a
        # proxy for it, and a misleading one at both ends -- it keeps creeping
        # long after the solution is reached (which used to report failure), and
        # it goes quiet when the amplitudes are near zero even though the
        # residual is still large (which used to report success). Modes with
        # a ~ 0 are not lasing, so the condition does not apply to them and their
        # residual is legitimately nonzero.
        lasing = [i for i in range(n) if a[i] > SALT_LASING_AMPLITUDE]
        if lasing:
            residuals = salt_residuals(graph, ks, a, fields, D0, pump, seed=seed)
            if all(residuals[i] <= residual_tol for i in lasing):
                converged = True
                break
        elif change <= 1e-6 * (1.0 + sum(np.linalg.norm(f) for f in fields)):
            # Nothing is lasing, so there is no residual to drive to zero; the
            # only meaningful statement is that the iterate has stopped moving.
            converged = True
            break

    residuals = salt_residuals(graph, ks, a, fields, D0, pump, seed=seed)
    modes_out = [np.array([float(k), 0.0]) for k in ks]
    return SaltSolution(
        np.asarray(modes_out), np.asarray(a), fields, residuals, converged, iterations
    )


def compute_modal_intensities_full_salt_newton(*args, **kwargs):
    """Operator-level full-SALT L--I curves (public entry).

    Thin wrapper that temporarily lowers ``DENSE_EIG_MAX`` so the saturated
    eigensolves -- on the *banded*, oversampled graph, targeting isolated lasing
    modes -- run through ARPACK shift-invert (~flat in N, far cheaper than dense
    O(N^3) at the medium/large sizes oversampling produces). The global default is
    left high so dense-spectrum mode *finding* on 2D graphs keeps the robust dense
    path. See :data:`~netsalt.quantum_graph.DENSE_EIG_MAX`.
    """
    from netsalt import quantum_graph as _qg

    saved = _qg.DENSE_EIG_MAX
    _qg.DENSE_EIG_MAX = min(saved, NEWTON_DENSE_EIG_MAX)
    try:
        return _full_salt_newton_impl(*args, **kwargs)
    finally:
        _qg.DENSE_EIG_MAX = saved


def _full_salt_newton_impl(
    graph,
    modes_df,
    max_pump_intensity,
    D0_steps=30,
    max_iter=30,
    tol=1e-8,
    oversample_size=None,
    inner_max_iter=25,
    inner_damping=0.8,
    seed=42,
    quality_method="eigenvalue",
    oversample_resolution=12,
    oversample_node_cap=3000,
    residual_tol=None,
):
    r"""Operator-level full-SALT L--I curves by pump continuation.

    Steps the pump and, at each step, maintains the set of lasing modes and
    solves them with :func:`solve_salt_fixed_set`. The two layers are kept
    separate on purpose: the solve is a well-posed ``2N``-equation root-find that
    needs no heuristics, while *discovering* which modes lase is the hard part
    and is where any doubt belongs.

    **Nothing here corrects the physics.** Earlier revisions carried a
    total-output ratchet (holding a collapsing mode at its previous amplitude so
    the L--I curve stayed monotone) and a wrong-basin guard (reverting an
    activation that suppressed an established mode). Both imposed the expected
    answer on the numerics, which makes a non-monotone curve or a mode swap --
    exactly the things worth investigating -- unobservable. They are gone. What
    replaces them is reporting: every pump step records its residuals, active
    set, and convergence into ``modes_df.attrs["salt_diagnostics"]``, so the
    curve can be audited rather than trusted.

    The two rules that remain are about *termination*, not physics: a candidate
    that is admitted and immediately dies is not retried at the same pump, and
    the number of active-set sweeps per pump is bounded.

    Active-set rules, each applied once per sweep:

    * **Drop** any mode whose amplitude falls below
      :data:`SALT_LASING_AMPLITUDE` -- it has stopped lasing.
    * **Bootstrap** an empty set from the lowest noninteracting threshold. That
      is exact on the unsaturated background, and the gain probe below would
      wrongly reject a mode sitting exactly *at* its threshold, where alpha = 0
      rather than < 0.
    * **Add** the most above-threshold candidate that shows net gain
      (``alpha <`` :data:`SALT_GAIN_MARGIN`) on the current saturated
      background -- one per sweep, since a simultaneous multi-mode add hands the
      coupled solve a multistable warm start.

    Amplitudes are reported in the linear modal-intensity unit via
    :func:`_newton_onset_unit_scale`, an analytic first-order change of
    variables. The factor comes out near 1, and it is recorded per mode in
    ``modes_df.attrs["salt_unit_scale"]``. **Because the reported intensity is
    scaled to the linear model's unit, near-threshold agreement with ``linear``
    is not independent validation** -- use the residuals for that.

    Args:
        graph: pumped quantum graph.
        modes_df: threshold modes with ``lasing_thresholds``.
        max_pump_intensity: top of the pump sweep.
        D0_steps: number of pump points.
        oversample_size: sub-edge size for the within-edge hole burning. None
            auto-picks a wavelength-resolving size; 0 keeps the bare edges, which
            over-clamps and suppresses co-lasing modes.
        residual_tol: SALT residual below which a solve is accepted. None ties it
            to the graph's ``quality_threshold`` -- the same bar the passive mode
            search uses to call something a mode, so the two halves of the
            pipeline are held to one standard. Demanding more of the SALT solve
            than of the modes going into it would be incoherent.
        oversample_resolution, oversample_node_cap: passed to
            :func:`_auto_oversample_size`.
        seed: fixes ARPACK start vectors, making the whole sweep deterministic.

    Returns:
        ``modes_df`` with the L--I columns attached, plus
        ``attrs["salt_diagnostics"]`` (a per-pump dataframe) and
        ``attrs["salt_unit_scale"]``.
    """
    del max_iter, tol, inner_max_iter, inner_damping, quality_method  # interface parity

    # Two different numbers, deliberately. The *solver* drives the residual as
    # far as its method allows (SALT_RESIDUAL_TARGET); the *acceptance* bar is
    # the same one the passive mode search uses to call something a mode, since
    # demanding more of the SALT solve than of the modes going into it would be
    # incoherent. Using the acceptance bar as the stopping rule instead would
    # stop the solve early and throw away accuracy that is nearly free.
    accept_tol = (
        graph.graph["params"].get("quality_threshold", 1e-4)
        if residual_tol is None
        else residual_tol
    )
    solve_tol = min(SALT_RESIDUAL_TARGET, accept_tol)

    lasing_thresholds = np.asarray(modes_df["lasing_thresholds"]).ravel()
    n_modes = len(modes_df)
    modal_intensities = pd.DataFrame(index=range(n_modes))
    interacting_lasing_thresholds = np.inf * np.ones(n_modes)
    candidates = [int(i) for i in np.where(lasing_thresholds < np.inf)[0]]
    if not candidates:
        return _finalise_modal_intensities(
            modes_df, modal_intensities, interacting_lasing_thresholds
        )

    if oversample_size is None:
        oversample_size = _auto_oversample_size(
            graph, modes_df, resolution=oversample_resolution, node_cap=oversample_node_cap
        )
    work_graph = graph if not oversample_size else oversample_graph(graph, oversample_size)
    pump = np.asarray(work_graph.graph["params"]["pump"], dtype=float)
    pump_mask = _get_mask_matrices(work_graph.graph["params"])[1]
    max_steps = 30

    # The linear competition matrix supplies the amplitude *unit* only (its
    # diagonal); the active set is found self-consistently from the saturated
    # operator, not borrowed from it.
    t_linear = compute_mode_competition_matrix(work_graph, modes_df)
    t_diag = np.array(
        [abs(t_linear[i, i]) if abs(t_linear[i, i]) > 1e-12 else 1.0 for i in range(n_modes)]
    )
    threshold_modes = modes_df["threshold_lasing_modes"].to_numpy()

    mode_state: dict[int, np.ndarray] = {}
    field_state: dict[int, np.ndarray] = {}
    a_state: dict[int, float] = {}
    unit_scale: dict[int, float] = {}

    cand_ks = np.sort([float(from_complex(threshold_modes[c])[0]) for c in candidates])
    k_cap = float(np.clip(0.2 * np.diff(cand_ks).min(), 1e-6, 0.1)) if len(cand_ks) > 1 else None

    def onset_amplitude(mode_id, d0):
        """Linear-slope estimate ``(D0 - thr)/(T_cc thr)``, used as a warm start.

        Starting at a floor instead sits in the basin of the trivial ``a = 0``
        root of the bounded solve; the physical near-threshold estimate keeps the
        solve on the lasing branch.
        """
        threshold = float(lasing_thresholds[mode_id])
        slope = (
            1.0 / (t_diag[mode_id] * threshold)
            if (t_diag[mode_id] > 0.0 and threshold > 0.0)
            else 0.0
        )
        return max(1e-3, slope * max(float(d0) - threshold, 0.0))

    def initialise(mode_id):
        mode = np.asarray(from_complex(threshold_modes[mode_id]), dtype=float)
        field = _single_mode_field_intensity(
            graph_with_pump(work_graph, float(lasing_thresholds[mode_id])), mode, pump_mask
        )
        mode_state[mode_id], field_state[mode_id], a_state[mode_id] = mode, field, 0.0
        unit_scale[mode_id] = _newton_onset_unit_scale(
            work_graph, mode, field, float(lasing_thresholds[mode_id]), t_diag[mode_id]
        )

    def solve(active_ids, d0):
        """Solve the given set at this pump and write the result back to state."""
        solution = solve_salt_fixed_set(
            work_graph,
            [mode_state[i] for i in active_ids],
            [
                a_state[i] if a_state[i] > SALT_LASING_AMPLITUDE else onset_amplitude(i, d0)
                for i in active_ids
            ],
            [field_state[i] for i in active_ids],
            d0,
            pump,
            pump_mask,
            seed=seed,
            max_steps=max_steps,
            k_window_cap=k_cap,
            residual_tol=solve_tol,
        )
        for j, mode_id in enumerate(active_ids):
            mode_state[mode_id] = solution.ks[j]
            field_state[mode_id] = solution.fields[j]
            a_state[mode_id] = float(solution.amplitudes[j])
        return solution

    active: list[int] = []
    first = float(np.min(lasing_thresholds[candidates]))
    diagnostics = []

    for D0 in np.linspace(first, max_pump_intensity, D0_steps):
        stillborn: set[int] = set()  # admitted then died at this pump: do not retry
        added: list[int] = []
        dropped: list[int] = []
        solution = None

        for _ in range(2 * len(candidates) + 2):
            if active:
                solution = solve(list(active), float(D0))
                gone = [i for i in active if a_state[i] <= SALT_LASING_AMPLITUDE]
                if gone:
                    dropped.extend(gone)
                    stillborn.update(gone)
                    active = [i for i in active if a_state[i] > SALT_LASING_AMPLITUDE]

            if not active:
                # Strictly above threshold: at D0 == D0_thr exactly the modal
                # intensity is 0 by definition, so there is no lasing solution to
                # find there. Admitting the mode anyway (the amplitude floor in
                # the solve makes it look lasing) left the first sweep point
                # chasing a residual it cannot drive to zero.
                eligible = [
                    c for c in candidates if lasing_thresholds[c] < D0 and c not in stillborn
                ]
                if not eligible:
                    break
                mode_id = min(eligible, key=lambda i: lasing_thresholds[i])
                if mode_id not in mode_state:
                    initialise(mode_id)
                a_state[mode_id] = onset_amplitude(mode_id, float(D0))
                active.append(mode_id)
                added.append(mode_id)
                continue

            # Net gain the not-yet-lasing candidates see on the saturated background.
            background = _saturated_graph_multi(
                work_graph,
                [mode_state[i] for i in active],
                [a_state[i] for i in active],
                float(D0),
                pump,
                [field_state[i] for i in active],
            )
            active_ks = [float(mode_state[i][0]) for i in active]
            best, best_alpha, best_k = None, SALT_GAIN_MARGIN, 0.0
            for c in candidates:
                if c in active or c in stillborn or lasing_thresholds[c] > D0:
                    continue
                if c not in mode_state:
                    initialise(c)
                gap = min(abs(float(mode_state[c][0]) - k) for k in active_ks)
                window = float(np.clip(0.2 * gap, 1e-6, 0.3))
                probed = _refine_local(
                    mode_state[c], background, 1e-9, max_steps, seed, k_window=window
                )
                if probed[1] < best_alpha:
                    best, best_alpha, best_k = c, float(probed[1]), float(probed[0])
            if best is None:
                break

            mode_state[best] = np.array([best_k, 0.0])
            # A newcomer crosses its *interacting* threshold here, so it turns on
            # from ~0. The bare-threshold estimate overshoots badly when
            # D0 >> its own threshold; the field is anchored by the incumbents,
            # so the trivial a = 0 root is not a risk.
            a_cap = 1e-2 * max(a_state[i] for i in active)
            a_state[best] = float(min(onset_amplitude(best, float(D0)), max(a_cap, 1e-3)))
            active.append(best)
            added.append(best)

        for mode_id in candidates:
            value = (
                max(a_state.get(mode_id, 0.0) * unit_scale.get(mode_id, 1.0), 0.0)
                if mode_id in active
                else 0.0
            )
            modal_intensities.loc[mode_id, D0] = value
            if (
                mode_id in active
                and a_state[mode_id] > 0
                and D0 < interacting_lasing_thresholds[mode_id]
            ):
                interacting_lasing_thresholds[mode_id] = D0

        diagnostics.append(
            {
                "D0": float(D0),
                "n_active": len(active),
                "active": tuple(sorted(active)),
                "added": tuple(added),
                "dropped": tuple(dropped),
                "converged": bool(solution.converged) if solution is not None else True,
                "max_residual": (
                    float(np.max(solution.residuals))
                    if solution is not None and len(solution.residuals)
                    else 0.0
                ),
                "iterations": solution.iterations if solution is not None else 0,
            }
        )

    diagnostics = pd.DataFrame(diagnostics)
    worst = float(diagnostics["max_residual"].max()) if len(diagnostics) else 0.0
    if worst > accept_tol:
        warnings.warn(
            f"full_salt_newton: worst SALT residual over the sweep is {worst:.3g}, above the "
            f"{accept_tol:.3g} tolerance -- the operator is not singular at those modes, so "
            "the affected points are not SALT solutions. See "
            "modes_df.attrs['salt_diagnostics'] for the per-pump breakdown.",
            stacklevel=2,
        )

    modes_df = _finalise_modal_intensities(
        modes_df, modal_intensities, interacting_lasing_thresholds
    )
    modes_df.attrs["salt_diagnostics"] = diagnostics
    modes_df.attrs["salt_unit_scale"] = dict(unit_scale)
    # What the within-edge hole burning was actually resolved at. On a large
    # graph ``oversample_node_cap`` binds and silently reduces the resolution
    # below ``oversample_resolution``, so record what was achieved rather than
    # what was asked for (issue #52).
    modes_df.attrs["salt_work_nodes"] = len(work_graph)
    modes_df.attrs["salt_oversample_size"] = float(oversample_size) if oversample_size else 0.0
    return modes_df


def pump_trajectories(modes_df, graph, return_approx=False, quality_method="eigenvalue"):
    """Track every mode's position as the pump ``D0`` is raised from 0 to ``D0_max``.

    Modes whose refinement fails at some pump are *frozen*: their last
    successfully refined position is carried through the remaining pump steps
    and the pump at which tracking was lost is recorded in the
    ``tracking_lost_at_D0`` column (``NaN`` for modes tracked all the way).

    Freezing rather than re-seeding matters. The previous behaviour substituted
    the last good position and kept feeding it to :func:`pump_linear` at the
    *new* pump, where it is no longer a mode — :func:`mode_on_nodes` then raised
    ``"Not a mode, as quality is too high"`` from inside the next iteration,
    killing the whole run with an error pointing at a mode that was fine. On the
    shipped ``examples/line_PRA`` config, changing only ``gamma_perp`` from 3.0
    to 1.5 was enough to hit it.
    """

    D0s = np.linspace(
        0,
        graph.graph["params"]["D0_max"],
        graph.graph["params"]["D0_steps"],
    )

    n_modes = len(modes_df)

    pumped_modes = [[from_complex(mode) for mode in modes_df["passive"]]]
    pumped_modes_approx = pumped_modes.copy()
    # D0 at which each mode stopped being trackable; NaN while still tracked.
    lost_at = np.full(n_modes, np.nan)
    # One pool for the whole sweep. Creating it per D0 step cost a fork +
    # teardown each time (~15 ms + ~8 ms at 4 workers, ~690 ms at 80), which
    # dominates the tail of the sweep where only a handful of modes are left.
    # Pickling the graph to the workers is cheap by comparison (<1 ms even at
    # buffon size), so nothing else has to change.
    with (
        _scoped_warning_filters(),
        multiprocessing.Pool(graph.graph["params"]["n_workers"]) as pool,
    ):
        for d in range(len(D0s) - 1):
            L.info(
                "Step %s / %s, computing for D0= %s",
                str(d + 1),
                str(len(D0s) - 1),
                str(D0s[d + 1]),
            )
            tracked = [m for m in range(n_modes) if np.isnan(lost_at[m])]
            pumped_modes_approx.append(pumped_modes[-1].copy())
            # The linear pump step is one eigensolve per mode and was run
            # serially in the parent while the pool sat idle (30% of this step).
            approx = pool.imap(
                partial(pump_linear, graph=graph, D0_0=D0s[d], D0_1=D0s[d + 1]),
                [pumped_modes[-1][m] for m in tracked],
            )
            for m, mode_approx in zip(tracked, approx, strict=True):
                pumped_modes_approx[-1][m] = mode_approx

            worker_modes = WorkerModes(
                pumped_modes_approx[-1],
                graph,
                D0s=n_modes * [D0s[d + 1]],
                quality_method=quality_method,
            )
            refined = list(tqdm(pool.imap(worker_modes, tracked), total=len(tracked)))

            # Frozen modes keep their last position; newly-lost ones join them.
            pumped_modes.append(pumped_modes[-1].copy())
            for m, mode in zip(tracked, refined, strict=True):
                if mode is None:
                    lost_at[m] = D0s[d + 1]
                else:
                    pumped_modes[-1][m] = mode

    n_lost = int(np.count_nonzero(~np.isnan(lost_at)))
    if n_lost:
        warnings.warn(
            f"{n_lost} of {n_modes} modes could not be tracked over the whole pump sweep and "
            "were frozen at their last refined position; see the 'tracking_lost_at_D0' column. "
            "Consider a finer D0_steps or a looser quality_threshold.",
            stacklevel=2,
        )

    if "mode_trajectories" in modes_df:
        del modes_df["mode_trajectories"]
    for D0, pumped_mode in zip(D0s, pumped_modes, strict=True):
        modes_df["mode_trajectories", D0] = [to_complex(mode) for mode in pumped_mode]

    if return_approx:
        if "mode_trajectories_approx" in modes_df:
            del modes_df["mode_trajectories_approx"]
        for D0, pumped_mode_approx in zip(D0s, pumped_modes_approx, strict=True):
            modes_df["mode_trajectories_approx", D0] = [
                to_complex(mode) for mode in pumped_mode_approx
            ]

    modes_df["tracking_lost_at_D0"] = lost_at
    return modes_df


def _get_new_D0(arg, graph=None, D0_steps=0.1):
    """Internal function for multiprocessing."""
    mode_id, new_mode, D0 = arg
    # Both helpers below need the mode's overlap with the pump at this same D0.
    # Computing it here and passing it in halves the work: it was previously
    # evaluated twice per call with byte-identical arguments (52% of this
    # function's cost).
    overlapping_factor = compute_overlapping_factor(new_mode, graph_with_pump(graph, D0))
    increment = lasing_threshold_linear(new_mode, graph, D0, overlapping_factor=overlapping_factor)
    if increment > -D0_steps:
        new_D0 = abs(D0 + increment)
        new_D0 = min(new_D0, D0_steps + D0)
    else:
        L.debug("Intensity increment is negative, we set step to half max step.")
        new_D0 = D0 + 0.5 * D0_steps

    L.debug("Mode %s at intensity %s", mode_id, new_D0)
    new_modes_approx = pump_linear(
        new_mode, graph, D0, new_D0, overlapping_factor=overlapping_factor
    )
    return mode_id, new_D0, new_modes_approx


def find_threshold_lasing_modes(modes_df, graph, quality_method="eigenvalue"):
    """Find the threshold lasing modes and associated lasing thresholds.

    Modes whose refinement fails part-way up the pump are dropped from the
    search with their threshold left at ``inf`` (the existing "never reached
    threshold" encoding) and reported in a warning, rather than crashing the
    run — see the comment at the refinement result loop below.
    """
    stepsize = graph.graph["params"]["search_stepsize"]
    D0_steps = graph.graph["params"]["D0_max"] / graph.graph["params"]["D0_steps"]
    new_modes = modes_df["passive"].to_numpy()

    threshold_lasing_modes = np.zeros([len(modes_df), 2])
    lasing_thresholds = np.inf * np.ones(len(modes_df))
    D0s = np.zeros(len(modes_df))
    current_modes = np.arange(len(modes_df))
    lost_modes: list[int] = []
    stuck_modes_count = 0
    max_modes = len(current_modes)
    prev_n_modes = 0
    # One pool for the whole search. It used to create *two* per while-loop
    # iteration, and the tail iterations carry one or two modes each yet still
    # paid a full fork + teardown twice -- over half the iteration at that point.
    with multiprocessing.Pool(graph.graph["params"]["n_workers"]) as pool:
        while len(current_modes) > 0:
            if len(current_modes) == prev_n_modes:
                stuck_modes_count += 1
            prev_n_modes = len(current_modes)
            if max_modes > stuck_modes_count > 100:
                warnings.warn("We stop here, some modes got stuck.", stacklevel=2)
                current_modes = []
                continue
            L.info("%s modes left to find", len(current_modes))

            new_D0s = np.zeros(len(modes_df))
            new_modes_approx = np.empty([len(new_modes), 2])
            args = ((mode_id, new_modes[mode_id], D0s[mode_id]) for mode_id in current_modes)
            for mode_id, new_D0, new_mode_approx in pool.imap(
                partial(_get_new_D0, graph=graph, D0_steps=D0_steps), args
            ):
                new_D0s[mode_id] = new_D0
                new_modes_approx[mode_id] = new_mode_approx

            # this is a trick to reduce the stepsizes as we are near the solution.
            # Passed explicitly to WorkerModes (applied to its per-call params copy)
            # rather than stashed on the shared graph.graph["params"].
            search_stepsize = (
                stepsize * np.mean(abs(new_D0s[new_D0s > 0] - D0s[new_D0s > 0])) / D0_steps
            )

            L.debug("Current search_stepsize: %s", search_stepsize)
            worker_modes = WorkerModes(
                new_modes_approx,
                graph,
                D0s=new_D0s,
                search_stepsize=search_stepsize,
                quality_method=quality_method,
            )
            new_modes_tmp = np.zeros([len(modes_df), 2])

            refined = list(tqdm(pool.imap(worker_modes, current_modes), total=len(current_modes)))

            # ``refine_mode`` returns None when it fails to converge. Assigning that
            # straight into the float array raised an opaque numpy "inhomogeneous
            # shape" ValueError, and the `is None` check below it could never fire
            # (a row of a float array is never None), so the intended recovery was
            # dead code. Keep the last known position for a failed mode and stop
            # tracking it: its threshold stays inf, which is how the rest of the
            # pipeline already represents "never reached threshold".
            to_delete = []
            for i, (mode_index, mode) in enumerate(zip(current_modes, refined, strict=True)):
                if mode is None:
                    # ``new_modes`` holds complex passive modes on the first pass and
                    # [k, alpha] pairs afterwards; from_complex normalises both.
                    new_modes_tmp[mode_index] = from_complex(new_modes[mode_index])
                    lost_modes.append(int(mode_index))
                    to_delete.append(i)
                    continue
                new_modes_tmp[mode_index] = mode
                if abs(new_modes_tmp[mode_index][1]) < 1e-6:
                    to_delete.append(i)
                    threshold_lasing_modes[mode_index] = new_modes_tmp[mode_index]
                    lasing_thresholds[mode_index] = new_D0s[mode_index]

                elif new_D0s[mode_index] > graph.graph["params"]["D0_max"]:
                    to_delete.append(i)

            current_modes = np.delete(current_modes, to_delete)
            D0s = new_D0s.copy()
            new_modes = new_modes_tmp.copy()

    # A mode whose threshold comes out at (essentially) zero was already at or
    # above threshold with no pump -- a near-zero-loss trapped mode, alpha ~ 0.
    # The whole near-threshold model divides by alpha (q_value = k / 2*alpha), so
    # it says nothing about such a mode, and feeding it downstream produces
    # absurd intensities rather than an error (observed: 7e7 where the real modes
    # sit at ~1e2). Exclude them the same way modes that never reach threshold
    # are excluded, and say so.
    threshold_floor = 1e-6 * graph.graph["params"]["D0_max"]
    already_lasing = [
        int(i)
        for i in np.where(lasing_thresholds <= threshold_floor)[0]
        if np.isfinite(lasing_thresholds[i])
    ]
    if already_lasing:
        warnings.warn(
            f"Mode(s) {already_lasing} reach threshold at D0 <= {threshold_floor:.3g}, i.e. they "
            "already lase with no pump (alpha ~ 0, a trapped or numerically marginal mode). The "
            "near-threshold model divides by alpha and cannot describe them, so they are excluded "
            "from the lasing set. Narrow the scan window (alpha_min > 0) or tighten "
            "quality_threshold if these are numerical artefacts.",
            stacklevel=2,
        )
        lasing_thresholds[already_lasing] = np.inf
        threshold_lasing_modes[already_lasing] = 0.0

    if lost_modes:
        warnings.warn(
            f"Refinement failed for mode(s) {sorted(set(lost_modes))} part-way up the pump; "
            "their lasing threshold is reported as inf. Consider a finer D0_steps or a "
            "looser quality_threshold.",
            stacklevel=2,
        )

    modes_df["threshold_lasing_modes"] = [to_complex(mode) for mode in threshold_lasing_modes]
    modes_df["lasing_thresholds"] = lasing_thresholds

    # we remove duplicated threshold lasing modes (we keep first appearance)
    prec = graph.graph["params"]["quality_threshold"]
    modes_df["th"] = prec * (abs(modes_df["threshold_lasing_modes"]) / prec).round(0)
    val, count = np.unique(modes_df["th"].to_numpy(), return_counts=True)
    for v in val[count > 1]:
        mask = modes_df[modes_df["th"] == v].index
        if len(mask) > 1:
            modes_df.loc[mask[1:], "threshold_lasing_modes"] = 0.0
            modes_df.loc[mask[1:], "lasing_thresholds"] = np.inf

    return modes_df.drop(columns=["th"])


def lasing_threshold_linear(mode, graph, D0, overlapping_factor=None):
    """Find the linear approximation of the pump increment to reach threshold.

    ``overlapping_factor`` is optional and only exists so a caller that also
    needs it (:func:`_get_new_D0`) can compute it once; see
    :func:`pump_linear`.
    """
    graph = graph_with_pump(graph, D0)
    if overlapping_factor is None:
        overlapping_factor = compute_overlapping_factor(mode, graph)
    return 1.0 / (
        q_value(mode)
        * -1
        * np.imag(gamma(to_complex(mode), graph.graph["params"]))
        * np.real(overlapping_factor)
    )


def get_node_transfer(k, graph, input_flow):
    """Compute node transfer from a given input flow."""
    return sc.sparse.linalg.spsolve(construct_laplacian(k, graph), graph.graph["ks"] * input_flow)


def get_edge_transfer(k, graph, input_flow):
    """Compute edge transfer from a given input flow."""
    set_wavenumber(graph, k)
    BT, B = construct_incidence_matrix(graph)
    _r = get_node_transfer(k, graph, BT.dot(input_flow))
    Winv = construct_weight_matrix(graph, with_k=False)
    return Winv.dot(B).dot(_r)


def estimate_boundary_flow(graph, input_flow, k_frac=1e-2):
    """Estimate boundary flow for static simulations.

    Arguments:
        graph: QG graph
        input_flow: edge input flow
        k_frac: fraction of estimated small wavenumber to evaluate the flows
    """
    # estimate a small wavenumber for the graph
    k = k_frac * np.mean(
        np.array(graph.graph["params"]["inner"], dtype=float)
        * graph.graph["params"]["c"]
        / graph.graph["lengths"]
    )

    e_deg = np.array([len(graph[v]) for u, v in graph.edges])
    output_ids = list(np.argwhere(e_deg == 1).flatten())
    output_ids += list(2 * np.argwhere(e_deg == 1).flatten() + 1)

    # get the flows on all nodes
    flows = np.abs(get_edge_transfer(k, graph, input_flow))
    # remove on inner nodes
    flows[[i for i in range(len(flows)) if i not in output_ids]] = 0
    # get total output flow per unit of input flow
    total = flows[output_ids].sum() / input_flow.sum()
    # remove input flow to get a global conservation
    flows -= total * input_flow
    return flows, k

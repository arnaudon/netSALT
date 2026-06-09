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
        from .contour import find_modes_contour

        # Reasonable defaults; callers can override via kwargs.
        contour_defaults = {
            "n_k": kwargs.pop("n_k", None),
            "n_alpha": kwargs.pop("n_alpha", 2),
            "n_quad": kwargs.pop("n_quad", 80),
            "probe_dim": kwargs.pop("probe_dim", None),
        }
        if contour_defaults["n_k"] is None:
            # Rule of thumb: roughly one sub-cell per ~5 expected modes.
            # Without an accurate prior we fall back to 1 cell per unit k
            # (sensible for the small ranges netsalt normally scans).
            k_min = graph.graph["params"]["k_min"]
            k_max = graph.graph["params"]["k_max"]
            contour_defaults["n_k"] = max(int(round(k_max - k_min)), 1)
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


def pump_linear(mode_0, graph, D0_0, D0_1):
    """Find the linear approximation of the new wavenumber."""
    graph = graph_with_pump(graph, D0_0)
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


def _compute_mode_competition_element(lengths, params, data, with_gamma=True):
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

    chunksize = max(1, int(0.1 * len(input_data) / n_workers))
    with multiprocessing.Pool(n_workers) as pool:
        output_data = list(
            tqdm(
                pool.imap(
                    partial(
                        _compute_mode_competition_element,
                        graph.graph["lengths"],
                        graph.graph["params"],
                        with_gamma=with_gamma,
                    ),
                    input_data,
                    chunksize=chunksize,
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

    return np.real(mode_competition_matrix)


def _scatter_competition_block(block, lasing_mask, n_total):
    """Place a lasing-only competition block back into a full n_total matrix."""
    full = np.zeros([n_total, n_total])
    full[np.ix_(lasing_mask, lasing_mask)] = block
    return full


def compute_mode_competition_matrix(graph, modes_df, with_gamma=True):
    """Compute the mode competition matrix, or T matrix.

    Each mode's profile is evaluated at its own lasing threshold pump (the
    linearised, near-threshold model). See
    :func:`compute_mode_competition_matrix_at_pump` for the self-consistent
    variant that evaluates all modes at a common operating pump.
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


def compute_mode_competition_matrix_at_pump(
    graph, modes_df, pump_intensity, with_gamma=True, follow_modes=True
):
    """Competition matrix with every mode profile evaluated at ``pump_intensity``.

    Relaxes the frozen-threshold-profile approximation (#2): rather than each
    mode sitting at its own threshold, all modes are evaluated at the common
    operating pump ``pump_intensity``. Used by
    :func:`compute_modal_intensities_self_consistent`. Reduces to
    :func:`compute_mode_competition_matrix` when ``pump_intensity`` equals every
    mode's threshold.

    With ``follow_modes`` (default) each mode is first **refined to the actual
    mode of the operating-pump operator** (:func:`_refine_local`, warm-started
    from its threshold position) before its profile is taken. This matters far
    above threshold: the frozen threshold-frequency field is no longer an
    eigenmode of the strongly-pumped operator (its ``|λ₁|`` grows large), and the
    distortion -- worst for the lowest-threshold / highest-gain mode -- inflates
    that mode's self-saturation and can spuriously flip the mode ordering.
    Following the mode keeps every profile physical. Refining at a mode's own
    threshold is a no-op, so the reduction to the linear matrix is preserved.
    """
    threshold_modes_all = modes_df["threshold_lasing_modes"].to_numpy()
    lasing_thresholds_all = modes_df["lasing_thresholds"].to_numpy()
    lasing_mask = lasing_thresholds_all < np.inf

    threshold_modes = threshold_modes_all[lasing_mask]
    pumps = np.full(len(threshold_modes), float(pump_intensity))

    if follow_modes and len(threshold_modes):
        threshold_modes = _follow_modes_to_pump(
            graph, threshold_modes, lasing_thresholds_all[lasing_mask], float(pump_intensity)
        )

    block = _mode_competition_matrix_block(
        graph, threshold_modes, pumps, with_gamma=with_gamma, check_quality=False
    )
    return _scatter_competition_block(block, lasing_mask, len(threshold_modes_all))


def _follow_modes_to_pump(graph, modes_complex, thresholds, pump_intensity, n_steps=5, seed=42):
    """Refine each (complex) mode to the operating-pump operator's nearby mode.

    Returns the refined modes in the same complex ``k - i·alpha`` storage format.
    A mode pumped well above its threshold sits deep in the gain half-plane, too
    far for a single refine to reach from the threshold position, so it is tracked
    by **continuation** -- a few warm-started refines through intermediate pumps
    from its own threshold up to ``pump_intensity``. The real-``k`` excursion of
    each refine is capped below the inter-mode spacing so a mode cannot hop onto a
    neighbour (only the imaginary part moves much, as the mode goes into gain).
    """
    ks = np.array([np.real(z) for z in modes_complex])
    if len(ks) > 1:
        gaps = np.abs(ks[:, None] - ks[None, :])
        gaps[np.diag_indices(len(ks))] = np.inf
        k_window = float(np.clip(0.4 * gaps.min(), 0.05, 1.0))
    else:
        k_window = 1.0

    refined = []
    for z, threshold in zip(modes_complex, thresholds, strict=True):
        mode = from_complex(z)
        # ramp from the mode's own threshold to the operating pump (a single step
        # if the operating pump is at or below threshold)
        start = min(float(threshold), pump_intensity)
        for d0 in np.linspace(start, pump_intensity, n_steps):
            mode = _refine_local(mode, graph_with_pump(graph, float(d0)), 1e-9, 100, seed, k_window)
        refined.append(to_complex(mode))
    return np.array(refined)


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
                * sub_mode_comp_matrix_mu_inv.dot(1.0 / lasing_thresholds[lasing_mode_ids])
            )
            _int_thresh = lasing_thresholds[mu] * factor
            if _int_thresh > pump_intensity and _int_thresh > lasing_thresholds[mu]:
                interacting_lasing_thresholds[mu] = _int_thresh

    next_lasing_mode_id = np.argmin(interacting_lasing_thresholds)
    next_lasing_threshold = interacting_lasing_thresholds[next_lasing_mode_id]
    return next_lasing_mode_id, next_lasing_threshold


def _intensity_slopes_shifts(mode_competition_matrix, lasing_thresholds, lasing_mode_ids):
    """Linear modal-intensity solve for the active set.

    Returns ``(slopes, shifts)`` such that the modal intensities at pump ``D0``
    are ``slopes * D0 - shifts``. Shared by the ``linear`` and
    ``self_consistent`` solvers (they differ only in which competition matrix
    they feed in).
    """
    mode_competition_matrix_inv = np.linalg.pinv(
        mode_competition_matrix[np.ix_(lasing_mode_ids, lasing_mode_ids)]
    )
    slopes = mode_competition_matrix_inv.dot(1.0 / lasing_thresholds[lasing_mode_ids])
    shifts = mode_competition_matrix_inv.sum(1)
    return slopes, shifts


def _nonneg_active_set(mode_competition_matrix, lasing_thresholds, lasing_mode_ids, pump_intensity):
    """Prune the active set so every modal intensity at ``pump_intensity`` is >= 0.

    The SALT intensity equations only admit a physical solution with all modal
    intensities non-negative. For the constant linear competition matrix the
    event-driven sweep already guarantees this, so this returns the active set
    unchanged (the linear result is byte-identical). With a *pump-dependent*
    matrix (``self_consistent`` / ``full_salt``) the raw linear solve can return
    negative intensities -- a mode that should have switched off, or an
    ill-conditioned rebuild; drop the most-negative mode and re-solve until the
    survivors are all non-negative (a small active-set / non-negative-least-
    squares step). At least the dominant mode is always kept.
    """
    ids = list(lasing_mode_ids)
    while len(ids) > 1:
        slopes, shifts = _intensity_slopes_shifts(mode_competition_matrix, lasing_thresholds, ids)
        intensities = slopes * pump_intensity - shifts
        worst = int(np.argmin(intensities))
        if intensities[worst] >= -1e-12:
            break
        del ids[worst]
    return ids


def compute_modal_intensities(modes_df, max_pump_intensity, mode_competition_matrix):
    """Compute the modal intensities of the modes up to D0, with D0_steps.

    Thin wrapper over :func:`_modal_intensity_sweep` with a *fixed*
    (pump-independent) competition matrix -- the original near-threshold SALT
    model. :func:`compute_modal_intensities_self_consistent` reuses the same
    sweep with a pump-dependent matrix.
    """
    return _modal_intensity_sweep(
        modes_df, max_pump_intensity, lambda _pump, _ids: mode_competition_matrix
    )


def _modal_intensity_sweep(modes_df, max_pump_intensity, get_matrix):
    """Event-driven modal-intensity sweep over the pump strength.

    ``get_matrix(pump, lasing_mode_ids)`` returns the full mode-competition
    matrix to use at the given operating pump and active set. For the linear
    model it ignores both arguments and returns a constant matrix (so the result
    is byte-identical to the historical implementation); the self-consistent
    model rebuilds the matrix from the operating-pump mode profiles; the
    full-SALT model additionally saturates it with the lasing field. The mode
    activation / vanishing event logic is shared by all three.
    """
    lasing_thresholds = np.asarray(modes_df["lasing_thresholds"]).ravel()

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
    # safety cap: the linear model terminates in <~2*n_modes events, but a
    # pump-dependent matrix (self_consistent / full_salt) could in principle
    # chatter; bound the loop so it always returns.
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

        # competition matrix at the current operating pump (constant for linear)
        mode_competition_matrix = get_matrix(pump_intensity, lasing_mode_ids)

        # enforce the physical non-negativity constraint: a pump-dependent matrix
        # can drive the linear solve negative (a no-op for the constant linear
        # matrix, so its result is unchanged).
        lasing_mode_ids = _nonneg_active_set(
            mode_competition_matrix, lasing_thresholds, lasing_mode_ids, pump_intensity
        )

        # 1) compute the current mode intensities
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


def compute_modal_intensities_self_consistent(
    graph,
    modes_df,
    max_pump_intensity,
    D0_steps=30,
    max_iter=20,
    tol=1e-6,
    damping=0.5,
    quality_method="eigenvalue",
):
    r"""Modal intensities with mode profiles re-evaluated at the operating pump.

    Relaxes the frozen-threshold-profile approximation (issue #42, #2): instead
    of a single competition matrix built once with every mode at its own
    threshold, the matrix is rebuilt at each operating pump with all modes
    **followed to that pump** (each refined to the actual mode of the pumped
    operator, not its frozen threshold field -- see
    :func:`compute_mode_competition_matrix_at_pump`). It then reuses the exact
    same event-driven activation / mode-vanishing sweep as
    :func:`compute_modal_intensities` (via :func:`_modal_intensity_sweep`), so it
    inherits the linear model's competition bookkeeping and reduces to it as the
    matrix becomes pump-independent.

    The *linear* gain saturation is kept, so at a fixed operating pump the matrix
    depends only on the pump (through the profiles) and not on the intensities --
    there is no inner fixed point. ``D0_steps``/``max_iter``/``tol``/``damping``
    are accepted only for interface parity with
    :func:`compute_modal_intensities_full_salt`.

    Args:
        graph: pumped quantum graph (with ``pump``/``D0_max`` in its params).
        modes_df: threshold-modes dataframe (``threshold_lasing_modes``,
            ``lasing_thresholds``).
        max_pump_intensity (float): top of the pump sweep.
    """
    del max_iter, tol, damping, quality_method  # linear saturation: no inner loop

    # Rebuild the (expensive) competition matrix only on a bounded pump grid: the
    # event sweep runs at fine resolution for the intensities, but snapping the
    # matrix to ``D0_steps`` points keeps it piecewise-constant and bounds the
    # number of rebuilds (the event spacing is otherwise unbounded once T varies
    # with pump). The grid includes the first threshold, so the matrix there
    # equals the linear one and the reduction-at-threshold check still holds.
    snap = _pump_snapper(modes_df, max_pump_intensity, D0_steps)
    cache: dict[float, np.ndarray] = {}

    def get_matrix(pump, _lasing_mode_ids):
        key = snap(pump)
        if key not in cache:
            cache[key] = compute_mode_competition_matrix_at_pump(graph, modes_df, key)
        return cache[key]

    return _modal_intensity_sweep(modes_df, max_pump_intensity, get_matrix)


def _pump_snapper(modes_df, max_pump_intensity, D0_steps):
    """Return a function snapping a pump to a bounded grid above first threshold."""
    lasing_thresholds = np.asarray(modes_df["lasing_thresholds"]).ravel()
    finite = lasing_thresholds[lasing_thresholds < np.inf]
    first = float(finite.min()) if finite.size else 0.0
    grid = np.linspace(first, float(max_pump_intensity), max(int(D0_steps), 2))

    def snap(pump):
        return float(grid[np.argmin(np.abs(grid - float(pump)))])

    return snap


def compute_modal_intensities_full_salt(
    graph,
    modes_df,
    max_pump_intensity,
    D0_steps=30,
    max_iter=30,
    tol=1e-7,
    damping=0.7,
    oversample_size=None,
    quality_method="eigenvalue",
):
    r"""Best-effort nonlinear-SALT modal intensities with spatial hole burning.

    *Experimental, opt-in.* Relaxes both linearised-SALT approximations
    (issue #42): mode profiles are taken at the operating pump (as in
    :func:`compute_modal_intensities_self_consistent`, #2) *and* the gain is
    saturated by the lasing field through the per-edge hole-burning denominator
    :math:`1 + \sum_\nu \Gamma_\nu a_\nu |\Psi_\nu(x)|^2` (#1), which clamps the
    gain and bends the L--I curves over.

    Implementation: the same event-driven sweep as the other two solvers
    (:func:`_modal_intensity_sweep`) is reused, so the activation / mode-vanishing
    bookkeeping is identical. At each operating pump a damped fixed point in the
    active intensities saturates the competition matrix: each lasing mode's
    effective gain is reduced by its pump-weighted hole-burning factor
    :math:`g_\mu\in(0,1]`, which *inflates* its row of the competition matrix and
    so lowers its intensity. As the intensities go to zero ``g`` goes to one and
    the result reduces to :func:`compute_modal_intensities_self_consistent`
    (hence to linear) -- asserted in the tests. Non-convergence never raises: the
    last iterate is kept and a warning emitted.

    ``oversample_size`` (forwarded to :func:`oversample_graph`) refines the
    per-edge-constant saturation toward the true within-edge field -- smaller is
    more accurate and slower. This solver is validated on the small ``line_PRA``
    example; on large graphs it is best treated as exploratory.
    """
    del quality_method  # frozen-threshold profiles: no mode re-solve needed

    lasing_thresholds = np.asarray(modes_df["lasing_thresholds"]).ravel()
    work_graph = graph if oversample_size is None else oversample_graph(graph, oversample_size)
    threshold_modes = modes_df["threshold_lasing_modes"].to_numpy()

    # snap the (expensive) matrix/profile rebuilds onto a bounded pump grid; see
    # compute_modal_intensities_self_consistent for the rationale.
    snap = _pump_snapper(modes_df, max_pump_intensity, D0_steps)
    matrix_cache: dict[float, np.ndarray] = {}
    profile_cache: dict[tuple, tuple] = {}

    def _profiles(pump, ids):
        """(weight, mean_e2_n, gains, denom_norm) for ``ids`` at ``pump``."""
        key = (snap(pump), tuple(ids))
        if key not in profile_cache:
            pumped = graph_with_pump(work_graph, key[0])
            pump_profile = np.asarray(pumped.graph["params"]["pump"], dtype=float)
            mean_e2 = np.array(
                [
                    np.abs(mean_mode_on_edges(threshold_modes[i], pumped, check_quality=False))
                    for i in ids
                ]
            )
            gains = np.array(
                [abs(gamma(to_complex(threshold_modes[i]), pumped.graph["params"])) for i in ids]
            )
            weight = mean_e2 * pump_profile[None, :]
            denom_norm = weight.sum(1)
            denom_norm[denom_norm == 0] = 1.0
            profile_cache[key] = (weight, mean_e2 / denom_norm[:, None], gains, denom_norm)
        return profile_cache[key]

    def get_matrix(pump, lasing_mode_ids):
        key = snap(pump)
        if key not in matrix_cache:
            matrix_cache[key] = compute_mode_competition_matrix_at_pump(work_graph, modes_df, key)
        base = matrix_cache[key]
        ids = list(lasing_mode_ids)
        if not ids:
            return base

        weight, mean_e2_n, gains, denom_norm = _profiles(pump, ids)
        idx = np.ix_(ids, ids)
        a = np.zeros(len(ids))
        saturated = base
        for _ in range(max_iter):
            # per-edge spatial-hole-burning denominator -> per-mode gain clamp
            sat = 1.0 + (gains[:, None] * a[:, None] * mean_e2_n).sum(0)  # (n_edges,)
            g = (weight / sat[None, :]).sum(1) / denom_norm  # in (0, 1], -> 1 as a->0
            saturated = base.copy()
            saturated[idx] = base[idx] / g[:, None]  # inflate mode-mu rows -> lower intensity
            slopes, shifts = _intensity_slopes_shifts(saturated, lasing_thresholds, ids)
            a_new = np.clip(slopes * pump - shifts, 0.0, None)
            if np.linalg.norm(a_new - a) <= tol * (np.linalg.norm(a) + tol):
                a = a_new
                break
            a = (1.0 - damping) * a + damping * a_new
        else:
            warnings.warn(
                f"full_salt hole-burning did not converge at D0={pump:.4g}; keeping last iterate.",
                stacklevel=2,
            )
        return saturated

    return _modal_intensity_sweep(modes_df, max_pump_intensity, get_matrix)


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


def _solve_active_set(
    graph, modes0, fields0, a0, D0, pump, pump_mask, max_steps, seed, outer=6, damping=0.7
):
    """Frozen-field trust-region ``(k, a)`` solve for a *fixed* active set.

    Block iteration: (i) freeze the saturated background fields, (ii) solve every
    mode's ``(k_μ real, a_μ ≥ 0)`` with one bounded trust-region least-squares on
    the clean :func:`_salt_block_residual` (``k`` confined to a window below the
    inter-mode spacing so modes cannot hop, ``a`` to ``[0, a_max]``), (iii) refresh
    the fields, repeat. This replaces the old decoupled amplitude least-squares
    whose residual re-ran an inner fixed point -- a noisy Jacobian that made the
    multimode amplitudes chatter. Returns ``(modes, fields, a, converged)``.
    """
    n = len(modes0)
    k0 = np.array([float(m[0]) for m in modes0])
    a = np.clip(np.asarray(a0, dtype=float), 1e-3, None)
    fields = [np.asarray(f, dtype=float) for f in fields0]
    # Confine k to a *tight* window: frequency pulling above threshold is small,
    # and a loose window lets the trust region zero a mode's residual by drifting
    # its k to a spurious nearby root with a=0 (collapsing multimode to one mode)
    # instead of raising its amplitude to lase. The window tracks the spacing
    # (0.2 * min gap) with only a numerical floor, so it stays below the spacing
    # even for a dense/near-degenerate spectrum (a fixed floor that exceeded the
    # spacing made near-degenerate modes collide -> collapse or divergence).
    if n > 1:
        gaps = np.abs(k0[:, None] - k0[None, :])
        gaps[np.diag_indices(n)] = np.inf
        window = float(np.clip(0.2 * gaps.min(), 1e-6, 0.1))
    else:
        window = 0.1
    a_max = max(1.0e3 * max(float(np.max(a)), 1.0e-3), 1.0e3)
    lo = np.concatenate([k0 - window, np.zeros(n)])
    hi = np.concatenate([k0 + window, np.full(n, a_max)])
    ks = k0.copy()
    converged = False
    for _ in range(outer):
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
        g = _saturated_graph_multi(graph, [[k, 0.0] for k in ks], a, D0, pump, fields)
        new = [_single_mode_field_intensity(g, [ks[i], 0.0], pump_mask) for i in range(n)]
        change = sum(np.linalg.norm(new[i] - fields[i]) for i in range(n))
        fields = [(1.0 - damping) * fields[i] + damping * new[i] for i in range(n)]
        if change <= 1e-6 * (1.0 + sum(np.linalg.norm(f) for f in fields)):
            converged = True
            break
    modes = [np.array([float(ks[i]), 0.0]) for i in range(n)]
    return modes, fields, a, converged


def _newton_onset_unit_scale(
    graph, mode0, field0, threshold, t_self, pump, pump_mask, max_steps, seed
):
    """Per-mode factor converting the Newton amplitude to the linear-intensity unit.

    The Newton amplitude lives in the integral-normalised (``∫|Ê|^2 = 1``)
    convention with a per-edge-mean saturation, while the competition matrix
    integrates the true ``|E|^4``; the two differ by a graph-dependent within-edge
    form factor. Match the single-mode onset slope (linear: ``1/(T_μμ·D0_thr)``) by
    probing the isolated mode just above threshold. Returns 1.0 if degenerate.
    """
    s_linear = 1.0 / (t_self * threshold)
    eps = 0.05
    _, _, a_probe, _ = _solve_active_set(
        graph,
        [mode0],
        [field0],
        [s_linear * threshold * eps],
        threshold * (1.0 + eps),
        pump,
        pump_mask,
        max_steps,
        seed,
    )
    s_newton = float(a_probe[0]) / (threshold * eps)
    if not np.isfinite(s_newton) or s_newton <= 0.0:
        return 1.0
    return s_linear / s_newton


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
    if est_nodes > node_cap:  # keep the oversampled graph (and its eigensolves) bounded
        target *= est_nodes / node_cap
    return float(target)


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
):
    r"""Operator-level full-SALT L--I curves with a self-consistent active set.

    Solves the real nonlinear SALT eigenproblem: at each pump it finds, for every
    *lasing* mode, the real frequency ``k_μ`` and amplitude ``a_μ ≥ 0`` such that
    the shared saturated operator ``L_sat``
    (:func:`~netsalt.physics.dispersion_relation_pump_saturated`) is singular at
    each real ``k_μ`` simultaneously.

    Two ingredients make the multimode solve robust (see ``doc/source/lasing.rst``):

    * **Frozen-field trust-region solve** (:func:`_solve_active_set`). For a fixed
      active set the background fields are frozen while a bounded trust-region
      least-squares solves all ``(k_μ, a_μ)``; the fields are then refreshed and the
      step repeated. The frozen field makes the residual a single clean eigensolve
      per mode, so the Jacobian is noise-free -- unlike the old decoupled solve
      whose residual re-ran an inner fixed point and chattered.
    * **Self-consistent active set.** The pump is stepped up; at each step the
      confirmed lasing set is solved, modes whose amplitude vanishes are dropped,
      and a non-lasing candidate is added when it has net gain (``α < 0``) on the
      *current saturated background*. The active set is thus found self-consistently
      from the saturated operator, not borrowed from the linear model. (This is only
      faithful when the hole burning is resolved -- see below; with the bare-edge
      mean it over-clamps and drops modes that should co-lase.)

    Amplitudes are reported in the **linear modal-intensity unit**
    (:func:`_newton_onset_unit_scale`), so the curves are directly comparable to
    the other solvers and reduce to the linear onset slope at threshold.

    **What this solver is validated for: the lasing count and frequency pulling,
    not the above-threshold magnitudes.** The operator-level solve gives a
    self-consistent gain-clamping *active set* (which modes lase) and the lasing
    *frequencies* ``k_μ`` that the competition-matrix solvers cannot -- on
    ``line_PRA`` it lases the two modes of Ge-Chong-Stone (PRA 82, 063824, Eq. 28)
    where ``self_consistent`` over-suppresses to one. But the *magnitude* is read
    off the bare amplitude ``a`` in the saturation denominator ``1 + Γ a |Ê|²``,
    which is **not** the SALT modal intensity: Ge/Stone obtain intensities from the
    single-pole-approximation matrix equation ``D0/D0_thr - 1 = Σ_ν Γ_ν χ_μν I_ν``
    (exactly netSALT's competition-matrix solvers). The bare ``a`` reduces to the
    linear intensity at threshold but **grows super-linearly above it on
    multi-loop graphs** (verified: ~1--2× above linear on chord/ring networks,
    where ``self_consistent``/``full_salt`` correctly saturate *below* linear),
    because the local-saturation clamp with a non-uniform standing-wave profile is
    not the projected SPA intensity. **For quantitative L--I magnitudes use
    ``linear`` / ``self_consistent`` / ``full_salt`` (Ge's SPA method);** treat
    ``full_salt_newton`` as the gain-clamping count + frequency-pulling diagnostic.

    **Within-edge hole burning must be resolved.** The saturation samples
    ``|E_ν(x)|^2`` per edge; with one sample per edge the per-edge *mean*
    over-estimates the spatial overlap (it washes out the standing-wave
    nodes/antinodes) and **over-clamps**, spuriously suppressing co-lasing modes --
    it lased one mode on ``line_PRA`` where Ge-Chong-Stone (PRA 82, 063824, Eq. 28)
    and the competition matrix lase two. ``oversample_size=None`` therefore
    auto-picks a wavelength-resolving sub-edge size (:func:`_auto_oversample_size`);
    with it newton reproduces the two-mode result and reduces to linear near
    threshold. Pass ``oversample_size=0`` for the old (over-clamping) bare-edge
    behaviour, or a float to set it explicitly.

    **Best for sparse spectra (few, resolved modes).** The coupled ``(k, a)``
    solve confines each mode's ``k`` to a window ``0.2 * min_spacing`` so a mode
    cannot drift to a neighbour's root; the window tracks the actual spacing (no
    fixed floor), so it stays robust as the spectrum tightens -- on a near-degenerate
    triplet (Δk ~ 1e-4) it lases the cluster instead of collapsing or diverging.
    Two practical limits remain on a genuinely *dense* spectrum (e.g. a buffon
    network, ~10^2--10^3 modes per unit ``k``): (i) **cost** -- the active set is
    re-solved at every pump with a numerical Jacobian over all lasing ``(k, a)``,
    each residual an operator eigensolve, so the work grows steeply with the
    number of co-lasing modes and the graph size (minutes for a few dozen modes on
    a 200-node graph); (ii) **near-degeneracy** -- the spacing is so small that any
    physically broad gain window holds dozens of modes within ~1e-4 of each other,
    and the per-mode amplitudes become ill-conditioned. The competition-matrix
    solvers (``linear`` / ``self_consistent`` / ``full_salt``) remain the right
    tool at buffon scale; ``full_salt_newton`` is aimed at sparse-spectrum cavities
    (lines, rings, chord networks) and small mode counts.

    It never raises -- a step that fails to fully converge keeps its iterate and
    warns. (``max_iter``, ``tol``, ``inner_max_iter``, ``inner_damping`` are
    accepted for interface parity.)
    """
    del max_iter, tol, inner_max_iter, inner_damping, quality_method  # interface parity

    lasing_thresholds = np.asarray(modes_df["lasing_thresholds"]).ravel()
    n_modes = len(modes_df)
    modal_intensities = pd.DataFrame(index=range(n_modes))
    interacting_lasing_thresholds = np.inf * np.ones(n_modes)
    candidates = [int(i) for i in np.where(lasing_thresholds < np.inf)[0]]
    if not candidates:
        return _finalise_modal_intensities(
            modes_df, modal_intensities, interacting_lasing_thresholds
        )

    # Resolve the within-edge field for the hole burning: the bare-edge (per-edge
    # mean) sampling over-clamps and spuriously suppresses co-lasing modes (it
    # disagreed with Ge-Chong-Stone Eq. 28 on line_PRA, lasing one mode where two
    # lase). ``oversample_size=None`` now auto-picks a wavelength-resolving size;
    # pass 0 to force the old bare-edge behaviour.
    if oversample_size is None:
        oversample_size = _auto_oversample_size(graph, modes_df)
    work_graph = graph if not oversample_size else oversample_graph(graph, oversample_size)
    pump = np.asarray(work_graph.graph["params"]["pump"], dtype=float)
    pump_mask = _get_mask_matrices(work_graph.graph["params"])[1]
    # small, fixed budget for each trust-region sub-solve (a local, warm-started
    # solve converges in tens of evaluations; independent of params["max_steps"])
    max_steps = 30

    # linear competition matrix only for the amplitude *unit* (diagonal) -- the
    # active set itself is found self-consistently, not borrowed from it
    t_linear = compute_mode_competition_matrix(work_graph, modes_df)
    t_diag = np.array(
        [abs(t_linear[i, i]) if abs(t_linear[i, i]) > 1e-12 else 1.0 for i in range(n_modes)]
    )
    threshold_modes = modes_df["threshold_lasing_modes"].to_numpy()

    # NB: do not add per-mode "0 at its own threshold" baseline columns -- a
    # mode's threshold is off the shared pump grid, so the *other* modes are then
    # undefined (NaN -> 0) at that pump, putting a spurious dip in their curves.
    # The grid already records every mode at every pump, with 0 below activation.

    mode_state: dict[int, np.ndarray] = {}
    field_state: dict[int, np.ndarray] = {}
    a_state: dict[int, float] = {}
    unit_scale: dict[int, float] = {}

    def _init(i):
        mode = np.asarray(from_complex(threshold_modes[i]), dtype=float)
        field = _single_mode_field_intensity(
            graph_with_pump(work_graph, float(lasing_thresholds[i])), mode, pump_mask
        )
        mode_state[i], field_state[i], a_state[i] = mode, field, 0.0
        unit_scale[i] = _newton_onset_unit_scale(
            work_graph,
            mode,
            field,
            float(lasing_thresholds[i]),
            t_diag[i],
            pump,
            pump_mask,
            max_steps,
            seed,
        )

    active: list[int] = []  # confirmed lasing ids, carried along the continuation
    first = float(np.min(lasing_thresholds[candidates]))
    for D0 in np.linspace(first, max_pump_intensity, D0_steps):
        for _ in range(len(candidates) + 1):  # active-set sweeps until stable
            if active:
                modes_out, fields_out, a_out, converged = _solve_active_set(
                    work_graph,
                    [mode_state[i] for i in active],
                    [field_state[i] for i in active],
                    [max(a_state[i], 1e-3) for i in active],
                    D0,
                    pump,
                    pump_mask,
                    max_steps,
                    seed,
                )
                if not converged:
                    warnings.warn(
                        f"full_salt_newton field loop did not fully converge at D0={D0:.4g}.",
                        stacklevel=2,
                    )
                for j, i in enumerate(active):
                    mode_state[i], field_state[i], a_state[i] = (
                        modes_out[j],
                        fields_out[j],
                        float(a_out[j]),
                    )
                active = [i for i in active if a_state[i] > 1e-4]  # drop vanished
            # gain the not-yet-lasing candidates see on the current saturated background
            background = _saturated_graph_multi(
                work_graph,
                [mode_state[i] for i in active],
                [a_state[i] for i in active],
                D0,
                pump,
                [field_state[i] for i in active],
            )
            added = False
            for c in candidates:
                if c in active or lasing_thresholds[c] > D0:
                    continue
                if c not in mode_state:
                    _init(c)
                kc = _refine_local(mode_state[c], background, 1e-9, max_steps, seed, k_window=0.3)
                # alpha = mode[1] < 0 => net gain => the mode lases. Use a *tight*
                # margin: gain clamping pins an above-threshold mode's alpha at ~0^-
                # (the lasing modes hold it right at threshold), often only ~1e-4
                # negative. A looser cutoff (e.g. -1e-4, the same magnitude) then adds
                # the mode many pump steps late and snaps it to its already-large
                # amplitude -- a spurious jump in its L--I curve and a matching dip in
                # the others. Adding right at the crossing makes it ramp continuously;
                # the a < 1e-4 drop rule above is the safety net against false adds.
                if kc[1] < -1e-6:
                    mode_state[c] = np.array([float(kc[0]), 0.0])
                    a_state[c] = 1e-3
                    active.append(c)
                    added = True
            if not added:
                break
        for i in candidates:
            value = max(a_state.get(i, 0.0) * unit_scale.get(i, 1.0), 0.0) if i in active else 0.0
            modal_intensities.loc[i, D0] = value
            if i in active and a_state[i] > 0 and D0 < interacting_lasing_thresholds[i]:
                interacting_lasing_thresholds[i] = D0

    return _finalise_modal_intensities(modes_df, modal_intensities, interacting_lasing_thresholds)


def pump_trajectories(modes_df, graph, return_approx=False, quality_method="eigenvalue"):
    """For a sequence of D0s, find the mode positions of the modes modes."""

    D0s = np.linspace(
        0,
        graph.graph["params"]["D0_max"],
        graph.graph["params"]["D0_steps"],
    )

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
            pumped_modes_approx[-1][m] = pump_linear(pumped_modes[-1][m], graph, D0s[d], D0s[d + 1])

        worker_modes = WorkerModes(
            pumped_modes_approx[-1],
            graph,
            D0s=n_modes * [D0s[d + 1]],
            quality_method=quality_method,
        )
        with (
            _scoped_warning_filters(),
            multiprocessing.Pool(graph.graph["params"]["n_workers"]) as pool,
        ):
            pumped_modes.append(list(tqdm(pool.imap(worker_modes, range(n_modes)), total=n_modes)))
        for i, mode in enumerate(pumped_modes[-1]):
            if mode is None:
                L.info("Mode not be updated, consider changing the search parameters.")
                pumped_modes[-1][i] = pumped_modes[-2][i]

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


def find_threshold_lasing_modes(modes_df, graph, quality_method="eigenvalue"):
    """Find the threshold lasing modes and associated lasing thresholds."""
    stepsize = graph.graph["params"]["search_stepsize"]
    D0_steps = graph.graph["params"]["D0_max"] / graph.graph["params"]["D0_steps"]
    new_modes = modes_df["passive"].to_numpy()

    threshold_lasing_modes = np.zeros([len(modes_df), 2])
    lasing_thresholds = np.inf * np.ones(len(modes_df))
    D0s = np.zeros(len(modes_df))
    current_modes = np.arange(len(modes_df))
    stuck_modes_count = 0
    max_modes = len(current_modes)
    prev_n_modes = 0
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
        with multiprocessing.Pool(graph.graph["params"]["n_workers"]) as pool:
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

        with multiprocessing.Pool(graph.graph["params"]["n_workers"]) as pool:
            new_modes_tmp[current_modes] = list(
                tqdm(pool.imap(worker_modes, current_modes), total=len(current_modes))
            )

        to_delete = []
        for i, mode_index in enumerate(current_modes):
            if new_modes_tmp[mode_index] is None:
                L.info("A mode could not be updated, consider modifying the search parameters.")
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


def lasing_threshold_linear(mode, graph, D0):
    """Find the linear approximation of the new wavenumber."""
    graph = graph_with_pump(graph, D0)
    return 1.0 / (
        q_value(mode)
        * -1
        * np.imag(gamma(to_complex(mode), graph.graph["params"]))
        * np.real(compute_overlapping_factor(mode, graph))
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

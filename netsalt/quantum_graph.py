"""Quantum graph construction module.

A quantum graph is a networkx graph with additional parameters in graph.graph['param']
and specific node/edges attributes.
"""

import copy
import logging
import warnings

import networkx as nx
import numpy as np
import scipy as sc

from .params import NetSaltParams
from .physics import (
    set_dielectric_constant,
    set_dispersion_relation,
    update_params_dielectric_constant,
)
from .utils import to_complex

L = logging.getLogger(__name__)

# Quantum-graph laplacians at or below this dimension use a direct dense
# eigensolve instead of ARPACK shift-invert: ``eigs(sigma=0)`` carries a fixed
# per-call overhead (a sparse LU factorisation + Arnoldi restart, ~2.5 ms) that
# only dominates for small graphs, where ``np.linalg.eig`` on the dense matrix is
# faster and returns the same nearest-zero eigenpair. Above the crossover (~50
# nodes, measured) ARPACK wins decisively: it is ~flat in N on the banded
# quantum-graph laplacian (~2.5 ms at N=60, ~6 ms at N=1000), whereas dense is
# O(N^3) (~24 ms at N=120, ~160 ms at N=250). Kept high (256) here for robust
# *mode finding* on dense-spectrum 2D graphs (e.g. buffon): the grid scan probes
# many near-singular k, where ARPACK shift-invert (sigma=0, an LU of a near-singular
# matrix) is slow/unstable, while dense is robust. ``full_salt_newton`` lowers it
# *locally* (NEWTON_DENSE_EIG_MAX) for its own banded, oversampled saturated solves,
# where ARPACK is both fast (~flat in N) and stable (isolated lasing modes).
DENSE_EIG_MAX = 256
# Above this dimension ``laplacian_quality(method="singularvalue")`` falls back to
# the sparse ``svds`` path. Below it, a dense SVD is far cheaper: ``svds(which="SM")``
# converges very slowly on quantum-graph laplacians (measured 109 ms vs 0.267 ms at
# n=61 -- a 410x penalty), because the smallest singular value is exactly what
# Lanczos-type methods are worst at.
DENSE_SVD_MAX = 1000


def create_quantum_graph(
    graph, params=None, positions=None, lengths=None, seed=42, noise_level=0.001
):
    """Extend a networkx graph with necessary attributes for being a quantum graph.

    Args:
        graph (networkx graph): pure networkx graph to consider as a quantum graph
        params (dict): specific parameters to setup the quantum graph (depends on use cases)
        positions (list): node positions, if Non networkx.spring_layout is used
        lengths (list) node lengths, if not None, it will override the lengths from positions
        seed (int): seed for rng
        noise_level (float): adds some noise if too manuy edges of equal lengths are found
    """
    _set_node_positions(graph, positions)
    _set_edge_lengths(graph, lengths=lengths)
    _verify_lengths(graph, seed=seed, noise_level=noise_level, from_positions=lengths is None)
    if params is None:
        params = graph.graph["params"]
    set_inner_edges(graph, params)
    update_parameters(graph, params)


def _verify_lengths(graph, seed=42, noise_level=0.001, from_positions=True):
    """Jitter the geometry when too many edges share a length.

    Equal edge lengths put every edge's ``k_e l_e`` on the secular matrix's pole
    at the same ``k`` (see the module docstring of ``examples/audit``), so the
    modes there are lost. This breaks the tie — but it is a *physics* change,
    not a numerical nudge: at the default ``noise_level=0.001`` the measured
    price is ~1e-4 relative in every mode's ``k`` and a ~1e-3 relative splitting
    of exact degeneracies. Hence the warning rather than a silent log line.

    Args:
        graph (graph): quantum graph
        seed (int): seed for the jitter
        noise_level (float): jitter scale, relative to the shortest edge. 0 disables.
        from_positions (bool): jitter node positions and recompute lengths from
            them. False when the caller supplied explicit ``lengths``, in which
            case the lengths are jittered directly — recomputing from positions
            would throw the supplied lengths away entirely.
    """
    if noise_level <= 0.0:
        return
    lengths = np.array([graph[u][v]["length"] for u, v in graph.edges])
    # ``np.unique(..., return_counts=True)`` returns ``(values, counts)``; taking
    # ``np.max`` over the pair compared the largest edge *length* against a
    # threshold on the *count*, so any graph whose edges were longer than
    # ``0.2 * n_edges`` got jittered even with every length distinct.
    _, counts = np.unique(np.around(lengths, 5), return_counts=True)
    if counts.max() <= 0.2 * len(graph.edges):
        return

    warnings.warn(
        f"{counts.max()} of {len(graph.edges)} edges share a length; jittering the geometry "
        f"by noise_level={noise_level} so the equal-length modes are not lost to the secular "
        "matrix's pole at k*l in pi*Z. This perturbs every mode (~1e-4 relative in k at the "
        "default) and splits exact degeneracies. Pass noise_level=0 to keep the geometry "
        "exactly as given, and see issue #45.",
        stacklevel=3,
    )
    rng = np.random.default_rng(seed)
    if from_positions:
        for u in graph:
            graph.nodes[u]["position"][0] += rng.normal(0, noise_level * lengths.min())
        _set_edge_lengths(graph)
    else:
        _set_edge_lengths(
            graph, lengths=lengths + rng.normal(0, noise_level * lengths.min(), len(lengths))
        )


def _not_equal(data1, data2, force=False):
    """Check if datasets are the same."""
    if force:
        return True
    if isinstance(data1, np.ndarray):
        return all(data1 != data2)
    return data1 != data2


def update_parameters(graph, params, force=False):
    """Set the parameter dictionary to the graph, validating via pydantic.

    Unknown keys are allowed (``NetSaltParams`` uses ``extra="allow"``), but
    typed fields (``k_min``, ``D0``, …) get validated on assignment so a
    wrong-type value fails loudly at the seam rather than silently floating
    through the compute path.

    Args:
        graph (graph): quantum graph
        params (dict or NetSaltParams): parameters to merge in.
        force (bool): if True, overwrite values for keys in ``warning_params``
            that would otherwise be preserved when the graph already has them.
    """
    warning_params = {
        "k_min",
        "k_max",
        "k_n",
        "alpha_min",
        "alpha_max",
        "alpha_n",
        "k_a",
        "gamma_perp",
        "dielectric_params",
        "edgelabel",
    }
    validated = NetSaltParams.from_dict(params)
    if "params" not in graph.graph:
        graph.graph["params"] = validated
        return
    existing = graph.graph["params"]
    if not isinstance(existing, NetSaltParams):
        existing = NetSaltParams.from_dict(existing)
        graph.graph["params"] = existing
    for param, value in validated.items():
        if param not in existing:
            existing[param] = value
        elif _not_equal(existing[param], value, force=force):
            if param in warning_params:
                if force:
                    existing[param] = value
            else:
                existing[param] = value


def get_total_length(graph):
    """Get the total length of a quantum graph.

    Args:
        graph (graph): quantum graph
    """
    return sum(graph[u][v]["length"] for u, v in graph.edges())


def get_total_inner_length(graph):
    """Get the total inner length of the graph (considering inner edges only).

    Inner edges are defined as edges without degree one nodes.

    Args:
        graph (graph): quantum graph
    """
    return sum(graph[u][v]["length"] for u, v in graph.edges() if graph[u][v]["inner"])


def set_total_length(graph, total_length=None, max_extent=None, inner=True, with_position=True):
    """Set the (inner) total lengths of the graph to a given value.

    Args:
        graph (graph): quantum graph
        total_length (float): total length to set
        max_extent (float): only if total_length is None, set the maximal extent
        inner (bool): if True, only consider inner edges
        with_position (bool): if True, also rescale node positions
    """
    if total_length is not None and max_extent is not None:
        raise ValueError("only one of total_length or max_extent is allowed")
    length_ratio = 1.0
    if total_length is not None:
        if inner:
            original_total_length = get_total_inner_length(graph)
        else:
            original_total_length = get_total_length(graph)
        length_ratio = total_length / original_total_length

    if max_extent is not None:
        _min_pos = min(
            np.array(
                [graph.nodes[u]["position"] for u in graph.nodes() if len(graph[u]) > 1]
            ).flatten()
        )
        _max_pos = max(
            np.array(
                [graph.nodes[u]["position"] for u in graph.nodes() if len(graph[u]) > 1]
            ).flatten()
        )
        _extent = _max_pos - _min_pos
        length_ratio = max_extent / _extent

    for u, v in graph.edges:
        graph[u][v]["length"] *= length_ratio
    if with_position:
        for u in graph:
            graph.nodes[u]["position"] *= length_ratio

    graph.graph["lengths"] = np.array([graph[u][v]["length"] for u, v in graph.edges])


def _set_pump_on_graph(graph):
    """Set the pump values on the graph from params."""
    if "pump" not in graph.graph["params"]:
        graph.graph["params"]["pump"] = np.ones(len(graph.edges))
    for ei, e in enumerate(graph.edges):
        graph[e[0]][e[1]]["pump"] = graph.graph["params"]["pump"][ei]


def _set_pump_on_params(graph, params):
    """Set the pump values on the graph from params."""
    params["pump"] = np.ones(len(graph.edges))
    for ei, e in enumerate(graph.edges):
        params["pump"][ei] = graph[e[0]][e[1]]["pump"]


def simplify_graph(graph):
    """Remove degree 2 nodes.

    Args:
        graph (graph): quantum graph
    """
    nodes_to_remove = []
    edges_to_add = []
    if all(len(graph[u]) == 2 for u in graph.nodes):
        return graph
    for u in graph.nodes:
        if len(graph[u]) == 2:
            neighs = list(graph[u].keys())
            edges_to_add.append((neighs[0], neighs[1]))
            nodes_to_remove.append(u)
    graph.add_edges_from(edges_to_add)
    graph.remove_nodes_from(nodes_to_remove)
    return nx.convert_node_labels_to_integers(graph)


def oversample_graph(graph, edge_size):
    """Oversample a graph by adding points on edges.

    The input graph is deep-copied before any mutation: ``_set_pump_on_graph``
    writes per-edge ``pump`` attributes, and the post-copy
    ``_set_pump_on_params`` call rewrites ``params['pump']`` to the
    oversampled-edge count. Without the deep copy these mutations leak back to
    the caller via ``graph.graph["params"]`` (a shared reference) and break any
    subsequent ``compute_mode_*`` call that re-reads those arrays.

    Each sub-edge **inherits** its parent's ``inner`` flag and ``edgelabel``
    rather than having them re-derived. Re-deriving is wrong here:
    :func:`set_inner_edges` calls an edge outer when one of its endpoints has
    degree 1, which after subdivision is true only of the single sub-edge
    touching the terminal node — so most of an open graph's vacuum leads would
    be relabelled *inner*. On the shipped Fabry-Perot line that moves the
    "inner" length from the cavity's 0.5 to 0.589, an 18% error in every
    integral normalised over the cavity (``_newton_onset_unit_scale`` is one).
    netsalt's own hole-burning path masks with ``pump * inner`` and so was
    insulated, but the flag is public and consumers read it.

    Inheriting ``edgelabel`` also keeps the sub-edge → parent-edge map, which
    is the only way to fold a work-graph quantity back onto the original graph.

    Args:
        graph (graph): quantum graph (left untouched)
        edge_size (float): edge size to sample the graph
    """
    graph = copy.deepcopy(graph)
    _set_pump_on_graph(graph)
    oversampled_graph = graph.copy()
    for ei, (u, v) in enumerate(graph.edges):
        last_node = len(oversampled_graph)
        n_nodes = int(graph[u][v]["length"] / edge_size)
        if n_nodes > 1:
            dielectric_constant = graph[u][v].get("dielectric_constant", None)
            pump = graph[u][v]["pump"]
            inner = graph[u][v].get("inner", True)
            edgelabel = graph[u][v].get("edgelabel", ei)
            oversampled_graph.remove_edge(u, v)

            for node_index in range(n_nodes - 1):
                node_position_x = graph.nodes[u]["position"][0] + (node_index + 1) / n_nodes * (
                    graph.nodes[v]["position"][0] - graph.nodes[u]["position"][0]
                )
                node_position_y = graph.nodes[u]["position"][1] + (node_index + 1) / n_nodes * (
                    graph.nodes[v]["position"][1] - graph.nodes[u]["position"][1]
                )
                node_position = np.array([node_position_x, node_position_y])

                if node_index == 0:
                    first, last = u, last_node
                else:
                    first, last = last_node + node_index - 1, last_node + node_index

                oversampled_graph.add_node(last, position=node_position)
                oversampled_graph.add_edge(
                    first,
                    last,
                    dielectric_constant=dielectric_constant,
                    pump=pump,
                    inner=inner,
                    edgelabel=edgelabel,
                )

            oversampled_graph.add_edge(
                last_node + node_index,
                v,
                dielectric_constant=dielectric_constant,
                pump=pump,
                inner=inner,
                edgelabel=edgelabel,
            )

    oversampled_graph = nx.convert_node_labels_to_integers(oversampled_graph)
    _set_edge_lengths(oversampled_graph)
    params = oversampled_graph.graph["params"]
    params["inner"] = [bool(oversampled_graph[u][v]["inner"]) for u, v in oversampled_graph.edges]
    oversampled_graph.graph["edgelabel"] = np.array(
        [oversampled_graph[u][v]["edgelabel"] for u, v in oversampled_graph.edges]
    )
    update_params_dielectric_constant(oversampled_graph, params)
    _set_pump_on_params(oversampled_graph, params)
    update_parameters(oversampled_graph, params, force=True)
    return oversampled_graph


def construct_laplacian(wavenumber, graph):
    """Construct quantum laplacian from a graph.

    The quantum laplacian is L(k) = B^T(k) W^{-1}(k) B(k), with quantum incidence and weight matrix.

    Args:
        wavenumber (complex): wavenumber
        graph (graph): quantum graph
    """
    set_wavenumber(graph, wavenumber)
    BT, B = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph)
    laplacian = BT.dot(Winv).dot(B)

    node_loss = graph.graph["params"].get("node_loss", 0)
    if node_loss > 0:
        laplacian -= node_loss * sc.sparse.diags(
            [graph[u].get("node_loss", node_loss) for u in graph.nodes()]
        )

    return laplacian


def set_wavenumber(graph, wavenumber):
    """Set edge wavenumbers with dispersion relation defined in graph['dispersion_relation'].

    Args:
        wavenumber (complex): wavenumber
        graph (graph): quantum graph
    """
    graph.graph["ks"] = graph.graph["dispersion_relation"](wavenumber, params=graph.graph["params"])


def graph_with_params(graph, **overrides):
    """Return a shallow copy of ``graph`` with ``params`` field overrides applied.

    Rather than mutating ``graph.graph["params"]`` in place — which leaks values
    into shared state and imposes a fragile set-then-read ordering on every
    downstream consumer (``mode_on_nodes``, ``flux_on_edges``, the
    ``graph.graph["ks"]`` read-backs) — callers that need the laplacian (and its
    derived quantities) at specific parameter values build them on this
    throwaway copy.

    Graph structure and node / edge attributes are shared by reference; only
    ``graph.graph`` is a fresh dict (``nx.Graph.copy`` semantics) with a fresh
    ``params`` swapped in, so writes to ``params`` / ``ks`` /
    ``_incidence_topology`` on the copy never touch the original graph.

    Args:
        graph (graph): quantum graph
        **overrides: ``params`` fields to override on the copy (e.g. ``D0=0.7``,
            ``search_stepsize=0.02``).
    """
    local = graph.copy()
    params = graph.graph["params"]
    if isinstance(params, NetSaltParams):
        local.graph["params"] = params.model_copy(update=dict(overrides))
    else:
        local.graph["params"] = {**params, **overrides}
    return local


def graph_with_pump(graph, D0):
    """Return a shallow copy of ``graph`` whose ``params`` carry pump ``D0``.

    Thin wrapper over :func:`graph_with_params`; see it for the copy semantics.
    The dispersion relations build the laplacian from ``params["D0"]``, so this
    is how callers evaluate a mode at a specific pump without mutating shared
    state.
    """
    return graph_with_params(graph, D0=D0)


def _csr_pattern(rows, cols, n_rows):
    """Return ``(perm, indices, indptr)`` reproducing ``csr_matrix((data, (rows, cols)))``.

    With them, ``csr_matrix((data[perm], indices, indptr))`` is *bit-identical* to
    the COO construction, because the COO path is a pure reordering here: the
    quantum incidence matrices have one entry per (bond, node) pair, so there
    are no duplicates to sum. Returns None if duplicates do exist (a self-loop),
    so the caller falls back to the COO path rather than silently dropping them.
    """
    order = np.lexsort((cols, rows))
    sorted_rows, sorted_cols = rows[order], cols[order]
    duplicated = np.any((np.diff(sorted_rows) == 0) & (np.diff(sorted_cols) == 0))
    if duplicated:
        return None
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n_rows), out=indptr[1:])
    return order, sorted_cols, indptr


def _incidence_topology(graph):
    """Precompute the k-independent arrays used by ``construct_incidence_matrix``.

    Row / column indices, node degrees, and the open-model boundary mask
    depend only on the graph topology and ``params["open_model"]`` — not on
    the wavenumber. The inner loop of :func:`scan_frequencies` and the
    Brownian-ratchet refinement reconstruct these arrays tens of thousands
    of times per run; caching them on ``graph.graph["_incidence_topology"]``
    removes that overhead.
    """
    m = len(graph.edges)
    edges = list(graph.edges)
    row = np.repeat(np.arange(2 * m), 2)
    col = np.repeat(edges, 2, axis=0).flatten()
    deg_u = np.array([len(graph[e[0]]) for e in edges])
    deg_v = np.array([len(graph[e[1]]) for e in edges])
    topology = {
        "row": row,
        "col": col,
        "open_mask": np.logical_or(deg_u == 1, deg_v == 1),
        "m": m,
        "n": len(graph.nodes),
    }
    # CSR patterns for B and B^T. Building a csr_matrix from COO triplets on
    # every call re-sorts the indices and re-runs scipy's index-dtype and format
    # checks, which is most of what construct_laplacian costs at these sizes
    # (the cost is nearly flat in the graph size, i.e. pure bookkeeping). The
    # pattern depends only on the topology, so it is cached here and each call
    # becomes a permutation of the data array.
    topology["b_pattern"] = _csr_pattern(row, col, 2 * m)
    topology["bt_pattern"] = _csr_pattern(col, row, len(graph.nodes))
    graph.graph["_incidence_topology"] = topology
    return topology


def construct_incidence_matrix(graph):
    """Construct the quantum incidence matrix B(k).

    Args:
        graph (graph): quantum graph
    """
    topo = graph.graph.get("_incidence_topology")
    if topo is None or topo["m"] != len(graph.edges):
        topo = _incidence_topology(graph)
    m, n = topo["m"], topo["n"]
    row, col = topo["row"], topo["col"]

    expl = np.exp(1.0j * graph.graph["lengths"] * graph.graph["ks"])
    ones = np.ones(m)
    data = np.dstack([-ones, expl, expl, -ones])[0].flatten()
    data_out = data.copy()

    open_model = graph.graph["params"]["open_model"]
    if open_model == "open":
        mask = topo["open_mask"]
        data_out[1::4][mask] = 0
        data_out[2::4][mask] = 0
    elif open_model == "directed":
        data_out[2::4] = 0
        data_out[3::4] = 0
    elif open_model == "directed_reversed":
        data[2::4] = 0
        data[3::4] = 0

    b_pattern, bt_pattern = topo["b_pattern"], topo["bt_pattern"]
    if b_pattern is None or bt_pattern is None:  # self-loops: no cached pattern
        BT = sc.sparse.csr_matrix((data_out, (col, row)), shape=(n, 2 * m), dtype=np.complex128)
        B = sc.sparse.csr_matrix((data, (row, col)), shape=(2 * m, n), dtype=np.complex128)
        return BT, B

    perm, indices, indptr = bt_pattern
    BT = sc.sparse.csr_matrix(
        (data_out[perm], indices, indptr), shape=(n, 2 * m), dtype=np.complex128
    )
    perm, indices, indptr = b_pattern
    B = sc.sparse.csr_matrix((data[perm], indices, indptr), shape=(2 * m, n), dtype=np.complex128)
    return BT, B


def construct_weight_matrix(graph, with_k=True):
    """Construct the quantum matrix W^{-1}(k).

    The with_k argument is needed for the graph laplcian, not for computing the edge amplitudes.

    Args:
        graph (graph): quantum graph
        with_k (bool): multiplies or not the laplacian by k
    """
    data_tmp = 1.0 / (np.exp(2.0j * graph.graph["lengths"] * graph.graph["ks"]) - 1.0)
    # ``data_tmp`` is complex, and numpy's ``>`` on complex compares the real
    # part, so the old ``(data_tmp > 1e5)`` missed a blown-up entry unless it
    # happened to be large *and positive real* -- exactly two thirds of the
    # cases. This guard exists to catch an edge sitting on the pole
    # ``k_e l_e in pi*Z``, where the entry is large in modulus and of any phase.
    if (np.abs(data_tmp) > 1e5).any():
        L.info("Large values in Winv, it may not work!")
    if with_k:
        data_tmp *= graph.graph["ks"]

    # A diagonal matrix in CSC is trivially its own pattern; going through
    # ``sc.sparse.diags`` builds a DIA matrix and converts it on every call.
    diagonal = np.repeat(data_tmp, 2)
    size = diagonal.shape[0]
    return sc.sparse.csc_matrix(
        (diagonal, np.arange(size), np.arange(size + 1)),
        shape=(size, size),
        dtype=np.complex128,
    )


def set_inner_edges(graph, params=None, outer_edges=None):
    """Set the inner edges based on ``params['open_model']``.

    Writes an ``inner`` list into ``params`` and tags each edge with an
    ``inner`` boolean and an ``edgelabel`` integer. Callers are responsible
    for persisting ``params`` onto the graph via :func:`update_parameters`
    afterwards; this is the existing two-step pattern used by
    :func:`create_quantum_graph`.

    Args:
        graph (graph): quantum graph
        params (dict or NetSaltParams): must contain ``open_model`` as one of
            ``open``, ``closed``, ``custom``, ``directed``, ``directed_reversed``.
        outer_edges (list): if ``open_model == "custom"``, list of outer edges.
    """
    if params["open_model"] not in ["open", "closed", "custom", "directed", "directed_reversed"]:
        raise ValueError(f"open_model value not understood:{params['open_model']}")

    params["inner"] = []
    for ei, (u, v) in enumerate(graph.edges()):
        if params["open_model"] == "open" and (len(graph[u]) == 1 or len(graph[v]) == 1):
            graph[u][v]["inner"] = False
            params["inner"].append(False)
        elif params["open_model"] == "custom" and (u, v) in outer_edges:
            graph[u][v]["inner"] = False
            params["inner"].append(False)
        else:
            graph[u][v]["inner"] = True
            params["inner"].append(True)
        graph[u][v]["edgelabel"] = ei
    graph.graph["edgelabel"] = np.array([graph[u][v]["edgelabel"] for u, v in graph.edges])


def _set_node_positions(graph, positions=None):
    """Set the position to the networkx graph."""
    if positions is None:
        positions = nx.spring_layout(graph)
        L.warning("No node positions given, plots will have random positions from spring_layout")

    for i, u in enumerate(graph.nodes()):
        graph.nodes[u]["position"] = positions[i]


def _set_edge_lengths(graph, lengths=None):
    """Set lengths of edges."""
    for ei, e in enumerate(list(graph.edges())):
        (u, v) = e[:2]
        if lengths is None:
            graph[u][v]["length"] = np.linalg.norm(
                graph.nodes[u]["position"] - graph.nodes[v]["position"]
            )
        else:
            graph[u][v]["length"] = lengths[ei]

    graph.graph["lengths"] = np.array([graph[u][v]["length"] for u, v in graph.edges])


def laplacian_quality(laplacian, method="eigenvalue", rng=None):
    """Return the quality of a mode encoded in the quantum laplacian.

    If quality is low, the wavenumber of the laplacian is close to a solution of the quantum graph.

    Args:
        laplacian (sparse matrix): laplacian matrix
        method (str): method for quality evaluation. One of:

            * ``"eigenvalue"`` (default) — returns ``|λ₁|``, the magnitude of
              the smallest eigenvalue.
            * ``"complex_eigenvalue"`` — returns the signed complex ``λ₁``.
              Used by :func:`refine_mode_root` to drive
              ``(Re λ₁, Im λ₁) = 0``.
            * ``"singularvalue"`` — returns the smallest singular value.
            * ``"determinant"`` — returns a scaled determinant.
        rng: optional ``numpy.random.Generator`` used to draw the ARPACK
            starting vector. If None, a fresh generator with fresh entropy is
            created. Pass a seeded generator for reproducibility.
    """
    # Dense fast path for small matrices (see DENSE_EIG_MAX): the nearest-zero
    # eigenvalue is the smallest-magnitude one, identical to ``eigs(sigma=0)`` but
    # without ARPACK's per-call overhead. ``rng`` is irrelevant here (no ARPACK
    # start vector), keeping the result deterministic.
    if method in ("eigenvalue", "complex_eigenvalue") and laplacian.shape[0] <= DENSE_EIG_MAX:
        dense = laplacian.toarray()
        # a root-finder can probe a ``k`` whose operator overflows to inf/NaN;
        # ``np.linalg.eigvals`` raises there, so signal "not a mode" (quality 1)
        # exactly as the ARPACK branch does on non-convergence.
        if not np.isfinite(dense).all():
            return 1.0 if method == "eigenvalue" else 1.0 + 0j
        eigenvalues = np.linalg.eigvals(dense)
        lam = eigenvalues[np.argmin(np.abs(eigenvalues))]
        return abs(lam) if method == "eigenvalue" else complex(lam)

    if rng is None:
        rng = np.random.default_rng()
    v0 = rng.random(laplacian.shape[0])
    tol = 0.0  # ARPACK default (machine precision)

    def _eigs(return_vec=False):
        try:
            return sc.sparse.linalg.eigs(
                laplacian,
                k=1,
                sigma=0,
                return_eigenvectors=return_vec,
                which="LM",
                v0=v0,
                tol=tol,
            )
        except sc.sparse.linalg.ArpackNoConvergence:
            return None
        except RuntimeError:
            L.info("Runtime error, we add a small diagonal to laplacian, but things may be bad!")
            return sc.sparse.linalg.eigs(
                laplacian + 1e-6 * sc.sparse.eye(laplacian.shape[0]),
                k=1,
                sigma=0,
                return_eigenvectors=return_vec,
                which="LM",
                v0=v0,
                tol=tol,
            )

    if method == "eigenvalue":
        result = _eigs(return_vec=False)
        return 1.0 if result is None else abs(result)[0]

    if method == "complex_eigenvalue":
        result = _eigs(return_vec=False)
        return 1.0 + 0j if result is None else complex(result[0])

    if method == "determinant":
        logdet = np.linalg.slogdet(laplacian.todense())[1]
        return np.exp(np.real(logdet / laplacian.shape[0]))

    if method == "singularvalue":
        # ``svds(which="SM")`` converges very slowly on these matrices: measured
        # 109 ms versus 0.267 ms for a dense SVD of the same 61-node laplacian,
        # a 410x penalty. Take the dense route while the matrix is small enough
        # for it to be the cheaper option.
        if laplacian.shape[0] <= DENSE_SVD_MAX:
            return np.linalg.svd(laplacian.toarray(), compute_uv=False)[-1]
        return sc.sparse.linalg.svds(
            laplacian,
            k=1,
            which="SM",
            return_singular_vectors=False,
            v0=v0,
        )[0]
    return 1.0


def mode_quality(mode, graph, quality_method="eigenvalue", rng=None):
    """Quality of a mode, small means good quality, thus the mode is close to a correct mode.

    Args:
        mode (complex): complex mode
        graph (graph): quantum graph
        quality_method (str): method for quality evaluation (eig, singular value or det)
        rng: optional ``numpy.random.Generator`` threaded to
            :func:`laplacian_quality` for reproducible results.
    """
    laplacian = construct_laplacian(to_complex(mode), graph)
    return laplacian_quality(laplacian, method=quality_method, rng=rng)


class QuantumGraph(nx.Graph):
    """A :class:`networkx.Graph` carrying quantum-graph state, with method
    sugar over the module-level functions.

    This is a *thin, additive* layer requested in issue #28: it lets callers
    write ``qg.laplacian(k)`` instead of ``construct_laplacian(k, graph)``
    without threading a bare graph through every call. Because it subclasses
    ``nx.Graph``, all state still lives in ``graph.graph[...]`` and node / edge
    attributes, so JSON (``node_link_data``) serialisation, pickling to
    ``multiprocessing.Pool`` workers, and every existing procedural call site
    keep working unchanged — a ``QuantumGraph`` *is-a* ``nx.Graph``.

    Build instances with :meth:`from_networkx` (not ``__init__``): the inherited
    ``nx.Graph.__init__`` is what pickle, ``node_link_graph`` and ``.copy()``
    use to reconstruct, so it must stay a plain graph constructor.

    The methods cover the common workflow on a single object — set up physics
    (:meth:`set_dispersion_relation`, :meth:`set_dielectric_constant`), build
    matrices (:meth:`laplacian`, :meth:`weight_matrix`, :meth:`incidence_matrix`),
    evaluate quality (:meth:`mode_quality`), and run the scan/solve
    (:meth:`scan_frequencies`, :meth:`mode_on_nodes`). Each one delegates to the
    existing free function, so behaviour is identical; the class is ergonomic
    sugar only.
    """

    @classmethod
    def from_networkx(
        cls, graph, params=None, positions=None, lengths=None, seed=42, noise_level=0.001
    ):
        """Build a :class:`QuantumGraph` from a plain networkx graph.

        Wraps :func:`create_quantum_graph`; see it for argument semantics.
        """
        qg = cls(graph)  # nx.Graph copy-constructor copies structure + all attrs
        create_quantum_graph(
            qg,
            params=params,
            positions=positions,
            lengths=lengths,
            seed=seed,
            noise_level=noise_level,
        )
        return qg

    # --- state accessors (read graph.graph, like the free functions do) ---
    @property
    def params(self):
        """The :class:`~netsalt.params.NetSaltParams` stored on the graph."""
        return self.graph["params"]

    @property
    def total_length(self):
        return get_total_length(self)

    @property
    def total_inner_length(self):
        return get_total_inner_length(self)

    # --- physics setup (return self for chaining) ---
    def set_dispersion_relation(self, dispersion_relation):
        set_dispersion_relation(self, dispersion_relation)
        return self

    def set_dielectric_constant(self, custom_values=None, rng=None):
        set_dielectric_constant(self, self.params, custom_values=custom_values, rng=rng)
        return self

    # --- setters / mutators (return self for chaining) ---
    def update_parameters(self, params, force=False):
        update_parameters(self, params, force=force)
        return self

    def set_total_length(self, total_length=None, max_extent=None, inner=True, with_position=True):
        set_total_length(
            self,
            total_length=total_length,
            max_extent=max_extent,
            inner=inner,
            with_position=with_position,
        )
        return self

    def set_inner_edges(self, params=None, outer_edges=None):
        set_inner_edges(self, params if params is not None else self.params, outer_edges)
        return self

    def set_wavenumber(self, wavenumber):
        set_wavenumber(self, wavenumber)
        return self

    # --- matrix builders: delegate, reading state off self ---
    def laplacian(self, wavenumber):
        return construct_laplacian(wavenumber, self)

    def weight_matrix(self, with_k=True):
        return construct_weight_matrix(self, with_k=with_k)

    def incidence_matrix(self):
        return construct_incidence_matrix(self)

    def mode_quality(self, mode, quality_method="eigenvalue", rng=None):
        return mode_quality(mode, self, quality_method=quality_method, rng=rng)

    # --- mode search / solve (lazy imports: modes imports this module) ---
    def scan_frequencies(self, quality_method="eigenvalue"):
        """Scan the complex-frequency grid and return the quality matrix."""
        from .modes import scan_frequencies

        return scan_frequencies(self, quality_method=quality_method)

    def mode_on_nodes(self, mode):
        """Return the mode field evaluated on the graph nodes."""
        from .modes import mode_on_nodes

        return mode_on_nodes(mode, self)

    # --- structural ops return a NEW graph (the free functions already return
    # a subclass-preserving copy when called on a QuantumGraph) ---
    def with_pump(self, D0):
        """Return a copy of this graph whose ``params`` carry pump ``D0``.

        Wraps :func:`graph_with_pump`; the original graph is left untouched.
        """
        return graph_with_pump(self, D0)

    def oversample(self, edge_size):
        return oversample_graph(self, edge_size)

    def simplify(self):
        return simplify_graph(self)

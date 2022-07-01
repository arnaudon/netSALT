"""Quantum graph construction module.

A quantum graph is a networkx graph with additional parameters in graph.graph['param']
and specific node/edges attributes.
"""
import logging

import networkx as nx
import numpy as np
import scipy as sc

from .physics import update_params_dielectric_constant
from .utils import to_complex

L = logging.getLogger(__name__)


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
    _verify_lengths(graph, seed=seed, noise_level=noise_level)
    if params is None:
        params = graph.graph["params"]
    set_inner_edges(graph, params)
    update_parameters(graph, params)


def _verify_lengths(graph, seed=42, noise_level=0.001):
    """Add noise to lengths if many edges have equal."""
    if noise_level > 0.0:
        lengths = [graph[u][v]["length"] for u, v in graph.edges]
        np.random.seed(seed)
        if np.max(np.unique(np.around(lengths, 5), return_counts=True)) > 0.2 * len(graph.edges):
            L.info(
                """You have more than 20% of edges of the same length,
                so we add some small noise for safety for the numerics."""
            )
            for u in graph:
                graph.nodes[u]["position"][0] += np.random.normal(0, noise_level * np.min(lengths))
            _set_edge_lengths(graph)


def _not_equal(data1, data2, force=False):
    """Check if datasets are the same."""
    if force:
        return True
    if isinstance(data1, np.ndarray):
        return all(data1 != data2)
    return data1 != data2


def update_parameters(graph, params, force=False):
    """Set the parameter dictionary to the graph.

    TODO: improve this implementation

    Args:
        graph (graph): quantum graph
        params (dict): specific parameters to setup the quantum graph (depends on use cases)
        force (bool): I forgot
    """
    warning_params = [
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
    ]
    if "params" not in graph.graph:
        graph.graph["params"] = params
    else:
        for param, value in params.items():
            if param not in graph.graph["params"]:
                graph.graph["params"][param] = value
            elif _not_equal(graph.graph["params"][param], value, force=force):
                if param in warning_params:
                    if force:
                        graph.graph["params"][param] = value
                    else:
                        pass
                else:
                    graph.graph["params"][param] = value


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
        raise Exception("only one of total_length or max_extent is allowed")
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


def oversample_graph(graph, edge_size):  # pylint: disable=too-many-locals
    """Oversample a graph by adding points on edges.

    Args:
        graph (graph): quantum graph
        edge_size (float):  edge size to sample the graph
    """
    _set_pump_on_graph(graph)
    oversampled_graph = graph.copy()
    for ei, (u, v) in enumerate(graph.edges):
        last_node = len(oversampled_graph)
        n_nodes = int(graph[u][v]["length"] / edge_size)
        if n_nodes > 1:
            dielectric_constant = graph[u][v]["dielectric_constant"]
            pump = graph[u][v]["pump"]
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
                    inner=True,
                    dielectric_constant=dielectric_constant,
                    pump=pump,
                    edgelabel=ei,
                )

            oversampled_graph.add_edge(
                last_node + node_index,
                v,
                inner=True,
                dielectric_constant=dielectric_constant,
                pump=pump,
                edgelabel=ei,
            )

    oversampled_graph = nx.convert_node_labels_to_integers(oversampled_graph)
    _set_edge_lengths(oversampled_graph)
    params = {"inner": [oversampled_graph[u][v]["inner"] for u, v in oversampled_graph.edges]}
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
    BT, Bout = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph)
    return BT.dot(Winv).dot(Bout)


def set_wavenumber(graph, wavenumber):
    """Set edge wavenumbers with dispersion relation defined in graph['dispersion_relation'].

    Args:
        wavenumber (complex): wavenumber
        graph (graph): quantum graph
    """
    graph.graph["ks"] = graph.graph["dispersion_relation"](wavenumber, params=graph.graph["params"])


def construct_incidence_matrix(graph):
    """Construct the quantum incidence matrix B(k).

    Args:
        graph (graph): quantum graph
    """
    row = np.repeat(np.arange(len(graph.edges) * 2), 2)
    col = np.repeat(graph.edges, 2, axis=0).flatten()
    expl = np.exp(1.0j * graph.graph["lengths"] * graph.graph["ks"])
    ones = np.ones(len(graph.edges))

    data = np.dstack([-ones, expl, expl, -ones])[0].flatten()

    deg_u = np.array([len(graph[e[0]]) for e in graph.edges])
    deg_v = np.array([len(graph[e[1]]) for e in graph.edges])

    data_out = data.copy()
    mask = np.logical_or(deg_u == 1, deg_v == 1)
    data_out[1::4][mask] = 0
    data_out[2::4][mask] = 0

    m = len(graph.edges)
    n = len(graph.nodes)
    BT = sc.sparse.csr_matrix((data, (col, row)), shape=(n, 2 * m), dtype=np.complex128)
    Bout = sc.sparse.csr_matrix((data_out, (row, col)), shape=(2 * m, n), dtype=np.complex128)
    return BT, Bout


def construct_weight_matrix(graph, with_k=True):
    """Construct the quantum matrix W^{-1}(k).

    The with_k argument is needed for the graph laplcian, not for computing the edge amplitudes.

    Args:
        graph (graph): quantum graph
        with_k (bool): multiplies or not the laplacian by k
    """
    mask = abs(graph.graph["ks"]) > 0
    data_tmp = np.zeros(len(graph.edges), dtype=np.complex128)
    data_tmp[mask] = 1.0 / (
        np.exp(2.0j * graph.graph["lengths"][mask] * graph.graph["ks"][mask]) - 1.0
    )
    if any(data_tmp > 1e5):
        L.info("Large values in Winv, it may not work!")
    if with_k:
        data_tmp[mask] *= graph.graph["ks"][mask]
    data_tmp[~mask] = -0.5 * graph.graph["lengths"][~mask]

    row = np.arange(len(graph.edges) * 2)
    data = np.repeat(data_tmp, 2)

    m = len(graph.edges)
    return sc.sparse.csc_matrix((data, (row, row)), shape=(2 * m, 2 * m), dtype=np.complex128)


def set_inner_edges(graph, params=None, outer_edges=None):
    """Set the inner edges to True, according to a given model in params['open_model'].

    WARNING: this modifies params, which has to be set to graph with update_parameters
    TODO: improve implementation along with update_parameters

    Args:
        graph (graph): quantum graph
        params (dict): has to contain 'open_model' of the form open, closed, custom
        outer_edges (list): if open_model == custom, pass the list of outer edges.
    """
    if params["open_model"] not in ["open", "closed", "custom"]:
        raise Exception(f"open_model value not understood:{params['open_model']}")

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
        Warning("No node positions given, plots will have random positions from spring_layout")

    for u in graph.nodes():
        graph.nodes[u]["position"] = positions[u]


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


def laplacian_quality(laplacian, method="eigenvalue"):
    """Return the quality of a mode encoded in the quantum laplacian.

    If quality is low, the wavenumber of the laplacian is close to a solution of the quantum graph.

    Args:
        laplacian (sparse matrix): laplacian matrix
        method (str): method for quality evaluation (eigenvalue, singular value or determinant)
    """
    v0 = np.random.random(laplacian.shape[0])
    if method == "eigenvalue":
        try:
            return abs(
                sc.sparse.linalg.eigs(
                    laplacian, k=1, sigma=0, return_eigenvectors=False, which="LM", v0=v0
                )
            )[0]
        except sc.sparse.linalg.ArpackNoConvergence:
            # If eigenvalue solver did not converge, set to 1.0,
            return 1.0
        except RuntimeError:
            L.info("Runtime error, we add a small diagonal to laplacian, but things may be bad!")
            return abs(
                sc.sparse.linalg.eigs(
                    laplacian + 1e-6 * sc.sparse.eye(laplacian.shape[0]),
                    k=1,
                    sigma=0,
                    return_eigenvectors=False,
                    which="LM",
                    v0=v0,
                )
            )[0]

    if method == "determinant":
        logdet = np.linalg.slogdet(laplacian.todense())[1]
        return np.exp(np.real(logdet / laplacian.shape[0]))

    if method == "singularvalue":
        return sc.sparse.linalg.svds(
            laplacian,
            k=1,
            which="SM",
            return_singular_vectors=False,
            v0=v0,
        )[0]
    return 1.0


def mode_quality(mode, graph, quality_method='eigenvalue'):
    """Quality of a mode, small means good quality, thus the mode is close to a correct mode.

    Args:
        mode (complex): complex mode
        graph (graph): quantum graph
        quality_method (str): method for quality evaluation (eig, singular value or det)
    """
    laplacian = construct_laplacian(to_complex(mode), graph)
    return laplacian_quality(laplacian, method=quality_method)

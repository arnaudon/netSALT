"""graph construction methods"""
import networkx as nx
import numpy as np
import scipy as sc

from .dispersion_relations import update_params_dielectric_constant


def create_naq_graph(graph, params, positions=None, lengths=None):
    """append a networkx graph with necessary attributes for being a NAQ graph"""
    set_node_positions(graph, positions)
    set_edge_lengths(graph, lengths=lengths)
    _verify_lengths(graph)
    set_inner_edges(graph, params)
    update_parameters(graph, params)


def _verify_lengths(graph):
    """Add noise to lenghts if many are equal."""
    lengths = [graph[u][v]["length"] for u, v in graph.edges]
    if np.max(np.unique(np.around(lengths, 5), return_counts=True)) > 0.2 * len(
        graph.edges
    ):
        print(
            """WARNING: you have more than 20% of edges of the same length,
               so we add some small noise for safety for the numerics."""
        )
        for u in graph:
            graph.nodes[u]["position"][0] += np.random.normal(0, 0.01 * np.min(lengths))
        set_edge_lengths(graph)


def _not_equal(data1, data2, force=False):
    """Check if datasets are the same."""
    if force:
        return True
    if isinstance(data1, np.ndarray):
        return all(data1 != data2)
    return data1 != data2


def update_parameters(graph, params, force=False):
    """Set the parameter dictionary to the graph."""
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
    ]
    if "params" not in graph.graph:
        graph.graph["params"] = params
    else:
        for param, value in params.items():
            if param not in graph.graph["params"]:
                # print("Adding new parameter:", param)
                graph.graph["params"][param] = value
            elif _not_equal(graph.graph["params"][param], value, force=force):
                if param in warning_params:
                    if force:
                        # print(
                        #    "WARNING: you have forced the update of parameter",
                        #    param,
                        #    "so things may not work anymore.",
                        # )
                        graph.graph["params"][param] = value
                    else:
                        pass
                        # print(
                        #    "You are trying to update parmeter:",
                        #    param,
                        #    "but this may break the pipeline, so we will not update it. Use argument force=True to update if you really want it.",
                        # )
                else:
                    # print("Parameter:", param, "is updated.")
                    graph.graph["params"][param] = value


def get_total_length(graph):
    """Get the total lenght of the graph."""
    return sum([graph[u][v]["length"] for u, v in graph.edges()])


def get_total_inner_length(graph):
    """Get the total lenght of the graph."""
    return sum(
        [graph[u][v]["length"] for u, v in graph.edges() if graph[u][v]["inner"]]
    )


def set_total_length(graph, total_length, inner=True, with_position=True):
    """Set the inner total lenghts of the graph to a given value."""
    if inner:
        original_total_lenght = get_total_inner_length(graph)
    else:
        original_total_lenght = get_total_length(graph)

    length_ratio = total_length / original_total_lenght
    for u, v in graph.edges:
        graph[u][v]["length"] *= length_ratio
    if with_position:
        for u in graph:
            graph.nodes[u]["position"] *= length_ratio

    graph.graph["lengths"] = np.array([graph[u][v]["length"] for u, v in graph.edges])


def _set_pump_on_graph(graph, params):
    """set the pump values on the graph from params"""
    if "pump" not in graph.graph["params"]:
        graph.graph["params"]["pump"] = np.zeros(len(graph.edges))
    for ei, e in enumerate(graph.edges):
        graph[e[0]][e[1]]["pump"] = graph.graph["params"]["pump"][ei]


def _set_pump_on_params(graph, params):
    """set the pump values on the graph from params"""
    params["pump"] = np.zeros(len(graph.edges))
    for ei, e in enumerate(graph.edges):
        params["pump"][ei] = graph[e[0]][e[1]]["pump"]


def oversample_graph(graph, params):
    """oversample a graph by adding points on edges"""

    _set_pump_on_graph(graph, params)
    oversampled_graph = graph.copy()
    for u, v in graph.edges:
        last_node = len(oversampled_graph)
        if graph[u][v]["inner"]:
            n_nodes = int(graph[u][v]["length"] / params["plot_edgesize"])
            if n_nodes > 1:
                dielectric_constant = graph[u][v]["dielectric_constant"]
                pump = graph[u][v]["pump"]
                oversampled_graph.remove_edge(u, v)

                for node_index in range(n_nodes - 1):
                    node_position_x = graph.nodes[u]["position"][0] + (
                        node_index + 1
                    ) / n_nodes * (
                        graph.nodes[v]["position"][0] - graph.nodes[u]["position"][0]
                    )
                    node_position_y = graph.nodes[u]["position"][1] + (
                        node_index + 1
                    ) / n_nodes * (
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
                    )

                oversampled_graph.add_edge(
                    last_node + node_index,
                    v,
                    inner=True,
                    dielectric_constant=dielectric_constant,
                    pump=pump,
                )

    oversampled_graph = nx.convert_node_labels_to_integers(oversampled_graph)
    set_edge_lengths(oversampled_graph)
    params['inner'] = [oversampled_graph[u][v]['inner'] for u, v in oversampled_graph.edges]
    update_params_dielectric_constant(oversampled_graph, params)
    _set_pump_on_params(oversampled_graph, params)
    update_parameters(oversampled_graph, params, force=True)
    return oversampled_graph


def construct_laplacian(freq, graph):
    """construct naq laplacian from a graph"""
    set_wavenumber(graph, freq)
    BT, Bout = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph)
    return BT.dot(Winv).dot(Bout)


def set_wavenumber(graph, freq):
    """set edge wavenumbers from frequency and dispersion relation"""
    graph.graph["ks"] = graph.graph["dispersion_relation"](freq)


def construct_incidence_matrix(graph):
    """Construct the incidence matrix B"""
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
    Bout = sc.sparse.csr_matrix(
        (data_out, (row, col)), shape=(2 * m, n), dtype=np.complex128
    )
    return BT, Bout


def construct_weight_matrix(graph, with_k=True):
    """Construct the matrix W^{-1}
    with_k: multiplies or not by k (needed for graph laplcian, not for edge flux)"""
    mask = abs(graph.graph["ks"]) > 0
    data_tmp = np.zeros(len(graph.edges), dtype=np.complex128)
    data_tmp[mask] = 1.0 / (
        np.exp(2.0j * graph.graph["lengths"][mask] * graph.graph["ks"][mask]) - 1.0
    )
    if any(data_tmp > 1e5):
        print("WARNING: large values in Winv, it may not work!")
    if with_k:
        data_tmp[mask] *= graph.graph["ks"][mask]
    data_tmp[~mask] = -0.5 * graph.graph["lengths"][~mask]

    row = np.arange(len(graph.edges) * 2)
    data = np.repeat(data_tmp, 2)

    m = len(graph.edges)
    return sc.sparse.csc_matrix(
        (data, (row, row)), shape=(2 * m, 2 * m), dtype=np.complex128
    )


def set_inner_edges(graph, params, outer_edges=None):
    """set the inner edges to True, according to a model"""
    if params["open_model"] not in ["open_ends", "closed", "custom"]:
        raise Exception(
            "open_model value not understood:{}".format(params["open_model"])
        )

    params["inner"] = []
    for u, v in graph.edges():
        if params["open_model"] == "open_ends" and (
            len(graph[u]) == 1 or len(graph[v]) == 1
        ):
            graph[u][v]["inner"] = False
            params["inner"].append(False)
        elif params["open_model"] == "custom" and (u, v) in outer_edges:
            graph[u][v]["inner"] = False
            params["inner"].append(False)
        else:
            graph[u][v]["inner"] = True
            params["inner"].append(True)


def set_node_positions(graph, positions=None):
    """set the position to the networkx graph"""
    if positions is None:
        positions = nx.spring_layout(graph)
        Warning(
            "No node positions given, plots will have random positions from spring_layout"
        )

    for u in graph.nodes():
        graph.nodes[u]["position"] = positions[u]


def set_edge_lengths(graph, lengths=None):
    """set lengths of edges"""
    for ei, e in enumerate(list(graph.edges())):
        (u, v) = e[:2]
        if lengths is None:
            graph[u][v]["length"] = np.linalg.norm(
                graph.nodes[u]["position"] - graph.nodes[v]["position"]
            )
        else:
            graph[u][v]["length"] = lengths[ei]

    graph.graph["lengths"] = np.array([graph[u][v]["length"] for u, v in graph.edges])

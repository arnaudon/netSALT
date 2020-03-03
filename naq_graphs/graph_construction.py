"""graph construction methods"""
import numpy as np
import scipy as sc
import networkx as nx


def create_naq_graph(graph, params, positions=None, lengths=None):
    """append a networkx graph with necessary attributes for being a NAQ graph"""
    set_node_positions(graph, positions)

    if lengths is None:
        set_edge_lengths(graph)
    else:
        set_edge_lengths(graph, lengths=lengths)

    set_inner_edges(graph, params)


def oversample_graph(graph, edgesize=1.0e-2):
    """oversample a graph by adding points on edges"""
    oversampled_graph = graph.copy()
    for u, v in graph.edges():
        last_node = len(oversampled_graph)
        if graph[u][v]["inner"]:

            n_nodes = int(graph[u][v]["length"] / edgesize)
            if n_nodes > 1:
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
                    oversampled_graph.add_edge(first, last, inner=True)

                oversampled_graph.add_edge(last_node + node_index, v, inner=True)

    set_edge_lengths(oversampled_graph)

    return nx.convert_node_labels_to_integers(oversampled_graph)


def construct_laplacian(freq, graph):
    """construct naq laplacian from a graph"""
    set_wavenumber(graph, freq)
    BT, Bout = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph)
    return BT.dot(Winv).dot(Bout)


def set_wavenumber(graph, freq):
    """set edge wavenumbers from frequency and dispersion relation"""
    for ei, e in enumerate(list(graph.edges())):
        graph[e[0]][e[1]]["k"] = graph.graph["dispersion_relation"](freq, ei)


def construct_incidence_matrix(graph):
    """Construct the incidence matrix B"""
    row = []
    col = []
    data_B = []
    data_Bout = []
    for ei, e in enumerate(list(graph.edges())):
        (u, v) = e[:2]

        expl = np.exp(1.0j * graph[u][v]["length"] * graph[u][v]["k"])

        row.append(2 * ei)
        col.append(u)
        data_B.append(-1)
        data_Bout.append(-1)

        row.append(2 * ei)
        col.append(v)
        data_B.append(expl)
        if len(graph[u]) == 1 or len(graph[v]) == 1:
            data_Bout.append(0)
        else:
            data_Bout.append(expl)

        row.append(2 * ei + 1)
        col.append(u)
        data_B.append(expl)
        if len(graph[u]) == 1 or len(graph[v]) == 1:
            data_Bout.append(0)
        else:
            data_Bout.append(expl)

        row.append(2 * ei + 1)
        col.append(v)
        data_B.append(-1)
        data_Bout.append(-1)

    m = len(graph.edges)
    n = len(graph.nodes)
    B = sc.sparse.coo_matrix((data_B, (row, col)), shape=(2 * m, n))
    Bout = sc.sparse.coo_matrix((data_Bout, (row, col)), shape=(2 * m, n))

    return B.T.asformat("csc"), Bout.asformat("csc")


def construct_weight_matrix(graph, with_k=True):
    """Construct the matrix W^{-1}
    with_k: multiplies or not by k (needed for graph laplcian, not for edge flux)"""
    row = []
    data = []
    for ei, e in enumerate(list(graph.edges())):
        (u, v) = e[:2]

        if abs(graph[u][v]["k"]) > 0.0:
            w = 1 / (np.exp(2.0j * graph[u][v]["length"] * graph[u][v]["k"]) - 1.0)
            if with_k:
                w *= graph[u][v]["k"]
        else:
            w = -0.5 * graph[u][v]["length"]

        row.append(2 * ei)
        row.append(2 * ei + 1)
        data.append(w)
        data.append(w)

    m = len(graph.edges)
    return sc.sparse.coo_matrix((data, (row, row)), shape=(2 * m, 2 * m)).asformat(
        "csc"
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

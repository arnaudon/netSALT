"""graph construction methods"""
import numpy as np
import scipy as sc

def construct_laplacian(freq, graph, dispersion_relation):
    """construct naq laplacian from a graph"""
    set_wavenumber(graph, freq, dispersion_relation)
    BT, Bout = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph)
    return BT.dot(Winv).dot(Bout)


def set_wavenumber(graph, freq, dispersion_relation):
    """set edge wavenumbers from frequency and dispersion relation"""
    for ei, e in enumerate(list(graph.edges())):
        graph[e[0]][e[1]]["k"] = dispersion_relation(freq, ei)


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


def construct_weight_matrix(graph):
    """Construct the matrix W^{-1}"""
    row = []
    data = []
    for ei, e in enumerate(list(graph.edges())):
        (u, v) = e[:2]

        if abs(graph[u][v]["k"]) > 0:
            w = graph[u][v]["k"] / (
                np.exp(2.0 * graph[u][v]["length"] * graph[u][v]["k"]) - 1.0
            )
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

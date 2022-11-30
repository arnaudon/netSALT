"""Module for transfer quantum graphs."""
import numpy as np
from netsalt.quantum_graph import (
    construct_weight_matrix,
    construct_incidence_matrix,
    construct_laplacian,
)
from scipy.sparse import linalg


def get_edge_transfer_matrix(k, graph, input_flow):
    """Compute edge transfer matrix from a given input flow."""
    L = construct_laplacian(k, graph)
    BT, B = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=False)
    K = np.append(graph.graph["ks"], graph.graph["ks"])
    _input_flow = BT.dot(K * input_flow)
    _r = linalg.spsolve(L, _input_flow)
    return Winv.dot(B).dot(_r)


def get_static_boundary_flow(graph, input_flow, k_frac=0.1):
    """Get static boundary flow for static simulations, edge to node."""

    # get the small wavenumber for the graph
    k = k_frac * np.mean(
        np.array(graph.graph["params"]["inner"], dtype=float)
        * graph.graph["lengths"]
        * graph.graph["params"]["c"]
    )

    flows = np.real(get_edge_transfer_matrix(k, graph, input_flow))

    e_deg = np.array([len(graph[v]) for u, v in graph.edges])
    output_ids = np.argwhere(e_deg == 1).flatten()
    n_ids = [list(graph.edges)[i][1] for i in output_ids]

    # normalise flow so it sums to 0
    out_flow = flows[2 * output_ids]
    out_sum = out_flow[out_flow > 0].sum()
    out_flow[out_flow < 0] -= out_sum + out_flow[out_flow < 0].sum()
    _out_flow = np.zeros(len(graph))
    _out_flow[n_ids] = -out_flow

    return _out_flow, k

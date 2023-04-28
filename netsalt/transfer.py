"""Module for transfer quantum graphs."""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from netsalt.quantum_graph import (
    construct_weight_matrix,
    construct_incidence_matrix,
    construct_laplacian,
)
from scipy.sparse import linalg


def get_node_transfer_matrix(k, graph, input_flow):
    """Compute edge transfer matrix from a given input flow."""
    L = construct_laplacian(k, graph)
    BT, B = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=False)
    K = np.append(graph.graph["ks"], graph.graph["ks"])
    _input_flow = BT.dot(K * input_flow)
    return linalg.spsolve(L, _input_flow)



def get_edge_transfer_matrix(k, graph, input_flow):
    """Compute edge transfer matrix from a given input flow."""
    _r = get_node_transfer_matrix(k, graph, input_flow)
    BT, B = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=False)
    return Winv.dot(B).dot(_r)


def mean_tranfer_mode_on_edges(k, graph, input_flow):
    edge_flux = get_edge_transfer_matrix(k, graph, input_flow)
    #return np.real(edge_flux[0::2])

    mean_edge_solution = np.zeros(len(graph.edges))
    for ei in range(len(graph.edges)):
        k = 1.0j * graph.graph["ks"][ei]
        length = graph.graph["lengths"][ei]
        z = np.zeros([2, 2], dtype=np.complex128)

        if abs(np.real(k)) > 0:  # in case we deal with closed graph, we have 0 / 0
            z[0, 0] = (np.exp(length * (k + np.conj(k))) - 1.0) / (length * (k + np.conj(k)))
        else:
            z[0, 0] = 1.0
        z[0, 1] = (np.exp(length * k) - np.exp(length * np.conj(k))) / (length * (k - np.conj(k)))
        z[1, 0] = z[1, 0]
        z[1, 1] = z[0, 0]
        mean_edge_solution[ei] = np.abs(
            edge_flux[2 * ei : 2 * ei + 2].T.dot(z.dot(np.conj(edge_flux[2 * ei : 2 * ei + 2])))
        )

    return mean_edge_solution


def plot_single_transfer_mode(
    graph, mode, input_flow, colorbar=True, ax=None, edge_vmin=None, edge_vmax=None, cmap="coolwarm"
):
    """Plot single mode on the graph."""
    ax = _plot_single_transfer_mode(
        graph,
        mode,
        input_flow,
        ax=ax,
        colorbar=colorbar,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        cmap=cmap,
    )
    ax.set_title("k = " + str(np.around(np.real(mode), 3) - 1j * np.around(np.imag(mode), 3)))


def _plot_single_transfer_mode(
    graph, mode, input_flow, ax=None, colorbar=True, edge_vmin=None, edge_vmax=None, cmap="coolwarm"
):
    positions = [graph.nodes[u]["position"] for u in graph]
    edge_solution = mean_tranfer_mode_on_edges(mode, graph, input_flow)

    if ax is None:
        plt.figure(figsize=(5, 4))  # 14,3
        ax = plt.gca()

    nx.draw(graph, pos=positions, node_size=0, width=0, ax=ax)

    cmap = plt.get_cmap(cmap)
    if edge_vmax is None:
        edge_vmax = max(abs(edge_solution))
    if edge_vmin is None:
        edge_vmin = -max(abs(edge_solution))
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        edge_color=edge_solution,
        width=2,
        edge_cmap=cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        ax=ax,
    )
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=edge_vmin, vmax=edge_vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=r"$|E|^2$ (a.u)", shrink=0.5)
    return ax


def get_static_boundary_flow(graph, input_flow, k_frac=0.01):
    """Get static boundary flow for static simulations, edge to node."""

    # get the small wavenumber for the graph
    k = k_frac * np.mean(
        np.array(graph.graph["params"]["inner"], dtype=float)
        * graph.graph["params"]["c"]
        / graph.graph["lengths"]
    )

    flows = np.real(get_edge_transfer_matrix(k, graph, input_flow))

    e_deg = np.array([min(len(graph[u]), len(graph[v])) for u, v in graph.edges])
    output_ids = np.argwhere(e_deg == 1).flatten()
    n_ids = [list(graph.edges)[i][1] for i in output_ids]

    # normalise flow so it sums to 0
    out_flow = flows[2 * output_ids]
    out_sum = out_flow[out_flow > 0].sum()
    out_flow[out_flow < 0] -= out_sum + out_flow[out_flow < 0].sum()
    _out_flow = np.zeros(len(graph))
    _out_flow[n_ids] = -out_flow

    return _out_flow, k

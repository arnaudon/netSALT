"""plotting function"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .utils import order_edges_by


def plot_scan(ks, alphas, qualities, modes=None):
    """plot the scan with the mode found"""
    plt.figure(figsize=(10, 5))

    plt.imshow(
        np.log(qualities.T),
        extent=(ks[0], ks[-1], alphas[0], alphas[-1]),
        aspect="auto",
        origin="lower",
        cmap=plt.get_cmap('Blues')
    )

    cbar = plt.colorbar()
    cbar.set_label("smallest singular value")

    plt.xlabel(r"$Real(k)$")
    plt.ylabel(r"$\alpha = -Im(k)$")

    if modes is not None:
        for mode in modes:
            plt.plot(mode[0], mode[1], "r+")

    plt.axis([ks[0], ks[-1], alphas[-1], alphas[0]])


def plot_naq_graph(graph, edge_colors=None, node_colors=None):
    """plot the graph"""
    positions = [graph.nodes[u]["position"] for u in graph]

    plt.figure(figsize=(5, 4))

    if node_colors is not None:
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            node_size=10,
            node_color=node_colors,
            vmin=0,
            vmax=np.max(node_colors),
            cmap=plt.get_cmap("plasma"),
        )
        nodes = plt.cm.ScalarMappable(
            norm=plt.cm.colors.Normalize(0, np.max(node_colors)),
            cmap=plt.get_cmap("plasma"),
        )

        plt.colorbar(nodes, label=r"node values")

    else:
        nx.draw_networkx_nodes(graph, pos=positions, node_size=10, node_color="k")

    nx.draw_networkx_edges(graph, pos=positions)

    if edge_colors is not None:
        for ei, e in enumerate(order_edges_by(graph, edge_colors)):
            nx.draw_networkx_edges(
                graph,
                pos=positions,
                edgelist=[e,],
                edge_color=[np.sort(edge_colors)[ei],],
                edge_cmap=plt.get_cmap("plasma"),
                width=5,
                alpha=0.7,
                edge_vmin=0,
                edge_vmax=np.max(edge_colors),
            )

        edges = plt.cm.ScalarMappable(
            norm=plt.cm.colors.Normalize(0, np.max(edge_colors)),
            cmap=plt.get_cmap("plasma"),
        )

        plt.colorbar(edges, label=r"edge values")

    out_nodes = []
    for e in graph.edges():
        if not graph[e[0]][e[1]]["inner"]:
            if len(graph[e[0]]) == 1:
                out_nodes.append(e[0])
            if len(graph[e[1]]) == 1:
                out_nodes.append(e[1])

    nx.draw_networkx_nodes(
        graph, nodelist=out_nodes, pos=positions, node_color="r", node_size=10
    )
    plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)


def plot_pump_traj(modes, new_modes, new_modes_approx=None):
    """plot pump trajectories"""

    plt.scatter(modes[:, 0], modes[:, 1], s=20, c="r")
    for i in range(len(modes)):
        plt.scatter(new_modes[:, i, 0], new_modes[:, i, 1], marker="o", s=10, c="b")
        plt.plot(new_modes[:, i, 0], new_modes[:, i, 1], c="b")
    if new_modes_approx is not None:
        for i in range(len(modes)):
            for j in range(len(new_modes_approx[:, i, 0])):
                plt.plot(
                    [new_modes[j, i, 0], new_modes_approx[j, i, 0]],
                    [new_modes[j, i, 1], new_modes_approx[j, i, 1]],
                    c="k",
                    lw=0.5,
                )
            plt.scatter(
                new_modes_approx[:, i, 0],
                new_modes_approx[:, i, 1],
                marker="+",
                s=10,
                c="k",
            )

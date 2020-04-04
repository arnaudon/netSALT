"""plotting function"""
from itertools import cycle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

from .modes import mean_mode_on_edges, mode_on_nodes
from .utils import get_scan_grid, lorentzian, order_edges_by


def plot_stem_spectra(graph, modes_df, pump_index):
    """Plot spectra with stem plots."""
    threshold_modes = modes_df["threshold_lasing_modes"]
    modal_amplitudes = modes_df["modal_intensities"].iloc[:, pump_index]

    plt.figure(figsize=(5, 2))
    markerline, stemlines, baseline = plt.stem(threshold_modes, modal_amplitudes, "-")
    markerline.set_markerfacecolor("white")
    plt.setp(baseline, "color", "grey", "linewidth", 1)
    plt.xlabel(r"$k$")
    plt.ylabel("Intensity (a.u.)")

    plt.twinx()
    ks = np.linspace(
        graph.graph["params"]["k_min"], graph.graph["params"]["k_max"], 1000
    )
    plt.plot(ks, lorentzian(ks, graph), "r--")
    plt.ylabel("Gain spectrum (a.u.)")


def plot_ll_curve(graph, modes_df):
    """Plot LL curves."""
    colors = cycle(["C{}".format(i) for i in range(10)])
    pump_intensities = modes_df["modal_intensities"].columns.values
    plt.figure(figsize=(5, 3))
    for i, intens in enumerate(modes_df["modal_intensities"].to_numpy()):
        if intens[-1] > 0:
            color = next(colors)
            plt.plot(pump_intensities, intens, label="$mode $" + str(i), c=color)
            plt.axvline(
                modes_df["lasing_thresholds"][i], c=color, ls="dotted", ymin=0, ymax=0.1
            )

    plt.legend()
    top = np.max(modes_df["modal_intensities"].to_numpy())
    plt.axis([pump_intensities[0], pump_intensities[-1], -0.02 * top, top])
    plt.xlabel(r"$D_0$")
    plt.ylabel(r"$I_\mu$")


def plot_scan(graph, qualities, modes_df=None, figsize=(10, 5)):
    """plot the scan with the mode found"""

    ks, alphas = get_scan_grid(graph)

    plt.figure(figsize=figsize)

    plt.imshow(
        np.log10(qualities.T),
        extent=(ks[0], ks[-1], alphas[0], alphas[-1]),
        aspect="auto",
        origin="lower",
        cmap=plt.get_cmap("Blues"),
    )

    cbar = plt.colorbar()
    cbar.set_label("smallest singular value")

    plt.xlabel(r"$Real(k)$")
    plt.ylabel(r"$\alpha = -Im(k)$")

    if modes_df is not None:
        modes = modes_df["passive"].to_numpy()
        plt.plot(np.real(modes), -np.imag(modes), "r+")

    plt.axis([ks[0], ks[-1], alphas[-1], alphas[0]])


def plot_naq_graph(graph, edge_colors=None, node_colors=None, node_size=1):
    """plot the graph"""
    positions = [graph.nodes[u]["position"] for u in graph]

    plt.figure(figsize=(5, 4))

    if node_colors is not None:
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            node_size=node_size,
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
        nx.draw_networkx_nodes(
            graph, pos=positions, node_size=node_size, node_color="k"
        )

    nx.draw_networkx_edges(graph, pos=positions)

    if edge_colors is not None:
        edge_colors = np.real(edge_colors)
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


def plot_pump_traj(modes_df):  # , new_modes, new_modes_approx=None):
    """plot pump trajectories"""
    passive_modes = modes_df["passive"].to_numpy()
    plt.scatter(np.real(passive_modes), -np.imag(passive_modes), s=20, c="r")

    pumped_modes = modes_df["mode_trajectories"].to_numpy()
    for pumped_mode in pumped_modes:
        plt.scatter(
            np.real(pumped_mode), -np.imag(pumped_mode), marker="o", s=10, c="b"
        )
        plt.plot(np.real(pumped_mode), -np.imag(pumped_mode), c="b")

    if "mode_trajectories_approx" in modes_df:
        pumped_modes_approx = modes_df["mode_trajectories_approx"].to_numpy()
        for pumped_mode_approx in pumped_modes_approx:
            plt.scatter(
                np.real(pumped_mode_approx),
                -np.imag(pumped_mode_approx),
                marker="+",
                s=10,
                c="k",
            )


def plot_modes(graph, modes_df, df_entry="passive", folder="modes"):
    """Plot modes on the graph"""
    # TODO: cleaning needed here
    positions = [graph.nodes[u]["position"] for u in graph]

    for i, mode in tqdm(enumerate(modes_df[df_entry]), total=len(modes_df)):
        if df_entry == "threshold_lasing_modes":
            graph.graph["params"]["D0"] = modes_df["lasing_thresholds"][i]

        node_solution = mode_on_nodes(mode, graph)
        edge_solution = mean_mode_on_edges(mode, graph)

        plt.figure(figsize=(6, 4))
        nodes = nx.draw_networkx_nodes(
            graph,
            pos=positions,
            node_color=abs(node_solution) ** 2,
            node_size=2,
            cmap=plt.get_cmap("Blues"),
        )
        plt.colorbar(nodes, label=r"$|E|^2$ (a.u)")
        edges_k = nx.draw_networkx_edges(
            graph,
            pos=positions,
            edge_color=edge_solution,
            width=2,
            edge_cmap=plt.get_cmap("Blues"),
        )
        plt.title(
            "k=" + str(np.around(np.real(mode), 3) - 1j * np.around(np.imag(mode), 3))
        )

        plt.savefig(folder + "/mode_" + str(i) + ".png")
        plt.close()

        if graph.graph["name"] == "line_PRA" or graph.graph["name"] == "line_semi":
            position_x = [graph.nodes[u]["position"][0] for u in graph]
            E_sorted = node_solution[np.argsort(position_x)]
            node_positions = np.sort(position_x - position_x[1])

            plt.figure()
            plt.plot(
                node_positions[1:-1], abs(E_sorted[1:-1]) ** 2
            )  # only plot over inner edges

            plt.title(
                "k="
                + str(np.around(np.real(mode), 3) - 1j * np.around(np.imag(mode), 3))
            )
            plt.savefig(folder + "/profile_mode_" + str(i) + ".svg")

            # naq.save_modes(node_positions, E_sorted, filename="modes/passivemode_" + str(i))

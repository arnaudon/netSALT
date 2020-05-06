"""plotting function"""
import os
from pathlib import Path
from itertools import cycle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import networkx as nx
import numpy as np
from tqdm import tqdm

from .modes import mean_mode_on_edges, mode_on_nodes
from .utils import get_scan_grid, lorentzian, order_edges_by, linewidth


def _savefig(graph, fig, folder, filename):
    """Save figures in subfolders and with different extensions."""
    if fig is not None:
        for ext in graph.graph["params"]["exts"]:
            folder_ext = Path(folder + "_" + ext.split(".")[-1])
            if not folder_ext.exists():
                os.mkdir(folder_ext)
            fig.savefig((folder_ext / filename).with_suffix(ext), bbox_inches="tight")


def plot_spectra(
    graph,
    modes_df,
    pump_index=-1,
    width=0.0005,
    ax=None,
    folder="plots",
    filename="spectra",
):
    """Plot spectra with linewidths."""
    threshold_modes = np.real(modes_df["threshold_lasing_modes"])
    modal_amplitudes = np.real(modes_df["modal_intensities"].iloc[:, pump_index])

    if ax == None:
        fig = plt.figure(figsize=(5, 2))
        ax = plt.gca()
    else:
        fig = None

    ks = np.linspace(
        graph.graph["params"]["k_min"], graph.graph["params"]["k_max"], 10000
    )
    spectra = np.zeros(len(ks))
    for mode, amplitude in zip(threshold_modes, modal_amplitudes):
        if amplitude > 0:
            spectra += amplitude * linewidth(ks, np.real(mode), width)

    ax.plot(ks, spectra)

    ax2 = ax.twinx()
    ks = np.linspace(
        graph.graph["params"]["k_min"], graph.graph["params"]["k_max"], 1000
    )
    ax2.plot(ks, lorentzian(ks, graph), "r--")
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel("Gain spectrum (a.u.)")

    _savefig(graph, fig, folder, filename)


def plot_stem_spectra(
    graph, modes_df, pump_index=-1, ax=None, folder="plots", filename="stem_spectra"
):
    """Plot spectra with stem plots."""
    threshold_modes = np.real(modes_df["threshold_lasing_modes"])
    modal_amplitudes = np.real(modes_df["modal_intensities"].iloc[:, pump_index])
    print(len(threshold_modes[modal_amplitudes > 0]), "lasing modes in spectrum")

    ks, alphas = get_scan_grid(graph)

    if ax == None:
        fig = plt.figure(figsize=(5, 2))
        ax = plt.gca()
    else:
        fig = None

    # markerline, stemlines, baseline = ax.stem(threshold_modes, modal_amplitudes, "-")
    markerline, stemlines, baseline = ax.stem(
        threshold_modes, modal_amplitudes, "-", linefmt="grey", markerfmt=" "
    )

    # colors = cycle(["C{}".format(i) for i in range(10)])
    markerline.set_markerfacecolor("white")
    plt.setp(baseline, "color", "grey", "linewidth", 1)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Intensity (a.u.)")

    ax.set_xlim(ks[0], ks[-1])
    ax.set_ylim(
        -0.05 * np.max(modal_amplitudes[~np.isnan(modal_amplitudes)]),
        np.max(modal_amplitudes[~np.isnan(modal_amplitudes)]) * 1.3,
    )

    ax2 = ax.twinx()
    ks = np.linspace(
        graph.graph["params"]["k_min"], graph.graph["params"]["k_max"], 1000
    )
    ax2.plot(ks, lorentzian(ks, graph), "r--")
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel("Gain spectrum (a.u.)")

    ax3 = ax.twiny()
    lams = 2 * np.pi / ks
    ax3.set_xlim(lams[0], lams[-1])

    _savefig(graph, fig, folder, filename)


def plot_ll_curve(
    graph,
    modes_df,
    with_legend=True,
    ax=None,
    with_colors=True,
    with_thresholds=True,
    folder="plots",
    filename="ll_curve",
):
    """Plot LL curves."""
    colors = cycle(["C{}".format(i) for i in range(10)])
    pump_intensities = modes_df["modal_intensities"].columns.values
    if ax == None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
    else:
        fig = None

    for index, mode in modes_df.iterrows():
        intens = np.real(mode["modal_intensities"].to_numpy())
        if with_colors:
            color = next(colors)
        else:
            color = "grey"
        if intens[-1] > 0:
            ax.plot(
                pump_intensities, intens, label="mode " + str(index), c=color, lw=0.5
            )
            if with_thresholds:
                ax.axvline(
                    modes_df["lasing_thresholds"][index],
                    c=color,
                    ls="dotted",
                    ymin=0,
                    ymax=0.2,
                )
    ax.axhline(0, lw=0.5, c="k")

    if with_legend:
        ax.legend()

    top = np.max(np.nan_to_num(modes_df["modal_intensities"].to_numpy()))
    ax.axis([0, pump_intensities[-1], -0.02 * top, top])
    ax.set_xlabel(r"$D_0$")
    ax.set_ylabel("Intensity (a.u)")

    _savefig(graph, fig, folder, filename)


def plot_scan(
    graph,
    qualities,
    modes_df=None,
    figsize=(10, 5),
    ax=None,
    with_trajectories=True,
    with_scatter=True,
    with_approx=True,
    folder="plots",
    filename="scan",
    relax_upper=False,
):
    """plot the scan with the mode found"""

    ks, alphas = get_scan_grid(graph)

    if ax == None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        fig = None

    im = ax.imshow(
        np.log10(qualities.T),
        extent=(ks[0], ks[-1], alphas[0], alphas[-1]),
        aspect="auto",
        origin="lower",
        cmap=plt.get_cmap("Blues_r"),
    )
    plt.axhline(0, c="k")
    cbaxes = inset_axes(ax, width="2%", height="40%", loc="lower center")
    cbar = plt.colorbar(im, cax=cbaxes)
    cbar.set_label(r"$log_{10}(quality)$", fontsize=8)

    ax.set_xlabel(r"$Real(k)$")
    ax.set_ylabel(r"$\alpha = -Im(k)$")

    if modes_df is not None:
        for index, modes in modes_df.iterrows():
            k = np.real(modes["passive"][0])
            alpha = -np.imag(modes["passive"][0])
            ax.scatter(k, alpha, marker="+", color="r")
            ax.annotate(index, (k, alpha))
        if "threshold_lasing_modes" in modes_df:
            ax.scatter(
                np.real(modes_df["threshold_lasing_modes"].to_numpy()),
                -np.imag(modes_df["threshold_lasing_modes"].to_numpy()),
                c="m",
            )

        if with_trajectories and "mode_trajectories" in modes_df:
            plot_pump_traj(
                modes_df, with_scatter=with_scatter, with_approx=with_approx, ax=ax
            )

    ax.axis([ks[0], ks[-1], alphas[-1], alphas[0]])

    if relax_upper:
        ax.set_ylim(
            graph.graph["params"]["alpha_max"],
            -np.max(np.imag(modes_df["mode_trajectories"].to_numpy())),
        )

    _savefig(graph, fig, folder, filename)
    return ax


def plot_naq_graph(
    graph,
    figsize=(5, 4),
    ax=None,
    edge_colors=None,
    node_colors=None,
    node_size=1,
    folder="plots",
    filename="original_graph",
):
    """plot the graph"""
    positions = [graph.nodes[u]["position"] for u in graph]

    if ax == None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        fig = None

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

    # nx.draw_networkx_edges(graph, pos=positions)

    if edge_colors is not None:
        edge_colors = np.real(edge_colors)
        for ei, e in enumerate(order_edges_by(graph, edge_colors)):
            nx.draw_networkx_edges(
                graph,
                pos=positions,
                edgelist=[e,],
                edge_color=[np.sort(edge_colors)[ei],],
                edge_cmap=plt.get_cmap("Accent_r"),  # plasma
                width=2,  # 5
                alpha=1,  # 0.7
                edge_vmin=0,
                edge_vmax=np.max(edge_colors),
            )

        edges = plt.cm.ScalarMappable(
            norm=plt.cm.colors.Normalize(0, np.max(edge_colors)),
            cmap=plt.get_cmap("Accent_r"),
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
        graph, nodelist=out_nodes, pos=positions, node_color="r", node_size=node_size
    )
    plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    _savefig(graph, fig, folder, filename)


def plot_pump_traj(modes_df, with_scatter=True, with_approx=True, ax=None):
    """plot pump trajectories"""
    if ax == None:
        ax = plt.gca()

    colors = cycle(["C{}".format(i) for i in range(10)])

    passive_modes = modes_df["passive"].to_numpy()

    pumped_modes = modes_df["mode_trajectories"].to_numpy()
    for pumped_mode in pumped_modes:
        if with_scatter:
            ax.scatter(
                np.real(pumped_mode), -np.imag(pumped_mode), marker="o", s=10, c="b"
            )
        ax.plot(np.real(pumped_mode), -np.imag(pumped_mode), c=next(colors))

    if "mode_trajectories_approx" in modes_df and with_approx:
        pumped_modes_approx = modes_df["mode_trajectories_approx"].to_numpy()
        for pumped_mode_approx in pumped_modes_approx:
            ax.scatter(
                np.real(pumped_mode_approx),
                -np.imag(pumped_mode_approx),
                marker="+",
                s=10,
                c="k",
            )


def plot_single_mode(
    graph, modes_df, index, df_entry="passive", colorbar=True, ax=None
):
    """Plot single mode on the graph."""
    positions = [graph.nodes[u]["position"] for u in graph]
    mode = modes_df[df_entry][index]

    if df_entry == "threshold_lasing_modes":
        graph.graph["params"]["D0"] = modes_df["lasing_thresholds"][index]

    node_solution = mode_on_nodes(mode, graph)
    edge_solution = mean_mode_on_edges(mode, graph)

    if ax == None:
        plt.figure(figsize=(5, 4))  # 14,3
        ax = plt.gca()

    nodes = nx.draw_networkx_nodes(
        graph,
        pos=positions,
        node_color=abs(node_solution) ** 2,
        node_size=0,
        cmap=plt.get_cmap("PuRd"),  # Blues
        ax=ax,
    )

    if colorbar:
        plt.colorbar(nodes, label=r"$|E|^2$ (a.u)")
    edges_k = nx.draw_networkx_edges(
        graph,
        pos=positions,
        edge_color=edge_solution,
        width=2,  # 5
        edge_cmap=plt.get_cmap("PuRd"),  # Blues
        ax=ax,
    )

    ax.set_title(
        "mode "
        + str(index)
        + ", k = "
        + str(np.around(np.real(mode), 3) - 1j * np.around(np.imag(mode), 3))
    )


def plot_modes(graph, modes_df, df_entry="passive", folder="modes", ext=".png"):
    """Plot modes on the graph."""

    for index, mode_data in tqdm(modes_df.iterrows(), total=len(modes_df)):
        plot_single_mode(graph, modes_df, index, df_entry)

        plt.savefig(folder + "/mode_" + str(index) + ext)
        plt.close()
        if graph.graph["name"] == "line_PRA" or graph.graph["name"] == "line_semi":
            plot_line_mode(graph, modes_df, index, df_entry)
            plt.savefig(folder + "/profile_mode_" + str(index) + ext)


def plot_line_mode(graph, modes_df, index, df_entry="passive"):
    """Plot single mode on the line."""

    mode = modes_df[df_entry][index]

    if df_entry == "threshold_lasing_modes":
        graph.graph["params"]["D0"] = modes_df["lasing_thresholds"][index]

    node_solution = mode_on_nodes(mode, graph)
    edge_solution = mean_mode_on_edges(mode, graph)

    position_x = [graph.nodes[u]["position"][0] for u in graph]
    E_sorted = node_solution[np.argsort(position_x)]
    node_positions = np.sort(position_x - position_x[1])

    plt.figure()
    plt.plot(node_positions[1:-1], abs(E_sorted[1:-1]) ** 2)

    plt.title(
        "mode "
        + str(index)
        + "k = "
        + str(np.around(np.real(mode), 3) - 1j * np.around(np.imag(mode), 3))
    )

"""plotting function"""
import logging
import os
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm

from .modes import mean_mode_on_edges, mode_on_nodes
from .utils import get_scan_grid, linewidth, lorentzian, order_edges_by

# pylint: disable=too-many-locals,too-many-arguments

L = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.INFO)


def _savefig(graph, fig, folder, filename):
    """Save figures in subfolders and with different extensions."""
    if fig is not None:
        for ext in graph.graph["params"]["exts"]:
            folder_ext = Path(folder + "_" + ext.split(".")[-1])
            if not folder_ext.exists():
                os.mkdir(folder_ext)
            fig.savefig((folder_ext / filename).with_suffix(ext), bbox_inches="tight")


def get_spectra(graph, modes_df, pump_index=-1, width=0.0005):
    """Compute spectra."""
    threshold_modes = np.real(modes_df["threshold_lasing_modes"])
    modal_amplitudes = np.real(modes_df["modal_intensities"].iloc[:, pump_index])
    ks = np.linspace(graph.graph["params"]["k_min"], graph.graph["params"]["k_max"], 10000)
    spectra = np.zeros(len(ks))
    for mode, amplitude in zip(threshold_modes, modal_amplitudes):
        if amplitude > 0:
            spectra += amplitude * linewidth(ks, np.real(mode), width)
    return ks, spectra


def plot_spectra(
    graph,
    modes_df,
    pump_index=-1,
    width=0.0005,
    ax=None,
    folder="plots",
    filename="spectra",
    save_option=False,
    with_lorenz=False,
    plot_kwargs=None,
):
    """Plot spectra with linewidths."""

    if ax is None:
        fig = plt.figure(figsize=(5, 2))
        ax = plt.gca()
    else:
        fig = None
    ks, spectra = get_spectra(graph, modes_df, pump_index=pump_index, width=width)
    ax.plot(ks, spectra, **plot_kwargs)

    if with_lorenz:
        ax2 = ax.twinx()
        ax2.plot(ks, lorentzian(ks, graph), "r--")
        ax2.set_xlabel(r"$\lambda$")
        ax2.set_ylabel("Gain spectrum (a.u.)")

    if save_option:
        _savefig(graph, fig, folder, filename)


def plot_stem_spectra(
    graph,
    modes_df,
    pump_index=-1,
    ax=None,
    folder="plots",
    filename="stem_spectra",
    save_option=False,
):
    """Plot spectra with stem plots."""
    threshold_modes = np.real(modes_df["threshold_lasing_modes"])
    modal_amplitudes = np.real(modes_df["modal_intensities"].iloc[:, pump_index])
    L.info("%s lasing modes in spectrum", len(threshold_modes[modal_amplitudes > 0]))

    ks, _ = get_scan_grid(graph)

    if ax is None:
        fig = plt.figure(figsize=(5, 3))
        ax = plt.gca()
    else:
        fig = None

    # add first and last point to have spectra spanning entire plot
    threshold_modes = np.insert(threshold_modes, 0, ks[0])
    modal_amplitudes = np.insert(modal_amplitudes, 0, 0)
    threshold_modes = np.append(threshold_modes, ks[-1])
    modal_amplitudes = np.append(modal_amplitudes, 0)
    markerline, _, baseline = ax.stem(
        threshold_modes, modal_amplitudes, "-", linefmt="grey", markerfmt=" "
    )

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
    ks = np.linspace(graph.graph["params"]["k_min"], graph.graph["params"]["k_max"], 1000)
    gain = lorentzian(ks, graph)
    ax2.plot(ks, gain, "r--")
    ax2.set_ylim(0, max(gain) * 1.1)
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel("Gain spectrum (a.u.)")

    ax3 = ax.twiny()
    lams = 2 * np.pi / ks
    ax3.set_xlim(lams[0], lams[-1])

    if save_option:
        _savefig(graph, fig, folder, filename)


def plot_ll_curve(
    graph,
    modes_df,
    with_legend=True,
    ax=None,
    with_colors=True,
    with_thresholds=False,
    folder="plots",
    filename="ll_curve",
    save_option=False,
):
    """Plot LL curves."""
    colors = cycle([f"C{i}" for i in range(10)])
    pump_intensities = modes_df["modal_intensities"].columns.values
    modes_df = modes_df.sort_values(
        by=[("modal_intensities", pump_intensities[-1])],
        axis=0,
        na_position="last",
        ascending=False,
    )
    if ax is None:
        fig = plt.figure(figsize=(5, 3))
        ax = plt.gca()
    else:
        fig = None

    for index, mode in modes_df.iterrows():
        intens = np.real(mode["modal_intensities"].to_numpy())
        if not any(~np.isnan(intens)):
            # do not plot non-lasing modes
            continue

        if with_colors:
            color = next(colors)
        else:
            color = "grey"
        ax.plot(pump_intensities, intens, label="mode " + str(index), c=color, lw=0.5)

        if with_thresholds:
            ax.axvline(
                modes_df["lasing_thresholds"][index],
                c=color,
                ls="dotted",
                ymin=0,
                ymax=0.2,
            )
    ax.axhline(0, lw=0.5, c="k", ls="--")

    if with_legend:
        ax.legend()

    top = np.max(np.nan_to_num(modes_df["modal_intensities"].to_numpy()))
    if pump_intensities[-1] < np.inf:
        ax.axis([0, pump_intensities[-1], -0.02 * top, top])
    ax.set_xlabel(r"$D_0$")
    ax.set_ylabel("Intensity (a.u)")

    if save_option:
        _savefig(graph, fig, folder, filename)


def plot_scan(
    graph,
    qualities,
    modes_df=None,
    figsize=None,
    ax=None,
    with_trajectories=True,
    with_scatter=True,
    with_approx=True,
    folder="plots",
    filename="scan",
    relax_upper=False,
    save_option=False,
):
    """plot the scan with the mode found"""
    ks, alphas = get_scan_grid(graph)

    if figsize is None:
        figsize = (len(ks) / len(alphas) * 3, 5)

    if ax is None:
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
            ax.annotate(index, (k, alpha), size="x-small")
        if "threshold_lasing_modes" in modes_df:
            ax.scatter(
                np.real(modes_df["threshold_lasing_modes"].to_numpy()),
                -np.imag(modes_df["threshold_lasing_modes"].to_numpy()),
                c="m",
            )

        if with_trajectories and "mode_trajectories" in modes_df:
            plot_pump_traj(modes_df, with_scatter=with_scatter, with_approx=with_approx, ax=ax)

    ax.axis([ks[0], ks[-1], alphas[-1], alphas[0]])

    if relax_upper:
        ax.set_ylim(
            graph.graph["params"]["alpha_max"],
            -np.max(np.imag(modes_df["mode_trajectories"].to_numpy())),
        )

    if save_option:
        _savefig(graph, fig, folder, filename)
    return ax


def plot_pump_profile(graph, pump, figsize=(5, 4), ax=None, node_size=1.0, c="0.8"):
    """Plot the pump profile on top of the graph structure."""
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    positions = [graph.nodes[u]["position"] for u in graph]
    pumped_edges = [e for e, _pump in zip(graph.edges, pump) if _pump > 0.0]
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        edgelist=pumped_edges,
        edge_color=c,
        width=10,
    )
    plot_quantum_graph(graph, ax=ax, node_size=node_size)


def plot_quantum_graph(
    graph,
    figsize=(5, 4),
    ax=None,
    edge_colors=None,
    node_colors=None,
    node_size=0.1,
    color_map="Accent_r",  # coolwarm plasma
    cbar_min=0,
    cbar_max=1,
    folder="plots",
    filename="original_graph",
    save_option=False,
):
    """plot the graph"""
    positions = [graph.nodes[u]["position"] for u in graph]

    if ax is None:
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
            cmap=plt.get_cmap(color_map),
        )
        nodes = plt.cm.ScalarMappable(
            norm=plt.cm.colors.Normalize(0, np.max(node_colors)),
            cmap=plt.get_cmap(color_map),
        )

        plt.colorbar(nodes, label=r"node values")

    else:
        nx.draw_networkx_nodes(graph, pos=positions, node_size=node_size, node_color="k")

    # nx.draw_networkx_edges(graph, pos=positions)
    # for edge labeling:
    # labels = nx.get_edge_attributes(graph,'edgelabel')
    # labels = dict([((u, v), i) for i, (u, v) in enumerate(graph.edges())])
    # nx.draw_networkx_edge_labels(graph, pos=positions, edge_labels=labels)

    if edge_colors is not None:
        edge_colors = np.real(edge_colors)
        for ei, e in enumerate(order_edges_by(graph, edge_colors)):
            nx.draw_networkx_edges(
                graph,
                pos=positions,
                edgelist=[e],
                edge_color=[np.sort(edge_colors)[ei]],
                edge_cmap=plt.get_cmap(color_map),
                width=2,  # 5
                alpha=1,  # 0.7
                edge_vmin=cbar_min,
                edge_vmax=cbar_max,
            )

        edges = plt.cm.ScalarMappable(
            norm=plt.cm.colors.Normalize(cbar_min, cbar_max),
            cmap=plt.get_cmap(color_map),
        )

        plt.colorbar(edges, label=r"edge values")
    else:
        nx.draw_networkx_edges(graph, pos=positions, width=2)

    out_nodes = []
    for e in graph.edges():
        if "inner" in graph[e[0]][e[1]] and not graph[e[0]][e[1]]["inner"]:
            if len(graph[e[0]]) == 1:
                out_nodes.append(e[0])
            if len(graph[e[1]]) == 1:
                out_nodes.append(e[1])

    nx.draw_networkx_nodes(
        graph, nodelist=out_nodes, pos=positions, node_color="r", node_size=node_size
    )
    plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    if save_option:
        _savefig(graph, fig, folder, filename)


def plot_pump_traj(
    modes_df, with_scatter=True, with_approx=True, ax=None, d0s_max=None, s=1, c="d0"
):
    """plot pump trajectories"""
    if ax is None:
        ax = plt.gca()

    colors = cycle([f"C{i}" for i in range(10)])

    pumped_modes = modes_df["mode_trajectories"].to_numpy()
    for pumped_mode in pumped_modes:
        ax.plot(np.real(pumped_mode), -np.imag(pumped_mode), c=next(colors))
        if with_scatter:
            vmax = None
            if c == "d0":
                c = modes_df["mode_trajectories"].columns.to_list()
                if d0s_max is None:
                    vmax = c[max(np.argmin(abs(np.imag(pumped_modes)), axis=1)) + 1]
                else:
                    vmax = d0s_max
            ax.scatter(
                np.real(pumped_mode),
                -np.imag(pumped_mode),
                marker="o",
                s=s,
                c=c,
                vmax=vmax,
                vmin=0,
                zorder=10,
            )

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
    graph,
    modes_df,
    index,
    df_entry="passive",
    colorbar=True,
    ax=None,
    edge_vmin=None,
    edge_vmax=None,
    cmap='coolwarm'
):
    """Plot single mode on the graph."""
    mode = modes_df[df_entry][index]
    if df_entry == "threshold_lasing_modes":
        graph.graph["params"]["D0"] = modes_df["lasing_thresholds"][index]
    ax = _plot_single_mode(
        graph, mode, ax=ax, colorbar=colorbar, edge_vmin=edge_vmin, edge_vmax=edge_vmax, cmap=cmap
    )
    ax.set_title(
        "mode "
        + str(index)
        + ", k = "
        + str(np.around(np.real(mode), 3) - 1j * np.around(np.imag(mode), 3))
    )


def _plot_single_mode(graph, mode, ax=None, colorbar=True, edge_vmin=None, edge_vmax=None, cmap='coolwarm'):
    positions = [graph.nodes[u]["position"] for u in graph]
    edge_solution = mean_mode_on_edges(mode, graph)

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


def plot_modes(graph, modes_df, df_entry="passive", folder="modes", ext=".png"):
    """Plot modes on the graph."""
    for index in tqdm(modes_df.index, total=len(modes_df)):
        plot_single_mode(graph, modes_df, index, df_entry)

        plt.savefig(folder + "/mode_" + str(index) + ext)
        plt.close()
        if "name" in graph.graph:
            if graph.graph["name"] == "line_PRA" or graph.graph["name"] == "line_semi":
                plot_line_mode(graph, modes_df, index, df_entry)
                plt.savefig(folder + "/profile_mode_" + str(index) + ext)


def plot_mode_evolution(graph, modes_df, index, folder="mode_evolution", ext=".png"):
    """Plot mode evolution as a function of pump power."""
    modes = modes_df.loc[index, "mode_trajectories"]
    Path(folder).mkdir(parents=True, exist_ok=True)
    for i, d0 in enumerate(modes.index):
        graph.graph["params"]["D0"] = d0
        _plot_single_mode(graph, modes.loc[d0])
        plt.suptitle(f"D0={np.around(d0, 3)}")
        plt.savefig(f"{folder}/mode_{index}_D0_{i:03d}{ext}")
        plt.close()


def plot_line_mode(graph, modes_df, index, df_entry="passive", ax=None):
    """Plot single mode on the line."""
    if ax is None:
        plt.figure(figsize=(5, 4))
        ax = plt.gca()

    mode = modes_df[df_entry][index]

    if df_entry == "threshold_lasing_modes":
        graph.graph["params"]["D0"] = modes_df["lasing_thresholds"][index]

    node_solution = mode_on_nodes(mode, graph)

    position_x = [graph.nodes[u]["position"][0] for u in graph]
    E_sorted = node_solution[np.argsort(position_x)]
    node_positions = np.sort(position_x - position_x[1])
    maxE2 = max(abs(E_sorted[1:-1]) ** 2)

    ax.plot(node_positions[1:-1], abs(E_sorted[1:-1]) ** 2 / maxE2)

    ax.set_title(
        "mode "
        + str(index)
        + "k = "
        + str(np.around(np.real(mode), 3) - 1j * np.around(np.imag(mode), 3))
    )

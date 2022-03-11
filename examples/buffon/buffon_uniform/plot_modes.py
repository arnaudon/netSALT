from netsalt.io import load_modes, load_graph
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from netsalt.modes import mean_mode_on_edges


def plot_single_mode(graph, modes_df, index, df_entry="passive", c="r", ax=None, th=0.7, lw=10):
    """Plot single mode on the graph."""
    mode = modes_df[df_entry][index]
    if df_entry == "threshold_lasing_modes":
        graph.graph["params"]["D0"] = modes_df["lasing_thresholds"][index]
    ax = _plot_single_mode(graph, mode, ax=ax, c=c, th=th, lw=lw)
    ax.set_title(
        "mode "
        + str(index)
        + ", k = "
        + str(np.around(np.real(mode), 3) - 1j * np.around(np.imag(mode), 3))
    )


def _plot_single_mode(graph, mode, ax=None, c="r", th=0.7, lw=10):

    positions = [graph.nodes[u]["position"] for u in graph]

    edge_solution = mean_mode_on_edges(mode, graph)
    edgelist = [e for e, s in zip(graph.edges, edge_solution) if s > th * max(edge_solution)]
    print(len(edgelist))
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        edgelist=edgelist,
        edge_color=c,
        width=lw,
        ax=ax,
    )
    return ax


if __name__ == "__main__":
    graph = load_graph("out/quantum_graph.gpickle")
    df = load_modes("out/passive_modes.h5")

    plt.figure()
    ax = plt.gca()
    positions = [graph.nodes[u]["position"] for u in graph]
    nx.draw(graph, pos=positions, node_size=0, width=0, ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        edge_color="k",
        width=3,
        ax=ax,
    )

    plot_single_mode(graph, df, 1, ax=ax, c="r", th=0.5, lw=10)
    plot_single_mode(graph, df, 2, ax=ax, c="b", th=0.5, lw=10)
    plt.savefig("mode.pdf")

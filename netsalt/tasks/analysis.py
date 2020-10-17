"""Tasks for analysis of results."""
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

from netsalt.plotting import plot_quantum_graph, plot_scan
from netsalt.io import load_graph, load_qualities

from .passive import CreateQuantumGraph, ScanFrequencies
from .netsalt_task import NetSaltTask


class PlotQuantumGraph(NetSaltTask):
    """Plot a quantum graph and print some informations."""

    def requires(self):
        """"""
        return CreateQuantumGraph()

    def run(self):
        """"""
        qg = load_graph(self.input().path)

        print("graph properties:")
        deg = nx.degree_histogram(qg)
        print("degree distribution", deg)
        c = nx.cycle_basis(qg)
        print("length cycle basis", len(c))

        print("number of nodes", len(qg.nodes()))
        print("number of edges", len(qg.edges()))
        print("number of inner edges", sum(qg.graph["params"]["inner"]))

        lengths = [qg[u][v]["length"] for u, v in qg.edges if qg[u][v]["inner"]]
        print("min edge length", np.min(lengths))
        print("max edge length", np.max(lengths))
        print("mean edge length", np.mean(lengths))

        cmap = get_cmap("Pastel1_r")
        newcolors = cmap(np.take(np.linspace(0, 1, 9), [0, 4, 2, 3, 1, 8, 6, 7, 5]))
        newcmp = ListedColormap(newcolors)
        plot_quantum_graph(
            qg,
            edge_colors=qg.graph["params"]["dielectric_constant"],
            node_size=5,
            color_map=newcmp,
            cbar_min=1,
            cbar_max=np.max(np.abs(qg.graph["params"]["dielectric_constant"])),
        )

        plt.savefig(self.target_path, bbox_inches="tight")


class PlotScanFrequencies(NetSaltTask):
    """Plot scan frequencies."""

    def requires(self):
        """"""
        return {"graph": CreateQuantumGraph(), "qualities": ScanFrequencies()}

    def run(self):
        """"""
        qg = load_graph(self.input()["graph"].path)
        scan_class = ScanFrequencies()
        qg.graph["params"].update(
            {
                "n_workers": scan_class.n_workers,
                "k_n": scan_class.k_n,
                "k_min": scan_class.k_min,
                "k_max": scan_class.k_max,
                "alpha_n": scan_class.alpha_n,
                "alpha_min": scan_class.alpha_min,
                "alpha_max": scan_class.alpha_max,
            }
        )

        qualities = load_qualities(filename=self.input()["qualities"].path)
        plot_scan(qg, qualities, filename=self.target_path)
        plt.savefig(self.target_path, bbox_inches="tight")

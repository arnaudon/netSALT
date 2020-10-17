"""Tasks for analysis of results."""
from pathlib import Path
import numpy as np
import luigi
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

from netsalt.plotting import plot_quantum_graph, plot_scan, plot_modes
from netsalt.io import load_graph, load_qualities, load_modes

from .passive import CreateQuantumGraph, ScanFrequencies, FindPassiveModes
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
        qg = ScanFrequencies().get_graph(self.input()["graph"].path)
        qualities = load_qualities(filename=self.input()["qualities"].path)
        plot_scan(qg, qualities, filename=self.target_path)
        plt.savefig(self.target_path, bbox_inches="tight")


class PlotPassiveModes(NetSaltTask):
    """Plot passive modes."""

    ext = luigi.Parameter(default=".png")

    def requires(self):
        """"""
        return {"graph": CreateQuantumGraph(), "modes": FindPassiveModes()}

    def run(self):
        """"""
        qg = FindPassiveModes().get_graph(self.input()["graph"].path)
        modes_df = load_modes(self.input()["modes"].path)

        if not Path(self.target_path).exists():
            Path(self.target_path).mkdir()
        plot_modes(
            qg, modes_df, df_entry="passive", folder=self.target_path, ext=self.ext
        )

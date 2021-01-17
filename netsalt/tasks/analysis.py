"""Tasks for analysis of results."""
from pathlib import Path
import pickle
import yaml

import luigi
import networkx as nx
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from netsalt.io import load_graph, load_modes, load_qualities, load_mode_competition_matrix
from netsalt.quantum_graph import oversample_graph
from netsalt.plotting import (
    plot_ll_curve,
    plot_modes,
    plot_quantum_graph,
    plot_scan,
    plot_stem_spectra,
    plot_pump_profile,
)

from .lasing import (
    ComputeModalIntensities,
    ComputeModeTrajectories,
    CreatePumpProfile,
    FindThresholdModes,
    ComputeModeCompetitionMatrix,
)
from .netsalt_task import NetSaltTask
from .passive import CreateQuantumGraph, FindPassiveModes, ScanFrequencies
from .pump import OptimizePump

matplotlib.use("Agg")


class PlotQuantumGraph(NetSaltTask):
    """Plot a quantum graph and print some informations."""

    plot_path = luigi.Parameter(default="figures/quantum_graph.pdf")

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
            color_map=newcmp,
            cbar_min=1,
            cbar_max=np.max(np.abs(qg.graph["params"]["dielectric_constant"])),
        )

        plt.savefig(self.output().path, bbox_inches="tight")

    def output(self):
        """"""
        return luigi.LocalTarget(self.plot_path)


class PlotScan(NetSaltTask):
    """Plot scan frequencies."""

    plot_path = luigi.Parameter(default="figures/scan_frequencies.pdf")

    def requires(self):
        """"""
        return {"graph": CreateQuantumGraph(), "qualities": ScanFrequencies()}

    def run(self):
        """"""
        qg = ScanFrequencies().get_graph(self.input()["graph"].path)
        qualities = load_qualities(filename=self.input()["qualities"].path)
        plot_scan(qg, qualities, filename=self.output().path)
        plt.savefig(self.output().path, bbox_inches="tight")

    def output(self):
        """"""
        return luigi.LocalTarget(self.plot_path)


class PlotPassiveModes(NetSaltTask):
    """Plot passive modes.

    Args:
        ext (str): extansion for saving plots
        n_modes (int): number of modes to plot (ordered by Q-values)
    """

    ext = luigi.Parameter(default=".pdf")
    n_modes = luigi.IntParameter(default=10)
    edge_size = luigi.FloatParameter(default=1.0)
    plot_path = luigi.Parameter(default="figures/passive_modes")

    def requires(self):
        """"""
        return {"graph": CreateQuantumGraph(), "modes": FindPassiveModes()}

    def run(self):
        """"""
        qg = FindPassiveModes().get_graph(self.input()["graph"].path)
        qg.graph["params"]["plot_edgesize"] = self.edge_size
        qg = oversample_graph(qg, qg.graph["params"])
        modes_df = load_modes(self.input()["modes"].path).head(self.n_modes)

        if not Path(self.output().path).exists():
            Path(self.output().path).mkdir()
        plot_modes(qg, modes_df, df_entry="passive", folder=self.output().path, ext=self.ext)

    def output(self):
        """"""
        return luigi.LocalTarget(self.plot_path)


class PlotScanWithModes(NetSaltTask):
    """Plot scan frequencies with modes."""

    plot_path = luigi.Parameter(default="figures/scan_frequencies_with_modes.pdf")

    def requires(self):
        """"""
        return {
            "graph": CreateQuantumGraph(),
            "qualities": ScanFrequencies(),
            "modes": FindPassiveModes(),
        }

    def run(self):
        """"""
        qg = ScanFrequencies().get_graph(self.input()["graph"].path)
        qualities = load_qualities(filename=self.input()["qualities"].path)
        modes_df = load_modes(self.input()["modes"].path)
        plot_scan(qg, qualities, modes_df, filename=self.output().path)
        plt.savefig(self.output().path, bbox_inches="tight")

    def output(self):
        """"""
        return luigi.LocalTarget(self.plot_path)


class PlotScanWithModeTrajectories(NetSaltTask):
    """Plot mode trajectories."""

    lasing_modes_id = luigi.ListParameter()
    plot_path = luigi.Parameter(default="figures/mode_trajectories.pdf")

    def requires(self):
        """"""
        return {
            "graph": CreateQuantumGraph(),
            "qualities": ScanFrequencies(),
            "trajectories": ComputeModeTrajectories(lasing_modes_id=self.lasing_modes_id),
        }

    def run(self):
        """"""
        qg = ScanFrequencies().get_graph(self.input()["graph"].path)
        qualities = load_qualities(filename=self.input()["qualities"].path)
        modes_df = load_modes(self.input()["trajectories"].path)

        plot_scan(qg, qualities, modes_df, relax_upper=True)
        plt.savefig(self.output().path, bbox_inches="tight")

    def output(self):
        """"""
        return luigi.LocalTarget(self.add_lasing_modes_id(self.plot_path))


class PlotScanWithThresholdModes(NetSaltTask):
    """"Plot threshold lasing modes."""

    lasing_modes_id = luigi.ListParameter()
    plot_path = luigi.Parameter(default="figures/threshold_modes.pdf")

    def requires(self):
        """"""
        return {
            "graph": CreateQuantumGraph(),
            "qualities": ScanFrequencies(),
            "thresholds": FindThresholdModes(lasing_modes_id=self.lasing_modes_id),
        }

    def run(self):
        """"""
        qg = ComputeModeTrajectories(lasing_modes_id=self.lasing_modes_id).get_graph(
            self.input()["graph"].path
        )
        qualities = load_qualities(filename=self.input()["qualities"].path)
        modes_df = load_modes(self.input()["thresholds"].path)

        plot_scan(qg, qualities, modes_df, relax_upper=True)
        plt.savefig(self.output().path, bbox_inches="tight")

    def output(self):
        """"""
        return luigi.LocalTarget(self.add_lasing_modes_id(self.plot_path))


class PlotThresholdModes(NetSaltTask):
    """Plot threshold modes.

    Args:
        ext (str): extansion for saving plots
        n_modes (int): number of modes to plot (ordered by Q-values)
    """

    lasing_modes_id = luigi.ListParameter(default=[])
    ext = luigi.Parameter(default=".pdf")
    n_modes = luigi.IntParameter(default=10)
    mode_ids = luigi.ListParameter(default=[])
    edge_size = luigi.FloatParameter(default=1.0)
    plot_path = luigi.Parameter(default="figures/threshold_modes")

    def requires(self):
        """"""
        return {
            "graph": CreateQuantumGraph(),
            "modes": FindThresholdModes(lasing_modes_id=self.lasing_modes_id),
            "pump": CreatePumpProfile(lasing_modes_id=self.lasing_modes_id),
        }

    def run(self):
        """"""
        qg = self.get_graph_with_pump(self.input()["graph"].path)
        qg.graph["params"]["plot_edgesize"] = self.edge_size
        qg = oversample_graph(qg, qg.graph["params"])

        if self.mode_ids:
            modes_df = load_modes(self.input()["modes"].path).loc[list(self.mode_ids)]
        else:
            modes_df = load_modes(self.input()["modes"].path).head(self.n_modes)
        print(modes_df)
        if not Path(self.output().path).exists():
            Path(self.output().path).mkdir()
        pd.options.mode.use_inf_as_na = True
        modes_df = modes_df[~modes_df["lasing_thresholds"].isna()]
        plot_modes(
            qg,
            modes_df,
            df_entry="threshold_lasing_modes",
            folder=self.output().path,
            ext=self.ext,
        )

    def output(self):
        """"""
        return luigi.LocalTarget(self.add_lasing_modes_id(self.plot_path))


class PlotLLCurve(NetSaltTask):
    """Plot LL curves from modal intensities."""

    lasing_modes_id = luigi.ListParameter()
    plot_path = luigi.Parameter(default="figures/ll_curve.pdf")

    def requires(self):
        """"""
        return {
            "modes": ComputeModalIntensities(lasing_modes_id=self.lasing_modes_id),
            "graph": CreateQuantumGraph(),
        }

    def run(self):
        """"""
        qg = ComputeModeTrajectories(lasing_modes_id=self.lasing_modes_id).get_graph(
            self.input()["graph"].path
        )
        modes_df = load_modes(self.input()["modes"].path)
        plot_ll_curve(qg, modes_df, with_legend=True)
        plt.savefig(self.output().path, bbox_inches="tight")

    def output(self):
        """"""
        return luigi.LocalTarget(self.add_lasing_modes_id(self.plot_path))


class PlotStemSpectra(NetSaltTask):
    """Plot LL curves from modal intensities."""

    lasing_modes_id = luigi.ListParameter()
    plot_path = luigi.Parameter(default="figures/stem_spectra.pdf")

    def requires(self):
        """"""
        return {
            "modes": ComputeModalIntensities(lasing_modes_id=self.lasing_modes_id),
            "graph": CreateQuantumGraph(),
        }

    def run(self):
        """"""
        qg = ComputeModeTrajectories(lasing_modes_id=self.lasing_modes_id).get_graph(
            self.input()["graph"].path
        )
        modes_df = load_modes(self.input()["modes"].path)
        plot_stem_spectra(qg, modes_df)
        plt.savefig(self.output().path, bbox_inches="tight")

    def output(self):
        """"""
        return luigi.LocalTarget(self.add_lasing_modes_id(self.plot_path))


class PlotOptimizedPump(NetSaltTask):
    """Plot info about optimized pump."""

    lasing_modes_id = luigi.ListParameter()
    plot_path = luigi.Parameter(default="figures/optimized_pump.pdf")

    def requires(self):
        """"""
        return {
            "pump": OptimizePump(lasing_modes_id=self.lasing_modes_id),
            "graph": CreateQuantumGraph(),
        }

    def run(self):
        """"""
        results = pickle.load(open(self.input()["pump"].path, "rb"))
        qg = load_graph(self.input()["graph"].path)

        with PdfPages(self.output().path) as pdf:

            plot_pump_profile(qg, results["optimal_pump"])
            pdf.savefig(bbox_inches="tight")

            plt.figure()
            plt.hist(results["costs"], bins=20)
            pdf.savefig(bbox_inches="tight")

            plt.figure(figsize=(20, 5))
            for lasing_mode in results["lasing_modes_id"]:
                plt.plot(results["pump_overlapps"][lasing_mode])

            plt.twinx()
            plt.plot(results["optimal_pump"], "r+")
            plt.gca().set_ylim(0.5, 1.5)
            pdf.savefig(bbox_inches="tight")

    def output(self):
        """"""
        return luigi.LocalTarget(self.add_lasing_modes_id(self.plot_path))


class PlotModeCompetitionMatrix(NetSaltTask):
    """Plot the mode competition matrix."""

    lasing_modes_id = luigi.ListParameter()
    plot_path = luigi.Parameter(default="figures/mode_competition_matrix.pdf")

    def requires(self):
        """"""
        return ComputeModeCompetitionMatrix(lasing_modes_id=self.lasing_modes_id)

    def run(self):
        """"""
        competition_matrix = load_mode_competition_matrix(filename=self.input().path)
        plt.figure(figsize=0.5 * np.array(np.shape(competition_matrix)))
        sns.heatmap(competition_matrix, ax=plt.gca(), square=True)
        plt.savefig(self.output().path)

    def output(self):
        """"""
        return luigi.LocalTarget(self.add_lasing_modes_id(self.plot_path))


class PlotPumpProfile(NetSaltTask):
    """Plot the pump profile."""

    plot_path = luigi.Parameter(default="figures/pump_profile.pdf")

    def requires(self):
        """"""
        return {"graph": CreateQuantumGraph(), "pump": CreatePumpProfile()}

    def run(self):
        """"""
        qg = ScanFrequencies().get_graph(self.input()["graph"].path)
        pump = yaml.safe_load(self.input()["pump"].open())
        plot_pump_profile(qg, pump)
        plt.savefig(self.output().path, bbox_inches="tight")

    def output(self):
        """"""
        return luigi.LocalTarget(self.plot_path)

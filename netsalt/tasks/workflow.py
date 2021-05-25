"""Main tasks to run entire workflows."""
import pickle

import luigi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from netsalt.io import load_modes

from .analysis import (
    PlotLLCurve,
    PlotModeCompetitionMatrix,
    PlotOptimizedPump,
    PlotPassiveModes,
    PlotPumpProfile,
    PlotQuantumGraph,
    PlotScan,
    PlotScanWithModes,
    PlotScanWithModeTrajectories,
    PlotScanWithThresholdModes,
    PlotStemSpectra,
    PlotThresholdModes,
)
from .lasing import (
    ComputeModalIntensities,
    ComputeModeCompetitionMatrix,
    ComputeModeTrajectories,
    CreatePumpProfile,
    FindThresholdModes,
)
from .netsalt_task import NetSaltTask
from .passive import CreateQuantumGraph, FindPassiveModes, ScanFrequencies

matplotlib.use("Agg")


class ComputePassiveModes(luigi.WrapperTask):
    """Run a workflow to compute passive modes of a graph."""

    rerun = luigi.BoolParameter(default=False)

    def requires(self):
        """ """
        return [
            CreateQuantumGraph(rerun=self.rerun),
            PlotQuantumGraph(rerun=self.rerun),
            ScanFrequencies(rerun=self.rerun),
            PlotScan(rerun=self.rerun),
            FindPassiveModes(rerun=self.rerun),
            PlotPassiveModes(rerun=self.rerun),
            PlotScanWithModes(rerun=self.rerun),
        ]


class ComputeLasingModes(luigi.WrapperTask):
    """Run a workflow to compute passive modes of a graph."""

    rerun = luigi.BoolParameter(default=False)
    rerun_all = luigi.BoolParameter(default=False)

    def requires(self):
        """ """
        if self.rerun_all:
            self.rerun = True

        tasks = ComputePassiveModes(self.rerun_all).requires()
        lasing_modes_id = CreatePumpProfile().lasing_modes_id
        tasks += [
            CreatePumpProfile(rerun=self.rerun),
            PlotPumpProfile(rerun=self.rerun),
            ComputeModeTrajectories(lasing_modes_id=lasing_modes_id, rerun=self.rerun),
            PlotScanWithModeTrajectories(lasing_modes_id=lasing_modes_id, rerun=self.rerun),
            FindThresholdModes(lasing_modes_id=lasing_modes_id, rerun=self.rerun),
            PlotScanWithThresholdModes(lasing_modes_id=lasing_modes_id, rerun=self.rerun),
            PlotThresholdModes(lasing_modes_id=lasing_modes_id, rerun=self.rerun),
            ComputeModeCompetitionMatrix(lasing_modes_id=lasing_modes_id, rerun=self.rerun),
            PlotModeCompetitionMatrix(lasing_modes_id=lasing_modes_id, rerun=self.rerun),
            ComputeModalIntensities(lasing_modes_id=lasing_modes_id, rerun=self.rerun),
            PlotLLCurve(lasing_modes_id=lasing_modes_id, rerun=self.rerun),
            PlotStemSpectra(lasing_modes_id=lasing_modes_id, rerun=self.rerun),
        ]
        return tasks


def compute_controllability(spectra_matrix):
    """Compute the controllability matrix and value from spectra."""
    single_mode_matrix = spectra_matrix.copy()
    single_mode_matrix[np.isnan(single_mode_matrix)] = 0
    for i, _ in enumerate(single_mode_matrix):
        single_mode_matrix[i] /= np.sum(single_mode_matrix[i])
    controllability = np.trace(single_mode_matrix) / len(single_mode_matrix)
    return single_mode_matrix, controllability


class ComputeControllability(NetSaltTask):
    """Run pump optimisation on several modes to see which can be single lased."""

    n_top_modes = luigi.IntParameter(default=4)
    pump_path = luigi.Parameter(default="pumps")
    single_mode_matrix_path = luigi.Parameter(default="out/single_mode_matrix.pkl")

    def requires(self):
        """ """
        return [FindPassiveModes(), ComputePassiveModes().requires()]

    def run(self):
        """ """

        modes_df = load_modes(self.input()[0].path)
        lasing_modes_id = modes_df.head(self.n_top_modes).index

        spectra_matrix = []
        for mode_id in lasing_modes_id:
            yield CreatePumpProfile(lasing_modes_id=[mode_id])
            yield PlotOptimizedPump(lasing_modes_id=[mode_id])
            yield FindThresholdModes(lasing_modes_id=[mode_id])
            yield ComputeModeCompetitionMatrix(lasing_modes_id=[mode_id])
            intensities_task = yield ComputeModalIntensities(lasing_modes_id=[mode_id])
            yield PlotLLCurve(lasing_modes_id=[mode_id])
            int_df = load_modes(intensities_task.path)

            spectra = int_df[
                "modal_intensities",
                int_df["modal_intensities"].columns[-1],
            ].to_numpy()

            spectra_matrix.append(spectra)

        spectra_matrix = np.array(spectra_matrix)
        single_mode_matrix, controllability = compute_controllability(spectra_matrix)
        pickle.dump(
            {
                "spectra_matrix": spectra_matrix,
                "single_mode_matrix": single_mode_matrix,
                "controllability": controllability,
            },
            open(self.output().path, "wb"),
        )

    def output(self):
        """ """
        return luigi.LocalTarget(self.single_mode_matrix_path)


class PlotControllability(NetSaltTask):
    """Plot controllability matrix."""

    plot_path = luigi.Parameter(default="figures/single_mode_control.pdf")

    def requires(self):
        """ """
        return ComputeControllability()

    def run(self):
        """ """
        data = pickle.load(open(self.input().path, "rb"))

        print(f"Controllability = {data['controllability']}")

        plt.figure(figsize=(6, 5))
        sns.heatmap(data["single_mode_matrix"], annot=False, fmt=".1f", cmap="Reds")
        plt.suptitle(f"Controllability = {data['controllability']}")
        plt.ylabel("Mode ids to single lase")
        plt.xlabel("Modal ids")
        plt.savefig(self.output().path, bbox_inches="tight")

    def output(self):
        """ """
        return luigi.LocalTarget(self.plot_path)

"""Main tasks to run entire workflows."""
import numpy as np
import luigi
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from netsalt.io import load_modes

from .analysis import (
    PlotLLCurve,
    PlotPassiveModes,
    PlotQuantumGraph,
    PlotScan,
    PlotScanWithModes,
    PlotScanWithModeTrajectories,
    PlotScanWithThresholdModes,
    PlotStemSpectra,
    PlotThresholdModes,
    PlotOptimizedPump,
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
from .pump import OptimizePump

matplotlib.use("Agg")


class ComputePassiveModes(luigi.WrapperTask):
    """Run a workflow to compute passive modes of a graph."""

    def requires(self):
        """"""
        return [
            CreateQuantumGraph(),
            PlotQuantumGraph(),
            ScanFrequencies(),
            PlotScan(),
            FindPassiveModes(),
            PlotPassiveModes(),
            PlotScanWithModes(),
        ]


class ComputeLasingModes(luigi.WrapperTask):
    """Run a workflow to compute passive modes of a graph."""

    def requires(self):
        """"""
        tasks = ComputePassiveModes().requires()
        lasing_modes_id = CreatePumpProfile().lasing_modes_id
        tasks += [
            CreatePumpProfile(),
            ComputeModeTrajectories(lasing_modes_id=lasing_modes_id),
            PlotScanWithModeTrajectories(lasing_modes_id=lasing_modes_id),
            FindThresholdModes(lasing_modes_id=lasing_modes_id),
            PlotScanWithThresholdModes(lasing_modes_id=lasing_modes_id),
            PlotThresholdModes(lasing_modes_id=lasing_modes_id),
            ComputeModeCompetitionMatrix(lasing_modes_id=lasing_modes_id),
            ComputeModalIntensities(lasing_modes_id=lasing_modes_id),
            PlotLLCurve(lasing_modes_id=lasing_modes_id),
            PlotStemSpectra(lasing_modes_id=lasing_modes_id),
        ]
        return tasks


class ComputeLasingModesWithPumpOptimization(luigi.WrapperTask):
    """Run a workflow to compute passive modes of a graph."""

    def requires(self):
        """"""
        tasks = ComputePassiveModes().requires()
        lasing_modes_id = CreatePumpProfile().lasing_modes_id
        tasks += [
            OptimizePump(lasing_modes_id=lasing_modes_id),
            PlotOptimizedPump(lasing_modes_id=lasing_modes_id),
            CreatePumpProfile(),
            ComputeModeTrajectories(lasing_modes_id=lasing_modes_id),
            PlotScanWithModeTrajectories(lasing_modes_id=lasing_modes_id),
            FindThresholdModes(lasing_modes_id=lasing_modes_id),
            PlotScanWithThresholdModes(lasing_modes_id=lasing_modes_id),
            PlotThresholdModes(lasing_modes_id=lasing_modes_id),
            ComputeModeCompetitionMatrix(lasing_modes_id=lasing_modes_id),
            ComputeModalIntensities(lasing_modes_id=lasing_modes_id),
            PlotLLCurve(lasing_modes_id=lasing_modes_id),
            PlotStemSpectra(lasing_modes_id=lasing_modes_id),
        ]
        return tasks


class ComputeControllability(NetSaltTask):
    """Run pump optimisation on several modes to see which can be single lased."""

    n_top_modes = luigi.IntParameter(default=4)
    pump_path = luigi.Parameter(default="pumps")
    plot_path = luigi.Parameter(default=" figures/single_mode_control.pdf")

    def requires(self):
        """"""
        return [FindPassiveModes(), ComputePassiveModes().requires()]

    def run(self):
        """"""

        modes_df = load_modes(self.input()[0].path)
        lasing_modes_id = modes_df.head(self.n_top_modes).index

        single_mode_matrix = []
        for mode_id in lasing_modes_id:
            yield CreatePumpProfile(lasing_modes_id=[mode_id])
            yield PlotOptimizedPump(lasing_modes_id=[mode_id])
            yield FindThresholdModes(lasing_modes_id=[mode_id])
            yield ComputeModeCompetitionMatrix(lasing_modes_id=[mode_id])
            intensities_task = yield ComputeModalIntensities(lasing_modes_id=[mode_id])
            yield PlotLLCurve(lasing_modes_id=[mode_id])
            int_df = load_modes(intensities_task.path)
            if (
                ComputeModalIntensities(lasing_modes_id=[mode_id]).D0_max
                in int_df["modal_intensities"].columns
            ):
                spectra = int_df[
                    "modal_intensities",
                    ComputeModalIntensities(lasing_modes_id=[mode_id]).D0_max,
                ].to_numpy()[: len(lasing_modes_id)]

                single_mode_matrix.append(spectra / np.sum(spectra[~np.isnan(spectra)]))
            # else:
            #    single_mode_matrix.append(np.zeros(len(lasing_modes_id)) * np.nan)

        plt.figure(figsize=(6, 5))
        sns.heatmap(single_mode_matrix, annot=False, fmt=".1f", cmap="Reds")
        plt.ylabel("Mode ids to single lase")
        plt.xlabel("Modal ids")
        plt.savefig(self.output().path, bbox_inches="tight")

        single_mode_matrix = np.array(single_mode_matrix)
        single_mode_matrix[np.isnan(single_mode_matrix)] = 0
        controlability = np.trace(single_mode_matrix) / len(single_mode_matrix)
        print(controlability)

    def output(self):
        """"""
        return luigi.LocalTarget(self.plot_path)

"""Main tasks to run entire workflows."""
import luigi
import matplotlib
import matplotlib.pyplot as plt

from netsalt.io import load_modes

from .analysis import (PlotLLCurve, PlotPassiveModes, PlotQuantumGraph,
                       PlotScan, PlotScanWithModes,
                       PlotScanWithModeTrajectories,
                       PlotScanWithThresholdModes, PlotStemSpectra,
                       PlotThresholdModes)
from .lasing import (ComputeModalIntensities, ComputeModeCompetitionMatrix,
                     ComputeModeTrajectories, CreatePumpProfile,
                     FindThresholdModes)
from .netsalt_task import NetSaltTask
from .passive import CreateQuantumGraph, FindPassiveModes, ScanFrequencies
from .pump import OptimizePump, PlotOptimizedPump

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
        tasks += [
            CreatePumpProfile(),
            ComputeModeTrajectories(),
            PlotScanWithModeTrajectories(),
            FindThresholdModes(),
            PlotScanWithThresholdModes(),
            PlotThresholdModes(),
            ComputeModeCompetitionMatrix(),
            ComputeModalIntensities(),
            PlotLLCurve(),
            PlotStemSpectra(),
        ]
        return tasks


class ComputeLasingModesWithPumpOptimization(luigi.WrapperTask):
    """Run a workflow to compute passive modes of a graph."""

    def requires(self):
        """"""
        tasks = ComputePassiveModes().requires()
        tasks += [
            OptimizePump(),
            PlotOptimizedPump(),
            CreatePumpProfile(),
            ComputeModeTrajectories(),
            PlotScanWithModeTrajectories(),
            FindThresholdModes(),
            PlotScanWithThresholdModes(),
            PlotThresholdModes(),
            ComputeModeCompetitionMatrix(),
            ComputeModalIntensities(),
            PlotLLCurve(),
            PlotStemSpectra(),
        ]
        return tasks


class ComputeControllability(NetSaltTask):
    """Run pump optimisation on several modes to see which can be single lased."""

    n_top_modes = luigi.IntParameter(default=4)
    pump_path = luigi.Parameter(default="pumps")

    def requires(self):
        """"""
        return FindPassiveModes()

    def run(self):
        """"""

        modes_df = load_modes(self.input().path)
        lasing_modes_id = modes_df.head(self.n_top_modes).index

        single_mode_matrix = []
        for mode_id in lasing_modes_id:
            yield CreatePumpProfile(lasing_modes_id=[mode_id])
            yield FindThresholdModes(lasing_modes_id=[mode_id])
            yield ComputeModeCompetitionMatrix(lasing_modes_id=[mode_id])
            intensities_task = yield ComputeModalIntensities(lasing_modes_id=[mode_id])
            int_df = load_modes(intensities_task.path)
            single_mode_matrix.append(
                int_df["modal_intensities", ComputeModalIntensities().D0_max].to_list()
            )

        plt.figure()
        plt.imshow(single_mode_matrix)
        plt.ylabel("Mode id to single lase")
        plt.xlabel("Modal intensities")
        plt.colorbar()
        plt.savefig(self.output().path, bbox_inches="tight")

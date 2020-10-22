"""Main tasks to run entire workflows."""
import luigi

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
)
from .lasing import (
    ComputeModalIntensities,
    ComputeModeCompetitionMatrix,
    ComputeModeTrajectories,
    CreatePumpProfile,
    FindThresholdModes,
)
from .passive import CreateQuantumGraph, FindPassiveModes, ScanFrequencies
from .pump import OptimizePump, PlotOptimizedPump


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

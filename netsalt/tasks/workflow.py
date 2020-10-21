"""Main tasks to run entire workflows."""
import luigi

from .passive import CreateQuantumGraph, ScanFrequencies, FindPassiveModes
from .lasing import (
    CreatePumpProfile,
    ComputeModeTrajectories,
    FindThresholdModes,
    ComputeModeCompetitionMatrix,
    ComputeModalIntensities,
)
from .analysis import (
    PlotQuantumGraph,
    PlotScan,
    PlotPassiveModes,
    PlotScanWithModes,
    PlotScanWithModeTrajectories,
    PlotScanWithThresholdModes,
    PlotThresholdModes,
    PlotLLCurve,
    PlotStemSpectra,
)


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

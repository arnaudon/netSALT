"""Main tasks to run entire workflows."""
import luigi

from .passive import CreateQuantumGraph, ScanFrequencies, FindPassiveModes
from .lasing import CreatePumpProfile, ComputeModeTrajectories
from .analysis import (
    PlotQuantumGraph,
    PlotScanFrequencies,
    PlotPassiveModes,
    PlotModeTrajectories,
)


class ComputePassiveModes(luigi.WrapperTask):
    """Run a workflow to compute passive modes of a graph."""

    def requires(self):
        """"""
        return [
            CreateQuantumGraph(),
            PlotQuantumGraph(),
            ScanFrequencies(),
            PlotScanFrequencies(),
            FindPassiveModes(),
            PlotPassiveModes(),
        ]


class ComputeLasingModes(luigi.WrapperTask):
    """Run a workflow to compute passive modes of a graph."""

    def requires(self):
        """"""
        tasks = ComputePassiveModes().requires()
        tasks += [
            CreatePumpProfile(),
            ComputeModeTrajectories(),
            PlotModeTrajectories(),
        ]
        return tasks

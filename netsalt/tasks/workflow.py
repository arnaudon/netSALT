"""Main tasks to run entire workflows."""
import luigi

from .passive import CreateQuantumGraph, ScanFrequencies, FindPassiveModes
from .analysis import PlotQuantumGraph, PlotScanFrequencies, PlotPassiveModes


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

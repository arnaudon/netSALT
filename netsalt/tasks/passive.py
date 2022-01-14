"""Tasks for passive modes."""
import luigi
import numpy as np
import yaml

from netsalt.io import load_graph, load_qualities, save_graph, save_modes, save_qualities
from netsalt.modes import find_modes, scan_frequencies
from netsalt.physics import (
    dispersion_relation_pump,
    set_dielectric_constant,
    set_dispersion_relation,
)
from netsalt.quantum_graph import (
    create_quantum_graph,
    oversample_graph,
    set_total_length,
    update_parameters,
    simplify_graph,
)

from .config import ModeSearchConfig
from .netsalt_task import NetSaltTask


class CreateQuantumGraph(NetSaltTask):
    """Create a quantum graph."""

    graph_path = luigi.Parameter(default="graph.gpickle")
    graph_mode = luigi.ChoiceParameter(default="open", choices=["open", "closed", "custom"])
    inner_total_length = luigi.FloatParameter(default=None)
    max_extent = luigi.FloatParameter(default=None)

    dielectric_mode = luigi.Parameter(default="refraction_params")
    method = luigi.Parameter(default="uniform")
    custom_index = luigi.Parameter(default="index.yaml")
    inner_value = luigi.FloatParameter(default=1.5)
    loss = luigi.FloatParameter(default=0.005)
    outer_value = luigi.FloatParameter(default=1.0)
    edge_size = luigi.FloatParameter(default=0.1)
    k_a = luigi.FloatParameter(default=15.0)
    gamma_perp = luigi.FloatParameter(default=3.0)

    noise_level = luigi.FloatParameter(default=0.001)

    quantum_graph_path = luigi.Parameter(default="out/quantum_graph.gpickle")

    def run(self):
        """ """
        params = {
            "open_model": self.graph_mode,
            self.dielectric_mode: {
                "method": self.method,
                "inner_value": self.inner_value,
                "loss": self.loss,
                "outer_value": self.outer_value,
            },
            "plot_edgesize": self.edge_size,
            "k_a": self.k_a,
            "gamma_perp": self.gamma_perp,
            "n_workers": ModeSearchConfig().n_workers,
        }

        quantum_graph = load_graph(self.graph_path)
        quantum_graph = simplify_graph(quantum_graph)
        positions = np.array([quantum_graph.nodes[u]["position"] for u in quantum_graph.nodes])
        create_quantum_graph(
            quantum_graph, params, positions=positions, noise_level=self.noise_level
        )

        set_total_length(
            quantum_graph, self.inner_total_length, max_extent=self.max_extent, inner=True
        )
        custom_index = None
        if self.method == "custom":
            with open(self.custom_index, "r") as yml:
                custom_index = yaml.safe_load(yml)

        set_dielectric_constant(quantum_graph, params, custom_values=custom_index)
        set_dispersion_relation(quantum_graph, dispersion_relation_pump)

        quantum_graph = oversample_graph(quantum_graph, params["plot_edgesize"])
        update_parameters(quantum_graph, params)
        save_graph(quantum_graph, self.output().path)

    def output(self):
        """ """
        return luigi.LocalTarget(self.quantum_graph_path)


class ScanFrequencies(NetSaltTask):
    """Scan frequencies to find passive modes."""

    qualities_path = luigi.Parameter(default="out/qualities.h5")

    def requires(self):
        """ """
        return CreateQuantumGraph()

    def run(self):
        """ """
        qg = self.get_graph(self.input().path)
        qualities = scan_frequencies(qg)
        save_qualities(qualities, filename=self.output().path)

    def output(self):
        """ """
        return luigi.LocalTarget(self.qualities_path)


class FindPassiveModes(NetSaltTask):
    """Find passive modes from quality scan."""

    passive_modes_path = luigi.Parameter(default="out/passive_modes.h5")

    def requires(self):
        """ """
        return {"graph": CreateQuantumGraph(), "qualities": ScanFrequencies()}

    def run(self):
        """ """
        qg = self.get_graph(self.input()["graph"].path)
        qualities = load_qualities(filename=self.input()["qualities"].path)
        modes_df = find_modes(qg, qualities)
        save_modes(modes_df, filename=self.output().path)

    def output(self):
        """ """
        return luigi.LocalTarget(self.passive_modes_path)

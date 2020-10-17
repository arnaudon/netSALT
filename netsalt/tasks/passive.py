"""Tasks for passive modes."""
import numpy as np
import luigi

from netsalt.io import load_graph, save_graph, save_qualities
from netsalt.quantum_graph import (
    create_quantum_graph,
    set_total_length,
    oversample_graph,
    update_parameters,
)
from netsalt.physics import (
    set_dielectric_constant,
    dispersion_relation_pump,
    set_dispersion_relation,
)
from netsalt.modes import scan_frequencies

from .netsalt_task import NetSaltTask


class CreateQuantumGraph(NetSaltTask):
    """Create a quantum graph."""

    graph_path = luigi.Parameter(default="graph.gpickle")
    graph_mode = luigi.ChoiceParameter(
        default="open", choices=["open", "closed", "custom"]
    )
    inner_total_length = luigi.FloatParameter(default=1.0)

    dielectric_mode = luigi.Parameter(default="refraction_params")
    method = luigi.Parameter(default="uniform")
    inner_value = luigi.FloatParameter(default=1.5)
    loss = luigi.FloatParameter(default=0.005)
    outer_value = luigi.FloatParameter(default=1.0)
    edge_size = luigi.FloatParameter(default=0.1)

    def run(self):
        """"""
        params = {
            "open_model": self.graph_mode,
            self.dielectric_mode: {
                "method": self.method,
                "inner_value": self.inner_value,
                "loss": self.loss,
                "outer_value": self.outer_value,
            },
            "plot_edgesize": self.edge_size,
        }
        quantum_graph = load_graph(self.graph_path)
        positions = np.array(
            [quantum_graph.nodes[u]["position"] for u in quantum_graph.nodes]
        )
        create_quantum_graph(quantum_graph, params, positions=positions)

        set_total_length(quantum_graph, self.inner_total_length, inner=True)
        set_dielectric_constant(quantum_graph, params)
        set_dispersion_relation(quantum_graph, dispersion_relation_pump, params)

        quantum_graph = oversample_graph(quantum_graph, params)
        update_parameters(quantum_graph, params)
        save_graph(quantum_graph, self.target_path)


class ScanFrequencies(NetSaltTask):
    """Scan frequencies to find passive modes."""

    n_workers = luigi.IntParameter(default=1)
    k_n = luigi.IntParameter(default=100)
    k_min = luigi.FloatParameter(default=10.0)
    k_max = luigi.FloatParameter(default=12.0)
    alpha_n = luigi.IntParameter(default=100)
    alpha_min = luigi.FloatParameter(default=0.0)
    alpha_max = luigi.FloatParameter(default=0.1)

    def requires(self):
        """"""
        return CreateQuantumGraph()

    def run(self):
        """"""
        qg = load_graph(self.input().path)
        qg.graph["params"].update(
            {
                "n_workers": self.n_workers,
                "k_n": self.k_n,
                "k_min": self.k_min,
                "k_max": self.k_max,
                "alpha_n": self.alpha_n,
                "alpha_min": self.alpha_min,
                "alpha_max": self.alpha_max,
            }
        )
        qualities = scan_frequencies(qg)
        save_qualities(qualities, filename=self.target_path)
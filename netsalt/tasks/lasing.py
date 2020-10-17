"""Tasks for lasing modes."""
import numpy as np
import yaml
import luigi

from netsalt.modes import pump_trajectories
from netsalt.io import load_modes, save_modes

from .passive import CreateQuantumGraph, FindPassiveModes
from .netsalt_task import NetSaltTask


class CreatePumpProfile(NetSaltTask):
    """Create a pump profile."""

    mode = luigi.ChoiceParameter(default="uniform", choices=["uniform"])

    def requires(self):
        """"""
        return {"graph": CreateQuantumGraph()}

    def run(self):
        """"""
        qg = FindPassiveModes().get_graph(self.input()["graph"].path)
        if self.mode == "uniform":
            pump = np.zeros(len(qg.edges()))
            for i, (u, v) in enumerate(qg.edges()):
                if qg[u][v]["inner"]:
                    pump[i] = 1
        yaml.dump(pump.tolist(), open(self.target_path, "w"))


class ComputeModeTrajectories(NetSaltTask):
    """Compute mode trajectories from passive modes."""

    D0_max = luigi.FloatParameter(default=0.05)
    D0_steps = luigi.IntParameter(default=10)
    k_a = luigi.FloatParameter(default=15.0)
    gamma_perp = luigi.FloatParameter(default=3.0)

    def requires(self):
        """"""
        return {
            "graph": CreateQuantumGraph(),
            "modes": FindPassiveModes(),
            "pump": CreatePumpProfile(),
        }

    def get_graph(self, graph_path):
        """To ensure we get all parameters."""
        qg = FindPassiveModes().get_graph(graph_path)
        qg.graph["params"].update(
            {
                "D0_max": self.D0_max,
                "D0_steps": self.D0_steps,
                "k_a": self.k_a,
                "gamma_perp": self.gamma_perp,
            }
        )
        return qg

    def run(self):
        """"""
        modes_df = load_modes(self.input()["modes"].path)
        qg = self.get_graph(self.input()["graph"].path)
        qg.graph["params"]["pump"] = np.array(
            yaml.full_load(self.input()["pump"].open())
        )

        modes_df = pump_trajectories(modes_df, qg, return_approx=True)
        save_modes(modes_df, filename=self.target_path)

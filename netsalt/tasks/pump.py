"""Tasks for lasing modes."""
import pickle

import luigi

from netsalt.io import load_modes
from netsalt.pump import optimize_pump_linear_programming
from netsalt.pump import optimize_pump_diff_evolution

from .netsalt_task import NetSaltTask
from .passive import CreateQuantumGraph, FindPassiveModes


class OptimizePump(NetSaltTask):
    """Optimise a pump profile to lase specific modes."""

    lasing_modes_id = luigi.ListParameter()
    optimisation_mode = luigi.ChoiceParameter(
        default="linear_programming", choices=["differential_evolution", "linear_programming"]
    )
    # params for differential evolution
    pump_min_frac = luigi.FloatParameter(default=1.0)
    maxiter = luigi.IntParameter(default=1000)
    popsize = luigi.IntParameter(default=5)
    seed = luigi.IntParameter(default=42)
    n_seeds = luigi.IntParameter(default=10)
    disp = luigi.BoolParameter(default=False)

    # params for linear programming
    eps_min = luigi.FloatParameter(default=5.0)
    eps_max = luigi.FloatParameter(default=10.0)
    eps_n = luigi.IntParameter(default=10)
    cost_diff_min = luigi.FloatParameter(default=1e-4)
    optimized_pump_path = luigi.Parameter(default="out/optimized_pump.pkl")

    def requires(self):
        """ """
        return {"graph": CreateQuantumGraph(), "modes": FindPassiveModes()}

    def run(self):
        """ """
        qg = self.get_graph(self.input()["graph"].path)
        modes_df = load_modes(self.input()["modes"].path)

        if self.optimisation_mode == "differential_evolution":
            optimal_pump, pump_overlapps, costs, final_cost = optimize_pump_diff_evolution(
                modes_df,
                qg,
                self.lasing_modes_id,
                pump_min_frac=self.pump_min_frac,
                maxiter=self.maxiter,
                popsize=self.popsize,
                seed=self.seed,
                n_seeds=self.n_seeds,
                disp=self.disp,
            )

        elif self.optimisation_mode == "linear_programming":
            optimal_pump, pump_overlapps, costs, final_cost = optimize_pump_linear_programming(
                modes_df,
                qg,
                self.lasing_modes_id,
                eps_min=self.eps_min,
                eps_max=self.eps_max,
                eps_n=self.eps_n,
                cost_diff_min=self.cost_diff_min,
            )
        else:
            raise ValueError(f"unknown optimisation mode {self.optimisation_mode}")

        results = {
            "optimal_pump": optimal_pump,
            "pump_overlapps": pump_overlapps,
            "costs": costs,
            "final_cost": final_cost,
            "lasing_modes_id": self.lasing_modes_id,
        }
        with open(self.output().path, "wb") as pkl:
            pickle.dump(results, pkl)

    def output(self):
        """ """
        return luigi.LocalTarget(self.add_lasing_modes_id(self.optimized_pump_path))

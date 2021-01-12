"""Tasks for lasing modes."""
import pickle

import luigi

from netsalt.io import load_modes
from netsalt.modes import optimize_pump

from .netsalt_task import NetSaltTask
from .passive import CreateQuantumGraph, FindPassiveModes


class OptimizePump(NetSaltTask):
    """Optimise a pump profile to lase specific modes."""

    lasing_modes_id = luigi.ListParameter()
    pump_min_frac = luigi.FloatParameter(default=0.1)
    maxiter = luigi.IntParameter(default=1000)
    popsize = luigi.IntParameter(default=5)
    seed = luigi.IntParameter(default=42)
    n_seeds = luigi.IntParameter(default=10)
    disp = luigi.BoolParameter(default=False)
    optimized_pump_path = luigi.Parameter(default="out/optimized_pump.pkl")

    def requires(self):
        """"""
        return {"graph": CreateQuantumGraph(), "modes": FindPassiveModes()}

    def run(self):
        """"""
        qg = self.get_graph(self.input()["graph"].path)
        modes_df = load_modes(self.input()["modes"].path)

        optimal_pump, pump_overlapps, costs, final_cost = optimize_pump(
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
        results = {
            "optimal_pump": optimal_pump,
            "pump_overlapps": pump_overlapps,
            "costs": costs,
            "final_cost": final_cost,
            "lasing_modes_id": self.lasing_modes_id,
        }
        pickle.dump(results, open(self.output().path, "wb"))

    def output(self):
        """"""
        return luigi.LocalTarget(self.optimized_pump_path)

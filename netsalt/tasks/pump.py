"""Tasks for lasing modes."""
import pickle

import luigi

from netsalt.io import load_modes
from netsalt.pump import optimize_pump

from .netsalt_task import NetSaltTask
from .passive import CreateQuantumGraph, FindPassiveModes


class OptimizePump(NetSaltTask):
    """Optimise a pump profile to lase specific modes."""

    lasing_modes_id = luigi.ListParameter()
    pump_min_frac = luigi.FloatParameter(default=1.0)
    maxiter = luigi.IntParameter(default=1000)
    popsize = luigi.IntParameter(default=5)
    seed = luigi.IntParameter(default=42)
    n_seeds = luigi.IntParameter(default=10)
    disp = luigi.BoolParameter(default=False)
    optimized_pump_path = luigi.Parameter(default="out/optimized_pump.pkl")
    use_modes = luigi.BoolParameter(default=False)

    def requires(self):
        """ """
        return {"graph": CreateQuantumGraph(), "modes": FindPassiveModes()}

    def run(self):
        """ """
        qg = self.get_graph(self.input()["graph"].path)
        modes_df = load_modes(self.input()["modes"].path)

        optimal_pump, pump_overlapps, costs, final_cost = optimize_pump(
            modes_df,
            qg,
            self.lasing_modes_id,
            #pump_min_frac=self.pump_min_frac,
            #maxiter=self.maxiter,
            #popsize=self.popsize,
            #seed=self.seed,
            #n_seeds=self.n_seeds,
            #disp=self.disp,
            #use_modes=self.use_modes,
        )
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

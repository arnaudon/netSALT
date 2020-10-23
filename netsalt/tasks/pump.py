"""Tasks for lasing modes."""
import pickle

import luigi
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from netsalt.io import load_graph, load_modes
from netsalt.modes import optimize_pump
from netsalt.plotting import plot_quantum_graph

from .netsalt_task import NetSaltTask
from .passive import CreateQuantumGraph, FindPassiveModes

matplotlib.use("Agg")


class OptimizePump(NetSaltTask):
    """Optimise a pump profile to lase specific modes."""

    lasing_modes_id = luigi.ListParameter(default=[0])
    pump_min_frac = luigi.FloatParameter(default=0.0)
    maxiter = luigi.IntParameter(default=100)
    popsize = luigi.IntParameter(default=5)
    seed = luigi.IntParameter(default=42)
    n_seeds = luigi.IntParameter(default=10)
    disp = luigi.BoolParameter(default=False)

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
            self.pump_min_frac,
            self.maxiter,
            self.popsize,
            self.seed,
            self.n_seeds,
            self.disp,
        )
        results = {
            "optimal_pump": optimal_pump,
            "pump_overlapps": pump_overlapps,
            "costs": costs,
            "final_cost": final_cost,
            "lasing_modes_id": self.lasing_modes_id,
        }
        pickle.dump(results, open(self.output().path, "wb"))


class PlotOptimizedPump(NetSaltTask):
    """Plot info about optimized pump."""

    lasing_modes_id = luigi.ListParameter(default=[0])

    def requires(self):
        """"""
        return {
            "pump": OptimizePump(lasing_modes_id=self.lasing_modes_id),
            "graph": CreateQuantumGraph(),
        }

    def run(self):
        """"""
        results = pickle.load(open(self.input()["pump"].path, "rb"))
        qg = load_graph(self.input()["graph"].path)

        with PdfPages(self.output().path) as pdf:
            plt.figure()
            plt.hist(results["costs"], bins=20)
            pdf.savefig(bbox_inches="tight")

            plt.figure(figsize=(20, 5))
            for lasing_mode in results["lasing_modes_id"]:
                plt.plot(results["pump_overlapps"][lasing_mode])

            plt.twinx()
            plt.plot(results["optimal_pump"], "r+")
            plt.gca().set_ylim(0.5, 1.5)
            pdf.savefig(bbox_inches="tight")

            plot_quantum_graph(
                qg,
                edge_colors=results["optimal_pump"],
                node_size=5,
                # color_map=newcmp,
                cbar_min=0,
                cbar_max=1,
            )

            pdf.savefig(bbox_inches="tight")

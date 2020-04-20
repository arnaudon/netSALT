import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import optimize

import naq_graphs as naq
from naq_graphs import plotting

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

os.chdir(graph_tpe)

graph = naq.load_graph()
modes_df = naq.load_modes()

def optimize_pump(
    modes_df,
    graph,
    lasing_modes_id,
    pump_min_size=0.5,
    maxiter=100,
    popsize=10,
    disp=True,
):
    """Optimise the pump for lasing a set of modes."""
    if "pump" not in graph.graph["params"]:
        graph.graph["params"]["pump"] = np.ones(len(graph.edges))

    pump_overlapps = np.empty([len(modes_df["passive"]), len(graph.edges)])
    for mode_id, mode in enumerate(modes_df["passive"]):
        pump_overlapps[mode_id] = (
            -naq.modes.q_value(mode)
            * naq.modes.compute_overlapping_single_edges(mode, graph)
            * np.imag(
                naq.dispersion_relations.gamma(
                    naq.utils.to_complex(mode), graph.graph["params"]
                )
            )
        )
        print(mode_id, mode, pump_overlapps[mode_id].sum())

    mode_mask = np.array(len(pump_overlapps) * [False])
    mode_mask[lasing_modes_id] = True
    pump_min_edge_number = int(
        pump_min_size * len(np.where(graph.graph["params"]["inner"])[0])
    )

    def cost(pump):
        """Cost function to minimize."""
        pump = np.round(pump, 0)
        if pump.sum() < pump_min_edge_number:
            return 1e10

        a = pump_overlapps[mode_mask].dot(pump)
        b = pump_overlapps[~mode_mask].dot(pump)
        return np.max(b) - np.min(a)

    bounds = len(graph.edges) * [(0, 1)]
    result = optimize.differential_evolution(
        cost, bounds, maxiter=maxiter, disp=disp, popsize=popsize
    )
    optimal_pump = np.round(result.x, 0)

    print("Final cost is:", cost(optimal_pump))
    if cost(optimal_pump) > 0:
        print("This pump may not provide single lasing!")
    return optimal_pump, pump_overlapps


lasing_modes_id = [2, 3]
optimal_pump, pump_overlapps = optimize_pump(
    modes_df, graph, lasing_modes_id=lasing_modes_id, pump_min_size=0.5
)
pickle.dump(optimal_pump, open("optimal_pump.pkl", "wb"))


plt.figure(figsize=(20, 5))
for lasing_mode in lasing_modes_id:
    plt.plot(pump_overlapps[lasing_mode])
for mode in range(len(pump_overlapps)):
    plt.plot(pump_overlapps[mode], lw=0.5, c="k")
plt.twinx()
plt.plot(optimal_pump, "+")
plt.gca().set_ylim(0.5, 1.5)
plt.savefig("pump.png")
plt.show()
import os
import pickle
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import optimize
import multiprocessing
from tqdm import tqdm

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

def cost(pump_min_edge_number, mode_mask, pump_overlapps, pump):
    """Cost function to minimize."""
    pump = np.round(pump, 0)
    if pump.sum() < pump_min_edge_number:
        return 1e10

    a = pump_overlapps[mode_mask][:, pump==1].sum(axis=1)
    b = pump_overlapps[~mode_mask][:, pump==1].sum(axis=1)
    return np.max(b) - np.min(a)

def overlap_matrix_element(graph, mode):
    return list(-naq.modes.q_value(mode) * naq.modes.compute_overlapping_single_edges(mode, graph) * np.imag(
                naq.dispersion_relations.gamma(
                    naq.utils.to_complex(mode), graph.graph["params"]
                )
            ))


def optimize_pump(
    modes_df,
    graph,
    lasing_modes_id,
    pump_min_size=0.5,
    maxiter=1000,
    popsize=10,
    disp=True,
):
    """Optimise the pump for lasing a set of modes."""
    if "pump" not in graph.graph["params"]:
        graph.graph["params"]["pump"] = np.ones(len(graph.edges))
    with multiprocessing.Pool(graph.graph['params']['n_workers']) as pool:
        overlapp_iter = pool.imap(partial(overlap_matrix_element, graph), modes_df["passive"])
        pump_overlapps = np.empty([len(modes_df["passive"]), len(graph.edges)])
        for mode_id, overlapp in tqdm(enumerate(overlapp_iter), total=len(pump_overlapps)):
            pump_overlapps[mode_id] = overlapp

    mode_mask = np.array(len(pump_overlapps) * [False])
    mode_mask[lasing_modes_id] = True
    pump_min_edge_number = int(
        pump_min_size * len(np.where(graph.graph["params"]["inner"])[0])
    )
    costf = partial(cost, pump_min_edge_number, mode_mask, pump_overlapps)
    bounds = len(graph.edges) * [(0, 1)]
    
    result = optimize.differential_evolution(
        costf, bounds, maxiter=maxiter, disp=disp, popsize=popsize, workers=graph.graph['params']['n_workers'] 
    )
    optimal_pump = np.round(result.x, 0)

    print("Final cost is:", costf(optimal_pump))
    if costf(optimal_pump) > 0:
        print("This pump may not provide single lasing!")
    return optimal_pump, pump_overlapps


lasing_modes_id = [2]

fig = plt.figure()
ax = plt.gca()
plotting.plot_single_mode(
    graph, modes_df, lasing_modes_id[0], df_entry="passive", colorbar=True, ax = ax
)

fig.savefig("mode_for_optimisation.png", bbox_inches="tight")
plt.show()

optimal_pump, pump_overlapps = optimize_pump(
    modes_df, graph, lasing_modes_id=lasing_modes_id, pump_min_size=0.5
)
pickle.dump(optimal_pump, open("optimal_pump.pkl", "wb"))


plt.figure(figsize=(20, 5))
for lasing_mode in lasing_modes_id:
    plt.plot(pump_overlapps[lasing_mode])
#for mode in range(len(pump_overlapps)):
#    plt.plot(pump_overlapps[mode], lw=0.5, c="k")
plt.twinx()
plt.plot(optimal_pump, "r+")
plt.gca().set_ylim(0.5, 1.5)
plt.savefig("pump.png")

D_invs = [];
for mode in range(len(pump_overlapps)):
    D_invs.append(pump_overlapps[mode])

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios':[4,1]})
fig.subplots_adjust(wspace=0, hspace=0)
im = ax0.imshow(D_invs, aspect='auto', cmap='plasma')
cbar = fig.colorbar(im, ax=ax0)
cbar.set_label('D_inv')
ax0.set(ylabel = r'$modes$')
ax0.set_yticks(np.arange(len(modes_df)))
ax0.set_ylim(len(modes_df)-0.5,-0.5)

impump = ax1.imshow(np.array([optimal_pump]*10),'gray')
cbar = fig.colorbar(impump, ax=ax1)
cbar.set_label('pump')
ax1.set(xlabel = r'$edges$')
ax1.set_yticks([])

plt.savefig('D_invs_matrix.png')
plt.show()

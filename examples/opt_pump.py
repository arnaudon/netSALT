import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

import netsalt
from netsalt import plotting

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

os.chdir(graph_tpe)

graph = netsalt.load_graph()
modes_df = netsalt.load_modes()

def plot_Dinvs(graph, pump_overlaps, folder="Dinvs", ext=".png"):
    """Plot Dinvs on the graph."""
    for mode_id in range(len(pump_overlapps)):
        plotting.plot_quantum_graph(
            graph,
            edge_colors=pump_overlapps[mode_id],
            node_size=0.1,
            color_map="viridis",
            cbar_min=np.min(pump_overlapps[mode_id]),
            cbar_max=np.max(pump_overlapps[mode_id]),
            save_option=False,
        )

        plt.savefig(folder + "/mode_" + str(mode_id) + ext)
        plt.close()


lasing_modes_id = [107] 

fig = plt.figure()
ax = plt.gca()
plotting.plot_single_mode(
    graph, modes_df, lasing_modes_id[0], df_entry="passive", colorbar=True, ax=ax
)
fig.savefig("mode_for_optimisation.png", bbox_inches="tight")
plt.show()

optimal_pump, pump_overlapps, costs = netsalt.optimize_pump(
    modes_df,
    graph,
    lasing_modes_id,
    pump_min_frac=0.6, #0.2
    maxiter=1000, #50
    popsize=10, #5
    seed=1,
    n_seeds=100, #10
    disp=True)

plt.figure()
plt.hist(costs, bins=20)
plt.savefig("opt_hist.png")
plt.show()

pickle.dump(optimal_pump, open("optimal_pump.pkl", "wb"))

plt.figure(figsize=(20, 5))
for lasing_mode in lasing_modes_id:
    plt.plot(pump_overlapps[lasing_mode])

plt.twinx()
plt.plot(optimal_pump, "r+")
plt.gca().set_ylim(0.5, 1.5)
plt.savefig("pump.png")

D_invs = []
for mode in range(len(pump_overlapps)):
    D_invs.append(pump_overlapps[mode])

fig, (ax0, ax1) = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [4, 1]})
im = ax0.imshow(D_invs, aspect="auto", cmap="plasma")
cbar = fig.colorbar(im, ax=ax0)
cbar.set_label("D_inv")
ax0.set(ylabel=r"$modes$")
ax0.set_yticks(np.arange(len(modes_df)))
ax0.set_ylim(len(modes_df) - 0.5, -0.5)

impump = ax1.imshow(np.array([optimal_pump] * 10), aspect="auto", cmap="gray")
cbar = fig.colorbar(impump, ax=ax1)
cbar.set_label("pump")
ax1.set(xlabel=r"$edges$")
ax1.set_yticks([])

plt.savefig("D_invs_matrix.png")
plt.show()

#if not os.path.isdir("Dinvs"):
#    os.mkdir("Dinvs")

#plot_Dinvs(graph, D_invs, folder="Dinvs")

import os
import pickle
import sys

import matplotlib as mp
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
tmp_list = list(range(0,300))
modes_df_sub = modes_df.iloc[tmp_list,:]
#modes_df_sub = modes_df
pump_overlapps = netsalt.modes.compute_pump_overlapping_matrix(graph, modes_df_sub)


def shiftedColorMap(cmap, midpoint=0.5, name='shiftedcmap'):
    '''
    Function to offset the center of a colormap 
    for data with a negative min and positive max.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      midpoint : The new center of the colormap =  
                 1 - vmax / (vmax + abs(vmin))
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(0, 1.0, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mp.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

# overlapping matrix
orig_cmap = mp.cm.coolwarm
cmap_min = np.min(pump_overlapps)
cmap_max = np.max(pump_overlapps)
mdpnt = 1 - cmap_max / (cmap_max + np.abs(cmap_min))
shifted_cmap = shiftedColorMap(orig_cmap, midpoint=mdpnt, name='shifted')

plt.figure()
plt.imshow(pump_overlapps, aspect="auto", cmap=shifted_cmap)
cbar = plt.colorbar()
cbar.set_label("D_inv")
ax = plt.gca()
ax.set(xlabel=r"$edges$")
ax.set(ylabel=r"$modes$")
ax.set_yticks(np.arange(len(modes_df_sub)))
ax.set_ylim(len(modes_df_sub) - 0.5, -0.5)
plt.savefig('overlapping_matrix.png')
plt.show()


def plot_Dinvs(graph, folder="Dinvs", ext=".png"):
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

# Dinvs on graph
#if not os.path.isdir("Dinvs"):
#    os.mkdir("Dinvs")
#plot_Dinvs(graph)
#plt.show()


#### lasing mode(s) to optimise ####
lasing_modes_id = [9] 


# mode for optimisation
fig = plt.figure()
ax = plt.gca()
plotting.plot_single_mode(
    graph, modes_df, lasing_modes_id[0], df_entry="passive", colorbar=True, ax=ax
)
fig.savefig("mode_for_optimisation.png", bbox_inches="tight")
plt.show()


optimal_pump, pump_overlapps, costs, final_cost = netsalt.optimize_pump(
    modes_df_sub,
    graph,
    lasing_modes_id,
    pump_min_frac=0., #0.5 0.2
    maxiter=200, #1000 50
    popsize=5, #10
    seed=1,
    n_seeds=100, #10
    disp=True)

pickle.dump(optimal_pump, open("optimal_pump.pkl", "wb"))
print('Final cost is:', final_cost)


# cost function histogram
plt.figure()
plt.hist(costs, bins=20)
plt.savefig("opt_hist.png")
plt.show()


# optimal pump
plt.figure(figsize=(20, 5))
for lasing_mode in lasing_modes_id:
    plt.plot(pump_overlapps[lasing_mode])

plt.twinx()
plt.plot(optimal_pump, "r+")
plt.gca().set_ylim(0.5, 1.5)
plt.savefig("pump.png")


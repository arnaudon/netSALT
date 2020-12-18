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

netsalt.modes.compute_IPRs(graph, modes_df)

netsalt.save_modes(modes_df)

passive_modes = modes_df["passive"]
modes_Qfactor = np.round(
    np.abs(
        np.real(passive_modes) / (2 * np.imag(passive_modes))
    )
)

#print('IPR = ', modes_df["IPR"])
#print('Qfactor = ', modes_Qfactor)

modes_gamma_q = netsalt.modes.compute_gamma_q_values(graph, modes_df)

### PLOT ###
plt.hist(
np.round(modes_df["IPR"], 1),
    bins = 50, 
    range = (0, 5)
    )
ax = plt.gca()
ax.set(xlabel=r"$IPR$")
ax.set(ylabel="number")
plt.savefig('IPR_histogram.png')
plt.show()

try:
    thresholds = modes_df["lasing_thresholds"]

    plt.figure()
    plt.scatter(
        modes_df["IPR"], #np.log10(modes_df["IPR"]),
        modes_Qfactor, #modes_gamma_q,
        c = thresholds,
        cmap = "viridis_r",
#        vmin = 0.,
        vmax = params["D0_max"],
        alpha = 0.5,
    )
    cbar = plt.colorbar()
    cbar.set_label("threshold")

    ax = plt.gca()
    #ax.set(xlabel=r"$log_{10}(IPR)$")
    ax.set(xlabel=r"$IPR$")
    #ax.set(ylabel=r"$Q$ factor x $\Gamma$")
    ax.set(ylabel=r"$Q$ factor")
    ax.axis([0, 5, 0, 200])

    plt.savefig('modes_IPR_thresholds.png')
    plt.show()
except:
    pass

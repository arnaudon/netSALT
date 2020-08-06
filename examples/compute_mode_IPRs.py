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


### PLOT ###
thresholds = modes_df["lasing_thresholds"]

plt.figure()
plt.scatter(
    np.log10(modes_df["IPR"]),
    modes_Qfactor,
    c = thresholds,
    cmap = "viridis_r",
#    vmin = 0.,
    vmax = params["D0_max"],
    alpha = 0.5,
)
cbar = plt.colorbar()
cbar.set_label("threshold")

ax = plt.gca()
ax.set(xlabel=r"$log_{10}(IPR)$")
ax.set(ylabel=r"$Q$ factor")

plt.savefig('modes_IPR_thresholds.png')
plt.show()


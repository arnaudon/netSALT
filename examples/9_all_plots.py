import os
import pickle as pickle
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import yaml

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
qualities = naq.load_qualities()


fig, axes = plt.subplots(nrows=2, ncols=1)
fig.subplots_adjust(wspace=0, hspace=0)
plotting.plot_stem_spectra(graph, modes_df, -1, ax=axes[0])
axes[0].set_xticks([])
plotting.plot_scan(graph, qualities, modes_df, ax=axes[1])
plotting.plot_pump_traj(modes_df, with_scatter=False, with_approx=False, ax=axes[1])

ll_axis = inset_axes(axes[0], width="30%", height="30%", borderpad=1, loc="upper left")
plotting.plot_ll_curve(graph, modes_df, with_legend=False, ax=ll_axis)
ll_axis.tick_params(axis="both", which="major", labelsize=5)
ll_axis.xaxis.label.set_size(8)
ll_axis.yaxis.label.set_size(8)

fig.savefig("final_plot.png", bbox_inches="tight")

plt.show()

"""compile figure with main results"""
import os as os
import pickle as pickle
import sys as sys

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import yaml as yaml
import pandas as pd

import naq_graphs as naq
from graph_generator import generate_graph, generate_index
from naq_graphs import plotting

# First load graph:
if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

os.chdir(graph_tpe)

graph = naq.load_graph()
#graph = naq.oversample_graph(graph, params)

custom_index = generate_index(graph_tpe, graph, params)
naq.set_dielectric_constant(graph, params, custom_values=custom_index)

# Load data:
modes_df = naq.load_modes()
lasing_mode_id = 8 #this has to be a single integer

#### PLOTTING ####
fig = plt.figure(figsize=(10, 8),constrained_layout=True)

gs = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
#ax0 = fig.add_subplot(1, 1, 1, aspect=1)
ax0 = fig.add_subplot(gs[0,0], aspect=1)
ax0.set_title('pump profile')
#plotting.plot_naq_graph(graph, edge_colors=params["dielectric_constant"], node_size=0.1, ax=ax0)
plotting.plot_naq_graph(graph, edge_colors=graph.graph['params']["pump"], node_size=0.5, ax=ax0)

ax1 = fig.add_subplot(gs[1,0], aspect=1)
ax1.set_title('mode profile')
plotting.plot_single_mode(graph, modes_df, lasing_mode_id, df_entry="passive", colorbar=True, ax=ax1)


ax2 = fig.add_subplot(gs[2,0], aspect=1)
ax2.set_title('pump profile')
plotting.plot_single_mode(graph, modes_df, lasing_mode_id, df_entry="threshold_lasing_modes", colorbar=True, ax=ax2)

ax3 = fig.add_subplot(gs[0,1:])
ax3.set_title('spectrum')
ax3.set_xlabel(r'$k (\mu m)$')
ax3.set_ylabel('Intensity')
D0s = modes_df["modal_intensities"].columns.values
D0_id = round(0.3*len(D0s)) #-1
plotting.plot_stem_spectra(graph, modes_df, D0_id, ax=ax3)

opt_mode_kth = np.real(modes_df["threshold_lasing_modes"].iloc[lasing_mode_id])
opt_modal_amplitude = np.real(modes_df["modal_intensities"].iloc[lasing_mode_id, D0_id])
ax3.stem(opt_mode_kth*np.ones(2), [0,opt_modal_amplitude], "C0-", markerfmt=" ")

ax4 = fig.add_subplot(gs[1:,1:])
ax4.set_title('LL')
ax4.set_xlabel(r'$D0$', fontsize=14)
ax4.set_ylabel('Intensity')
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10)

plotting.plot_ll_curve(
    graph, modes_df, with_legend=False, with_colors=False, with_thresholds=False, ax=ax4
)

opt_mode_df = np.real(modes_df["modal_intensities"].iloc[lasing_mode_id, :])
ax4.plot(D0s,opt_mode_df)

D0s = modes_df["modal_intensities"].columns.values
top = np.max(modes_df["modal_intensities"].to_numpy()[:, D0_id])
ax4.axis([D0s[0], D0s[ D0_id], -0.01, top ])

#ax4.tick_params(axis="both", which="major", labelsize=5)
#ax4.xaxis.label.set_size(8)
#ax4.yaxis.label.set_size(8)

fig.savefig("draft_figure.svg", bbox_inches = "tight")
plt.show()

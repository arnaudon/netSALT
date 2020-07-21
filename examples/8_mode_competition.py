import os
import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
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
netsalt.update_parameters(graph, params)

modes_df = netsalt.load_modes()

mode_competition_matrix = netsalt.load_mode_competition_matrix()

max_pump_intensity = 0.05 #0.015
modes_df = netsalt.compute_modal_intensities(
    modes_df, max_pump_intensity, mode_competition_matrix
)

netsalt.save_modes(modes_df)

plotting.plot_ll_curve(graph, modes_df, with_legend=False)

plotting.plot_stem_spectra(graph, modes_df, -1)

#plt.show()

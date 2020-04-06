import os
import pickle as pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

import naq_graphs as naq
from graph_generator import generate_graph
from naq_graphs import plotting

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

os.chdir(graph_tpe)

graph = naq.load_graph()
naq.update_parameters(graph, params)

modes_df = naq.load_modes()

modes_df = naq.find_threshold_lasing_modes(modes_df, graph)

naq.save_modes(modes_df)

qualities = naq.load_qualities()

ax = plotting.plot_scan(
    graph, qualities, modes_df, filename="scan_with_threshold_modes"
)
plt.show()

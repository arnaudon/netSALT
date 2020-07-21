import os
import pickle as pickle
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
netsalt.update_parameters(graph, params)

modes_df = netsalt.load_modes()
modes_df = modes_df.loc[[406, 413]]
print(modes_df)
modes_df = netsalt.find_threshold_lasing_modes(modes_df, graph)

netsalt.save_modes(modes_df)

qualities = netsalt.load_qualities()

ax = plotting.plot_scan(
    graph, qualities, modes_df, filename="scan_with_threshold_modes", relax_upper=True
)
#plt.show()

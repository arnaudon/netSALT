import os
import pickle as pickle
import sys

import matplotlib.pyplot as plt
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

qualities = netsalt.load_qualities()

modes_df = netsalt.find_modes(graph, qualities)

netsalt.save_modes(modes_df)

plotting.plot_scan(graph, qualities, modes_df, filename="scan_with_passive_modes")

#plt.show()

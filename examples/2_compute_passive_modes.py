import os
import pickle as pickle
import sys

import matplotlib.pyplot as plt
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
naq.update_parameters(graph, params)

qualities = naq.load_qualities()

modes_df = naq.find_modes(graph, qualities)

naq.save_modes(modes_df)

plotting.plot_scan(graph, qualities, modes_df) #, figsize=(30, 5))

plt.savefig("scan_with_modes.png")
plt.show()

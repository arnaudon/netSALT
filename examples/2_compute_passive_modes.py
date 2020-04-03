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

qualities = pickle.load(open("scan.pkl", "rb"))  # save it for later

modes = naq.find_modes(graph, qualities)

naq.save_modes(modes)

plotting.plot_scan(graph, qualities, modes, figsize=(30, 5))

plt.savefig("scan_with_modes.svg")
plt.show()

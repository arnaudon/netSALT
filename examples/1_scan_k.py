import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
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
naq.update_parameters(graph, params, force=True)

qualities = naq.scan_frequencies(graph)
pickle.dump(qualities, open("scan.pkl", "wb"))

plotting.plot_scan(graph, qualities)
plt.savefig("scan_nomodes.svg")
plt.show()

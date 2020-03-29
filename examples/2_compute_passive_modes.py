import os
import sys

import pickle as pickle
import yaml
import matplotlib.pyplot as plt

import naq_graphs as naq
from naq_graphs import plotting

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

os.chdir(graph_tpe)

graph, params = naq.load_graph()

ks, alphas, qualities = pickle.load(open("scan.pkl", "rb"))  # save it for later

modes = naq.find_modes(
    ks, alphas, qualities, graph, params, n_workers=params["n_workers"]
)
print("Found", len(modes), "mode(s)")

naq.save_modes(modes)

plotting.plot_scan(ks, alphas, qualities, modes)

plt.savefig("scan_with_modes.svg")
plt.show()

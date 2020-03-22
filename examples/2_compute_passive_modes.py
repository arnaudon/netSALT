import os
import sys

import pickle as pickle
import yaml
import matplotlib.pyplot as plt

from graph_generator import generate_graph

from naq_graphs import create_naq_graph, find_modes, save_modes
from naq_graphs.plotting import plot_scan
from naq_graphs.io import load_graph

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

os.chdir(graph_tpe)

graph, params = load_graph()

ks, alphas, qualities = pickle.load(open("scan.pkl", "rb"))  # save it for later

modes = find_modes(ks, alphas, qualities, graph, params, n_workers=params['n_workers'])
print("Found", len(modes), "mode(s)")

save_modes(modes)

plot_scan(ks, alphas, qualities, modes)

plt.savefig("scan_with_modes.svg")
plt.show()

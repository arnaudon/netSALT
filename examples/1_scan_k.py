import os
import sys

import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt

from graph_generator import generate_graph

from naq_graphs import create_naq_graph, scan_frequencies
from naq_graphs.plotting import plot_scan
from naq_graphs.io import load_graph

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

os.chdir(graph_tpe)

graph, params = load_graph()

ks, alphas, qualities = scan_frequencies(graph, params, n_workers=params['n_workers'])

pickle.dump([ks, alphas, qualities], open("scan.pkl", "wb"))

plot_scan(ks, alphas, qualities, np.array([[0, 0],]))
plt.savefig("scan_nomodes.svg")

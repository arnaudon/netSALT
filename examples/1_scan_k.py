import os
import sys

import pickle
import numpy as np
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

ks, alphas, qualities = naq.scan_frequencies(
    graph, params, n_workers=params["n_workers"]
)

pickle.dump([ks, alphas, qualities], open("scan.pkl", "wb"))

plotting.plot_scan(ks, alphas, qualities, np.array([[0, 0],]))
plt.savefig("scan_nomodes.svg")
plt.show()

import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
from tqdm import tqdm

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
# graph = naq.oversample_graph(graph, params)

modes_df = naq.load_modes()

if not os.path.isdir("modes"):
    os.mkdir("modes")

plotting.plot_modes(graph, modes_df)

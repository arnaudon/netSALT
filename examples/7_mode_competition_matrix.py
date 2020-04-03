import os
import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

import naq_graphs as naq
from graph_generator import generate_graph

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

os.chdir(graph_tpe)

graph = naq.load_graph()
naq.update_parameters(graph, params)

threshold_modes, lasing_thresholds = naq.load_modes(filename="threshold_modes")

mode_competition_matrix = naq.compute_mode_competition_matrix(
    graph, threshold_modes, lasing_thresholds
)
pickle.dump(mode_competition_matrix, open("mode_competition_matrix.pkl", "wb"))

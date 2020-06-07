import os
import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

import netsalt

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

os.chdir(graph_tpe)

graph = netsalt.load_graph()
netsalt.update_parameters(graph, params)

modes_df = netsalt.load_modes()

mode_competition_matrix = netsalt.compute_mode_competition_matrix(graph, modes_df)
netsalt.save_mode_competition_matrix(mode_competition_matrix)

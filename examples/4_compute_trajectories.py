import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

import naq_graphs as naq
from naq_graphs import plotting
from graph_generator import generate_pump

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

os.chdir(graph_tpe)

graph = naq.load_graph()
modes_df = naq.load_modes()

<<<<<<< HEAD
# LOAD FROM FILE
# graph.graph['params']["pump"] = pickle.load(open("optimal_pump.pkl", "rb"))
# graph.graph['params']["pump"] = pickle.load(open("pump_missing_edge60.pkl", "rb"))

# OR GENERATE PUMP PROFILE BASED ON PARAMS["PUMP_EDGES"]
generate_pump(graph_tpe, graph, params)
graph.graph['params']["pump"] = params["pump"]
=======
#graph.graph['params']["pump"] = pickle.load(open("optimal_pump.pkl", "rb"))
generate_pump(graph_tpe, graph, params)
>>>>>>> a51bd1c... moving files/functions

naq.update_parameters(graph, params)
naq.save_graph(graph)


plotting.plot_naq_graph(graph, edge_colors=graph.graph['params']["pump"], node_size=0.1, save_option=False)
plt.savefig("pump_profile.svg")
plt.show()

modes_df = naq.pump_trajectories(modes_df, graph, return_approx=True)
naq.save_modes(modes_df)

qualities = naq.load_qualities()

ax = plotting.plot_scan(
    graph, qualities, modes_df, filename="scan_with_trajectories", relax_upper=True
)
plt.show()

import os as os
import sys as sys

import numpy as np
import yaml as yaml
import pickle as pickle
import matplotlib.pyplot as plt
import networkx as nx

from graph_generator import generate_graph
from naq_graphs import io, utils, plotting
from naq_graphs.dispersion_relations import set_dialectric_constant, set_dispersion_relation, dispersion_relation_dielectric
from naq_graphs.main import *

if len(sys.argv)>1:
    graph_tpe = sys.argv[-1]
else:
    print('give me a type of graph please!')

params = yaml.full_load(open('graph_params.yaml','rb'))[graph_tpe]

graph, positions  = generate_graph(tpe=graph_tpe, params = params)

os.chdir(graph_tpe)

io.create_naq_graph(graph, params, positions=positions)

set_dialectric_constant(graph, params)
set_dispersion_relation(graph, dispersion_relation_dielectric, params)


modes = io.load_modes()

if not os.path.isdir('modes'):
    os.mkdir('modes')

for i, mode in enumerate(modes): 
    node_solution = mode_on_nodes(mode, graph) 

    plt.figure(figsize=(6,4))
    nodes = nx.draw_networkx_nodes(graph, pos=positions, node_color = abs(node_solution), node_size=20)
    plt.colorbar(nodes)
    edges_k = nx.draw_networkx_edges(graph, pos=positions, edge_color = '0.5', width=2)
    
    plt.savefig('modes/mode_' + str(i) + '.png')
    plt.close()

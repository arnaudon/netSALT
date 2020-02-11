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
from naq_graphs.graph_construction import *

if len(sys.argv)>1:
    graph_tpe = sys.argv[-1]
else:
    print('give me a type of graph please!')

params = yaml.full_load(open('graph_params.yaml','rb'))[graph_tpe]

graph, positions  = generate_graph(tpe=graph_tpe, params = params)

os.chdir(graph_tpe)

create_naq_graph(graph, params, positions=positions)

graph = oversample_graph(graph, edgesize=params['plot_edgesize'])
positions = [graph.nodes[u]['position'] for u in graph]

set_dialectric_constant(graph, params)
set_dispersion_relation(graph, dispersion_relation_dielectric, params)

modes = io.load_modes()

if not os.path.isdir('modes'):
    os.mkdir('modes')

for i, mode in enumerate(modes): 
    node_solution = mode_on_nodes(mode, graph) 
    edge_solution = mean_mode_on_edges(mode, graph) 

    plt.figure(figsize=(6,4))
    nodes = nx.draw_networkx_nodes(graph, pos=positions, node_color = abs(node_solution)**2, node_size=5, cmap=plt.get_cmap('Blues'))
    plt.colorbar(nodes, label=r'$|E|^2$ (a.u)')
    edges_k = nx.draw_networkx_edges(graph, pos=positions, edge_color = edge_solution, width=5, edge_cmap=plt.get_cmap('Blues'))
     
    plt.savefig('modes/mode_' + str(i) + '.png')
    plt.close()

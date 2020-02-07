import os as os
import sys as sys

import numpy as np
import yaml as yaml
import pickle as pickle
import matplotlib.pyplot as plt

from graph_generator import generate_graph
from naq_graphs import io, utils, plotting, dispersion_relations
from naq_graphs.dispersion_relations import set_dialectric_constant

if len(sys.argv)>1:
    graph_tpe = sys.argv[-1]
else:
    print('give me a type of graph please!')

params = yaml.full_load(open('graph_params.yaml','rb'))[graph_tpe]

graph, positions  = generate_graph(tpe=graph_tpe, params = params)

if not os.path.isdir(graph_tpe):
    os.mkdir(graph_tpe)
os.chdir(graph_tpe)

io.create_naq_graph(graph, params, positions=positions)

set_dialectric_constant(graph, params)

plotting.plot_naq_graph(graph, edge_colors=params['dialectric_constant'])

plt.savefig('original_graph.svg')

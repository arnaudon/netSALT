import os as os
import sys as sys

import numpy as np
import yaml as yaml
import pickle as pickle
import matplotlib.pyplot as plt
from functools import partial

from graph_generator import generate_graph
from naq_graphs import io, utils, plotting, dispersion_relations
from naq_graphs.main import *

if len(sys.argv)>1:
    graph_tpe = sys.argv[-1]
else:
    print('give me a type of graph please!')

params = yaml.full_load(open('graph_params.yaml','rb'))[graph_tpe]

graph, positions  = generate_graph(tpe=graph_tpe, params = params)

os.chdir(graph_tpe)

io.create_naq_graph(graph, params, positions=positions)

dispersion_relations.set_dialectric_constant(graph, params)
dispersion_relation = partial(dispersion_relations.dispersion_relation_dielectric, params=params)

ks, alphas, qualities = pickle.load(open('scan.pkl', 'rb')) #save it for later

modes = find_modes(ks, alphas, qualities, graph, dispersion_relation, params)
plotting.plot_scan(ks, alphas, qualities, modes)

print('Found', len(modes), 'mode(s)')
plt.savefig('scan_with_modes.svg')
plt.show()

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
from naq_graphs.dispersion_relations import set_dialectric_constant, set_dispersion_relation, dispersion_relation_dielectric
from naq_graphs.graph_construction import create_naq_graph

if len(sys.argv)>1:
    graph_tpe = sys.argv[-1]
else:
    print('give me a type of graph please!')

params = yaml.full_load(open('graph_params.yaml','rb'))[graph_tpe]

graph, positions  = generate_graph(tpe=graph_tpe, params = params)

os.chdir(graph_tpe)

create_naq_graph(graph, params, positions=positions)

set_dialectric_constant(graph, params)
set_dispersion_relation(graph, dispersion_relation_dielectric, params)

ks, alphas, qualities = pickle.load(open('scan.pkl', 'rb')) #save it for later

modes = find_modes(ks, alphas, qualities, graph, params, n_workers=4)
print('Found', len(modes), 'mode(s)')

io.save_modes(modes)

plotting.plot_scan(ks, alphas, qualities, modes)

plt.savefig('scan_with_modes.svg')
plt.show()

import os as os
import sys as sys

import numpy as np
import yaml as yaml
import pickle as pickle
import matplotlib.pyplot as plt
from functools import partial

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

ks, alphas, qualities = scan_frequencies(graph, params, n_workers=4)

pickle.dump([ks, alphas, qualities], open('scan.pkl', 'wb'))

plotting.plot_scan(ks, alphas, qualities, np.array([[0,0],]))
plt.savefig('scan_nomodes.svg')

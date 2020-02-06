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

if not os.path.isdir(graph_tpe):
    os.mkdir(graph_tpe)
os.chdir(graph_tpe)

io.create_naq_graph(graph, params, positions=positions)

dispersion_relations.set_dialectric_constant(graph, params)

freq = 1.
dispersion_params = {'c': len(graph.edges) * [1.]}
dispersion_relation = partial(dispersion_relations.dispersion_relation_linear, params=dispersion_params)
laplacian = construct_laplacian(freq, graph, dispersion_relation)
ks, alphas, qualities = scan_frequencies(graph, dispersion_relation, params, n_workers=4)

pickle.dump([ks, alphas, qualities], open('scan.pkl', 'wb')) #save it for later

#plot the fine scan and the mode found
plotting.plot_scan(ks, alphas, qualities, np.array([[0,0],]))

#plot the gain bandwidth
plt.twinx()

#naq_u1.pump_params = params['pump_params']
#plt.plot(Ks,naq_u1.lorentzian(Ks))

#plt.savefig('scan_nomodes.svg')
plt.show()

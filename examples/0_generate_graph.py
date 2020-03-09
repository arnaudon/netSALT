import os as os
import sys as sys

import numpy as np
import yaml as yaml

import pickle as pickle
import matplotlib.pyplot as plt

from graph_generator import generate_graph

from naq_graphs.plotting import plot_naq_graph
from naq_graphs.io import save_graph
from naq_graphs import (
    create_naq_graph,
    set_dielectric_constant,
    set_dispersion_relation,
)
from naq_graphs.dispersion_relations import dispersion_relation_pump
from naq_graphs.utils import set_total_length

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

graph, positions = generate_graph(tpe=graph_tpe, params=params)

if not os.path.isdir(graph_tpe):
    os.mkdir(graph_tpe)
os.chdir(graph_tpe)

create_naq_graph(graph, params, positions=positions)

set_total_length(graph, 1.0, inner=True)

if graph_tpe == 'line_PRA' and params["dielectric_params"]["method"] == "custom":
    custom_index = [] #line PRA example 
    for u, v in graph.edges:
        custom_index.append(3.0**2)
    custom_index[0] = 1.0**2
    custom_index[-1] = 1.0**2

    count_inedges = len(graph.edges)-2.;
    print('Number of inner edges', count_inedges)
    if count_inedges % 4 == 0:
        for i in range(round(count_inedges/4)):
            custom_index[i+1] = 1.5**2
    else:
        print('Change number of inner edges to be multiple of 4')
    set_dielectric_constant(graph, params, custom_values=custom_index)
else:
    set_dielectric_constant(graph, params) #for "uniform" and all other graphs

set_dispersion_relation(graph, dispersion_relation_pump, params)

save_graph(graph, params)

plot_naq_graph(graph, edge_colors=params["dielectric_constant"])

plt.savefig("original_graph.svg")
plt.show()

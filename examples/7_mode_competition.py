import os
import sys

import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt
import networkx as nx

from graph_generator import generate_graph

from naq_graphs import set_dielectric_constant, set_dispersion_relation
from naq_graphs.dispersion_relations import dispersion_relation_pump
from naq_graphs import (
    create_naq_graph,
    load_modes,
    oversample_graph,
)
from naq_graphs import mode_on_nodes, mean_mode_on_edges
from naq_graphs.io import load_graph
from naq_graphs.modes import compute_mode_competition_matrix, compute_modal_intensities

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

os.chdir(graph_tpe)

graph, params = load_graph()

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

elif graph_tpe == 'line_semi':  
    custom_index = [] #line OSA example 
    for u, v in graph.edges:
        custom_index.append(1.5**2)
    custom_index[0] = 100.0**2
    custom_index[-1] = 1.0**2
    set_dielectric_constant(graph, params, custom_values=custom_index)

else:
    set_dielectric_constant(graph, params) #for "uniform" and all other graphs

set_dispersion_relation(graph, dispersion_relation_pump, params)


#set pump profile for PRA example
if graph_tpe == 'line_PRA' and params["dielectric_params"]["method"] == "custom":
    pump_edges = round(len(graph.edges())/2)
    nopump_edges = len(graph.edges())-pump_edges
    params["pump"] = np.append(np.ones(pump_edges),np.zeros(nopump_edges))
    params["pump"][0] = 0 #first edge is outside
else:
    #params["pump"] = np.ones(len(graph.edges())) # uniform pump on ALL edges 
    params["pump"] = np.zeros(len(graph.edges())) # uniform pump on inner edges 
    for i, (u,v) in enumerate(graph.edges()): 
        if graph[u][v]["inner"]:
            params["pump"][i] = 1


#graph = oversample_graph(graph, params)
positions = [graph.nodes[u]["position"] for u in graph]

modes, lasing_thresholds = load_modes(filename="threshold_modes")
modes = np.array(modes)[np.argsort(lasing_thresholds)]
lasing_thresholds = np.array(lasing_thresholds)[np.argsort(lasing_thresholds)]
print('lasing threshold noninteracting',lasing_thresholds)
T_mu_all = compute_mode_competition_matrix(graph, params, modes, lasing_thresholds)

plt.figure()
plt.imshow(T_mu_all)#, origin='auto')
plt.colorbar()
plt.savefig('T_matrix.svg')
plt.show()
print('T matrix',T_mu_all)
D0_max = 1.0 #2.3
n_points = 100
pump_intensities = np.linspace(0, D0_max, n_points)
modal_intensities, interacting_lasing_thresholds = compute_modal_intensities(graph, params, modes, lasing_thresholds, pump_intensities)
print('Interacting thresholds:', interacting_lasing_thresholds)

pickle.dump([pump_intensities, modal_intensities, interacting_lasing_thresholds], open('modal_intensities_uniform.pkl','wb'))

plt.figure(figsize=(5,3))
cmap = plt.cm.get_cmap('tab10')    
   
n_lase =  0
for i, intens in enumerate(modal_intensities):
    #if intens[-1]>0:
    plt.plot(pump_intensities, intens,'-',c=cmap.colors[n_lase % 10], label='Mode: '+ str(i))#str(positive_id[sort_id[i]]))
    #plt.axvline(D0_th_sorted[i], c=cmap.colors[n_lase % 10], ls='--')

    n_lase+=1

plt.legend()
plt.title('Uniform mode '+str(n_lase) + ' lasing modes out of ' + str(len(modes)))
plt.xlabel(r'$D_0$')
plt.ylabel(r'$I_\mu$')

#plt.twinx()
#p=[]
#for d in range(n_points):
#    if len(I[:,d])>1:
#        p.append( 1- np.max(I[1:,d])/I[0,d] )
#    else:
#        p.append(I[0,d]/I[0,d])

#plt.plot(D0s, p,'--r')
#plt.axis([D0s[0],D0s[-1],-0.5,1.1])
plt.savefig('uniform_modal_pump.svg', bbox_inches ='tight')
plt.show()

ks, alphas, qualities = pickle.load(open("scan.pkl", "rb"))

def lorentzian(k, k0, gamma):
    #return 1./np.pi * gamma / ( ( k-k0)**2 + gamma**2 ) 
    return  gamma**2 / ( ( k-k0)**2 + gamma**2 ) 

gamma = 0.02
spectr = np.zeros(len(ks))
plt.figure(figsize=(5,2))
for i, intens in enumerate(modal_intensities):
    #if intens[-1]>0:
    center = modes[positive_id[sort_id[i]]][0]

    spectr += intens[-1]*lorentzian(ks,center,gamma)
    print(intens[-1])

#pickle.dump(spectr, open('uniform_spectra.pkl', 'wb'))

plt.plot(Ks, spectr, '-k')
plt.xlabel(r'$k$')
plt.ylabel('Mode amplitude (a.u.)')
#plt.axis([Ks[-1],Ks[0], 0, np.max(I)])
plt.savefig('uniform_spectra.svg', bbox_inches ='tight')
plt.show()

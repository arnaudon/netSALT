import networkx as nx
import numpy as np

def generate_graph(tpe='SM', params= {}):

    pos = [] 
    np.random.seed(params['seed'])

    if tpe == 'SM':
        G = nx.newman_watts_strogatz_graph(params['n'], params['k'], params['p'], seed = params['seed'])
        pos = [np.array([np.cos(2*np.pi*i/len(G)),np.sin(2*np.pi*i/len(G))]) for i in range(len(G))]

    elif tpe == 'ER':
        G = nx.erdos_renyi_graph(params['n'], params['p'])
        
    elif tpe == 'grid':
        G = nx.grid_2d_graph(params['n'], params['m'], periodic=True)
        G = nx.convert_node_labels_to_integers(G)

    elif tpe == 'SBM' or tpe == 'SBM_2':
        import SBM as sbm
        G = nx.stochastic_block_model(params['sizes'],np.array(params['probs'])/params['sizes'][0], seed=params['seed'])
        for i in G:
            G.node[i]['old_label'] = G.node[i]['block']
        
        #G,community_labels= sbm.SBM_graph(params['n'], params['n_comm'], params['p'])
        
    elif tpe == 'powerlaw':
        G = nx.powerlaw_cluster_graph(params['n'], params['m'], params['p'])

    elif tpe == 'geometric':
        G = nx.random_geometric_graph(params['n'], params['p'])

        
    #add infinite leads to the network with probability lead_prob
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pos)
    k = 0 
    for n in hull.simplices.flatten(): 
        p = np.random.rand()
        if p < params['lead_prob']:
            G.add_node(params['n']+k)
            G.add_edge(n, params['n']+k)
            pos.append(pos[n]*1.4)
            k += 1
             
    return G, pos

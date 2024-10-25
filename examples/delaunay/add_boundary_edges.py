import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from netsalt.io import load_graph, save_graph


def add_leads_boundarynodes(graph, lead_prob=1):

    k = 0
    n_tot = len(graph)
    pos = np.array([graph.nodes[u]["position"] for u in graph.nodes])

    from scipy.spatial import ConvexHull
    hull = ConvexHull(pos)
    hull_nodes = set(hull.simplices.flatten())
    print(len(hull_nodes), "hull nodes")

    for n in hull_nodes:
        p = np.random.rand()
        if p < lead_prob:
            graph.add_node(n_tot + k)
            graph.add_edge(n, n_tot + k)
            graph.nodes[n_tot + k]["position"] = pos[n] * 1.05
            k += 1

    # update positions
    pos = np.array([graph.nodes[u]["position"] for u in graph.nodes])

    return graph, pos


graph = load_graph('delaunay_graph.gpickle')

graph, pos = add_leads_boundarynodes(graph, lead_prob=1)

# Plot the network graph
fig, ax = plt.subplots()

nx.draw(graph, pos, node_size=1, width=0.2, node_color='black', with_labels=False, edge_color='green', alpha=1)
    
ax.set_xlim([-5.5,5.5])
ax.set_ylim([-5.5,5.5])
ax.set_aspect('equal', 'box')
ax.set_axis_on()
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

plt.title("Delaunay Network")
plt.savefig("delaunay_graph_with_boundary_edges.pdf")
#plt.show()

save_graph(graph, 'delaunay_graph.gpickle')

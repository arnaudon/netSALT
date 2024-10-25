import yaml
import numpy as np
import matplotlib.pyplot as plt
from netsalt.io import load_graph
from netsalt.plotting import plot_quantum_graph


def index_profile(graph, edgelist=None):

    """define index modification on edges based on their angle with respect to x axis

    Args:
        graph (graph): quantum graph
        edgelist (list): list of edges on which the index is modified
    """

    delta_n = np.zeros(len(graph.edges()))

    positions = [graph.nodes[u]["position"] for u in graph]

    # index change only on inner edges (physical links of network)
    for i, edge in enumerate(graph.edges()):
        if graph[edge[0]][edge[1]]["inner"]:
            edge_ = positions[edge[1]] - positions[edge[0]]
            costheta = np.abs(edge_[0]/np.linalg.norm(edge_))
            graph[edge[0]][edge[1]]["delta_n"] = 1
            delta_n[i] = costheta 
        else:
            graph[edge[0]][edge[1]]["delta_n"] = 0
            delta_n[i] = 0

    # if index change only on certain edges
    if edgelist is not None:
        for i, edge in enumerate(graph.edges()):
            if i in edgelist:
                graph[edge[0]][edge[1]]["delta_n"] = 1
                delta_n[i] = 1
            else:
                graph[edge[0]][edge[1]]["delta_n"] = 0
                delta_n[i] = 0

    return graph, delta_n


def plot_graph_with_index(graph, delta_n):

    """plotting

    Args:
        graph (graph): quantum graph
        delta_n (list): magnitude of index change on edges
    """

    fig = plt.figure(figsize=(5, 4),constrained_layout=True)
    #ax = plt.gca()

    # plots graph with edge color corresponding to index change
    plot_quantum_graph(
            graph, 
            edge_colors=delta_n, 
            node_size=2, 
            cbar_min=np.min(delta_n),
            cbar_max=np.max(delta_n)
            ) 

    plt.savefig("index_profile.pdf")


def write_index(complex_index):

    """write complex index to file

    Args:
        complex_index (list): custom edge values for dielectric constant
    """

    constant = np.real(complex_index)
    loss = np.imag(complex_index)
    complex_index_dict = {"constant": constant.tolist(), "loss": loss.tolist()}

    # Write the array to a YAML file
    with open("index.yaml", "w") as yml:
        yaml.dump(complex_index_dict, yml)


def read_index(fname="index.yaml"):
 
    """read complex index from file"""

    # Read the array of complex numbers from a YAML file
    with open("index.yaml", "r") as yml:
        index_dict = yaml.safe_load(yml)

    num_edges = len(index_dict["constant"])
    index_list = []
    for ei in range(num_edges):
        index_list.append(
                index_dict["constant"][ei] + 1.0j * index_dict["loss"][ei]
                )

    return index_list


"""MAIN PROGRAM"""

graph = load_graph("../delaunay/out/quantum_graph.gpickle")  # load graph that is already oversampled and has params loaded

# If only subset of edges have modified index use:
# ei = [0, 1, 2]  # list of edges with index change
# [graph, d_n ] = index_profile(graph, ei)

[graph, d_n ] = index_profile(graph)


### UPDATE INDEX PARAMS ###

factor = 0.05
edge_index = graph.graph["params"]["dielectric_constant"]
print("Original index values:", edge_index)
# modify real part of complex index
modified_edge_index = edge_index + d_n*factor 

### WRITE TO FILE ###

write_index(modified_edge_index.tolist())

### READ FROM FILE ###

index_list = read_index()
print("Modified index values:", index_list)

### PLOT INDEX ON GRAPH ###

plot_graph_with_index(graph, d_n*factor)


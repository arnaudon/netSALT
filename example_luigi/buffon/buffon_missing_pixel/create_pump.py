import yaml
import numpy as np
from netsalt.io import load_graph

if __name__ == "__main__":

    graph = load_graph("buffon.gpickle")
    lim_x = [-30, 0]
    lim_y = [-30, 0]
    edges = np.ones(len(graph.edges))
    for i, edge in enumerate(graph.edges):
        if (
            lim_x[0] < graph.nodes[edge[0]]["position"][0] < lim_x[1]
            and lim_y[0] < graph.nodes[edge[0]]["position"][1] < lim_y[1]
        ):
            edges[i] = 0
        if (
            lim_x[0] < graph.nodes[edge[1]]["position"][0] < lim_x[1]
            and lim_y[0] < graph.nodes[edge[1]]["position"][1] < lim_y[1]
        ):
            edges[i] = 0

    yaml.dump(edges.tolist(), open("pump_profile.yaml", "w"))

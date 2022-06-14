import numpy as np
import yaml
import networkx as nx
import netsalt

if __name__ == "__main__":
    # create the graph
    graph = nx.grid_2d_graph(11, 1, periodic=False)
    graph = nx.convert_node_labels_to_integers(graph)
    pos = np.array([[i / (len(graph) - 1), 0] for i in range(len(graph))])
    for n, _pos in zip(graph.nodes, pos):
        graph.nodes[n]["position"] = _pos
    netsalt.save_graph(graph, "graph.pkl")

    # create the index of refraction profile
    custom_index = len(graph.edges) * [3.0**2]
    custom_loss = len(graph.edges) * [0.0]
    custom_index[0] = 1.0**2
    custom_index[-1] = 1.0**2

    count_inedges = len(graph.edges) - 2.0
    if count_inedges % 4 == 0:
        for i in range(round(count_inedges / 4)):
            custom_index[i + 1] = 1.5**2

    yaml.dump({"constant": custom_index, "loss": custom_loss}, open("index.yaml", "w"))

    # create the pump profile
    pump_edges = round(len(graph.edges()) / 2)
    nopump_edges = len(graph.edges()) - pump_edges
    pump = np.append(np.ones(pump_edges), np.zeros(nopump_edges))
    pump[0] = 0
    yaml.dump(pump.astype(int).tolist(), open("pump.yaml", "w"))

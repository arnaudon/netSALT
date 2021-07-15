import yaml

import matplotlib.pyplot as plt
from netsalt.io import load_graph, save_graph

from netsalt.plotting import plot_quantum_graph
from netsalt.utils import remove_pixel
from netsalt.quantum_graph import set_edge_lengths, get_total_length

if __name__ == "__main__":

    graph = load_graph("buffon.gpickle")
    pix_size = 50.728
    box = [-0.5, 0.5, -0.5, 0.5]
    box = [pix_size*x for x in box]
    print(box)
    graph, pump = remove_pixel(graph, box)

    save_graph(graph, filename="buffon_missing_pixel.gpickle")
    yaml.dump(pump, open("pump_profile.yaml", "w"))

    plt.figure()
    ax = plt.gca()
    plot_quantum_graph(graph, ax=ax, node_size=10)

    plt.savefig("buffon_missing_pixel.pdf")


    """Get the total length of the graph with pump on."""

    set_edge_lengths(graph)
    tot_L = get_total_length(graph)
    pump_L = sum([graph[u][v]["length"] for u, v in graph.edges if graph[u][v]["pump"]])

    print("pump length", pump_L)
    print("total length", tot_L)
    print("pump fraction OFF = ", 1 - pump_L/tot_L)

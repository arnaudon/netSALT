import yaml

import matplotlib.pyplot as plt
from netsalt.io import load_graph, save_graph

from netsalt.plotting import plot_quantum_graph
from netsalt.utils import remove_pixel

if __name__ == "__main__":

    graph = load_graph("buffon.gpickle")
    box = [-30, 0, -30, 0]
    graph, pump = remove_pixel(graph, box)

    save_graph(graph, filename="buffon_missing_pixel.gpickle")
    yaml.dump(pump, open("pump_profile.yaml", "w"))

    plt.figure()
    ax = plt.gca()
    plot_quantum_graph(graph, ax=ax, node_size=10)

    plt.savefig("buffon_missing_pixel.pdf")

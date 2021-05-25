import os
import sys

import matplotlib.pyplot as plt
import yaml

import netsalt
from netsalt import plotting

if __name__ == "__main__":
    if len(sys.argv) > 1:
        graph_tpe = sys.argv[-1]
    else:
        print("give me a type of graph please!")

    params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

    os.chdir(graph_tpe)

    graph = netsalt.load_graph()
    netsalt.update_parameters(graph, params, force=True)

    qualities = netsalt.scan_frequencies(graph)
    netsalt.save_qualities(qualities)

    plotting.plot_scan(graph, qualities, filename="scan_no_modes")
    plt.show()

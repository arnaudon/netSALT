import os
import sys

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
    netsalt.update_parameters(graph, params)
    graph = netsalt.oversample_graph(graph, params)

    modes_df = netsalt.load_modes()

    if not os.path.isdir("modes"):
        os.mkdir("modes")

    plotting.plot_modes(graph, modes_df)

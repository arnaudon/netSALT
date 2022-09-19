import matplotlib.pyplot as plt
import networkx as nx
from netsalt.utils import make_buffon_graph

import numpy as np
if __name__ == "__main__":
    np.random.seed(42)
    buffon, pos = make_buffon_graph(n_lines=20, size=(-100.0, 100.0), resolution=1.0)

    plt.figure()
    nx.draw(buffon, pos=pos, node_size=0.00, width=0.2)
    ax = plt.gca()
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.savefig("buffon_graph.pdf")

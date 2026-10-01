"""Regenerate ``buffon.json``: a second Buffon realisation, seed 7.

The checked-in graph of ``buffon_competition`` is a single realisation, and
every result in ``examples/audit/results/mode_competition/`` rests on it. This
is an independent draw at the same parameters (20 lines over a 200 x 200 square,
intersections only), kept to its giant component, so the overlap-predicts-the-
switch claim can be tested on a graph it was not derived from.

``inner_total_length`` in ``../_base.yaml`` rescales both graphs to the same
optical length, so the mode density over the k window is comparable by
construction -- what differs is the geometry, and hence which modes overlap.
"""

import networkx as nx
import numpy as np

from netsalt.io import save_graph
from netsalt.utils import make_buffon_graph

SEED = 7

if __name__ == "__main__":
    graph, pos = make_buffon_graph(n_lines=20, size=(-100.0, 100.0), resolution=100.0, rng=SEED)
    giant = max(nx.connected_components(graph), key=len)
    graph = nx.convert_node_labels_to_integers(graph.subgraph(giant).copy(), label_attribute="old")
    positions = np.array([pos[graph.nodes[u]["old"]] for u in graph.nodes])
    for u in graph.nodes:
        del graph.nodes[u]["old"]
        graph.nodes[u]["position"] = [float(positions[u][0]), float(positions[u][1])]
    save_graph(graph, "buffon.json")
    print(f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

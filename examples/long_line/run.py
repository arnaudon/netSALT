"""Long multimode Fabry--Perot: many longitudinal modes by cavity length.

The same open 1D cavity as ``../line_fabry_perot`` but four times longer
(optical length ``nL = 6``): the longitudinal mode spacing
``dk = pi/(nL) ~ 0.52`` packs ~12 modes into the scan window, and the broad
gain (``gamma_perp = 3``) lets several of them reach threshold -- the textbook
multimode Fabry--Perot laser. Adjacent longitudinal modes have shifted
standing-wave patterns, so each burns its spatial holes in different places
and they co-lase despite sharing the cavity (the classic spatial-hole-burning
route to multimode operation in FP diodes).

Checks encoded here: the mode spacing matches ``pi/(nL)``, and ``linear`` and
``full_salt_newton`` agree on the lasing count near threshold.

Run from this directory (writes ``li_curves.png`` + ``mode_profiles.png``)::

    OMP_NUM_THREADS=1 python run.py
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import compare_and_plot, quantum_graph

TOTAL_LENGTH = 2.0  # 4x the short line -> dk = pi/(3*2) ~ 0.52


def build(n_edges=20, total_length=TOTAL_LENGTH):
    g = nx.path_graph(n_edges + 1)
    pos = np.array([[i, 0.0] for i in range(n_edges + 1)], dtype=float)
    return quantum_graph(g, pos, total_length)


if __name__ == "__main__":
    graph = build()
    inner = [graph[u][v]["length"] for u, v in graph.edges if graph[u][v]["inner"]]
    dk = np.pi / (3.0 * sum(inner))
    print(f"cavity length {sum(inner):.3f} -> expected longitudinal spacing dk = {dk:.3f}")
    compare_and_plot(graph, "long Fabry-Perot line", Path(__file__).resolve().parent)

"""Open 1D Fabry--Perot cavity: ``linear`` vs ``full_salt_newton`` L--I curves.

A path graph whose two end edges are vacuum leads (the radiative loss that sets
the lasing threshold). A short optical length keeps the longitudinal modes well
separated in ``k``; with the narrow gain here two modes near the line centre
lase under both solvers, and the newton curves bend below the linear ones above
threshold (the full-SALT gain saturation the near-threshold model omits).

Run from this directory (writes ``li_curves.png`` here)::

    OMP_NUM_THREADS=1 python run.py
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import compare_and_plot, quantum_graph


def build(n_edges=10, total_length=0.5):
    g = nx.path_graph(n_edges + 1)
    pos = np.array([[i, 0.0] for i in range(n_edges + 1)], dtype=float)
    return quantum_graph(g, pos, total_length)


if __name__ == "__main__":
    compare_and_plot(build(), "line (Fabry-Perot)", Path(__file__).resolve().parent)

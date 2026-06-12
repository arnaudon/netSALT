"""Binary-tree splitter: ``linear`` vs ``full_salt_newton`` L--I curves.

A depth-3 binary tree -- one input lead, several leaf leads. Lossy enough that
a single mode lases under both solvers (winner-take-all gain clamping).

Run from this directory (writes ``li_curves.png`` here)::

    OMP_NUM_THREADS=1 python run.py
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import compare_and_plot, quantum_graph


def build(total_length=1.0):
    g = nx.balanced_tree(2, 3)
    pos = nx.kamada_kawai_layout(g)
    positions = np.array([pos[u] for u in sorted(g.nodes)], dtype=float)
    return quantum_graph(g, positions, total_length)


if __name__ == "__main__":
    compare_and_plot(build(), "binary tree", Path(__file__).resolve().parent)

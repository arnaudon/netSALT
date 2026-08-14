"""Ring resonator with two leads: ``linear`` vs ``full_salt_newton`` L--I curves.

A closed loop made open by two pendant lead edges on opposite sides, so the
loop modes can radiate out. Both solvers lase the same modes; above threshold
the newton curves carry the full-SALT saturation bend.

Run from this directory (writes ``li_curves.png`` here)::

    OMP_NUM_THREADS=1 python run.py
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import compare_and_plot, quantum_graph


def build(n=12, total_length=1.0):
    g = nx.cycle_graph(n)
    pos = {u: [np.cos(2 * np.pi * u / n), np.sin(2 * np.pi * u / n)] for u in g.nodes}
    g.add_edge(0, n)
    pos[n] = [2.0, 0.0]
    g.add_edge(n // 2, n + 1)
    pos[n + 1] = [-2.0, 0.0]
    positions = np.array([pos[u] for u in sorted(g.nodes)], dtype=float)
    return quantum_graph(g, positions, total_length)


if __name__ == "__main__":
    compare_and_plot(build(), "ring + leads", Path(__file__).resolve().parent)

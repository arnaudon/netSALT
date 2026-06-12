"""Mini-buffon network: the dense-spectrum stress test for the two solvers.

A shrunk version of the Nat. Commun. buffon networks (10 random lines, giant
component: 39 nodes, 45 edges, 18 radiating lead ends) built with
:func:`netsalt.utils.make_buffon_graph` from a fixed seed. The spectrum is
genuinely dense -- ~25 modes per unit ``k`` (close to the Weyl estimate
``nL/pi = 28.6``), 10 modes in the scan window with disorder-spread losses
(``alpha = 0.006 - 0.064``) and a near-degenerate pair split by ``dk = 0.006``.

What the run shows (measured, not assumed):

* **Competition, not cost, limits the mode count.** Even with the gain
  broadened to cover all ten modes (``gamma_perp = 0.3``) only **two** lase
  under a *uniform* pump -- the extended disorder modes overlap strongly, so
  the first lasing modes clamp the gain for the rest. Multimode operation on
  buffon networks is achieved by **pump optimisation** (the Nat. Commun.
  route, ``netsalt.pump``), not by pumping everything; for many co-lasing
  modes by *design* see ``../ring_chain``.
* **The newton solver is still comfortable here**: the sweep takes ~25 s on
  the oversampled ~700-node work graph (vs ~0.1 s for ``linear``) and agrees
  with ``linear`` on the lasing set. The cost crossover the docs warn about
  sits at genuinely larger networks / larger co-lasing counts, not at this
  scale.

Run from this directory (writes ``li_curves.png`` + ``mode_profiles.png``)::

    OMP_NUM_THREADS=1 python run.py     # ~2 minutes, mostly the mode pipeline
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import compare_and_plot

import netsalt
from netsalt.physics import dispersion_relation_pump
from netsalt.quantum_graph import create_quantum_graph, set_total_length
from netsalt.utils import make_buffon_graph

HERE = Path(__file__).resolve().parent
N_LINES = 10
SEED = 4  # chosen by a small seed scan for a ~40-node giant component
D0_MAX = 0.4

PARAMS = {
    "open_model": "open",
    "c": 1.0,
    "k_a": 3.55,  # on the low-loss cluster found by the spectrum scan
    "gamma_perp": 0.3,
    "k_min": 3.3,
    "k_max": 3.7,
    "alpha_min": -0.05,
    "alpha_max": 0.3,
    "n_workers": 1,
    "n_modes_max": 60,
    "quality_threshold": 1e-3,
    "search_stepsize": 0.005,
    "max_steps": 1000,
    "max_tries_reduction": 50,
    "reduction_factor": 0.8,
    "D0_max": D0_MAX,
    "D0_steps": 14,
    "dielectric_params": {"method": "uniform", "inner_value": 9.0, "outer_value": 1.0, "loss": 0.0},
}


def build():
    g, pos = make_buffon_graph(n_lines=N_LINES, size=(-100.0, 100.0), resolution=100.0, rng=SEED)
    giant = max(nx.connected_components(g), key=len)
    g = nx.convert_node_labels_to_integers(g.subgraph(giant).copy(), label_attribute="old")
    positions = np.array([pos[g.nodes[u]["old"]] for u in g.nodes])
    create_quantum_graph(g, dict(PARAMS), positions=positions)
    set_total_length(g, 30.0)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    netsalt.set_dispersion_relation(g, dispersion_relation_pump)
    return g


if __name__ == "__main__":
    graph = build()
    n_leads = sum(1 for n in graph.nodes if len(graph[n]) == 1)
    inner = sum(graph[u][v]["length"] for u, v in graph.edges if graph[u][v]["inner"])
    print(
        f"mini-buffon: {len(graph)} nodes, {len(graph.edges)} edges, {n_leads} leads; "
        f"Weyl density nL/pi = {3 * inner / np.pi:.1f} modes per unit k"
    )
    t0 = time.perf_counter()
    compare_and_plot(
        graph, "mini-buffon", HERE, d0_max=D0_MAX, d0_steps=12, passive_method="contour"
    )
    print(f"total wall time {time.perf_counter() - t0:.0f}s")

"""Mode-search benchmark: contour single / subdivided / adaptive.

Two workloads:

1. **Easy** — buffon graph, ~20 modes in the rectangle (well under
   the ``probe_dim`` cap). Single contour (``n_k=1``) handles it;
   subdivision (``n_k>1``) and adaptive both work but add overhead.
2. **Stress** — same graph density extended to ~300 modes
   (8.4× over per-contour capacity). Single contour collapses;
   subdivision is mandatory; adaptive recovers most modes;
   ``tune_contour_parameters`` then ``find_modes_contour`` recovers
   all modes.

Usage::

    .venv/bin/python benchmark/bench_contour.py
"""

from __future__ import annotations

from time import perf_counter

import networkx as nx
import numpy as np

import netsalt
from netsalt.contour import (
    find_modes_contour,
    find_modes_contour_adaptive,
    tune_contour_parameters,
)
from netsalt.physics import dispersion_relation_dielectric
from netsalt.quantum_graph import create_quantum_graph, set_total_length
from netsalt.utils import make_buffon_graph


def buffon_graph(seed=2, n_lines=6, total_length=8.0):
    """Connected component of a random buffon planar graph."""
    g, pos = make_buffon_graph(
        n_lines=n_lines, size=(0.0, 1.0), resolution=0.1, rng=np.random.default_rng(seed)
    )
    biggest = max(nx.connected_components(g), key=len)
    g = g.subgraph(biggest).copy()
    pos_orig = {u: pos[u] for u in g.nodes}
    g = nx.convert_node_labels_to_integers(g, label_attribute="orig_label")
    pos_arr = np.array([pos_orig[g.nodes[u]["orig_label"]] for u in g.nodes])
    params = {
        "open_model": "open",
        "dielectric_params": {
            "method": "uniform",
            "inner_value": 4.0,
            "loss": 0.0,
            "outer_value": 1.0,
        },
        "c": 1.0,
    }
    create_quantum_graph(g, params, positions=pos_arr)
    set_total_length(g, total_length)
    netsalt.set_dispersion_relation(g, dispersion_relation_dielectric)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    return g


def time_block():
    class _T:
        def __enter__(self):
            self.t0 = perf_counter()
            return self

        def __exit__(self, *a):
            self.seconds = perf_counter() - self.t0

    return _T()


def workload(label, g, bounds, gold_n_k):
    """Run single / subdivided / adaptive on one graph and print one row each."""
    print(f"\n=== {label} ===")
    print(f"graph: {len(g)} nodes, {len(g.edges)} edges")
    probe_dim = min(40, len(g))

    # Gold reference: dense subdivision so we know the true mode count.
    with time_block() as t_g:
        gold = find_modes_contour(
            g,
            bounds=bounds,
            n_k=gold_n_k,
            n_alpha=2,
            n_quad=240,
            probe_dim=probe_dim,
            rng=np.random.default_rng(0),
        )
    print(f"gold ({len(gold)} modes, {t_g.seconds:.1f}s)")

    print(f"\n{'method':>22} | {'time (s)':>8} | {'modes':>5}")
    print("-" * 48)

    # Single contour (n_k=1).
    with time_block() as t:
        m = find_modes_contour(
            g,
            bounds=bounds,
            n_k=1,
            n_alpha=1,
            n_quad=200,
            probe_dim=probe_dim,
            rng=np.random.default_rng(0),
        )
    print(f"{'find_modes_contour n_k=1':>22} | {t.seconds:>8.2f} | {len(m):>5}")

    # Subdivided contour (n_k recommended by tune).
    rec_params, info = tune_contour_parameters(
        g, bounds=bounds, probe_dim=probe_dim, n_quad=200, max_depth=8, rng=np.random.default_rng(0)
    )
    with time_block() as t:
        m = find_modes_contour(g, bounds=bounds, **rec_params, rng=np.random.default_rng(0))
    print(
        f"{'find_modes_contour n_k=' + str(rec_params['n_k']):>22} | "
        f"{t.seconds:>8.2f} | {len(m):>5}"
    )

    # Adaptive (no manual sizing).
    with time_block() as t:
        m = find_modes_contour_adaptive(
            g,
            bounds=bounds,
            n_quad=200,
            probe_dim=probe_dim,
            max_depth=8,
            rng=np.random.default_rng(0),
        )
    print(f"{'_contour_adaptive':>22} | {t.seconds:>8.2f} | {len(m):>5}")


def main():
    # Easy: ~20 modes, single contour fits comfortably under probe_dim cap.
    workload("Easy: ~20 modes", buffon_graph(total_length=1.0), (0.5, 40.0, 0.0, 5.0), gold_n_k=4)

    # Stress: ~300 modes, single contour cannot fit.
    workload(
        "Stress: ~300 modes", buffon_graph(total_length=12.0), (0.5, 40.0, 0.0, 5.0), gold_n_k=24
    )


if __name__ == "__main__":
    main()

"""Shared helpers for the benchmark scripts.

Builds the dielectric line graphs used as the reference workload and
provides counters for ``mode_quality`` calls so the benchmarks can
report ARPACK invocations alongside wall-clock timings.
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

import networkx as nx
import numpy as np

import netsalt
from netsalt.physics import dispersion_relation_dielectric, dispersion_relation_pump
from netsalt.quantum_graph import create_quantum_graph, set_total_length
from netsalt.utils import make_buffon_graph


def line_graph(n_edges: int, dielectric: float = 4.0, total_length: float = 1.0):
    """Open dielectric line graph with ``n_edges`` segments of equal length."""
    g = nx.path_graph(n_edges + 1)
    pos = np.array([[float(i) / n_edges, 0.0] for i in range(n_edges + 1)])
    params = {
        "open_model": "open",
        "dielectric_params": {
            "method": "uniform",
            "inner_value": dielectric,
            "loss": 0.0,
            "outer_value": 1.0,
        },
        "c": 1.0,
    }
    create_quantum_graph(g, params, positions=pos)
    set_total_length(g, total_length)
    netsalt.set_dispersion_relation(g, dispersion_relation_dielectric)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    return g


def grid_planar_graph(
    rows: int = 3, cols: int = 3, dielectric: float = 4.0, total_length: float = 1.0
):
    """Small planar graph (``rows × cols`` grid, ``rows·cols`` nodes,
    deterministic topology). Less symmetric than the line graph — modes
    don't degenerate, so it stresses the refiners differently.
    """
    g = nx.grid_2d_graph(rows, cols)
    g = nx.convert_node_labels_to_integers(g)
    pos = np.array([[float(i % cols), float(i // cols)] for i in range(rows * cols)])
    params = {
        "open_model": "open",
        "dielectric_params": {
            "method": "uniform",
            "inner_value": dielectric,
            "loss": 0.0,
            "outer_value": 1.0,
        },
        "c": 1.0,
    }
    create_quantum_graph(g, params, positions=pos)
    set_total_length(g, total_length)
    netsalt.set_dispersion_relation(g, dispersion_relation_dielectric)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    return g


def buffon_planar_graph(
    n_lines: int = 4, dielectric: float = 4.0, total_length: float = 1.0, seed: int = 7
):
    """Small random buffon-style planar graph (intersecting line segments).

    Stress-test for the refiners on irregular topology where modes are
    nondegenerate and each one sits at a different ``alpha``.

    ``make_buffon_graph`` may return a disconnected graph; we keep the
    largest connected component so Beyn can find modes on a single
    laplacian. Node labels are then made contiguous via
    ``convert_node_labels_to_integers``.
    """
    rng = np.random.default_rng(seed)
    g, pos = make_buffon_graph(n_lines=n_lines, size=(0.0, 1.0), resolution=0.1, rng=rng)
    components = list(nx.connected_components(g))
    biggest = max(components, key=len)
    g = g.subgraph(biggest).copy()
    # Save the original-label position mapping before relabelling.
    pos_orig = {u: pos[u] for u in g.nodes}
    g = nx.convert_node_labels_to_integers(g, label_attribute="orig_label")
    pos_array = np.array([pos_orig[g.nodes[u]["orig_label"]] for u in g.nodes])
    params = {
        "open_model": "open",
        "dielectric_params": {
            "method": "uniform",
            "inner_value": dielectric,
            "loss": 0.0,
            "outer_value": 1.0,
        },
        "c": 1.0,
    }
    create_quantum_graph(g, params, positions=pos_array)
    set_total_length(g, total_length)
    netsalt.set_dispersion_relation(g, dispersion_relation_dielectric)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    return g


def line_graph_with_pump(n_edges: int = 6, dielectric: float = 4.0, total_length: float = 1.0):
    """Same line graph wired up for the pump dispersion (D0=0 → reduces to
    dielectric, but the dispersion relation function is the more complex
    one used in the production pipeline)."""
    g = line_graph(n_edges, dielectric=dielectric, total_length=total_length)
    netsalt.set_dispersion_relation(g, dispersion_relation_pump)
    netsalt.update_parameters(
        g,
        {
            "k_a": 5.0,
            "gamma_perp": 1.0,
            "D0": 0.0,
            "pump": np.ones(len(g.edges)),
        },
    )
    return g


@contextmanager
def count_calls(module, attr: str):
    """Patch ``module.attr`` to count its invocations.

    Yields a list whose first element is the call count after the block
    exits. This is used to count ``mode_quality`` calls inside a refiner
    without modifying the refiner.
    """
    original = getattr(module, attr)
    counter = [0]

    def wrapper(*args, **kwargs):
        counter[0] += 1
        return original(*args, **kwargs)

    setattr(module, attr, wrapper)
    try:
        yield counter
    finally:
        setattr(module, attr, original)


def time_block():
    """Return a context manager whose .seconds reports wall time."""

    class _T:
        def __enter__(self):
            self.t0 = perf_counter()
            return self

        def __exit__(self, *args):
            self.seconds = perf_counter() - self.t0

    return _T()


def fmt_complex(z: complex) -> str:
    return f"{z.real:.5f}{'+' if z.imag >= 0 else '-'}{abs(z.imag):.5f}j"

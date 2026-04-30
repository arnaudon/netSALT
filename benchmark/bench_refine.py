"""Mode-refinement benchmark: ``root`` vs ``brownian``.

Refines a batch of modes from a small perturbation of their true
location (Beyn-found ground truth). Reports per-method wall time,
``mode_quality`` evaluation count, and convergence count.

Usage::

    .venv/bin/python benchmark/bench_refine.py
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

import networkx as nx
import numpy as np

import netsalt
import netsalt.algorithm as alg
from netsalt.algorithm import refine_mode_brownian_ratchet, refine_mode_root
from netsalt.contour import find_modes_contour
from netsalt.physics import dispersion_relation_dielectric
from netsalt.quantum_graph import create_quantum_graph, set_total_length
from netsalt.utils import make_buffon_graph

REFINERS = [
    ("root", refine_mode_root, {}),
    ("brownian", refine_mode_brownian_ratchet, {"rng": np.random.default_rng(0)}),
]


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


@contextmanager
def count_calls(module, attr):
    original = getattr(module, attr)
    counter = [0]

    def wrapper(*a, **kw):
        counter[0] += 1
        return original(*a, **kw)

    setattr(module, attr, wrapper)
    try:
        yield counter
    finally:
        setattr(module, attr, original)


def main():
    g = buffon_graph()
    bounds = (0.5, 30.0, 0.0, 5.0)
    print(f"graph: {len(g)} nodes, {len(g.edges)} edges")

    truth = find_modes_contour(
        g, bounds=bounds, n_k=8, n_alpha=1, n_quad=200, probe_dim=40, rng=np.random.default_rng(0)
    )
    print(f"discovered {len(truth)} modes via Beyn (used as ground truth)")

    n_batch = min(20, len(truth))
    idx = np.linspace(0, len(truth) - 1, n_batch).astype(int)
    modes_truth = truth[idx]
    perturb = 0.01

    params = dict(g.graph["params"])
    params.update(
        {
            "quality_threshold": 1e-5,
            "search_stepsize": 0.01,
            "max_steps": 500,
            "k_min": bounds[0],
            "k_max": bounds[1],
            "alpha_min": bounds[2],
            "alpha_max": bounds[3],
        }
    )
    g.graph["params"] = params

    print(
        f"\n{'method':>10} | {'total (s)':>9} | {'ms/refine':>9} | "
        f"{'evals/refine':>12} | {'converged':>9}"
    )
    print("-" * 65)
    for name, fn, kwargs in REFINERS:
        with count_calls(alg, "mode_quality") as counter:
            t0 = perf_counter()
            converged = 0
            for true_mode in modes_truth:
                init = np.array([true_mode[0] + perturb, true_mode[1] + perturb])
                if isinstance(fn(init, g, dict(params), **kwargs), np.ndarray):
                    converged += 1
            wall = perf_counter() - t0
        print(
            f"{name:>10} | {wall:>9.2f} | {wall * 1e3 / n_batch:>9.1f} | "
            f"{counter[0] / n_batch:>12.1f} | {converged}/{n_batch:>3}"
        )


if __name__ == "__main__":
    main()

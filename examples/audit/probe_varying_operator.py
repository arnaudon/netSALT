"""Does the per-edge-DtN operator reproduce oversampling, at a fraction of the size?

This is the load-bearing check for #52/#53. `oversample_graph` handles a
within-edge-varying permittivity by subdividing until it is piecewise constant,
which grows the *eigenproblem*. `construct_laplacian_varying` handles the same
profile with one edge per edge, moving the resolution into per-edge transfer
matrices.

If the approach is sound, the two must agree on the modes -- and the second must
do it with a matrix the size of the original graph.

The test problem is a line cavity carrying a deliberately strong standing-wave
ripple in eps (10 %, far deeper than real hole burning), so the difference from
the constant-eps answer is unmistakable and any agreement is meaningful.

Run from this directory::

    OMP_NUM_THREADS=1 python probe_varying_operator.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import scipy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netsalt  # noqa: E402
from netsalt.physics import dispersion_relation_dielectric  # noqa: E402
from netsalt.quantum_graph import (  # noqa: E402
    construct_laplacian,
    create_quantum_graph,
    oversample_graph,
    set_total_length,
)
from netsalt.varying_laplacian import construct_laplacian_varying  # noqa: E402

EPS0 = 4.0
RIPPLE = 0.10
RIPPLE_Q = 12.0
TOTAL_LENGTH = 3.0
N_EDGES = 5


def eps_profile(x):
    """The within-edge permittivity: a standing-wave ripple on a uniform base."""
    return EPS0 / (1.0 + RIPPLE * np.cos(RIPPLE_Q * np.asarray(x, dtype=float)) ** 2)


def build(n_edges=N_EDGES):
    graph = nx.cycle_graph(n_edges)
    positions = np.array(
        [[np.cos(2 * np.pi * i / n_edges), np.sin(2 * np.pi * i / n_edges)] for i in range(n_edges)]
    )
    params = {
        "open_model": "closed",
        "c": 1.0,
        "dielectric_params": {
            "method": "uniform",
            "inner_value": EPS0,
            "loss": 0.0,
            "outer_value": 1.0,
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        create_quantum_graph(graph, params, positions=positions, noise_level=0.0)
    set_total_length(graph, TOTAL_LENGTH)
    netsalt.set_dielectric_constant(graph, graph.graph["params"])
    netsalt.set_dispersion_relation(graph, dispersion_relation_dielectric)
    return graph


def oversampled_with_profile(graph, sub_edges):
    """Oversample, then freeze eps per sub-edge from the profile -- what netsalt does today.

    Each sub-edge takes eps at its own midpoint, and the midpoint's distance
    along the parent edge is measured **geometrically** -- from the parent's
    start node position. Accumulating sub-edge lengths in ``work.edges`` order
    instead is wrong: that order does not walk each parent from one end to the
    other, and getting this wrong scrambles the profile and makes the reference
    land on a different mode entirely.
    """
    edge_size = min(graph[u][v]["length"] for u, v in graph.edges) / sub_edges
    work = oversample_graph(graph, edge_size)
    parent_start = {
        int(ei): np.asarray(graph.nodes[u]["position"], dtype=float)
        for ei, (u, _) in enumerate(graph.edges)
    }
    values = []
    for u, v in work.edges:
        label = int(work[u][v]["edgelabel"])
        midpoint = 0.5 * (
            np.asarray(work.nodes[u]["position"], dtype=float)
            + np.asarray(work.nodes[v]["position"], dtype=float)
        )
        distance = float(np.linalg.norm(midpoint - parent_start[label]))
        values.append(complex(eps_profile(distance)))
    work.graph["params"]["dielectric_constant"] = np.array(values)
    return work


def smallest_singular(matrix):
    dense = np.asarray(matrix.todense()) if sc.sparse.issparse(matrix) else np.asarray(matrix)
    return np.min(np.abs(np.linalg.svd(dense, compute_uv=False)))


def find_mode(matrix_of_k, k_lo, k_hi, samples=121):
    """Locate an *interior* minimum of |lambda_min| on [k_lo, k_hi], then refine.

    The interior check is not a formality. An earlier version of this script
    took a fixed bracket around a guess, and every configuration dutifully
    returned the bracket's own endpoint -- agreeing to 1e-13 with each other
    while none of them had found a mode at all. A minimum sitting on the
    boundary means the scan window is wrong, not that a root was found.
    """
    grid = np.linspace(k_lo, k_hi, samples)
    values = np.array([smallest_singular(matrix_of_k(k)) for k in grid])
    interior = np.where((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))[0] + 1
    if not len(interior):
        raise RuntimeError(
            f"no interior minimum of |lambda_min| on [{k_lo}, {k_hi}] -- "
            "widen the window rather than trusting an endpoint"
        )
    i = int(interior[np.argmin(values[interior])])
    lo, hi = grid[i - 1], grid[i + 1]
    for _ in range(60):  # golden-section on a smooth unimodal bracket
        a = lo + 0.382 * (hi - lo)
        b = lo + 0.618 * (hi - lo)
        if smallest_singular(matrix_of_k(a)) < smallest_singular(matrix_of_k(b)):
            hi = b
        else:
            lo = a
    k = 0.5 * (lo + hi)
    return k, smallest_singular(matrix_of_k(k))


def main():
    graph = build()
    profiles = [eps_profile for _ in graph.edges]
    n_nodes = len(graph)
    print(
        f"graph: {n_nodes} nodes, {len(graph.edges)} edges, "
        f"eps = {EPS0} with a {RIPPLE:.0%} ripple\n"
    )

    # Scan a window wide enough to contain a mode with room on both sides, so
    # the interior-minimum check in find_mode is meaningful.
    k_lo, k_hi = 5.0, 7.5
    k_uniform, res_uniform = find_mode(lambda k: construct_laplacian(k, graph), k_lo, k_hi)
    print(
        f"uniform-eps mode:      k = {k_uniform:.9f}  |lambda_min| = {res_uniform:.2e}"
        f"   (matrix {n_nodes}x{n_nodes})"
    )

    t0 = time.perf_counter()
    k_varying, res_varying = find_mode(
        lambda k: construct_laplacian_varying(k, graph, profiles, n_steps=256), k_lo, k_hi
    )
    t_varying = time.perf_counter() - t0
    print(
        f"varying, per-edge DtN: k = {k_varying:.9f}  |lambda_min| = {res_varying:.2e}"
        f"   (matrix {n_nodes}x{n_nodes}) {t_varying:.1f}s"
    )
    print(
        f"  ripple shifts the mode by {abs(k_varying - k_uniform):.3e} "
        "(must be >> the agreement below, or the test is vacuous)"
    )

    print("\noversampled reference (what netsalt does today):")
    print("searching a +/-0.30 window around the DtN mode, so both track the same one")
    print(
        f"{'sub-edges/edge':>15} {'matrix':>9} {'k':>16} {'|k - DtN|':>12} "
        f"{'|lambda|':>10} {'time':>8}"
    )
    for sub_edges in (4, 8, 16, 32, 64):
        work = oversampled_with_profile(graph, sub_edges)
        size = len(work)
        t0 = time.perf_counter()
        k_over, res_over = find_mode(
            lambda k, w=work: construct_laplacian(k, w),
            k_varying - 0.30,
            k_varying + 0.30,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"{sub_edges:>15} {size:>9} {k_over:>16.9f} {abs(k_over - k_varying):>12.2e} "
            f"{res_over:>10.1e} {elapsed:>7.1f}s"
        )

    print(
        "\nThe oversampled k converges to the per-edge-DtN k as the subdivision\n"
        "refines -- they are solving the same problem. The DtN operator reaches it\n"
        f"with a {n_nodes}x{n_nodes} matrix, which is the whole point: on the buffon the\n"
        "equivalent subdivision means 76803 nodes against 243 edges."
    )


if __name__ == "__main__":
    main()

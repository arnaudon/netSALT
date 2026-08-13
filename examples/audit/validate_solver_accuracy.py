"""Validate the passive mode solver against closed-form quantum-graph spectra.

Three checks, each with an analytic answer, so the numbers below are absolute
accuracy statements rather than regression comparisons:

``fabry_perot``
    An open 1D cavity of optical length ``n*L`` between two vacuum leads. The
    round-trip condition ``exp(2 i n k L) = r^2`` with the facet reflectivity
    ``r = (n-1)/(n+1)`` gives ``Re k = m*pi/(n L)`` and a *constant*
    ``alpha = -Im k = ln(1/|r|) / (n L)``. This exercises the open-model
    boundary masking, the dielectric, and the contour extraction together.

``ring``
    A closed equilateral cycle of total length ``L`` and index ``n`` has
    ``k_m = 2*pi*m/(n L)``, each doubly degenerate (the two travelling
    directions). This checks that Beyn's rank-revealing SVD resolves exact
    degeneracies.

``commensurate``
    The same ring, but reporting *which* of those modes the solver actually
    returns. The secular matrix ``L(k) = B^T W^{-1} B`` has per-edge weights
    ``k_e / (exp(2 i k_e l_e) - 1)``, which blow up whenever ``k_e l_e`` is a
    multiple of ``pi``. On an equilateral graph every edge hits that point at
    the same ``k``, and the corresponding modes are invisible to the solver --
    they are eigenfunctions that vanish at every vertex, so no vertex-based
    secular equation can see them. ``create_quantum_graph(noise_level=...)``
    dodges this by perturbing node positions when too many edges share a
    length; this check reports the blind spot with the dodge disabled.

Run from this directory::

    OMP_NUM_THREADS=1 python validate_solver_accuracy.py           # all checks
    OMP_NUM_THREADS=1 python validate_solver_accuracy.py ring      # one check
"""

from __future__ import annotations

import sys

import networkx as nx
import numpy as np

from netsalt.contour import find_modes_contour
from netsalt.physics import (
    dispersion_relation_pump,
    set_dielectric_constant,
    set_dispersion_relation,
)
from netsalt.quantum_graph import (
    create_quantum_graph,
    mode_quality,
    set_total_length,
    update_parameters,
)


def _finish(graph, params, total_length):
    """Apply the standard physics setup and return the configured graph."""
    set_total_length(graph, total_length, inner=True)
    set_dielectric_constant(graph, params)
    set_dispersion_relation(graph, dispersion_relation_pump)
    update_parameters(graph, params)
    return graph


def open_line(n_inner=2, total_length=1.0, n_index=1.5, k_max=40.0):
    """Fabry-Perot: a path graph whose two end edges are vacuum leads."""
    graph = nx.path_graph(n_inner + 2)
    positions = np.array([[i, 0.0] for i in range(n_inner + 2)], dtype=float)
    params = {
        "open_model": "open",
        "c": 1.0,
        "k_a": 0.5 * k_max,
        "gamma_perp": 10.0,
        "n_workers": 1,
        "refraction_params": {
            "method": "uniform",
            "inner_value": n_index,
            "loss": 0.0,
            "outer_value": 1.0,
        },
        "k_min": 1.0,
        "k_max": k_max,
        "alpha_min": 0.0,
        "alpha_max": 1.5,
        "quality_threshold": 1e-8,
    }
    create_quantum_graph(graph, params, positions=positions, noise_level=0.0)
    return _finish(graph, params, total_length)


def closed_ring(n_nodes=8, total_length=1.0, n_index=1.0, k_max=60.0):
    """Closed equilateral cycle graph."""
    graph = nx.cycle_graph(n_nodes)
    theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    positions = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    params = {
        "open_model": "closed",
        "c": 1.0,
        "k_a": 0.5 * k_max,
        "gamma_perp": 10.0,
        "n_workers": 1,
        "refraction_params": {
            "method": "uniform",
            "inner_value": n_index,
            "loss": 0.0,
            "outer_value": 1.0,
        },
        "k_min": 1.0,
        "k_max": k_max,
        "alpha_min": -0.5,
        "alpha_max": 0.5,
        "quality_threshold": 1e-8,
    }
    create_quantum_graph(graph, params, positions=positions, noise_level=0.0)
    return _finish(graph, params, total_length)


def check_fabry_perot():
    n_index, total_length, k_max = 1.5, 1.0, 40.0
    graph = open_line(n_index=n_index, total_length=total_length, k_max=k_max)
    modes = find_modes_contour(graph, n_k=12, n_alpha=1, n_quad=120)

    r = (n_index - 1.0) / (n_index + 1.0)
    alpha_exact = np.log(1.0 / abs(r)) / (n_index * total_length)
    k_exact = np.arange(1, 60) * np.pi / (n_index * total_length)
    k_exact = k_exact[(k_exact > 1.0) & (k_exact < k_max)]

    print("== Fabry-Perot (open line, analytic) ==")
    print(f"   exact alpha            {alpha_exact:.12f}  (same for every mode)")
    print(f"   modes found / expected {len(modes)} / {len(k_exact)}")
    err_k, err_a = [], []
    for k in k_exact:
        j = int(np.argmin(abs(modes[:, 0] - k)))
        err_k.append(abs(modes[j, 0] - k))
        err_a.append(abs(modes[j, 1] - alpha_exact))
    print(f"   max |dk|               {max(err_k):.3e}")
    print(f"   max |d alpha|          {max(err_a):.3e}")


def check_ring():
    n_index, total_length, k_max = 1.0, 1.0, 60.0
    graph = closed_ring(n_index=n_index, total_length=total_length, k_max=k_max)
    modes = find_modes_contour(graph, n_k=15, n_alpha=1, n_quad=120)
    k_exact = np.arange(1, 20) * 2 * np.pi / (n_index * total_length)
    k_exact = k_exact[k_exact < k_max]

    found = np.sort(modes[:, 0])
    print("== Closed ring (analytic, doubly degenerate) ==")
    print(f"   exact k                {np.round(k_exact, 4)}")
    print(f"   distinct k found       {np.round(np.unique(np.round(found, 6)), 4)}")
    print(f"   max |alpha| (should be 0) {np.max(abs(modes[:, 1])):.3e}")


def check_commensurate():
    """Report the blind spot at k * l_edge in pi * Z on an equilateral graph."""
    n_nodes, n_index, total_length, k_max = 8, 1.0, 1.0, 60.0
    edge_length = total_length / n_nodes
    graph = closed_ring(
        n_nodes=n_nodes, n_index=n_index, total_length=total_length, k_max=k_max
    )
    k_exact = np.arange(1, 20) * 2 * np.pi / (n_index * total_length)
    k_exact = k_exact[k_exact < k_max]

    print("== Commensurate-length blind spot ==")
    print(f"   equilateral ring, edge length l = {edge_length}")
    print("   k          k*l/pi   |lambda_1(L(k))|   visible?")
    missing = []
    for k in k_exact:
        quality = mode_quality([k, 0.0], graph)
        ratio = k * n_index * edge_length / np.pi
        visible = quality < 1e-6
        if not visible:
            missing.append(k)
        print(f"   {k:8.4f}   {ratio:6.3f}   {quality:16.3e}   {'yes' if visible else 'NO'}")
    if missing:
        print(
            f"   -> {len(missing)} exact mode(s) invisible to the secular equation, "
            "all at integer k*l/pi."
        )
        print(
            "   -> create_quantum_graph(noise_level>0) perturbs the geometry to break "
            "this;\n      examples/buffon/_base.yaml sets noise_level: 0.0, which disables it."
        )


CHECKS = {
    "fabry_perot": check_fabry_perot,
    "ring": check_ring,
    "commensurate": check_commensurate,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(CHECKS)
    for name in names:
        CHECKS[name]()
        print()

"""Convergence probe for the modal-intensity solvers.

A research result is only usable if it is invariant under knobs that carry no
physics. This script varies exactly those knobs and reports how much the
answer moves:

* ``D0_steps`` -- the pump-continuation grid. The steady state at a given pump
  is a property of the pump, not of how many pumps you visited on the way, so
  the L--I curves must be grid-independent.
* ``oversample_resolution`` -- how finely the within-edge standing wave is
  sampled for the spatial-hole-burning denominator. This is a discretisation
  of a continuum quantity, so the answer must *converge* as it is refined.

The ``linear`` (near-threshold, competition-matrix) solver is always run as
the reference. The operator-level ``full_salt_newton`` solver only exists on
the full-SALT branch; the script reports its absence and stops rather than
failing, so it can be run on either branch.

The test cavity is a 14-node ring with six fixed chords -- the buffon
mechanism shrunk to something that runs in minutes. The chord list is
hard-coded so the spectrum is reproducible.

Run from this directory::

    OMP_NUM_THREADS=1 python probe_full_salt_convergence.py             # D0_max = 2x threshold
    OMP_NUM_THREADS=1 python probe_full_salt_convergence.py 0.5         # deep above threshold
"""

from __future__ import annotations

import sys
import warnings

import networkx as nx
import numpy as np

import netsalt
from netsalt.modes import (
    compute_modal_intensities,
    compute_mode_competition_matrix,
    find_passive_modes,
    find_threshold_lasing_modes,
    pump_trajectories,
)
from netsalt.physics import dispersion_relation_pump
from netsalt.quantum_graph import create_quantum_graph, set_total_length

PARAMS = {
    "open_model": "open",
    "c": 1.0,
    "k_a": 2.81,
    "gamma_perp": 0.10,
    "k_min": 2.55,
    "k_max": 3.25,
    "alpha_min": -0.05,
    "alpha_max": 0.25,
    "n_workers": 1,
    "n_modes_max": 40,
    "quality_threshold": 1e-3,
    "search_stepsize": 0.005,
    "max_steps": 1000,
    "max_tries_reduction": 50,
    "reduction_factor": 0.8,
    "D0_max": 0.5,
    "D0_steps": 14,
    "dielectric_params": {
        "method": "uniform",
        "inner_value": 9.0,
        "outer_value": 1.0,
        "loss": 0.0,
    },
}
CHORDS = [(0, 7), (1, 5), (2, 12), (3, 5), (3, 10), (7, 11)]


def chaotic_ring(n_ring=14, total_length=12.0):
    """Single ring plus fixed chords, with two output leads."""
    graph = nx.cycle_graph(n_ring)
    graph.add_edges_from(CHORDS)
    graph.add_edge(0, n_ring)
    graph.add_edge(n_ring // 2, n_ring + 1)
    pos = {
        i: [np.cos(2 * np.pi * i / n_ring), np.sin(2 * np.pi * i / n_ring)]
        for i in range(n_ring)
    }
    pos[n_ring] = [1.6, 0.0]
    pos[n_ring + 1] = [-1.6, 0.0]
    positions = np.array([pos[i] for i in range(len(graph))])
    create_quantum_graph(graph, dict(PARAMS), positions=positions)
    set_total_length(graph, total_length)
    netsalt.set_dielectric_constant(graph, graph.graph["params"])
    netsalt.set_dispersion_relation(graph, dispersion_relation_pump)
    return graph


def intensity_curves(modes_df):
    """Return ``(pumps, intensities)`` sorted by pump."""
    cols = sorted(
        (c for c in modes_df.columns if isinstance(c, tuple) and c[0] == "modal_intensities"),
        key=lambda c: c[1],
    )
    pumps = np.array([c[1] for c in cols])
    data = np.nan_to_num(modes_df[cols].to_numpy(dtype=float))
    return pumps, data


def _report(label, intensities, n_warnings=None):
    order = np.argsort(intensities)[::-1]
    warn = "" if n_warnings is None else f"  non-convergence warnings={n_warnings}"
    print(
        f"   {label:34s} lasing={int(np.sum(intensities > 1e-8)):2d}  "
        f"total={intensities.sum():10.4f}  "
        f"top={np.round(intensities[order][:4], 4)}{warn}"
    )


def main(d0_max):
    graph = chaotic_ring()
    passive = find_passive_modes(graph, method="contour")
    graph.graph["params"]["pump"] = np.array(
        [1.0 if graph[u][v]["inner"] else 0.0 for u, v in graph.edges()]
    )
    trajectories = pump_trajectories(passive, graph, return_approx=True)
    threshold_df = find_threshold_lasing_modes(trajectories, graph)
    thresholds = np.asarray(threshold_df["lasing_thresholds"]).ravel()
    finite = thresholds[thresholds < np.inf]
    print(f"passive modes {len(passive)}, with a finite threshold {len(finite)}")
    print(f"thresholds {np.round(np.sort(finite), 4)}")
    print(f"D0_max = {d0_max} ({d0_max / finite.min():.1f}x the lowest threshold)\n")

    competition = compute_mode_competition_matrix(graph, threshold_df.copy())
    _, linear = intensity_curves(
        compute_modal_intensities(threshold_df.copy(), d0_max, competition)
    )
    _report("linear (reference)", linear[:, -1])

    try:
        from netsalt.modes import compute_modal_intensities_full_salt_newton
    except ImportError:
        print(
            "\nfull_salt_newton is not available on this branch -- run this script on the "
            "full-SALT branch to probe it."
        )
        return

    print("\n-- pump-grid independence (must not move) --")
    for steps in (5, 10, 20, 40):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            modes_df = compute_modal_intensities_full_salt_newton(
                graph, threshold_df.copy(), d0_max, D0_steps=steps, seed=42
            )
            n_warn = sum("full_salt_newton" in str(c.message) for c in caught)
        _, data = intensity_curves(modes_df)
        _report(f"newton D0_steps={steps}", data[:, -1], n_warn)

    print("\n-- within-edge resolution (must converge) --")
    for resolution in (6, 12, 20, 32):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modes_df = compute_modal_intensities_full_salt_newton(
                graph,
                threshold_df.copy(),
                d0_max,
                D0_steps=20,
                seed=42,
                oversample_resolution=resolution,
            )
        _, data = intensity_curves(modes_df)
        _report(f"newton lambda/{resolution}", data[:, -1])


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.05)

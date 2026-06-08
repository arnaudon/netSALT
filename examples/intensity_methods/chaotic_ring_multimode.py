"""Genuine multimode lasing on a *single* small ring with random chords.

``two_ring_multimode.py`` gets several modes to co-lase by joining two detuned
rings: the detuning localises each mode onto one ring, so they barely overlap.
But you do not need a multi-component graph for that -- a **single** ring with a
handful of chords (extra edges across it) already does the job, and on a much
smaller graph. This is the mechanism behind the multimode buffon networks, shrunk
to 14 nodes.

Why chords give multimode lasing:

* Each chord closes a new loop, so the cavity now supports many interfering
  path-length combinations -> a **dense, irregular spectrum** (no clean
  longitudinal comb).
* Those modes are **spatially distinct**: each one concentrates on a different
  subset of loops/chords (see the per-mode *participation ratio* printed below),
  so they burn their spatial holes in different places and leave gain for one
  another -> they co-lase instead of competing winner-take-all.

With a narrow gain centred on a cluster of these modes, ``full_salt_newton`` lases
**four** modes here. The chord layout is fixed (hard-coded, so the result is
reproducible regardless of the NumPy RNG); it came from a small seed scan picking
a graph whose spectrum has a clean four-mode cluster.

Run::

    OMP_NUM_THREADS=1 python chaotic_ring_multimode.py

Modes are found by Beyn's contour method (robust on this hand-built graph). The
four solvers disagree on the count -- that is the physics: ``linear`` has no gain
clamping, the surrogate ``full_salt`` over-clamps through its per-edge-mean hole
burning, and the operator-level ``full_salt_newton`` imposes the exact
self-consistent condition.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import netsalt
from netsalt.modes import (
    compute_modal_intensities,
    compute_modal_intensities_full_salt,
    compute_modal_intensities_full_salt_newton,
    compute_modal_intensities_self_consistent,
    compute_mode_competition_matrix,
    find_passive_modes,
    find_threshold_lasing_modes,
    mode_on_nodes,
    pump_trajectories,
)
from netsalt.physics import dispersion_relation_pump
from netsalt.quantum_graph import create_quantum_graph, set_total_length
from netsalt.utils import from_complex

HERE = Path(__file__).resolve().parent
M = 14  # ring nodes
# Chords across the ring, plus a lead on node 0 and node M // 2 for output
# coupling. Hard-coded (from a seed scan) so the spectrum is reproducible.
CHORDS = [(0, 7), (1, 5), (2, 12), (3, 5), (3, 10), (7, 11)]

# A narrow gain centred on the four-mode cluster near k = 2.81. Mode finding uses
# the contour method, so only k_min/k_max/alpha bounds matter for it.
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
    "dielectric_params": {"method": "uniform", "inner_value": 9.0, "outer_value": 1.0, "loss": 0.0},
}
D0_MAX = 0.5


def build_chaotic_ring():
    """A single ring with random chords and two leads (open cavity)."""
    g = nx.cycle_graph(M)
    g.add_edges_from(CHORDS)
    g.add_edge(0, M)  # lead
    g.add_edge(M // 2, M + 1)  # lead
    pos = {i: [np.cos(2 * np.pi * i / M), np.sin(2 * np.pi * i / M)] for i in range(M)}
    pos[M] = [1.6, 0.0]
    pos[M + 1] = [-1.6, 0.0]
    positions = np.array([pos[i] for i in range(len(g))])
    create_quantum_graph(g, dict(PARAMS), positions=positions)
    set_total_length(g, 12.0)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    netsalt.set_dispersion_relation(g, dispersion_relation_pump)
    return g


def _participation(mode, graph):
    """Effective number of nodes the mode lives on (1 / sum p_i^2)."""
    weight = np.abs(mode_on_nodes(from_complex(mode), graph, check_quality=False)) ** 2
    prob = weight / weight.sum()
    return 1.0 / np.sum(prob**2)


def main():
    graph = build_chaotic_ring()
    passive = find_passive_modes(graph, method="contour")
    if len(passive) == 0:
        raise SystemExit("no passive modes found")
    graph.graph["params"]["pump"] = np.array(
        [1.0 if graph[u][v]["inner"] else 0.0 for u, v in graph.edges()]
    )
    trajectories = pump_trajectories(passive, graph, return_approx=True)
    tdf = find_threshold_lasing_modes(trajectories, graph)
    thr = np.asarray(tdf["lasing_thresholds"]).ravel()
    threshold_modes = tdf["threshold_lasing_modes"].to_numpy()
    print("modes below the max pump (id, k, threshold, participation):")
    for i in range(len(tdf)):
        if thr[i] < D0_MAX:
            k = from_complex(threshold_modes[i])[0]
            print(
                f"  {i}: k={k:.3f}  thr={thr[i]:.3f}  participation={_participation(threshold_modes[i], graph):.1f}"
            )

    competition = compute_mode_competition_matrix(graph, tdf)
    solvers = {
        "linear": compute_modal_intensities(tdf.copy(), D0_MAX, competition),
        "self_consistent": compute_modal_intensities_self_consistent(
            graph, tdf.copy(), D0_MAX, D0_steps=12
        ),
        "full_salt": compute_modal_intensities_full_salt(graph, tdf.copy(), D0_MAX, D0_steps=12),
        "full_salt_newton": compute_modal_intensities_full_salt_newton(
            graph, tdf.copy(), D0_MAX, D0_steps=16
        ),
    }
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, (name, df) in zip(axes.ravel(), solvers.items(), strict=True):
        cols = np.array(
            sorted(c[1] for c in df.columns if isinstance(c, tuple) and c[0] == "modal_intensities")
        )
        data = np.nan_to_num(df[[("modal_intensities", c) for c in cols]].to_numpy(dtype=float))
        peak = max(data.max(), 1e-9)
        active = [m for m in range(data.shape[0]) if data[m].max() > 1e-2 * peak]
        for m in active:
            ax.plot(cols, data[m], ".-", color=cmap(m % 10), label=f"mode {m}")
        ax.set_title(f"{name}  ({len(active)} lasing)")
        ax.set_ylabel("modal intensity")
        if active:
            ax.legend(fontsize=7, ncol=2)
        print(
            f"{name}: {len(active)} lasing @max "
            + str({int(m): round(float(data[m, -1]), 3) for m in active})
        )
    for ax in axes[1]:
        ax.set_xlabel("pump $D_0$")
    fig.suptitle("Single ring + random chords: genuine multimode lasing", y=1.0)
    fig.tight_layout()
    out = HERE / "chaotic_ring_multimode.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

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

**Which solvers to trust here.** This deep-multimode regime (4--5 strongly
clustered thresholds) is exactly where the cheap solvers part ways:

* ``linear`` -- exact near-threshold model, gives clean piecewise-linear L--I
  curves, but has **no gain clamping** (it never asks whether a lasing mode still
  has net gain once the others saturate it), so its count can err either way --
  here it lases *one fewer* than newton, because its frozen-threshold competition
  matrix over-estimates how strongly the cluster suppresses the fourth mode.
* ``self_consistent`` / ``full_salt`` -- the event-driven sweep with a
  per-pump-rebuilt competition matrix becomes **numerically erratic** with this
  many strongly-competing modes (intensities go non-monotone, modes flick on and
  off). They are reliable near threshold / on weakly-multimode graphs (see
  ``compare_intensity_methods.py``), not here.
* ``full_salt_newton`` -- the operator-level solve stays smooth and physical and
  imposes the exact self-consistent gain clamping.

So this script plots only the two solvers that are sensible in this regime --
``linear`` (reference) and ``full_salt_newton`` (faithful) -- alongside the graph
geometry. It still *runs* the surrogate solvers and prints their endpoint counts
so you can see them disagree.

Run::

    OMP_NUM_THREADS=1 python chaotic_ring_multimode.py

Modes are found by Beyn's contour method (robust on this hand-built graph).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

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
D0_STEPS = 14


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


def _draw_geometry(ax, graph):
    """Draw the ring/chord/lead geometry in the plane."""
    pos = {n: graph.nodes[n]["position"] for n in graph.nodes}
    chord_set = {tuple(sorted(c)) for c in CHORDS}
    for u, v in graph.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        su, sv = min(u, v), max(u, v)
        if sv >= M:  # lead edge (nodes M, M + 1)
            ax.plot(x, y, color="crimson", lw=2.0, ls="--", zorder=1)
        elif (su, sv) in chord_set:
            ax.plot(x, y, color="royalblue", lw=2.2, zorder=2)
        else:  # ring edge
            ax.plot(x, y, color="0.6", lw=2.2, zorder=1)
    for n in graph.nodes:
        ax.scatter(*pos[n], s=220 if n < M else 160, color="white", edgecolor="black", zorder=3)
        ax.text(pos[n][0], pos[n][1], str(n), ha="center", va="center", fontsize=7, zorder=4)
    ax.legend(
        handles=[
            Line2D([0], [0], color="0.6", lw=2.2, label="ring edge"),
            Line2D([0], [0], color="royalblue", lw=2.2, label="random chord"),
            Line2D([0], [0], color="crimson", lw=2.0, ls="--", label="output lead"),
        ],
        loc="upper right",
        fontsize=7,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{M}-node ring + {len(CHORDS)} chords + 2 leads")


def _endpoint(df):
    """Modal intensities at the largest pump of an L--I dataframe."""
    cols = sorted(c[1] for c in df.columns if isinstance(c, tuple) and c[0] == "modal_intensities")
    return np.nan_to_num(df[("modal_intensities", cols[-1])].to_numpy(dtype=float))


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
            part = _participation(threshold_modes[i], graph)
            print(f"  {i}: k={k:.3f}  thr={thr[i]:.3f}  participation={part:.1f}")

    competition = compute_mode_competition_matrix(graph, tdf)

    # Sample linear on the same uniform pump grid newton uses: the event-driven
    # sweep otherwise only stores points at mode thresholds, which here all
    # cluster near 0.02 and collapse to a 2-point grid. linear is exact between
    # events, so endpoint-sampling a uniform grid simply draws the true line.
    first = float(thr[thr < np.inf].min())
    grid = np.linspace(first, D0_MAX, D0_STEPS)
    linear = np.zeros((len(tdf), grid.size))
    for j, d0 in enumerate(grid):
        linear[:, j] = _endpoint(compute_modal_intensities(tdf.copy(), d0, competition))

    newton_df = compute_modal_intensities_full_salt_newton(
        graph, tdf.copy(), D0_MAX, D0_steps=D0_STEPS
    )
    n_cols = np.array(
        sorted(
            c[1] for c in newton_df.columns if isinstance(c, tuple) and c[0] == "modal_intensities"
        )
    )
    newton = np.nan_to_num(
        newton_df[[("modal_intensities", c) for c in n_cols]].to_numpy(dtype=float)
    )

    # Also run the surrogate sweeps once, only to report their (unreliable) counts.
    sc = _endpoint(
        compute_modal_intensities_self_consistent(graph, tdf.copy(), D0_MAX, D0_steps=12)
    )
    fs = _endpoint(compute_modal_intensities_full_salt(graph, tdf.copy(), D0_MAX, D0_steps=12))

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    _draw_geometry(axes[0], graph)
    for ax, name, xs, data in (
        (axes[1], "linear (reference, no clamping)", grid, linear),
        (axes[2], "full_salt_newton (faithful)", n_cols, newton),
    ):
        peak = max(data.max(), 1e-9)
        active = [m for m in range(data.shape[0]) if data[m].max() > 1e-2 * peak]
        for m in active:
            ax.plot(xs, data[m], ".-", color=cmap(m % 10), label=f"mode {m}")
        ax.set_title(f"{name}  ({len(active)} lasing)")
        ax.set_xlabel("pump $D_0$")
        ax.set_ylabel("modal intensity")
        if active:
            ax.legend(fontsize=8)

    fig.suptitle("Single ring + random chords: genuine multimode lasing", y=1.02)
    fig.tight_layout()
    out = HERE / "chaotic_ring_multimode.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    def _count(arr):
        return int(np.sum(arr > 1e-2 * max(arr.max(), 1e-9)))

    print(f"linear: {_count(linear[:, -1])} lasing @max")
    print(f"full_salt_newton: {_count(newton[:, -1])} lasing @max")
    print(
        f"self_consistent: {_count(sc)} / full_salt: {_count(fs)} @max "
        "(event-sweep surrogates -- erratic in this deep-multimode regime, not plotted)"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

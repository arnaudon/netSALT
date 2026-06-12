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
* ``full_salt_newton`` -- the operator-level solve stays smooth and physical and
  imposes the exact self-consistent gain clamping.

So this script plots ``linear`` (dashed) and ``full_salt_newton`` (solid) in
three panels: the graph geometry, the full-range L--I, and a **zoom on the
onset** (the four thresholds sit in ``0.016--0.026``, marked by dotted lines).
The zoom shows the modes switching on in turn, and in particular that newton
turns the fourth mode on *well above* its bare threshold -- gain clamping delays
it until enough pump is present, which is exactly what ``linear`` (no clamping)
misses.

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
    compute_modal_intensities_full_salt_newton,
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
D0_STEPS = 22  # pump points over the full range
D0_ZOOM = 0.08  # onset window (the four thresholds sit in 0.016--0.026)
D0_STEPS_ZOOM = 22  # pump points within the zoom (fine, to resolve each turn-on)


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


def _curves(df):
    """(pumps, intensities[n_modes, n_pumps]) from an L--I dataframe."""
    cols = np.array(
        sorted(c[1] for c in df.columns if isinstance(c, tuple) and c[0] == "modal_intensities")
    )
    data = np.nan_to_num(df[[("modal_intensities", c) for c in cols]].to_numpy(dtype=float))
    return cols, data


def _linear_on_grid(tdf, competition, grid, n_modes):
    """Sample the (exact, piecewise-linear) linear model on a uniform pump grid."""
    out = np.zeros((n_modes, grid.size))
    for j, d0 in enumerate(grid):
        out[:, j] = _endpoint(compute_modal_intensities(tdf.copy(), d0, competition))
    return out


def _plot_li(ax, grid_lin, linear, n_cols, newton, thr, cmap, xmax=None):
    """Overlay linear (dashed) and newton (solid) L--I curves, coloured by mode id."""
    n_modes = newton.shape[0]
    active = [m for m in range(n_modes) if max(linear[m].max(), newton[m].max()) > 1e-3]
    for m in active:
        col = cmap(m % 10)
        if thr[m] < (xmax if xmax is not None else np.inf):
            ax.axvline(thr[m], color=col, ls=":", lw=0.8, alpha=0.6)
        ax.plot(grid_lin, linear[m], "--", color=col, lw=1.3, alpha=0.8)
        # markers show newton's discrete per-pump equilibria
        ax.plot(n_cols, newton[m], ".-", color=col, lw=1.8, ms=5, label=f"mode {m}")
    if xmax is not None:
        ax.set_xlim(0, xmax)
        top = max(
            (newton[m][n_cols <= xmax].max() if np.any(n_cols <= xmax) else 0) for m in active
        )
        ax.set_ylim(0, 1.15 * max(top, 1e-9))
    ax.set_xlabel("pump $D_0$")
    ax.set_ylabel("modal intensity")
    if active:
        ax.legend(fontsize=8)


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
    n_modes = len(tdf)
    first = float(thr[thr < np.inf].min())

    # Full range. Sample linear on the same uniform grid newton uses: the
    # event-driven sweep otherwise only stores points at mode thresholds, which
    # here all cluster near 0.02 and collapse to a 2-point grid. linear is exact
    # between events, so endpoint-sampling a uniform grid simply draws the line.
    grid = np.linspace(first, D0_MAX, D0_STEPS)
    linear = _linear_on_grid(tdf, competition, grid, n_modes)
    n_cols, newton = _curves(
        compute_modal_intensities_full_salt_newton(graph, tdf.copy(), D0_MAX, D0_steps=D0_STEPS)
    )

    # Zoom on the onset: the four thresholds sit in 0.016--0.026, so re-sample a
    # fine grid over a small pump window (a newton run with a small max_pump is
    # just a dense linspace there) to see each mode switch on in turn.
    grid_z = np.linspace(first, D0_ZOOM, D0_STEPS_ZOOM)
    linear_z = _linear_on_grid(tdf, competition, grid_z, n_modes)
    n_cols_z, newton_z = _curves(
        compute_modal_intensities_full_salt_newton(
            graph, tdf.copy(), D0_ZOOM, D0_steps=D0_STEPS_ZOOM
        )
    )

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    _draw_geometry(axes[0], graph)
    _plot_li(axes[1], grid, linear, n_cols, newton, thr, cmap)
    axes[1].set_title("L--I, full range (dashed = linear, solid = newton)")
    _plot_li(axes[2], grid_z, linear_z, n_cols_z, newton_z, thr, cmap, xmax=D0_ZOOM)
    axes[2].set_title("zoom on onset (dotted = thresholds)")

    fig.suptitle("Single ring + random chords: genuine multimode lasing", y=1.02)
    fig.tight_layout()
    out = HERE / "chaotic_ring_multimode.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    def _count(arr):
        return int(np.sum(arr > 1e-2 * max(arr.max(), 1e-9)))

    print(f"linear: {_count(linear[:, -1])} lasing @max")
    print(f"full_salt_newton: {_count(newton[:, -1])} lasing @max")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

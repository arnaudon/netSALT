"""Newton vs linear on a *denser* chord ring: consistency of the L--I curves.

A companion to ``../chaotic_ring/run.py`` that checks the operator-level
``full_salt_newton`` solver against the near-threshold ``linear`` model on a
bigger, more strongly-competing graph: a **16-node ring with 10 random chords**
(vs 14 nodes / 6 chords). The question this answers is *what should differ between
the two solvers, and why*:

* Both are built to share the same onset slope ``1/(T_μμ·D0_thr)`` at each mode's
  threshold, so just above threshold they **coincide**.
* ``linear`` then freezes every mode's profile at its own threshold and uses a
  fixed competition matrix ``T`` -> strictly piecewise-linear curves.
* ``full_salt_newton`` re-solves each lasing mode's ``(k_μ, a_μ)`` *and spatial
  profile* at the operating pump, under the shared saturated (hole-burnt)
  operator. As the pump rises the holes deepen and the profiles/overlaps shift, so
  the effective competition changes.

The visible consequences, printed and plotted below:

* the **dominant** mode tracks the linear curve closely (same onset, near-equal
  initial slope), then its slope drifts as saturation sets in;
* the **secondary** modes are *reshuffled* -- they switch on at different pumps and
  reach different intensities than the frozen-profile linear model predicts (here
  one secondary lights up earlier and stronger, another later and largely
  suppressed). That reshuffling is the operator-level mode interaction the linear
  model cannot see.

The chord layout is hard-coded (from a small seed scan) so the result is
reproducible regardless of the NumPy RNG.

Run::

    OMP_NUM_THREADS=1 python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import mode_profile_figure

import netsalt
from netsalt.modes import (
    compute_modal_intensities,
    compute_modal_intensities_full_salt_newton,
    compute_mode_competition_matrix,
    find_passive_modes,
    find_threshold_lasing_modes,
    pump_trajectories,
)
from netsalt.physics import dispersion_relation_pump
from netsalt.quantum_graph import create_quantum_graph, set_total_length
from netsalt.utils import from_complex

HERE = Path(__file__).resolve().parent
M = 16  # ring nodes
# 10 chords across the ring + a lead on node 0 and node M // 2 (hard-coded so the
# spectrum is reproducible regardless of the NumPy RNG).
CHORDS = [(0, 13), (1, 14), (2, 10), (3, 8), (4, 9), (6, 13), (7, 15), (9, 12), (10, 15), (11, 15)]

PARAMS = {
    "open_model": "open",
    "c": 1.0,
    "k_a": 2.9,
    "gamma_perp": 0.22,
    "k_min": 2.55,
    "k_max": 3.25,
    "alpha_min": -0.05,
    "alpha_max": 0.25,
    "n_workers": 1,
    "n_modes_max": 50,
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
D0_STEPS = 24


def build_dense_ring():
    """A 16-node ring with 10 chords and two leads (open cavity)."""
    g = nx.cycle_graph(M)
    g.add_edges_from(CHORDS)
    g.add_edge(0, M)  # lead
    g.add_edge(M // 2, M + 1)  # lead
    pos = {i: [np.cos(2 * np.pi * i / M), np.sin(2 * np.pi * i / M)] for i in range(M)}
    pos[M] = [1.7, 0.0]
    pos[M + 1] = [-1.7, 0.0]
    positions = np.array([pos[i] for i in range(len(g))])
    create_quantum_graph(g, dict(PARAMS), positions=positions)
    set_total_length(g, 12.0)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    netsalt.set_dispersion_relation(g, dispersion_relation_pump)
    return g


def _endpoint(df):
    cols = sorted(c[1] for c in df.columns if isinstance(c, tuple) and c[0] == "modal_intensities")
    return np.nan_to_num(df[("modal_intensities", cols[-1])].to_numpy(dtype=float))


def _curves(df):
    cols = np.array(
        sorted(c[1] for c in df.columns if isinstance(c, tuple) and c[0] == "modal_intensities")
    )
    return cols, np.nan_to_num(df[[("modal_intensities", c) for c in cols]].to_numpy(dtype=float))


def _onset_slope(pumps, y):
    """(onset pump, near-threshold slope) -- slope fit just above the turn-on."""
    idx = np.where(y > 1e-3)[0]
    if idx.size < 3:
        return np.nan, np.nan
    seg = idx[1 : min(idx.size, 5)]  # skip the activation point itself
    slope = np.polyfit(pumps[seg], y[seg], 1)[0]
    return float(pumps[idx[0]]), float(slope)


def _draw_geometry(ax, graph):
    pos = {n: graph.nodes[n]["position"] for n in graph.nodes}
    chord_set = {tuple(sorted(c)) for c in CHORDS}
    for u, v in graph.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        su, sv = min(u, v), max(u, v)
        if sv >= M:
            ax.plot(x, y, color="crimson", lw=2.0, ls="--", zorder=1)
        elif (su, sv) in chord_set:
            ax.plot(x, y, color="royalblue", lw=2.0, zorder=2)
        else:
            ax.plot(x, y, color="0.6", lw=2.0, zorder=1)
    for n in graph.nodes:
        ax.scatter(*pos[n], s=130 if n < M else 100, color="white", edgecolor="black", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{M}-node ring + {len(CHORDS)} chords")


def main():
    graph = build_dense_ring()
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
    n_modes = len(tdf)

    competition = compute_mode_competition_matrix(graph, tdf)
    first = float(thr[thr < np.inf].min())
    grid = np.linspace(first, D0_MAX, D0_STEPS)
    linear = np.zeros((n_modes, grid.size))
    for j, d0 in enumerate(grid):
        linear[:, j] = _endpoint(compute_modal_intensities(tdf.copy(), d0, competition))
    n_cols, newton = _curves(
        compute_modal_intensities_full_salt_newton(graph, tdf.copy(), D0_MAX, D0_steps=D0_STEPS)
    )

    peak = max(linear.max(), newton.max(), 1e-9)
    active = [m for m in range(n_modes) if max(linear[m].max(), newton[m].max()) > 1e-2 * peak]
    print("mode   k      thr     onset(lin/nwt)   slope(lin/nwt)    I@max(lin/nwt)")
    for m in active:
        k = from_complex(threshold_modes[m])[0]
        on_l, sl_l = _onset_slope(grid, linear[m])
        on_n, sl_n = _onset_slope(n_cols, newton[m])
        print(
            f"{m:>3}  {k:.3f}  {thr[m]:.3f}   {on_l:.3f} / {on_n:.3f}   "
            f"{sl_l:7.1f} / {sl_n:7.1f}   {linear[m, -1]:7.2f} / {newton[m, -1]:7.2f}"
        )

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _draw_geometry(axes[0], graph)
    for m in active:
        col = cmap(m % 10)
        axes[1].plot(grid, linear[m], "--", color=col, lw=1.3, alpha=0.8)
        axes[1].plot(n_cols, newton[m], ".-", color=col, lw=1.8, ms=4, label=f"mode {m}")
    axes[1].set_xlabel("pump $D_0$")
    axes[1].set_ylabel("modal intensity")
    axes[1].set_title("dashed = linear, solid = newton")
    axes[1].legend(fontsize=8)
    fig.suptitle("Denser chord ring: full_salt_newton vs linear", y=1.0)
    fig.tight_layout()
    cut = 1e-2 * peak
    mode_profile_figure(
        graph,
        tdf,
        D0_MAX,
        [m for m in range(n_modes) if linear[m, -1] > cut],
        [m for m in range(n_modes) if newton[m, -1] > cut],
        "dense chord ring",
        HERE,
        a0={m: float(newton[m, -1]) for m in range(n_modes)},
    )

    out = HERE / "dense_ring_compare.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

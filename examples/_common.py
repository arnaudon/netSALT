"""Shared machinery for the script-based per-graph examples.

The per-graph example folders (``line_fabry_perot/``, ``ring_leads/``,
``tree/``, ``two_ring/``, ``chaotic_ring/``, ``dense_ring/``) each hold a
``run.py`` that builds one graph in memory, runs the two modal-intensity
solvers (``linear`` and ``full_salt_newton``) and writes its figures *into
that folder* (figures are gitignored -- re-run the script to reproduce them).
This module carries what they share: the open-cavity parameter set, the
pipeline runner, curve extraction, and the standard geometry/per-mode/total
comparison figure used by the three simple cavities.
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
    compute_modal_intensities_full_salt_newton,
    compute_mode_competition_matrix,
    find_passive_modes,
    find_threshold_lasing_modes,
    pump_trajectories,
)
from netsalt.physics import dispersion_relation_pump
from netsalt.quantum_graph import create_quantum_graph, set_total_length

# Shared physics / search settings for the three simple open cavities (a gain
# line centred at ``k_a`` and a scan window straddling it). Kept small so each
# run.py finishes in about a minute on one core.
OPEN_CAVITY_PARAMS = {
    "open_model": "open",
    "c": 1.0,
    "k_a": 15.0,
    "gamma_perp": 3.0,
    "k_min": 12.0,
    "k_max": 18.0,
    "k_n": 100,
    "alpha_min": 0.0,
    "alpha_max": 1.0,
    "alpha_n": 30,
    "quality_threshold": 1.0e-4,
    "search_stepsize": 0.01,
    "max_steps": 10000,
    "max_tries_reduction": 50,
    "reduction_factor": 0.8,
    "n_workers": 1,
    "D0_max": 1.4,
    "D0_steps": 10,
    "salt_D0_steps": 10,
    "dielectric_params": {
        "method": "uniform",
        "inner_value": 9.0,  # epsilon = n^2, n = 3 inside the cavity
        "outer_value": 1.0,  # the leads are vacuum
        "loss": 0.0,
    },
}
D0_MAX = 1.4  # max pump for the simple-cavity L--I sweeps


def quantum_graph(nx_graph, positions, total_length, **overrides):
    """Wrap a networkx graph as a pumped, open netSALT quantum graph.

    ``overrides`` patch the shared :data:`OPEN_CAVITY_PARAMS`.
    """
    g = nx.convert_node_labels_to_integers(nx_graph)
    params = dict(OPEN_CAVITY_PARAMS)
    params.update(overrides)
    create_quantum_graph(g, params, positions=positions)
    set_total_length(g, total_length)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    netsalt.set_dispersion_relation(g, dispersion_relation_pump)
    return g


def threshold_modes(graph, method="grid"):
    """Shared pipeline: passive modes -> pump -> trajectories -> thresholds.

    ``method="grid"`` scans the frequency grid first (the simple cavities);
    ``method="contour"`` uses Beyn's contour search (the ring graphs).
    """
    if method == "grid":
        qualities = netsalt.scan_frequencies(graph)
        passive = netsalt.find_passive_modes(
            graph, qualities, method="grid", min_distance=2, threshold_abs=0.1
        )
    else:
        passive = find_passive_modes(graph, method="contour")
    if len(passive) == 0:
        raise SystemExit("no passive modes found")
    pump = np.array([1.0 if graph[u][v]["inner"] else 0.0 for u, v in graph.edges()])
    graph.graph["params"]["pump"] = pump
    trajectories = pump_trajectories(passive, graph, return_approx=True)
    return find_threshold_lasing_modes(trajectories, graph)


def curves(df):
    """(pumps, intensities[n_modes, n_pumps]) from an L--I dataframe."""
    cols = np.array(
        sorted(c[1] for c in df.columns if isinstance(c, tuple) and c[0] == "modal_intensities")
    )
    data = np.nan_to_num(df[[("modal_intensities", c) for c in cols]].to_numpy(dtype=float))
    return cols, data


def endpoint(df):
    """Modal intensities at the largest pump of an L--I dataframe."""
    return curves(df)[1][:, -1]


def linear_on_grid(tdf, competition, grid, n_modes):
    """Sample the (exact, piecewise-linear) linear model on a uniform pump grid.

    The event-driven sweep only stores points at activation events; endpoint-
    sampling a uniform grid simply draws the same line at the plot resolution.
    """
    out = np.zeros((n_modes, grid.size))
    for j, d0 in enumerate(grid):
        out[:, j] = endpoint(compute_modal_intensities(tdf.copy(), d0, competition))
    return out


def draw_geometry(ax, graph, name):
    """Cavity (inner) edges grey, leads red-dashed, using the stored positions."""
    pos = {n: np.asarray(graph.nodes[n]["position"], dtype=float) for n in graph.nodes}
    for u, v in graph.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        if graph[u][v]["inner"]:
            ax.plot(x, y, color="0.55", lw=2.2, zorder=1)
        else:
            ax.plot(x, y, color="crimson", lw=2.0, ls="--", zorder=1)
    for n in graph.nodes:
        ax.scatter(*pos[n], s=80, color="white", edgecolor="black", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{name}  ({len(graph)} nodes)")


def compare_and_plot(graph, name, outdir, d0_max=D0_MAX, d0_steps=26, passive_method="grid"):
    """Run linear + newton on ``graph`` and write the standard 3-panel figure.

    Panels: geometry | per-mode L--I (linear dashed, newton solid, colours keyed
    by mode) | total L--I for both solvers. Writes ``<outdir>/li_curves.png``
    and prints the lasing counts. Both solvers share the linear modal-intensity
    unit (the newton amplitude reduces to it analytically at threshold), so the
    curves overlay directly.
    """
    import time

    outdir = Path(outdir)
    tdf = threshold_modes(graph, method=passive_method)
    n_modes = len(tdf)
    competition = compute_mode_competition_matrix(graph, tdf)
    thr = np.asarray(tdf["lasing_thresholds"]).ravel()
    first = float(thr[thr < np.inf].min())
    grid = np.linspace(first, d0_max, d0_steps)
    t0 = time.perf_counter()
    linear = linear_on_grid(tdf, competition, grid, n_modes)
    t_linear = time.perf_counter() - t0
    t0 = time.perf_counter()
    n_cols, newton = curves(
        compute_modal_intensities_full_salt_newton(graph, tdf.copy(), d0_max, D0_steps=d0_steps)
    )
    t_newton = time.perf_counter() - t0
    print(f"{name}: linear sweep {t_linear:.1f}s, newton sweep {t_newton:.1f}s")

    peak = max(linear.max(), newton.max(), 1e-9)
    active = [m for m in range(n_modes) if max(linear[m].max(), newton[m].max()) > 1e-2 * peak]
    n_lin = int(np.sum(linear[:, -1] > 1e-2 * peak))
    n_nwt = int(np.sum(newton[:, -1] > 1e-2 * peak))
    print(f"{name}: linear lases {n_lin}, full_salt_newton lases {n_nwt}")

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    draw_geometry(axes[0], graph, name)
    for m in active:
        col = cmap(m % 10)
        axes[1].plot(grid, linear[m], "--", color=col, lw=1.3, alpha=0.8)
        axes[1].plot(n_cols, newton[m], ".-", color=col, lw=1.8, ms=4, label=f"mode {m}")
    axes[1].set_xlabel("pump $D_0$")
    axes[1].set_ylabel("modal intensity")
    axes[1].set_title(f"per mode: dashed = linear ({n_lin}), solid = newton ({n_nwt})")
    if active:
        axes[1].legend(fontsize=8, ncol=2)
    axes[2].plot(grid, linear.sum(axis=0), "o-", ms=3, color="tab:blue", label="linear")
    axes[2].plot(n_cols, newton.sum(axis=0), "o-", ms=3, color="tab:red", label="full_salt_newton")
    axes[2].set_xlabel("pump $D_0$")
    axes[2].set_ylabel("total modal intensity")
    axes[2].set_title("total")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{name}: full_salt_newton vs linear", y=1.02)
    fig.tight_layout()
    out = outdir / "li_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    ids_linear = [m for m in range(n_modes) if linear[m, -1] > 1e-2 * peak]
    ids_newton = [m for m in range(n_modes) if newton[m, -1] > 1e-2 * peak]
    mode_profile_figure(
        graph,
        tdf,
        d0_max,
        ids_linear,
        ids_newton,
        name,
        outdir,
        a0={m: newton[m, -1] for m in ids_newton},
    )


def _draw_profile(ax, work, values, cmap, vmin, vmax):
    """Edges of ``work`` coloured by the per-edge ``values`` (intensity map)."""
    from matplotlib.collections import LineCollection

    pos = {n: np.asarray(work.nodes[n]["position"], dtype=float) for n in work.nodes}
    segs = [(pos[u], pos[v]) for u, v in work.edges]
    lc = LineCollection(segs, cmap=cmap, linewidths=3.0)
    lc.set_array(np.asarray(values, dtype=float))
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")
    return lc


def mode_profile_figure(graph, tdf, d0_max, ids_linear, ids_newton, name, outdir, a0=None):
    """Per-mode profile comparison: linear vs saturated (newton) vs difference.

    For every mode lasing under *either* solver this draws three columns:

    * **linear** -- the mode's intensity profile ``|E(x)|^2`` frozen at its own
      threshold (the field the competition matrix is built from);
    * **full_salt_newton** -- the profile of the same mode at the operating pump
      ``d0_max`` under the shared *saturated* operator (all modes' spatial hole
      burning included), obtained by re-solving the coupled ``(k, a)``
      equilibrium with the union of both lasing sets active, warm-started from
      the linear profiles and the newton endpoint amplitudes;
    * **difference** (saturated - linear) -- where the deepening holes reshape
      the mode. This is the quantity the linear model freezes, so it is the
      *reason* the two solvers can disagree on counts and interacting
      thresholds: a suppressed/late mode is one whose saturated profile is
      pushed off the gain, an early one finds gain the linear overlap
      over-counted.

    All profiles share the ``int_pump |E|^2 = 1`` normalisation (per-edge values
    on the oversampled work graph), so the columns are directly comparable.
    Writes ``<outdir>/mode_profiles.png``.
    """
    import netsalt.modes as _m
    import netsalt.quantum_graph as _qg
    from netsalt.quantum_graph import graph_with_pump, oversample_graph
    from netsalt.utils import from_complex

    ids = sorted({int(i) for i in ids_linear} | {int(i) for i in ids_newton})
    if not ids:
        print(f"{name}: no lasing modes, skipping mode-profile figure")
        return
    thr = np.asarray(tdf["lasing_thresholds"]).ravel()
    tms = tdf["threshold_lasing_modes"].to_numpy()

    size = _m._auto_oversample_size(graph, tdf)
    work = oversample_graph(graph, size) if size else graph
    pump = np.asarray(work.graph["params"]["pump"], dtype=float)
    pump_mask = _m._get_mask_matrices(work.graph["params"])[1]

    saved = _qg.DENSE_EIG_MAX
    _qg.DENSE_EIG_MAX = min(saved, _m.NEWTON_DENSE_EIG_MAX)
    try:
        modes0, fields0 = [], []
        for i in ids:
            mode = np.asarray(from_complex(tms[i]), dtype=float)
            fields0.append(
                _m._single_mode_field_intensity(
                    graph_with_pump(work, float(thr[i])), mode, pump_mask
                )
            )
            modes0.append(mode)
        start = [max(float(a0[i]), 1e-3) for i in ids] if a0 is not None else [1.0] * len(ids)
        modes, fields, amps, _ = _m._solve_active_set(
            work, modes0, fields0, start, float(d0_max), pump, pump_mask, 30, 42
        )
    finally:
        _qg.DENSE_EIG_MAX = saved

    fig, axes = plt.subplots(len(ids), 3, figsize=(13, 3.4 * len(ids)), squeeze=False)
    for j, i in enumerate(ids):
        row = j
        lin, sat = np.asarray(fields0[j]), np.asarray(fields[j])
        vmax = max(lin.max(), sat.max())
        diff = sat - lin
        dmax = max(abs(diff).max(), 1e-12)
        lases = ("linear" if i in ids_linear else "") + (
            ("+newton" if i in ids_linear else "newton") if i in ids_newton else ""
        )
        lc0 = _draw_profile(axes[row, 0], work, lin, "viridis", 0.0, vmax)
        axes[row, 0].set_title(
            f"mode {i} (k={modes0[j][0]:.3f}, thr={thr[i]:.3g}, lases: {lases})\n"
            f"linear profile (at own threshold)",
            fontsize=9,
        )
        lc1 = _draw_profile(axes[row, 1], work, sat, "viridis", 0.0, vmax)
        axes[row, 1].set_title(
            f"saturated profile at D0={d0_max:g} (newton, a={amps[j]:.3g})", fontsize=9
        )
        lc2 = _draw_profile(axes[row, 2], work, diff, "coolwarm", -dmax, dmax)
        axes[row, 2].set_title("difference (saturated - linear)", fontsize=9)
        for lc, ax in ((lc0, axes[row, 0]), (lc1, axes[row, 1]), (lc2, axes[row, 2])):
            fig.colorbar(lc, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle(f"{name}: hole burning reshapes the mode profiles", y=1.0)
    fig.tight_layout()
    out = Path(outdir) / "mode_profiles.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

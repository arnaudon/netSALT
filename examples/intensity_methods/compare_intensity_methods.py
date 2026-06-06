"""Compare the four modal-intensity solvers on several quantum-graph topologies.

netSALT can turn the threshold modes + competition matrix into lasing L--I curves
with four ``intensity_method`` solvers, each relaxing more of the spatial-hole-
burning approximation (see ``doc/source/lasing.rst`` and issue #42):

* ``linear``           -- near-threshold competition matrix, a linear solve
                          (piecewise-linear curves);
* ``self_consistent``  -- competition matrix rebuilt at the operating pump;
* ``full_salt``        -- per-edge hole-burning surrogate (curves bend over);
* ``full_salt_newton`` -- operator-level nonlinear SALT (gain-clamping mode
                          suppression -- can lase *fewer* modes than ``linear``).

This script builds a few small **open** graphs (leads at the degree-1 nodes give
the radiative loss that sets a lasing threshold), runs the shared passive ->
pump -> trajectories -> threshold -> competition pipeline once per graph, then
overlays the four L--I curves. It writes one PDF per graph plus a combined panel
and a per-mode breakdown, and prints a small summary table.

Run from this directory::

    OMP_NUM_THREADS=1 python compare_intensity_methods.py

It is intentionally self-contained (graphs are built in memory, nothing is
cached to disk) so it doubles as a worked example of the library API.
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
)
from netsalt.physics import dispersion_relation_pump
from netsalt.quantum_graph import create_quantum_graph, set_total_length

HERE = Path(__file__).resolve().parent

# Shared physics / search settings (a gain line centred at ``k_a`` and a scan
# window straddling it). Kept small so the whole script runs in well under a
# minute on one core.
PARAMS = {
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
    # the SALT solvers march their (expensive) rebuilds on this coarser grid
    "salt_D0_steps": 10,
    "dielectric_params": {
        "method": "uniform",
        "inner_value": 9.0,  # epsilon = n^2, n = 3 inside the cavity
        "outer_value": 1.0,  # the leads are vacuum
        "loss": 0.0,
    },
}
D0_MAX = 1.4  # max pump for the L--I sweep (== intensities_D0_max)


def _quantum_graph(nx_graph, positions, total_length):
    """Wrap a networkx graph as a pumped, open netSALT quantum graph."""
    g = nx.convert_node_labels_to_integers(nx_graph)
    create_quantum_graph(g, dict(PARAMS), positions=positions)
    set_total_length(g, total_length)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    netsalt.set_dispersion_relation(g, dispersion_relation_pump)
    return g


def make_line(n_edges=10, total_length=1.0):
    """Open 1D Fabry--Perot cavity (path graph; the two end edges are leads)."""
    g = nx.path_graph(n_edges + 1)
    pos = np.array([[i, 0.0] for i in range(n_edges + 1)], dtype=float)
    return _quantum_graph(g, pos, total_length)


def make_ring_with_leads(n=12, total_length=1.0):
    """A closed loop made open by two pendant lead edges (a ring resonator)."""
    g = nx.cycle_graph(n)
    pos = {u: [np.cos(2 * np.pi * u / n), np.sin(2 * np.pi * u / n)] for u in g.nodes}
    # attach two leads on opposite sides so the loop modes can radiate out
    g.add_edge(0, n)
    pos[n] = [2.0, 0.0]
    g.add_edge(n // 2, n + 1)
    pos[n + 1] = [-2.0, 0.0]
    positions = np.array([pos[u] for u in sorted(g.nodes)], dtype=float)
    return _quantum_graph(g, positions, total_length)


def make_tree(total_length=1.0):
    """A small binary tree: one input lead, several leaf leads (a splitter)."""
    g = nx.balanced_tree(2, 3)  # depth-3 binary tree
    pos = nx.kamada_kawai_layout(g)
    positions = np.array([pos[u] for u in sorted(g.nodes)], dtype=float)
    return _quantum_graph(g, positions, total_length)


GRAPHS = {
    "line (Fabry-Perot)": make_line,
    "ring + leads": make_ring_with_leads,
    "binary tree": make_tree,
}

METHODS = ("linear", "self_consistent", "full_salt", "full_salt_newton")
COLORS = {
    "linear": "tab:blue",
    "self_consistent": "tab:orange",
    "full_salt": "tab:green",
    "full_salt_newton": "tab:red",
}


def _threshold_modes(graph):
    """Shared pipeline: scan -> passive modes -> pump -> trajectories -> thresholds."""
    qualities = netsalt.scan_frequencies(graph)
    passive = netsalt.find_passive_modes(
        graph, qualities, method="grid", min_distance=2, threshold_abs=0.1
    )
    # uniform pump on every inner (cavity) edge
    pump = np.array([1.0 if graph[u][v]["inner"] else 0.0 for u, v in graph.edges()])
    graph.graph["params"]["pump"] = pump
    trajectories = netsalt.pump_trajectories(passive, graph, return_approx=True)
    return netsalt.find_threshold_lasing_modes(trajectories, graph)


def _ll_curves(graph, threshold_df):
    """Return ``{method: (pumps, data)}`` with ``data`` shape ``(n_modes, n_pumps)``."""
    competition = compute_mode_competition_matrix(graph, threshold_df)
    solvers = {
        "linear": lambda: compute_modal_intensities(threshold_df.copy(), D0_MAX, competition),
        "self_consistent": lambda: compute_modal_intensities_self_consistent(
            graph, threshold_df.copy(), D0_MAX, D0_steps=PARAMS["salt_D0_steps"]
        ),
        "full_salt": lambda: compute_modal_intensities_full_salt(
            graph, threshold_df.copy(), D0_MAX, D0_steps=PARAMS["salt_D0_steps"]
        ),
        "full_salt_newton": lambda: compute_modal_intensities_full_salt_newton(
            graph, threshold_df.copy(), D0_MAX, D0_steps=PARAMS["salt_D0_steps"]
        ),
    }
    out = {}
    for method, run in solvers.items():
        df = run()
        cols = [c for c in df.columns if isinstance(c, tuple) and c[0] == "modal_intensities"]
        pumps = np.array(sorted(c[1] for c in cols))
        data = np.nan_to_num(df[[("modal_intensities", pp) for pp in pumps]].to_numpy(dtype=float))
        out[method] = (pumps, data)
    return out


def _plot_per_mode(name, curves, out):
    """One subplot per method, each showing every lasing mode's L--I curve.

    Within a method the amplitude unit is consistent, so this panel makes the
    *qualitative* differences plain: ``linear`` is piecewise-linear with kinks at
    each activation, ``full_salt`` bends the curves over via saturation, and
    ``full_salt_newton`` clamps the gain so some modes never switch on.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, method in zip(axes.ravel(), METHODS, strict=True):
        pumps, data = curves[method]
        active = np.where(data.max(axis=1) > 1e-9)[0]
        for mu in active:
            ax.plot(pumps, data[mu], ".-", ms=4, label=f"mode {mu}")
        ax.set_title(f"{method}  ({len(active)} lasing)")
        ax.set_ylabel("modal intensity")
        if active.size:
            ax.legend(fontsize=7, ncol=2)
    for ax in axes[1]:
        ax.set_xlabel("pump $D_0$")
    fig.suptitle(f"Per-mode L--I on the {name} graph", y=1.0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    n_graphs = len(GRAPHS)
    fig, axes = plt.subplots(1, n_graphs, figsize=(5 * n_graphs, 4), squeeze=False)
    print(f"{'graph':22s} {'method':18s} {'n_lasing':>9s} {'n_active@max':>13s} {'total@max':>11s}")
    print("-" * 76)

    first_curves = first_name = None
    for ax, (name, builder) in zip(axes[0], GRAPHS.items(), strict=True):
        graph = builder()
        threshold_df = _threshold_modes(graph)
        n_lasing = int(np.sum(np.asarray(threshold_df["lasing_thresholds"]) < np.inf))
        curves = _ll_curves(graph, threshold_df)
        if first_curves is None:
            first_curves, first_name = curves, name

        for method in METHODS:
            pumps, data = curves[method]
            total = data.sum(axis=0)
            n_active = int(np.sum(data[:, -1] > 1e-9))
            ax.plot(pumps, total, "o-", ms=3, color=COLORS[method], label=method)
            print(f"{name:22s} {method:18s} {n_lasing:>9d} {n_active:>13d} {total[-1]:>11.3e}")
        ax.set_title(f"{name}\n({len(graph)} nodes, {n_lasing} lasing modes)")
        ax.set_xlabel("pump $D_0$")
        ax.set_ylabel("total modal intensity")
        ax.legend(fontsize=8)

    fig.suptitle("Modal-intensity approximations across graph topologies", y=1.02)
    fig.tight_layout()
    out = HERE / "intensity_methods_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # a per-mode breakdown on the first graph makes the activation / bend-over /
    # gain-clamping differences between the methods explicit
    _plot_per_mode(first_name, first_curves, HERE / "intensity_methods_per_mode.pdf")


if __name__ == "__main__":
    main()

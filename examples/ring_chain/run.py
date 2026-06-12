"""Detuned ring chain: the mode count as a dial (one lasing mode per ring).

A chain of four rings of increasing size, bridged in sequence. Detuning shifts
each ring's mode comb, so near the gain centre each ring contributes one
**localised** mode (100 % of its intensity on its home ring, found by the
parameter scan below). The design trick is the output coupling: each bridge
carries a midpoint node with a lead. Hybridised (delocalised) modes have weight
on the bridges, localised ring modes avoid them -- so the bridge-mounted leads
*selectively damp the hybrids* (cold-cavity loss 0.02--0.04) while the
one-per-ring quartet keeps only the uniform material-loss floor (~0.004, set by
``dielectric_params['loss']``). With the gain (``k_a = 3.71``,
``gamma_perp = 0.18``) covering the quartet at k = 3.61 / 3.66 / 3.73 / 3.83,
the four localised modes lase one per ring with staggered onsets (ring 2's
mode is the leakiest of the quartet and turns on last), and the
hybrids stay below threshold: **the number of co-lasing modes equals the number
of rings**, by construction. (With leads mounted on the rings instead, the loss
hierarchy inverts -- the hybrids lase first and the story muddies; that failed
design is why the leads sit on the bridges.)

Both solvers should lase the quartet; ``full_salt_newton`` adds the saturated
mode-profile reshaping (see ``mode_profiles.png``).

Run from this directory (writes ``ring_chain.png`` + ``mode_profiles.png``)::

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
from _common import curves, linear_on_grid, mode_profile_figure

import netsalt
from netsalt.modes import (
    compute_modal_intensities_full_salt_newton,
    compute_mode_competition_matrix,
    find_passive_modes,
    find_threshold_lasing_modes,
    mean_mode_on_edges,
    pump_trajectories,
)
from netsalt.physics import dispersion_relation_pump
from netsalt.quantum_graph import create_quantum_graph, set_total_length
from netsalt.utils import from_complex

HERE = Path(__file__).resolve().parent
RADII = [0.8, 0.95, 1.1, 1.25]  # ~18 % detuning ring to ring
RING_NODES = [7, 8, 9, 10]

PARAMS = {
    "open_model": "open",
    "c": 1.0,
    "k_a": 3.71,  # centred on the one-per-ring quartet (3.61/3.66/3.73/3.83)
    "gamma_perp": 0.18,
    "k_min": 3.2,
    "k_max": 3.9,
    "alpha_min": -0.05,
    "alpha_max": 0.25,
    "n_workers": 1,
    "n_modes_max": 40,
    "quality_threshold": 1e-3,
    "search_stepsize": 0.005,
    "max_steps": 1000,
    "max_tries_reduction": 50,
    "reduction_factor": 0.8,
    "D0_max": 0.08,
    "D0_steps": 14,
    # the uniform material loss sets the threshold floor of the (otherwise
    # essentially lossless) localised ring modes
    "dielectric_params": {
        "method": "uniform",
        "inner_value": 9.0,
        "outer_value": 1.0,
        "loss": 0.02,
    },
}
D0_MAX = 0.08
D0_STEPS = 16


def build_ring_chain():
    """Chain of detuned rings, bridges carrying midpoint-mounted leads."""
    g = nx.Graph()
    pos = {}
    ring_nodes = []
    offset, x0 = 0, 0.0
    for r, n in zip(RADII, RING_NODES, strict=True):
        nodes = list(range(offset, offset + n))
        ring_nodes.append(nodes)
        g.add_edges_from([(nodes[i], nodes[(i + 1) % n]) for i in range(n)])
        for i, nd in enumerate(nodes):
            pos[nd] = [x0 + r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n)]
        x0 += 2.6 * r
        offset += n
    nid = offset
    for a, b in zip(ring_nodes[:-1], ring_nodes[1:], strict=True):
        ra = max(a, key=lambda nd: pos[nd][0])
        rb = min(b, key=lambda nd: pos[nd][0])
        mid, lead = nid, nid + 1
        nid += 2
        pos[mid] = [(pos[ra][0] + pos[rb][0]) / 2, (pos[ra][1] + pos[rb][1]) / 2]
        pos[lead] = [pos[mid][0], pos[mid][1] - 1.4]
        g.add_edge(ra, mid)
        g.add_edge(mid, rb)
        g.add_edge(mid, lead)
    positions = np.array([pos[i] for i in range(len(g))])
    create_quantum_graph(g, dict(PARAMS), positions=positions)
    set_total_length(g, 18.0)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    netsalt.set_dispersion_relation(g, dispersion_relation_pump)
    return g, ring_nodes


def _home_ring(mode, graph, ring_sets):
    """Index of the ring holding most of the mode's (length-weighted) intensity."""
    e2 = np.abs(mean_mode_on_edges(mode, graph, check_quality=False))
    weights = []
    for s in ring_sets:
        weights.append(
            sum(
                e2[ei] * graph[u][v]["length"]
                for ei, (u, v) in enumerate(graph.edges)
                if u in s and v in s
            )
        )
    weights = np.array(weights)
    frac = weights / max(weights.sum(), 1e-12)
    return int(np.argmax(frac)), float(frac.max())


def main():
    graph, ring_nodes = build_ring_chain()
    ring_sets = [set(n) for n in ring_nodes]
    passive = find_passive_modes(graph, method="contour")
    if len(passive) == 0:
        raise SystemExit("no passive modes found")
    graph.graph["params"]["pump"] = np.array(
        [1.0 if graph[u][v]["inner"] else 0.0 for u, v in graph.edges()]
    )
    trajectories = pump_trajectories(passive, graph, return_approx=True)
    tdf = find_threshold_lasing_modes(trajectories, graph)
    thr = np.asarray(tdf["lasing_thresholds"]).ravel()
    tms = tdf["threshold_lasing_modes"].to_numpy()
    n_modes = len(tdf)

    homes = {}
    print("modes below max pump (id, k, threshold, home ring, localisation):")
    for i in range(n_modes):
        if thr[i] < D0_MAX:
            home, frac = _home_ring(tms[i], graph, ring_sets)
            homes[i] = home
            print(
                f"  {i}: k={from_complex(tms[i])[0]:.4f}  thr={thr[i]:.4f}"
                f"  ring {home} ({100 * frac:.0f}%)"
            )

    competition = compute_mode_competition_matrix(graph, tdf)
    first = float(thr[thr < np.inf].min())
    grid = np.linspace(first, D0_MAX, D0_STEPS)
    linear = linear_on_grid(tdf, competition, grid, n_modes)
    n_cols, newton = curves(
        compute_modal_intensities_full_salt_newton(graph, tdf.copy(), D0_MAX, D0_steps=D0_STEPS)
    )
    peak = max(linear.max(), newton.max(), 1e-9)
    ids_linear = [m for m in range(n_modes) if linear[m, -1] > 1e-2 * peak]
    ids_newton = [m for m in range(n_modes) if newton[m, -1] > 1e-2 * peak]
    rings_lasing = sorted({homes.get(m, -1) for m in ids_newton})
    print(f"linear lases {len(ids_linear)}: {ids_linear}")
    print(f"full_salt_newton lases {len(ids_newton)}: {ids_newton} (rings {rings_lasing})")

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    pos = {n: np.asarray(graph.nodes[n]["position"], dtype=float) for n in graph.nodes}
    for u, v in graph.edges():
        x, y = zip(pos[u], pos[v], strict=True)
        axes[0].plot(x, y, color="0.55" if graph[u][v]["inner"] else "crimson", lw=2.0)
    axes[0].set_aspect("equal")
    axes[0].axis("off")
    axes[0].set_title(f"{len(RADII)} detuned rings, bridge-mounted leads")
    active = sorted(set(ids_linear) | set(ids_newton))
    for m in active:
        col = cmap(m % 10)
        axes[1].plot(grid, linear[m], "--", color=col, lw=1.3, alpha=0.8)
        axes[1].plot(
            n_cols,
            newton[m],
            ".-",
            color=col,
            lw=1.8,
            ms=4,
            label=f"mode {m} (ring {homes.get(m, '?')})",
        )
    axes[1].set_xlabel("pump $D_0$")
    axes[1].set_ylabel("modal intensity")
    axes[1].set_title(f"dashed = linear ({len(ids_linear)}), solid = newton ({len(ids_newton)})")
    axes[1].legend(fontsize=8)
    fig.suptitle("Ring chain: one lasing mode per ring", y=1.02)
    fig.tight_layout()
    out = HERE / "ring_chain.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    mode_profile_figure(
        graph,
        tdf,
        D0_MAX,
        ids_linear,
        ids_newton,
        "ring chain",
        HERE,
        a0={m: float(newton[m, -1]) for m in range(n_modes)},
    )


if __name__ == "__main__":
    main()

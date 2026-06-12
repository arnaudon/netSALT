"""Chord-count sweep: density alone does not buy lasing modes.

The chaotic-ring family made systematic: one 14-node ring with two leads and a
growing number of hard-coded random chords (6 -> 10 -> 14, nested sets so the
sweep changes one thing at a time). The naive expectation -- more chords, more
loops, denser spectrum, more lasing modes -- is **falsified** by the measured
sweep: 6 chords lase 3 (linear) / 4 (newton) modes, while 10 and 14 chords
collapse to **single-mode** lasing despite holding more modes under the gain.

The mechanism is in the participation column: adding chords *delocalises*
the modes (the mean participation ratio grows monotonically, 7.9 -> 8.4 ->
8.9) and reshuffles the cluster under the gain, so the spatial overlap rises
and the first lasing mode clamps the gain for the others -- winner-take-all.
Multimode lasing needs *localised* modes (weak overlap), which raw loop
density does not provide: compare
``../chaotic_ring`` (6 chords, localised cluster, 4 modes),
``../ring_chain`` (localisation by detuning, one mode per ring) and
``../mini_buffon`` (extended disorder modes, 2 of 10 lase).

Run from this directory (writes ``chord_sweep.png``; takes tens of minutes)::

    OMP_NUM_THREADS=1 python run.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import curves, linear_on_grid, threshold_modes

import netsalt
from netsalt.modes import (
    compute_modal_intensities_full_salt_newton,
    compute_mode_competition_matrix,
    mode_on_nodes,
)
from netsalt.physics import dispersion_relation_pump
from netsalt.quantum_graph import create_quantum_graph, set_total_length
from netsalt.utils import from_complex

HERE = Path(__file__).resolve().parent
M = 14  # ring nodes
# Nested chord sets: the first 6 are ../chaotic_ring's, extended twice.
CHORDS = [
    (0, 7),
    (1, 5),
    (2, 12),
    (3, 5),
    (3, 10),
    (7, 11),  # 6 (chaotic_ring)
    (4, 9),
    (6, 13),
    (2, 8),
    (0, 10),  # -> 10
    (5, 12),
    (1, 9),
    (8, 13),
    (4, 11),  # -> 14
]
SWEEP = [6, 10, 14]
D0_MAX = 0.5
D0_STEPS = 12

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
    "n_modes_max": 60,
    "quality_threshold": 1e-3,
    "search_stepsize": 0.005,
    "max_steps": 1000,
    "max_tries_reduction": 50,
    "reduction_factor": 0.8,
    "D0_max": D0_MAX,
    "D0_steps": 14,
    "dielectric_params": {"method": "uniform", "inner_value": 9.0, "outer_value": 1.0, "loss": 0.0},
}


def build(n_chords):
    g = nx.cycle_graph(M)
    g.add_edges_from(CHORDS[:n_chords])
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
    """Effective number of nodes the mode lives on (1 / sum p_i^2).

    Threshold modes sit at the lasing point where the operator is exactly
    singular and ARPACK shift-invert cannot factorise it; a tiny imaginary
    offset lifts the singularity (same guard as the field-intensity helper).
    """
    k, alpha = from_complex(mode)
    nudged = [float(k), float(alpha) if abs(alpha) > 1e-7 else 1e-7]
    weight = np.abs(mode_on_nodes(nudged, graph, check_quality=False)) ** 2
    prob = weight / weight.sum()
    return 1.0 / np.sum(prob**2)


def main():
    rows = []
    print(
        f"{'chords':>6} {'modes':>6} {'min dk':>8} {'<partic>':>8}"
        f" {'lin lases':>9} {'nwt lases':>9} {'t_nwt':>7}"
    )
    for n_chords in SWEEP:
        graph = build(n_chords)
        tdf = threshold_modes(graph, method="contour")
        thr = np.asarray(tdf["lasing_thresholds"]).ravel()
        tms = tdf["threshold_lasing_modes"].to_numpy()
        n_modes = len(tdf)
        ks = np.sort([from_complex(tms[i])[0] for i in range(n_modes)])
        min_dk = float(np.diff(ks).min()) if len(ks) > 1 else np.nan
        partic = float(np.mean([_participation(tms[i], graph) for i in range(n_modes)]))

        competition = compute_mode_competition_matrix(graph, tdf)
        first = float(thr[thr < np.inf].min())
        grid = np.linspace(first, D0_MAX, D0_STEPS)
        linear = linear_on_grid(tdf, competition, grid, n_modes)
        t0 = time.perf_counter()
        _, newton = curves(
            compute_modal_intensities_full_salt_newton(graph, tdf.copy(), D0_MAX, D0_steps=D0_STEPS)
        )
        t_newton = time.perf_counter() - t0
        peak = max(linear.max(), newton.max(), 1e-9)
        n_lin = int(np.sum(linear[:, -1] > 1e-2 * peak))
        n_nwt = int(np.sum(newton[:, -1] > 1e-2 * peak))
        rows.append((n_chords, n_modes, min_dk, partic, n_lin, n_nwt, t_newton))
        print(
            f"{n_chords:>6} {n_modes:>6} {min_dk:>8.4f} {partic:>8.1f}"
            f" {n_lin:>9} {n_nwt:>9} {t_newton:>6.0f}s"
        )

    rows = np.array(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(rows[:, 0], rows[:, 1], "s--", color="0.5", label="modes in window")
    axes[0].plot(rows[:, 0], rows[:, 4], "o-", color="tab:blue", label="linear lases")
    axes[0].plot(rows[:, 0], rows[:, 5], "x-", color="tab:red", label="newton lases")
    axes[0].set_xlabel("number of chords")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)
    axes[0].set_title("denser spectrum, fewer lasing modes")
    axes[1].plot(rows[:, 0], rows[:, 3], "o-", color="tab:purple")
    axes[1].set_xlabel("number of chords")
    axes[1].set_ylabel("mean participation ratio")
    axes[1].set_title("the mechanism: modes delocalise")
    fig.suptitle("Ring + chords: density alone does not buy lasing modes", y=1.02)
    fig.tight_layout()
    out = HERE / "chord_sweep.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

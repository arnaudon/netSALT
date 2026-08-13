"""Genuine multimode lasing on a detuned two-ring "photonic molecule".

The simple single-graph examples (``../line_fabry_perot``, ``../ring_leads``, ``../tree``) are all
*single-mode* under faithful SALT: their modes overlap strongly, so the dominant
mode clamps the gain and holds the others below threshold (``full_salt_newton``
correctly reports one mode). To get
genuine *multimode* lasing you need modes that occupy **different regions** of the
graph, so each burns its own spatial hole and leaves gain for the others.

This builds two rings of **different sizes** (radii 0.9 and 1.25, i.e. ~1.4x
detuned) joined by a single bridge edge, with a lead on each ring for output
coupling. The detuning is the whole point:

* If the rings were *identical* the mirror symmetry would force the eigenmodes to
  be symmetric/antisymmetric (bonding/antibonding) combinations spread over *both*
  rings -- delocalised, high mutual overlap, strong competition.
* Detuning shifts the two rings' mode combs, so at a given frequency only one ring
  is near-resonant and the mode **localises** onto that ring. Two localised modes
  (one per ring) barely overlap -> they co-lase.

With a narrow gain centred on a pair of nearby modes from *different* rings, all
four solvers -- including ``full_salt_newton`` -- lase several modes at once.

Run::

    OMP_NUM_THREADS=1 python run.py

Modes are found by Beyn's contour method (robust on this hand-built graph). The
solve is the operator-level Newton; this is the multimode case it was built for.
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
    mean_mode_on_edges,
    pump_trajectories,
)
from netsalt.physics import dispersion_relation_pump
from netsalt.quantum_graph import create_quantum_graph, set_total_length
from netsalt.utils import from_complex

HERE = Path(__file__).resolve().parent
N_A, N_B = 7, 9  # ring node counts -- different sizes => detuned => localised modes

# A narrow gain line centred on a cross-ring mode pair near k = 3.57. Mode finding
# uses the contour method, so only k_min/k_max/alpha bounds matter for it.
PARAMS = {
    "open_model": "open",
    "c": 1.0,
    "k_a": 3.567,
    "gamma_perp": 0.12,
    "k_min": 3.45,
    "k_max": 3.66,
    # A passive open cavity has alpha > 0; the floor also excludes the
    # near-trapped very high-Q modes (alpha ~ 1e-9) this graph supports, which
    # lase at essentially zero pump where the near-threshold model does not
    # apply. See tests/test_functional.py for the longer note.
    "alpha_min": 5e-4,
    "alpha_max": 0.15,
    "n_workers": 1,
    "n_modes_max": 10,
    "quality_threshold": 1e-3,
    "search_stepsize": 0.005,
    "max_steps": 1000,
    "max_tries_reduction": 50,
    "reduction_factor": 0.8,
    "D0_max": 1.0,
    "D0_steps": 14,
    "dielectric_params": {"method": "uniform", "inner_value": 9.0, "outer_value": 1.0, "loss": 0.0},
}
D0_MAX = 1.0


def build_two_ring():
    """Two detuned rings joined by a bridge, with a lead on each (open cavity)."""
    g = nx.disjoint_union(nx.cycle_graph(N_A), nx.cycle_graph(N_B))
    g.add_edge(0, N_A)  # bridge between the rings
    g.add_edge(2, N_A + N_B)  # lead on ring A
    g.add_edge(N_A + 4, N_A + N_B + 1)  # lead on ring B
    pos = {}
    for i in range(N_A):
        pos[i] = [-1.6 + 0.9 * np.cos(2 * np.pi * i / N_A), 0.9 * np.sin(2 * np.pi * i / N_A)]
    for j in range(N_B):
        pos[N_A + j] = [
            1.7 + 1.25 * np.cos(2 * np.pi * j / N_B),
            1.25 * np.sin(2 * np.pi * j / N_B),
        ]
    pos[N_A + N_B] = [-1.6, -2.2]
    pos[N_A + N_B + 1] = [1.7, -2.6]
    positions = np.array([pos[i] for i in range(len(g))])
    create_quantum_graph(g, dict(PARAMS), positions=positions)
    set_total_length(g, 9.0)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    netsalt.set_dispersion_relation(g, dispersion_relation_pump)
    return g


def _which_ring(mode, graph):
    """'A' or 'B' -- the ring holding most of the mode's intensity."""
    e2 = np.abs(mean_mode_on_edges(mode, graph, check_quality=False))
    ring_a = sum(e2[ei] for ei, (u, v) in enumerate(graph.edges) if u < N_A and v < N_A)
    ring_b = sum(
        e2[ei] for ei, (u, v) in enumerate(graph.edges) if N_A <= u and N_A <= v < N_A + N_B
    )
    return "A" if ring_a > ring_b else "B"


def main():
    graph = build_two_ring()
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
    rings = {i: _which_ring(threshold_modes[i], graph) for i in range(len(tdf)) if thr[i] < np.inf}
    print("lasing modes (id, k, threshold, ring):")
    for i, r in rings.items():
        print(f"  {i}: k={from_complex(threshold_modes[i])[0]:.3f}  thr={thr[i]:.3f}  ring {r}")

    competition = compute_mode_competition_matrix(graph, tdf)
    solvers = {
        "linear": compute_modal_intensities(tdf.copy(), D0_MAX, competition),
        "full_salt_newton": compute_modal_intensities_full_salt_newton(
            graph, tdf.copy(), D0_MAX, D0_steps=18
        ),
    }
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    lasing_at_max = {}
    endpoint = {}
    for ax, (name, df) in zip(axes.ravel(), solvers.items(), strict=True):
        cols = np.array(
            sorted(c[1] for c in df.columns if isinstance(c, tuple) and c[0] == "modal_intensities")
        )
        data = np.nan_to_num(df[[("modal_intensities", c) for c in cols]].to_numpy(dtype=float))
        peak = max(data.max(), 1e-9)
        active = [m for m in range(data.shape[0]) if data[m].max() > 1e-2 * peak]
        lasing_at_max[name] = [m for m in active if data[m, -1] > 1e-2 * peak]
        endpoint[name] = {m: float(data[m, -1]) for m in active}
        for m in active:
            ax.plot(
                cols,
                data[m],
                ".-",
                color=cmap(m % 10),
                label=f"mode {m} (ring {rings.get(m, '?')})",
            )
        ax.set_title(f"{name}  ({len(active)} lasing)")
        ax.set_ylabel("modal intensity")
        if active:
            ax.legend(fontsize=7)
        print(
            f"{name}: {len(active)} lasing @max "
            + str({int(m): round(float(data[m, -1]), 3) for m in active})
        )
    for ax in axes.ravel():
        ax.set_xlabel("pump $D_0$")
    fig.suptitle("Detuned two-ring photonic molecule: genuine multimode lasing", y=1.0)
    fig.tight_layout()
    out = HERE / "two_ring_multimode.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    mode_profile_figure(
        graph,
        tdf,
        D0_MAX,
        lasing_at_max["linear"],
        lasing_at_max["full_salt_newton"],
        "two-ring molecule",
        HERE,
        a0=endpoint["full_salt_newton"],
    )


if __name__ == "__main__":
    main()

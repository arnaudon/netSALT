"""Mini-buffon network: the dense-spectrum stress test for the two solvers.

A shrunk version of the Nat. Commun. buffon networks (10 random lines, giant
component: 39 nodes, 45 edges, 18 radiating lead ends) built with
:func:`netsalt.utils.make_buffon_graph` from a fixed seed. The spectrum is
genuinely dense -- ~25 modes per unit ``k`` (close to the Weyl estimate
``nL/pi = 28.6``), 10 modes in the scan window with disorder-spread losses
(``alpha = 0.006 - 0.064``) and a near-degenerate pair split by ``dk = 0.006``.

What the run shows (measured, not assumed) -- three stages:

* **Low uniform pump (D0 <= 0.4): gain clamping wins.** Ten modes sit under
  the broadened gain but only **two** lase -- the extended disorder modes
  overlap strongly and the winners clamp the gain. But the suppression is
  not absolute: the losers' *interacting* thresholds are finite, just 10-40x
  their bare ones.
* **High uniform pump (D0 <= 1.2): pump strength buys the modes back.** The
  same uniform pump swept 3x further lases **5 (linear) / 4 (newton)** -- on
  this network *more pump*, not pump shaping, is the simplest route to more
  co-lasing modes.
* **Pump shaping selects, it does not multiply.** A mode-resolved pump
  (greedy low-cross-saturation targets from the competition matrix, pumping
  the edges each target dominates) was probed at several target counts and
  ownership margins: at the *same* pump ceiling every shaped pump lased
  *fewer* modes (2-3) than uniform (4-5), because removing pump area raises
  all thresholds faster than the decoupling pays back. Shaping is the tool
  for choosing *which* modes lase (the Nat. Commun. optimisation route,
  ``netsalt.pump``); the shaped stage is kept here to document that
  trade-off honestly.

Throughout, ``full_salt_newton`` stays comfortable (~25-40 s per sweep on the
~700-node oversampled work graph vs ~0.1 s for ``linear``) -- at this scale
competition physics, not solver cost, is the constraint. For many co-lasing
modes by *design* see ``../ring_chain``.

**Known artifact -- the per-mode kink at high pump.** The dominant mode's
newton curve shows a sharp drop near ``D0 ~ 0.3-0.5``. It is *not* physics
(the summed intensity through it is continuous; a falling total with rising
pump would be unphysical): modes 5 and 6 are a near-degenerate pair
(``k = 3.5351 / 3.5390``, ``dk = 0.004``) and at an active-set event the
coupled solve hands the amplitude from one label to the other -- a
mode-identity swap. This is the documented near-degeneracy limit of
``full_salt_newton`` showing up in data: per-mode curves are unreliable
*across active-set events* for modes closer than the solve can keep apart;
the cluster (pair-sum) intensity and the total are the trustworthy
quantities there.

Run from this directory (writes ``li_curves.png`` + ``mode_profiles.png``)::

    OMP_NUM_THREADS=1 python run.py     # ~2 minutes, mostly the mode pipeline
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import compare_and_plot

import netsalt
from netsalt.physics import dispersion_relation_pump
from netsalt.quantum_graph import create_quantum_graph, set_total_length
from netsalt.utils import make_buffon_graph

HERE = Path(__file__).resolve().parent
N_LINES = 10
SEED = 4  # chosen by a small seed scan for a ~40-node giant component
D0_MAX = 0.4

PARAMS = {
    "open_model": "open",
    "c": 1.0,
    "k_a": 3.55,  # on the low-loss cluster found by the spectrum scan
    "gamma_perp": 0.3,
    "k_min": 3.3,
    "k_max": 3.7,
    "alpha_min": -0.05,
    "alpha_max": 0.3,
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


def build():
    g, pos = make_buffon_graph(n_lines=N_LINES, size=(-100.0, 100.0), resolution=100.0, rng=SEED)
    giant = max(nx.connected_components(g), key=len)
    g = nx.convert_node_labels_to_integers(g.subgraph(giant).copy(), label_attribute="old")
    positions = np.array([pos[g.nodes[u]["old"]] for u in g.nodes])
    create_quantum_graph(g, dict(PARAMS), positions=positions)
    set_total_length(g, 30.0)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    netsalt.set_dispersion_relation(g, dispersion_relation_pump)
    return g


def selective_pump(graph, tdf, n_targets=4, margin=0.7):
    """Mode-resolved pump: cover the edges each low-overlap target dominates.

    Greedy target selection on the (uniform-pump) competition matrix: start
    from the lowest-threshold mode, then repeatedly add the candidate whose
    worst normalised cross-saturation ``T_ij / sqrt(T_ii T_jj)`` against the
    already-selected set is smallest -- spatially distinct modes by
    construction. The pump then covers exactly the inner edges whose largest
    per-edge intensity (among *all* candidates) belongs to a target: each
    target keeps the gain where it is strongest, the non-targets are starved.
    """
    from netsalt.modes import compute_mode_competition_matrix, mean_mode_on_edges

    thr = np.asarray(tdf["lasing_thresholds"]).ravel()
    tms = tdf["threshold_lasing_modes"].to_numpy()
    candidates = [int(i) for i in np.where(thr < np.inf)[0]]
    T = compute_mode_competition_matrix(graph, tdf)

    def xsat(i, j):
        return abs(T[i, j]) / max(np.sqrt(abs(T[i, i]) * abs(T[j, j])), 1e-12)

    targets = [min(candidates, key=lambda i: thr[i])]
    while len(targets) < min(n_targets, len(candidates)):
        rest = [c for c in candidates if c not in targets]
        targets.append(min(rest, key=lambda c: max(xsat(c, t) for t in targets)))
    print(f"selective-pump targets (low mutual cross-saturation): {sorted(targets)}")

    from netsalt.utils import from_complex

    fields = {}
    for i in candidates:
        e2 = np.abs(mean_mode_on_edges(from_complex(tms[i]), graph, check_quality=False))
        fields[i] = e2 / max(e2.sum(), 1e-12)
    inner = np.array([1.0 if graph[u][v]["inner"] else 0.0 for u, v in graph.edges()])
    # pump an edge when a *target* is (close to) the strongest candidate on it:
    # strict winner-take-all starves the targets of pump area, so allow any
    # target within ``margin`` of the top share
    top = np.array([max(fields[i][e] for i in candidates) for e in range(len(inner))])
    near = np.array(
        [any(fields[t][e] >= margin * top[e] for t in targets) for e in range(len(inner))]
    )
    pump = inner * near
    print(f"selective pump covers {int(pump.sum())} of {int(inner.sum())} inner edges")
    return pump


if __name__ == "__main__":
    graph = build()
    n_leads = sum(1 for n in graph.nodes if len(graph[n]) == 1)
    inner = sum(graph[u][v]["length"] for u, v in graph.edges if graph[u][v]["inner"])
    print(
        f"mini-buffon: {len(graph)} nodes, {len(graph.edges)} edges, {n_leads} leads; "
        f"Weyl density nL/pi = {3 * inner / np.pi:.1f} modes per unit k"
    )
    t0 = time.perf_counter()
    uniform = compare_and_plot(
        graph,
        "mini-buffon (uniform pump, low)",
        HERE,
        d0_max=D0_MAX,
        d0_steps=12,
        passive_method="contour",
        prefix="uniform_low_",
    )
    compare_and_plot(
        graph,
        "mini-buffon (uniform pump, high)",
        HERE,
        d0_max=3.0 * D0_MAX,
        d0_steps=12,
        passive_method="contour",
        prefix="uniform_high_",
    )
    inner_pump = np.array([1.0 if graph[u][v]["inner"] else 0.0 for u, v in graph.edges()])
    graph.graph["params"]["pump"] = inner_pump  # design against the uniform operator
    pump = selective_pump(graph, uniform["tdf"])
    compare_and_plot(
        graph,
        "mini-buffon (selective pump)",
        HERE,
        d0_max=3.0 * D0_MAX,
        d0_steps=12,
        passive_method="contour",
        pump=pump,
        prefix="selective_",
    )
    print(f"total wall time {time.perf_counter() - t0:.0f}s")

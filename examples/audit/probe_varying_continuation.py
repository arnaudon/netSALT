"""Pump continuation on the un-oversampled operator: is the L-I curve physical?

Issue #53's complaint about the oversampled solver was a non-monotone summed
output (-40.8% across one pump step on mini_buffon) and residuals of 1e-2 deep
above threshold. This runs the same kind of sweep through
``netsalt.salt_varying.solve_salt_varying``, which never oversamples: the
matrix stays 8x8 while the within-edge resolution lives in per-edge transfer
matrices.

Measured on the Fabry-Perot fixture, 15 pumps from 1.05x to 3.0x threshold:
amplitude strictly monotone 0.037 -> 1.372, k stable to 1e-5, residuals
3e-7..9e-7, converged at every pump, 11.4 s.

Note the continuation matters: solving each pump from a scan-derived guess
instead of from the previous solution makes the solver land on *different*
modes (k jumping 11.4 -> 10.4 -> 9.5), each a genuine root. Carrying the
solution forward is what keeps it on one branch.

Run from this directory::

    OMP_NUM_THREADS=1 python probe_varying_continuation.py
"""

from __future__ import annotations

import time
import warnings

import networkx as nx
import numpy as np

import netsalt
from netsalt.physics import dispersion_relation_pump
from netsalt.quantum_graph import create_quantum_graph, set_total_length
from netsalt.salt_varying import _lam_varying, saturated_eps_profiles, solve_salt_varying


def build(n_inner=5):
    n_edges = n_inner + 2
    g = nx.path_graph(n_edges + 1)
    pos = np.array([[float(i), 0.0] for i in range(n_edges + 1)])
    params = {
        "open_model": "open",
        "c": 1.0,
        "k_a": 10.0,
        "gamma_perp": 3.0,
        "dielectric_params": {
            "method": "uniform",
            "inner_value": 9.0,
            "loss": 0.0,
            "outer_value": 1.0,
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        create_quantum_graph(g, params, positions=pos, noise_level=0.0)
    set_total_length(g, 1.0, inner=True)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    netsalt.set_dispersion_relation(g, dispersion_relation_pump)
    pump = np.array([0.0 if not g[u][v]["inner"] else 1.0 for u, v in g.edges])
    g.graph["params"]["pump"] = pump
    return g, pump


g, pump = build()
NS = 32
zero = [[np.zeros(NS + 1) for _ in g.edges]]


def thr_scan():
    best = (np.inf, None, None)
    for D0 in np.linspace(0.30, 0.55, 26):
        prof = saturated_eps_profiles(g, [10.0], [0.0], zero, D0, pump)
        grid = np.linspace(10.2, 10.7, 51)
        vals = [abs(_lam_varying(g, k, prof, NS)) for k in grid]
        i = int(np.argmin(vals))
        if vals[i] < best[0]:
            best = (vals[i], D0, grid[i])
    return best


res_thr, D0_thr, k_thr = thr_scan()
print(f"threshold: D0={D0_thr:.5f}  k={k_thr:.6f}  min|lambda|={res_thr:.2e}")
print(f"graph: {len(g)} nodes / {len(g.edges)} edges (never oversampled)\n")

print(f"{'D0/thr':>7} {'k':>13} {'a':>12} {'residual':>11} {'conv':>6} {'iters':>6}")
k, a = k_thr, 0.02
t0 = time.perf_counter()
prev_a, drops = None, []
for factor in np.linspace(1.05, 3.0, 15):
    D0 = factor * D0_thr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_salt_varying(g, [k], [a], D0, pump, n_steps=NS, outer=25)
    k, a = float(sol.ks[0]), float(sol.amplitudes[0])
    if prev_a is not None and a < prev_a * (1 - 1e-6):
        drops.append((factor, prev_a, a))
    prev_a = a
    print(
        f"{factor:>7.2f} {k:>13.7f} {a:>12.7f} {sol.residuals[0]:>11.2e} "
        f"{str(sol.converged):>6} {sol.iterations:>6}"
    )
elapsed = time.perf_counter() - t0
print(f"\nsweep {elapsed:.1f}s   monotone in a: {not drops}")
for f_, b_, a_ in drops:
    print(f"   DROP at {f_:.2f}x: {b_:.6f} -> {a_:.6f}")

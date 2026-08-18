"""Step 5v: the *varying-operator* SALT solve on the same cavity.

The counterpart of ``step5_netsalt_salt_run.py``, which drives netsalt's
oversampled ``solve_salt_fixed_set``. This one drives
:func:`netsalt.salt_varying.solve_salt_varying`, which never subdivides the
graph: the matrix stays ``len(graph)`` square and the within-edge hole burning
lives in per-edge transfer matrices.

Why this script exists. The varying path's above-threshold accuracy had only
ever been checked against an amplitude continuation *through its own operator*
-- same ``construct_laplacian_varying``, same ``saturated_eps_profiles``, same
``_lam_varying``. That is a real check of which solution branch is selected and
no check at all of the model: a systematic error in the operator would be shared
by both and they would agree perfectly. ``indep_salt.py`` shares no code with
netsalt (numpy and scipy only), so running the varying path against it is the
first test of the varying *physics* rather than the varying *solver*.

Output schema matches step 5 exactly, so ``step8_compare.py`` reads it
unchanged. The convention-free quantity is again the physical modal intensity

    I_mu = int_cavity |Psi_mu(x)|^2 dx = a_mu * int_cavity f_mu(x) dx ,

but here ``f_mu`` is *sampled along* each edge rather than one scalar per edge,
so the edge integral is a Simpson quadrature over the sample grid instead of
``l_e * f_e``. Simpson rather than trapezoid because the profiles are resolved
to a fourth-order propagator and an O(h^2) quadrature would throw that away.

Usage: python step5v_netsalt_varying_run.py <n_steps> <n_modes> <D0_max> <out> <D0_step>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from setup_case import GAMMA_PERP, K_A, build_graph

from netsalt.salt_varying import (
    net_gain_alpha,
    saturated_eps_profiles,
    solve_salt_varying,
)

TAG = os.environ.get("NS_TAG", "")

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
N_MODES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
D0_MAX = float(sys.argv[3]) if len(sys.argv) > 3 else 1.4
OUT = sys.argv[4] if len(sys.argv) > 4 else f"results_step5v_varying_n{N_STEPS}.json"
D0_STEP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02

tdf = pd.read_hdf(f"netsalt_threshold_modes{TAG}.h5", key="modes")
graph = build_graph()
thr = np.asarray(tdf["lasing_thresholds"]).ravel()
tms = tdf["threshold_lasing_modes"].to_numpy()

pump = np.asarray(graph.graph["params"]["pump"], dtype=float)
lengths = np.asarray(graph.graph["lengths"], dtype=float)
# The cavity is the pumped region -- the same definition step 5 uses.
inner = pump > 0
print(f"nodes={len(graph)} edges={len(graph.edges)} n_steps={N_STEPS}", flush=True)
print("thresholds", thr, flush=True)


def gain_clamp(k):
    return GAMMA_PERP**2 / ((k - K_A) ** 2 + GAMMA_PERP**2)


def intensity(field, amplitude):
    """``a * int_cavity f(x) dx`` with the profile sampled inside each edge."""
    total = 0.0
    for edge_index in np.flatnonzero(inner):
        samples = np.asarray(field[edge_index], dtype=float)
        if samples.size < 2:
            total += float(lengths[edge_index]) * float(np.mean(samples))
            continue
        grid = np.linspace(0.0, float(lengths[edge_index]), samples.size)
        total += float(simpson(samples, x=grid))
    return float(amplitude) * total


d0_grid = np.round(np.arange(0.58, D0_MAX + 1e-9, D0_STEP), 4)
state: dict[int, tuple[float, float]] = {}
active: list[int] = []
records = []

for d0 in d0_grid:
    solution = None
    while True:
        if active:
            solution = solve_salt_varying(
                graph,
                [state[i][0] for i in active],
                [state[i][1] for i in active],
                float(d0),
                pump,
                n_steps=N_STEPS,
                residual_tol=1e-8,
            )
            for slot, i in enumerate(active):
                state[i] = (float(solution.ks[slot]), float(solution.amplitudes[slot]))
            res = float(np.max(solution.residuals)) if solution.residuals.size else 0.0
        else:
            res = 0.0
        cand = [
            i for i in np.argsort(thr) if i not in active and thr[i] < d0 and len(active) < N_MODES
        ]
        best, best_alpha, best_k = None, -1e-9, 0.0
        for c in cand:
            k_c = float(np.real(tms[c]))
            if active and solution is not None:
                profiles = saturated_eps_profiles(
                    graph, solution.ks, solution.amplitudes, solution.fields, float(d0), pump
                )
                gap = min(abs(k_c - float(k)) for k in solution.ks)
                window = float(np.clip(0.2 * gap, 1e-6, 0.3))
                kk, alpha = net_gain_alpha(graph, k_c, profiles, n_steps=N_STEPS, k_window=window)
            else:
                alpha, kk = -1.0, k_c
            if alpha < best_alpha:
                best, best_alpha, best_k = c, alpha, kk
        if best is None:
            break
        a0 = 1e-3 if not active else 1e-2 * max(state[i][1] for i in active)
        state[best] = (best_k, a0)
        active.append(int(best))
        active.sort()
        print(f"  D0={d0}: admitted {int(best)} (alpha={best_alpha:.2e})", flush=True)

    rec = {"D0": float(d0), "active": [int(i) for i in active], "residual": res}
    for slot, i in enumerate(active):
        k, a = state[i]
        rec[f"k_{i}"] = float(k)
        rec[f"a_raw_{i}"] = float(a)
        rec[f"I_{i}"] = intensity(solution.fields[slot], a)
        rec[f"Gam_{i}"] = gain_clamp(float(k))
    records.append(rec)
    print(
        f"D0={d0:.3f} active={active} "
        + " ".join(f"[m{i}: k={rec[f'k_{i}']:.9f} I={rec[f'I_{i}']:.6e}]" for i in active)
        + f"  res={res:.1e} conv={solution.converged if solution else True}",
        flush=True,
    )

json.dump(
    {"resolution": N_STEPS, "work_nodes": len(graph), "records": records},
    open(OUT, "w"),
    indent=1,
)
print("wrote", OUT)

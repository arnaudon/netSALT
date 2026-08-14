"""Step 5: netsalt's full-SALT solve on the same cavity, at a *given* active set.

Uses the public ``solve_salt_fixed_set`` (no discovery layer), at a series of
pumps and for several oversampling resolutions, and reports the two
convention-free quantities:

  * the real lasing frequencies ``k_mu``;
  * the physical modal intensity ``I_mu = int_cavity |Psi_mu|^2 dx``, which in
    netsalt's own variables is ``a_mu * sum_{inner e} l_e f_mu,e`` -- because
    the hole-burning denominator is ``1 + sum_nu Gamma_nu a_nu f_nu(x)``, the
    product ``a_nu f_nu`` *is* ``|Psi_nu(x)|^2``, so it carries no normalisation
    convention.

Usage: python step5_netsalt_salt_run.py <resolution> <n_modes>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from setup_case import GAMMA_PERP, K_A, build_graph

import netsalt.modes as M
import netsalt.quantum_graph as QG
from netsalt.modes import _auto_oversample_size  # noqa: PLC2701
from netsalt.quantum_graph import graph_with_pump, oversample_graph
from netsalt.utils import from_complex

TAG = os.environ.get("NS_TAG", "")

RESOLUTION = int(sys.argv[1]) if len(sys.argv) > 1 else 12
N_MODES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
D0_MAX = float(sys.argv[3]) if len(sys.argv) > 3 else 1.4
OUT = sys.argv[4] if len(sys.argv) > 4 else f"results_step5_netsalt_res{RESOLUTION}.json"

tdf = pd.read_hdf(f"netsalt_threshold_modes{TAG}.h5", key="modes")
g = build_graph()
thr = np.asarray(tdf["lasing_thresholds"]).ravel()
tms = tdf["threshold_lasing_modes"].to_numpy()
print("thresholds", thr)

size = _auto_oversample_size(g, tdf, resolution=RESOLUTION, node_cap=100000)
work = oversample_graph(g, size)
pump = np.asarray(work.graph["params"]["pump"], dtype=float)
pump_mask = M._get_mask_matrices(work.graph["params"])[1]
lengths = np.asarray(work.graph["lengths"], dtype=float)
inner = np.asarray(work.graph["params"]["inner"], dtype=bool)
# NB: after oversampling netsalt re-derives ``inner`` from node degree, so only
# the single sub-edge touching each degree-1 node stays "outer" and most of the
# vacuum lead is relabelled inner.  The cavity is the *pumped* region.
inner = pump > 0
print(
    f"resolution={RESOLUTION} oversample_size={size:.5g} work nodes={len(work)} edges={len(work.edges)}",
    flush=True,
)
# x coordinate of each edge midpoint, shifted so the cavity starts at x = 0
_pos = {n: float(work.nodes[n]["position"][0]) for n in work.nodes}
_xmid = np.array([0.5 * (_pos[u] + _pos[v]) for u, v in work.edges])
_x0 = min(min(_pos[u], _pos[v]) for (u, v), p in zip(work.edges, pump, strict=True) if p > 0)
_xmid = _xmid - _x0

QG.DENSE_EIG_MAX = min(QG.DENSE_EIG_MAX, M.NEWTON_DENSE_EIG_MAX)

# ---- initial per-mode profiles (each at its own threshold) ----------------
modes0, fields0 = [], []
for i in range(len(tdf)):
    mode = np.asarray(from_complex(tms[i]), dtype=float)
    fields0.append(
        M._single_mode_field_intensity(graph_with_pump(work, float(thr[i])), mode, pump_mask)
    )
    modes0.append(mode)

order = list(np.argsort(thr))  # turn-on order
D0_STEP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02
D0_START = float(os.environ.get("NS_D0_START", 0.58))
d0_grid = np.round(np.arange(D0_START, D0_MAX + 1e-9, D0_STEP), 4)

state = {}  # mode_id -> (mode, amp, field)
active: list[int] = []
records = []


def gain_clamp(k):
    return GAMMA_PERP**2 / ((k - K_A) ** 2 + GAMMA_PERP**2)


for d0 in d0_grid:
    # admit modes whose *interacting* turn-on has passed: probe net gain on the
    # current saturated background, exactly as netsalt's own discovery does.
    while True:
        if active:
            sol = M.solve_salt_fixed_set(
                work,
                [state[i][0] for i in active],
                [state[i][1] for i in active],
                [state[i][2] for i in active],
                float(d0),
                pump,
                pump_mask,
                seed=42,
                max_steps=60,
                outer=40,
                residual_tol=1e-9,
            )
            for j, i in enumerate(active):
                state[i] = (sol.ks[j], float(sol.amplitudes[j]), sol.fields[j])
            res = float(np.max(sol.residuals)) if sol.residuals.size else 0.0
        else:
            res = 0.0
        cand = [i for i in order if i not in active and thr[i] < d0 and len(active) < N_MODES]
        best, best_alpha, best_k = None, -1e-9, 0.0
        for c in cand:
            if active:
                background = M._saturated_graph_multi(
                    work,
                    [state[i][0] for i in active],
                    [state[i][1] for i in active],
                    float(d0),
                    pump,
                    [state[i][2] for i in active],
                )
                gap = min(abs(float(modes0[c][0]) - float(state[i][0][0])) for i in active)
                probed = M._refine_local(
                    modes0[c],
                    background,
                    1e-10,
                    60,
                    42,
                    k_window=float(np.clip(0.2 * gap, 1e-6, 0.3)),
                )
                alpha, kk = float(probed[1]), float(probed[0])
            else:
                alpha, kk = -1.0, float(modes0[c][0])
            if alpha < best_alpha:
                best, best_alpha, best_k = c, alpha, kk
        if best is None:
            break
        a0 = 1e-3 if not active else 1e-2 * max(state[i][1] for i in active)
        state[best] = (np.array([best_k, 0.0]), a0, fields0[best])
        active.append(best)
        active.sort()
        print(
            f"  D0={d0}: admitted {int(best)} (alpha={best_alpha:.2e}) -> {[int(i) for i in active]}",
            flush=True,
        )

    rec = {"D0": float(d0), "active": [int(i) for i in active], "residual": res}
    for i in active:
        mode, a, f = state[i]
        k = float(mode[0])
        rec[f"k_{i}"] = k
        rec[f"a_raw_{i}"] = a
        rec[f"I_{i}"] = float(a * np.sum(lengths[inner] * np.asarray(f)[inner]))
        rec[f"Gam_{i}"] = gain_clamp(k)
        rec[f"psi2_{i}"] = (a * np.asarray(f)[inner]).tolist()
    records.append(rec)
    print(
        f"D0={d0:.3f} active={active} "
        + " ".join(f"[m{i}: k={rec[f'k_{i}']:.9f} I={rec[f'I_{i}']:.6e}]" for i in active)
        + f"  res={res:.1e}",
        flush=True,
    )

json.dump(
    {
        "resolution": RESOLUTION,
        "oversample_size": float(size),
        "work_nodes": len(work),
        "x_inner": _xmid[inner].tolist(),
        "len_inner": lengths[inner].tolist(),
        "records": records,
    },
    open(OUT, "w"),
    indent=1,
)
print("wrote", OUT)

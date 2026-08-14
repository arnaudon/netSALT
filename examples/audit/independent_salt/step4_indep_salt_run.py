"""Step 4: MY independent full-SALT pump sweep (1, 2 and 3 co-lasing modes).

Continuation in D0.  A mode is admitted to the active set when the *saturated*
background at the current D0 has net gain for it -- i.e. when the single-mode
secular function of the saturated medium, restricted to real k near that mode,
crosses zero from the lossy side.  In practice we simply admit each candidate
once D0 passes its interacting turn-on, detected by solving with it included
and checking that its amplitude comes out positive.

Outputs a json with, per pump and per mode: real lasing frequency k_mu and the
physical intensity a_mu = int_cavity |Psi_mu|^2 dx (both convention free).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from indep_salt import FDSalt, field_tm, gamma, gamma_clamp

TAG = os.environ.get("NS_TAG", "")
L, EPS, N_IDX = 0.5, 9.0, 3.0
K_A = float(os.environ.get("NS_K_A", 15.0))
GP = float(os.environ.get("NS_GAMMA_PERP", 3.0))

N_GRID = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OUT = sys.argv[2] if len(sys.argv) > 2 else f"results_step4_indep_N{N_GRID}.json"

thr = json.load(open(f"results_step3_thresholds{TAG}.json"))
K_THR = np.array(thr["k_thr"])  # [m=6, m=7, m=8]
D0_THR = np.array(thr["D0_thr"])

N_INNER = 9
PUMP_EDGES = [
    int(v)
    for v in os.environ.get("NS_PUMP_EDGES", ",".join(str(i) for i in range(N_INNER))).split(",")
]
PUMP_SLAB = np.array([1.0 if i in PUMP_EDGES else 0.0 for i in range(N_INNER)])
WIDTHS9 = np.full(N_INNER, L / N_INNER)
PUMP_GRID = np.repeat(PUMP_SLAB, N_GRID // N_INNER + 1)[: N_GRID + 1]
s = FDSalt(L, EPS, N_GRID, K_A, GP, pump=PUMP_GRID)


def seed(mode_id, amp):
    k = K_THR[mode_id]
    eps_slab = EPS + gamma(k, K_A, GP) * D0_THR[mode_id] * PUMP_SLAB
    psi = field_tm(k, WIDTHS9, eps_slab, s.x)
    nrm = np.sqrt(np.sum(s.w * np.abs(psi) ** 2))
    phi = psi / nrm
    i0 = int(np.argmax(np.abs(phi)))
    phi = phi * np.exp(-1j * np.angle(phi[i0]))
    return phi, float(amp), float(k), i0


def net_gain_alpha(active, phis, amps, ks, d0, k_probe):
    """alpha = -Im k of the probe mode on the current saturated background.

    Solves the *linear* problem in the medium saturated by the active modes,
    i.e. exactly the 'does this mode see net gain' question.
    """
    from scipy.optimize import root

    sden = np.ones(s.N + 1)
    for phi, a, k in zip(phis, amps, ks, strict=True):
        sden = sden + gamma_clamp(k, K_A, GP) * a * np.abs(phi) ** 2
    # piecewise-constant slabs on the FD grid (cell centred)
    widths = np.diff(s.x)
    dmid = 0.5 * (sden[:-1] + sden[1:])
    pmid = 0.5 * (PUMP_GRID[:-1] + PUMP_GRID[1:])

    def res(v):
        kk = v[0] + 1j * v[1]
        eps_sl = EPS + gamma(kk, K_A, GP) * d0 * pmid / dmid
        from indep_salt import secular_tm

        f = secular_tm(kk, widths, eps_sl)
        return [f.real, f.imag]

    sol = root(res, [k_probe, 0.0], method="hybr", tol=1e-12)
    return sol.x[0], -sol.x[1]  # (k, alpha)


order = np.argsort(D0_THR)  # turn-on order by non-interacting threshold
D0_MAX = float(sys.argv[3]) if len(sys.argv) > 3 else 1.4
D0_STEP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
D0_START = float(os.environ.get("NS_D0_START", 0.58))
NSAMP = int(sys.argv[5]) if len(sys.argv) > 5 else 401
d0_grid = np.round(np.arange(D0_START, D0_MAX + 1e-9, D0_STEP), 4)

active: list[int] = []
state: dict[int, tuple] = {}
records = []

for d0 in d0_grid:
    # admit any candidate below its non-interacting threshold check first
    while True:
        if active:
            phis = [state[i][0] for i in active]
            amps = [state[i][1] for i in active]
            ks = [state[i][2] for i in active]
            i0s = [state[i][3] for i in active]
            z0 = s.pack(phis, np.array(amps), np.array(ks))
            z, phis, amps, ks, nrm = s.solve(z0, len(active), float(d0), i0s)
            for j, i in enumerate(active):
                state[i] = (phis[j], float(amps[j]), float(ks[j]), i0s[j])
            if nrm > 1e-6:
                print(f"  WARNING D0={d0} residual {nrm:.2e}")
        # look for a new mode with net gain on the saturated background
        cand = [i for i in order if i not in active and D0_THR[i] < d0]
        best, best_alpha, best_k = None, -1e-9, 0.0
        for c in cand:
            if active:
                kk, al = net_gain_alpha(
                    active,
                    [state[i][0] for i in active],
                    [state[i][1] for i in active],
                    [state[i][2] for i in active],
                    float(d0),
                    float(K_THR[c]),
                )
            else:
                kk, al = float(K_THR[c]), -1.0
            if al < best_alpha:
                best, best_alpha, best_k = c, al, kk
        if best is None:
            break
        amp0 = 1e-3 if not active else 1e-2 * max(state[i][1] for i in active)
        phi, _, _, i0 = seed(best, amp0)
        state[best] = (phi, amp0, best_k, i0)
        active.append(best)
        active.sort()
        print(
            f"  D0={d0}: admitted mode {int(best)} (alpha={best_alpha:.3e}) -> active {[int(i) for i in active]}",
            flush=True,
        )

    rec = {
        "D0": float(d0),
        "active": [int(i) for i in active],
        "residual": float(nrm) if active else 0.0,
    }
    for i in active:
        phi, a, k, _ = state[i]
        rec[f"k_{i}"] = float(k)
        rec[f"a_{i}"] = float(a)
        rec[f"Gam_{i}"] = float(gamma_clamp(k, K_A, GP))
        # physical |Psi|^2 on a fixed 401-point sample of the cavity
        idx = np.linspace(0, s.N, NSAMP).astype(int)
        rec[f"psi2_{i}"] = (a * np.abs(phi[idx]) ** 2).tolist()
    records.append(rec)
    print(
        f"D0={d0:.3f} active={[int(i) for i in active]} "
        + " ".join(f"[m{i}: k={state[i][2]:.9f} a={state[i][1]:.9e}]" for i in active)
        + f"  |R|={nrm:.1e}",
        flush=True,
    )

json.dump({"N_grid": N_GRID, "n_samp": NSAMP, "records": records}, open(OUT, "w"), indent=1)
print("wrote", OUT)

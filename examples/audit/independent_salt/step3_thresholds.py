"""Step 3: independent non-interacting lasing thresholds, vs netsalt."""

import json
import os

import numpy as np
from indep_salt import FDSalt, threshold_tm

N_INNER = 9
PUMP_EDGES = [
    int(v)
    for v in os.environ.get("NS_PUMP_EDGES", ",".join(str(i) for i in range(N_INNER))).split(",")
]
WIDTHS = np.full(N_INNER, 0.5 / N_INNER)
PUMP = np.array([1.0 if i in PUMP_EDGES else 0.0 for i in range(N_INNER)])

TAG = os.environ.get("NS_TAG", "")
L, EPS, N_IDX = 0.5, 9.0, 3.0
K_A = float(os.environ.get("NS_K_A", 15.0))
GP = float(os.environ.get("NS_GAMMA_PERP", 3.0))

ns = json.load(open(f"results_step2_netsalt_passive{TAG}.json"))
thr_ns = np.array(ns["thresholds"])
kthr_ns = np.array([c[0] for c in ns["k_thr"]])

m_list = [int(v) for v in os.environ.get("NS_M_LIST", "6,7,8").split(",")]
k_pass = np.array([np.pi * m / (N_IDX * L) for m in m_list])

print("non-interacting thresholds: independent (exact transfer matrix) vs netsalt")
rows = []
for i, (m, _k0) in enumerate(zip(m_list, k_pass, strict=True)):
    k_t, d0_t, res = threshold_tm(
        kthr_ns[i], EPS, L, K_A, GP, d0_guess=thr_ns[i], pump=PUMP, widths=WIDTHS
    )
    rows.append((m, k_t, d0_t, res))
    print(
        f"  m={m}: indep D0_thr={d0_t:.12f} k_thr={k_t:.12f}  (|secular|={res:.1e})\n"
        f"        netsalt D0_thr={thr_ns[i]:.12f} k_thr={kthr_ns[i]:.12f}\n"
        f"        rel err  D0: {abs(d0_t - thr_ns[i]) / d0_t:.3e}   k: {abs(k_t - kthr_ns[i]) / k_t:.3e}"
    )

print("\nsame thresholds from MY finite-difference discretisation (grid convergence):")
for n_grid in [1000, 2000, 4000, 8000]:
    s = FDSalt(L, EPS, n_grid, K_A, GP, pump=np.repeat(PUMP, n_grid // N_INNER + 1)[: n_grid + 1])
    out = []
    for i in range(len(m_list)):
        kf, df = s.threshold_fd(kthr_ns[i], d0_guess=thr_ns[i])
        out.append((kf, df))
    print(
        f"  N={n_grid:5d}  "
        + "  ".join(f"dD0={abs(out[i][1] - rows[i][2]):.2e}" for i in range(len(m_list)))
        + "   "
        + "  ".join(f"dk={abs(out[i][0] - rows[i][1]):.2e}" for i in range(len(m_list)))
    )

json.dump(
    {
        "m": m_list,
        "k_thr": [r[1] for r in rows],
        "D0_thr": [r[2] for r in rows],
        "pump_edges": PUMP_EDGES,
    },
    open(f"results_step3_thresholds{TAG}.json", "w"),
    indent=1,
)
print("\nwrote results_step3_thresholds.json")

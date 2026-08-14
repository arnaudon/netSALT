"""Step 1: verify MY solver's passive layer against the closed form.

Analytic passive modes of a uniform slab of index n in vacuum, outgoing both
ends:  exp(2 i q L) = ((q+k)/(q-k))^2 with q = n k, i.e. for n = 3
exp(2 i n k L) = 4, so   k_m = pi m /(n L) - i ln(4)/(2 n L).
"""

import json

import numpy as np
from indep_salt import FDSalt, passive_modes_tm

L, EPS, N_IDX = 0.5, 9.0, 3.0
K_A, GP = 15.0, 3.0

m_list = [6, 7, 8]
k_exact = np.array([np.pi * m / (N_IDX * L) - 1j * np.log(4.0) / (2 * N_IDX * L) for m in m_list])

k_tm = passive_modes_tm(k_exact + 0.01, EPS, L)
print("analytic vs transfer-matrix passive modes")
for m, ke, kt in zip(m_list, k_exact, k_tm, strict=True):
    print(f"  m={m}  exact k={ke.real:.15f} {ke.imag:+.15f}i   TM err={abs(kt - ke):.3e}")

print("\nfinite-difference grid convergence (|k_FD - k_exact|)")
rows = []
for n_grid in [500, 1000, 2000, 4000, 8000, 16000]:
    s = FDSalt(L, EPS, n_grid, K_A, GP)
    errs = []
    for ke in k_exact:
        kfd = s.passive_mode_fd(ke)
        errs.append(abs(kfd - ke))
    rows.append((n_grid, errs))
    print(f"  N={n_grid:6d}  " + "  ".join(f"{e:.3e}" for e in errs))
print("\n  ratios between successive N (expect ~4 for O(h^2)):")
for j in range(1, len(rows)):
    print(
        f"  N {rows[j - 1][0]}->{rows[j][0]}: "
        + "  ".join(f"{rows[j - 1][1][i] / rows[j][1][i]:.2f}" for i in range(len(m_list)))
    )

out = {
    "k_exact": [[k.real, k.imag] for k in k_exact],
    "k_tm": [[k.real, k.imag] for k in k_tm],
    "fd_convergence": [[n, e] for n, e in rows],
}
with open("results_step1_passive_indep.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote results_step1_passive_indep.json")

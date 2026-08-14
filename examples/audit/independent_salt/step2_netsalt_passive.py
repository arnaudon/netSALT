"""Step 2: netsalt's passive modes + non-interacting thresholds on the same cavity."""

import json
import os

import numpy as np
from setup_case import L_CAV, N_CAV, build_graph

from netsalt.modes import find_passive_modes, find_threshold_lasing_modes, pump_trajectories
from netsalt.utils import to_complex

TAG = os.environ.get("NS_TAG", "")

g = build_graph()

passive = find_passive_modes(g, method="contour")
ks = np.array([to_complex(m) for m in passive["passive"]])
order = np.argsort(ks.real)
ks = ks[order]
passive = passive.iloc[order].reset_index(drop=True)
print("netsalt passive modes in [11,19]:")
for k in ks:
    print(f"  k = {k.real:.15f} {k.imag:+.15f}i")

m_list = [int(v) for v in os.environ.get("NS_M_LIST", "6,7,8").split(",")]
k_exact = np.array(
    [np.pi * m / (N_CAV * L_CAV) - 1j * np.log(4.0) / (2 * N_CAV * L_CAV) for m in m_list]
)
print("\ncomparison with the closed form (uniform slab, outgoing both ends):")
matched = []
for ke in k_exact:
    j = int(np.argmin(np.abs(ks - ke)))
    matched.append(j)
    print(
        f"  exact {ke.real:.12f}{ke.imag:+.12f}i   netsalt {ks[j].real:.12f}{ks[j].imag:+.12f}i"
        f"   |dk|={abs(ks[j] - ke):.3e}  rel={abs(ks[j] - ke) / abs(ke):.3e}"
    )

# ---- thresholds -------------------------------------------------------
sub = passive.iloc[matched].reset_index(drop=True)
traj = pump_trajectories(sub, g, return_approx=True)
tdf = find_threshold_lasing_modes(traj, g)
thr = np.asarray(tdf["lasing_thresholds"]).ravel()
tms = np.array([to_complex(m) for m in tdf["threshold_lasing_modes"]])
print("\nnetsalt non-interacting thresholds:")
for i, (t, km) in enumerate(zip(thr, tms, strict=True)):
    print(f"  mode {i}: D0_thr = {t:.12f}   k_thr = {km.real:.12f}{km.imag:+.3e}i")

tdf.to_hdf(f"netsalt_threshold_modes{TAG}.h5", key="modes", format="fixed", mode="w")
json.dump(
    {
        "passive_k": [[k.real, k.imag] for k in ks],
        "matched_index": matched,
        "thresholds": [float(t) for t in thr],
        "k_thr": [[k.real, k.imag] for k in tms],
    },
    open(f"results_step2_netsalt_passive{TAG}.json", "w"),
    indent=1,
)
print("\nwrote results_step2_netsalt_passive.json / netsalt_threshold_modes.h5")

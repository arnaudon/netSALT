"""Step 7: reverse cross-residual -- feed MY solution into netsalt's operator.

Takes the ``(k_mu, |Psi_mu(x)|^2)`` my finite-difference solver produced, maps
``|Psi_mu|^2`` onto netsalt's work-graph edges (per-edge means) and calls the
public ``netsalt.salt_residuals``.  A genuine SALT solution makes netsalt's
saturated operator singular at every ``k_mu``.

Usage: python step7_reverse_residual.py <indep-json> <resolution>
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from setup_case import build_graph

import netsalt.modes as M
import netsalt.quantum_graph as QG
from netsalt.modes import _auto_oversample_size, salt_residuals  # noqa: PLC2701
from netsalt.quantum_graph import oversample_graph

INDEP = sys.argv[1] if len(sys.argv) > 1 else "results_step4_indep_N2000.json"
RESOLUTION = int(sys.argv[2]) if len(sys.argv) > 2 else 12

d = json.load(open(INDEP))
tdf = pd.read_hdf("netsalt_threshold_modes.h5", key="modes")
g = build_graph()
size = _auto_oversample_size(g, tdf, resolution=RESOLUTION, node_cap=100000)
work = oversample_graph(g, size)
QG.DENSE_EIG_MAX = min(QG.DENSE_EIG_MAX, M.NEWTON_DENSE_EIG_MAX)
pump = np.asarray(work.graph["params"]["pump"], dtype=float)
lengths = np.asarray(work.graph["lengths"], dtype=float)
pos = {n: float(work.nodes[n]["position"][0]) for n in work.nodes}
lo = np.array([min(pos[u], pos[v]) for u, v in work.edges])
hi = np.array([max(pos[u], pos[v]) for u, v in work.edges])
x0 = lo[pump > 0].min()
lo, hi = lo - x0, hi - x0
print(f"resolution={RESOLUTION}, {int((pump > 0).sum())} pumped sub-edges, work nodes={len(work)}")

x_samp = np.linspace(0.0, 0.5, d.get("n_samp", 401))


def edge_means(psi2):
    """Per-edge mean of my |Psi|^2 profile (0 outside the pumped cavity)."""
    out = np.zeros(len(lengths))
    for e in range(len(lengths)):
        if pump[e] <= 0:
            continue
        xs = np.linspace(lo[e], hi[e], 81)
        out[e] = np.trapezoid(np.interp(xs, x_samp, psi2), xs) / (hi[e] - lo[e])
    return out


print(f"{'D0':>6} {'n':>3} " + " ".join(f"{'|lam| m' + str(i):>12}" for i in range(3)))
rows = []
for r in d["records"]:
    if not r["active"]:
        continue

    ks = [r[f"k_{i}"] for i in r["active"]]
    fields = [edge_means(np.array(r[f"psi2_{i}"])) for i in r["active"]]
    amps = [1.0] * len(ks)
    res = salt_residuals(work, ks, amps, fields, float(r["D0"]), pump, seed=42)
    # sensitivity: d|lambda|/dk from netsalt's own operator, to convert to a dk
    gsat = M._saturated_graph_multi(
        work, [[k, 0.0] for k in ks], np.array(amps), float(r["D0"]), pump, fields
    )
    dk = []
    for j, k in enumerate(ks):
        h = 1e-5
        lp = M._lam_real_k(gsat, k + h, 42)
        lm = M._lam_real_k(gsat, k - h, 42)
        slope = abs(lp - lm) / (2 * h)
        dk.append(res[j] / slope if slope > 0 else np.nan)
    rows.append({"D0": r["D0"], "active": r["active"], "res": res.tolist(), "dk": dk})
    print(
        f"{r['D0']:6.2f} {len(ks):3d} "
        + " ".join(f"{res[j]:.3e}(dk={dk[j]:.1e})" for j in range(len(ks)))
    )

json.dump(rows, open(f"results_step7_reverse_res{RESOLUTION}.json", "w"), indent=1)

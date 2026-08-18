"""Step 8v: the varying-operator sweep against the independent solver.

``step8_compare.py`` Richardson-extrapolates netsalt's *oversampled* sweep in the
oversampling resolution before comparing. The varying path has no oversampling
to extrapolate in -- its convergence parameter is ``n_steps``, the per-edge
sample count -- so this compares directly at the resolution given, and reports
the spread the way step 8 does.

The quantities are convention-free: the real lasing frequency ``k_mu`` and the
physical modal intensity ``I_mu = int_cavity |Psi_mu|^2 dx``.

Usage: python step8v_compare_varying.py [indep.json] [varying.json ...]
"""

from __future__ import annotations

import json
import sys

import numpy as np

indep_path = sys.argv[1] if len(sys.argv) > 1 else "results_step4_indep.json"
varying_paths = sys.argv[2:] or ["results_step5v_varying_n64.json"]

indep = json.load(open(indep_path))
indep_by_d0 = {round(float(r["D0"]), 4): r for r in indep["records"]}

for path in varying_paths:
    data = json.load(open(path))
    print(f"\n=== {path}  (n_steps = {data['resolution']}, {data['work_nodes']} nodes) ===")
    dk, di, rows = [], [], []
    for rec in data["records"]:
        d0 = round(float(rec["D0"]), 4)
        ref = indep_by_d0.get(d0)
        if ref is None:
            continue
        for i in rec["active"]:
            kk, ii = f"k_{i}", f"I_{i}"
            if kk not in ref or ii not in ref:
                continue
            if not np.isfinite(ref[ii]) or ref[ii] <= 0:
                continue
            rel_k = abs(rec[kk] - ref[kk]) / max(abs(ref[kk]), 1e-300)
            rel_i = abs(rec[ii] - ref[ii]) / max(abs(ref[ii]), 1e-300)
            dk.append(rel_k)
            di.append(rel_i)
            rows.append((d0, i, rec[kk], ref[kk], rec[ii], ref[ii], rel_i))
    if not rows:
        print("  no overlapping pump points -- check the two grids match")
        continue
    dk, di = np.array(dk), np.array(di)
    print(
        f"  {len(rows)} (pump, mode) comparisons over D0 = "
        f"{min(r[0] for r in rows):.2f} .. {max(r[0] for r in rows):.2f}"
    )
    print(f"  lasing frequency k_mu : median {np.median(dk):.2e}   max {dk.max():.2e}")
    print(f"  modal intensity I_mu  : median {np.median(di):.2e}   max {di.max():.2e}")
    worst = max(rows, key=lambda r: r[6])
    print(
        f"  worst intensity point : D0={worst[0]:.2f} mode {worst[1]}  "
        f"varying {worst[4]:.6e} vs indep {worst[5]:.6e}  ({worst[6]:.1e})"
    )
    print(
        f"\n  {'D0':>6} {'mode':>4} {'k varying':>14} {'k indep':>14} "
        f"{'I varying':>13} {'I indep':>13} {'rel I':>9}"
    )
    for row in rows[:: max(1, len(rows) // 12)]:
        print(
            f"  {row[0]:6.2f} {row[1]:4d} {row[2]:14.9f} {row[3]:14.9f} "
            f"{row[4]:13.6e} {row[5]:13.6e} {row[6]:9.1e}"
        )

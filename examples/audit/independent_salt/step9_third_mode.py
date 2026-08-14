"""Step 9: how close is the THIRD mode to lasing?

From a saved sweep (either solver's), rebuild the saturated medium at each pump
and solve for the complex k of the not-yet-lasing candidate near k ~ 12.9 with
the exact transfer matrix.  alpha = -Im k < 0 means net gain, i.e. it should
have turned on.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from indep_salt import gamma, secular_tm
from scipy.optimize import root

EPS, K_A = 9.0, 15.0
GP = float(os.environ.get("NS_GAMMA_PERP", 3.0))
path = sys.argv[1]
k_probe = float(sys.argv[2]) if len(sys.argv) > 2 else 12.90
d = json.load(open(path))

if "x_inner" in d:  # netsalt output: profiles live on the pumped sub-edges
    x = np.array(d["x_inner"])
    srt = np.argsort(x)
    widths = np.array(d["len_inner"])[srt]
    key = "psi2_"
else:  # independent output: profiles on a uniform sample of the cavity
    n = d.get("n_samp", 401)
    xs = np.linspace(0.0, 0.5, n)
    widths = np.diff(xs)
    srt = None
    key = "psi2_"

print(f"{'D0':>7} {'k_probe':>13} {'alpha':>13}   (alpha<0 => should lase)")
out = []
for r in d["records"]:
    if not r["active"] or 0 in r["active"]:
        continue
    if srt is not None:
        denom = np.ones(len(widths))
        for i in r["active"]:
            denom = denom + r[f"Gam_{i}"] * np.array(r[key + str(i)])[srt]
    else:
        denom = np.ones(len(xs))
        for i in r["active"]:
            denom = denom + r[f"Gam_{i}"] * np.array(r[key + str(i)])
        denom = 0.5 * (denom[:-1] + denom[1:])
    d0 = r["D0"]

    def res(v, _d=denom, _d0=d0):
        kk = v[0] + 1j * v[1]
        f = secular_tm(kk, widths, EPS + gamma(kk, K_A, GP) * _d0 / _d)
        return [f.real, f.imag]

    sol = root(res, [k_probe, -0.05], method="hybr", tol=1e-13)
    k, alpha = sol.x[0], -sol.x[1]
    out.append({"D0": d0, "k": k, "alpha": alpha})
    print(f"{d0:7.2f} {k:13.7f} {alpha:13.3e}")

json.dump(out, open(path.replace(".json", "_thirdmode.json"), "w"), indent=1)

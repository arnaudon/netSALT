"""Step 6: cross-residual test -- feed netsalt's SALT solution into MY equations.

This is unit-free and assumption-free apart from one identification: in netsalt
the hole-burning denominator is ``1 + sum_nu Gamma_nu a_nu f_nu(x)``, so the
product ``a_nu f_nu(x)`` *is* the physical ``|Psi_nu(x)|^2``.  Given that, the
saturated permittivity profile is fully determined, and an independent solver
must find the operator singular at netsalt's reported ``k_mu``.

We evaluate my exact transfer-matrix secular function on netsalt's own
piecewise-constant saturated profile.  For scale, we also report the secular
function's sensitivity dF/dk, so |F| can be converted into an equivalent error
in k.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from indep_salt import gamma, secular_tm

EPS, K_A = 9.0, 15.0
GP = float(os.environ.get("NS_GAMMA_PERP", 3.0))
path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/quick.json"
d = json.load(open(path))
x = np.array(d["x_inner"])
lens = np.array(d["len_inner"])
srt = np.argsort(x)
lens = lens[srt]
print(f"{path}: {len(lens)} inner sub-edges, total length {lens.sum():.6f}")

print(f"{'D0':>6} {'mode':>5} {'k_netsalt':>15} {'|F(k)|':>11} {'|dF/dk|':>11} {'dk_equiv':>11}")
out = []
for r in d["records"]:
    d0 = r["D0"]
    denom = np.ones(len(lens))
    for i in r["active"]:
        psi2 = np.array(r[f"psi2_{i}"])[srt]
        denom = denom + r[f"Gam_{i}"] * psi2
    for i in r["active"]:
        k = r[f"k_{i}"]
        eps_slabs = EPS + gamma(k, K_A, GP) * d0 / denom
        f0 = secular_tm(k, lens, eps_slabs)
        h = 1e-6
        df = (secular_tm(k + h, lens, eps_slabs) - secular_tm(k - h, lens, eps_slabs)) / (2 * h)
        dk = abs(f0) / abs(df)
        out.append({"D0": d0, "mode": i, "k": k, "absF": abs(f0), "dk": dk})
        print(f"{d0:6.2f} {i:5d} {k:15.9f} {abs(f0):11.3e} {abs(df):11.3e} {dk:11.3e}")

json.dump(out, open(path.replace(".json", "_crossres.json"), "w"), indent=1)

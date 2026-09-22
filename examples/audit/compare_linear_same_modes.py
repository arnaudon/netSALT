"""Split linear's error into "wrong mode count" and "wrong intensities".

Full SALT lases five modes above 1.085x; linear lases six, and its five
survivors come out 22-82 % below SALT's by 1.16x. Those are not independent: if
linear is made to lase the SAME five -- by striking the extinguished mode from
its candidate set -- its remaining modes inherit the gain it was consuming.

  * linear-5 ~= full SALT   -> linear's only real error is the mode count, and
                              everything downstream follows from it.
  * linear-5 still below    -> there is a genuine intensity error on top, i.e.
                              physics in the saturation that the linearised
                              competition matrix cannot represent.

Also reports the competition matrix's conditioning, since netsalt already warns
that a near-degenerate pair leaves the split between them unresolvable -- worth
knowing whether the linear model flags this case itself.
"""

import os  # noqa: E402
from pathlib import Path  # noqa: E402

# The fixture these run on: the checked-in buffon over the production k window.
# Resolved from this file rather than hardcoded, so the script works from any
# working directory and any checkout. `bash run.sh` there builds out/ first.
os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition")


import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.simplefilter("ignore")
from netsalt import pipeline as pl  # noqa: E402
from netsalt.config_loader import load_config  # noqa: E402
from netsalt.io import load_modes  # noqa: E402
from netsalt.modes import (  # noqa: E402
    competition_conditioning,
    compute_modal_intensities,
    compute_mode_competition_matrix,
)

p = load_config("config.yaml")
p["out_folder"] = "out"
qg = pl.step_create_quantum_graph(p)
md = load_modes("out/passive_modes.h5")
pump = pl.step_create_pump_profile(p, qg, md, None)
qg = pl._attach_pump_to_graph(p, qg, pump)
tr = pl.step_compute_mode_trajectories(p, qg, md, pump, None)
th = pl.step_find_threshold_modes(p, qg, tr, pump, None)
thr = np.asarray(th["lasing_thresholds"]).ravel()
tlm = th["threshold_lasing_modes"].to_numpy()
thr0 = float(np.nanmin(thr))
order = list(np.argsort(thr))[:6]
T = compute_mode_competition_matrix(qg, th)
DOOMED = order[3]

cand = [int(i) for i in np.where(np.isfinite(thr))[0]]
print(
    f"competition matrix conditioning over {len(cand)} candidates: "
    f"{competition_conditioning(T, cand):.3e}"
)
pair = [order[0], order[3]]
print(
    f"  restricted to the near-degenerate pair (k={float(np.real(tlm[pair[0]])):.6f}, "
    f"{float(np.real(tlm[pair[1]])):.6f}): {competition_conditioning(T, pair):.3e}"
)
print(
    f"  the other five                       : "
    f"{competition_conditioning(T, [i for i in order if i != DOOMED]):.3e}"
)

th5 = th.copy()
th5["lasing_thresholds"] = thr.copy()
th5.loc[DOOMED, "lasing_thresholds"] = np.inf


def intensities(frame, mult):
    o = compute_modal_intensities(frame, thr0 * mult, T)
    cols = [c for c in o.columns if c[0] == "modal_intensities"]
    return np.nan_to_num(o[cols].to_numpy(dtype=float))[:, -1]


d = np.load("out/ladder.npz")
x, salt = d["mult"], d["salt"]
keep = [j for j in range(6) if j != 3]
print(
    f"\n{'D0/thr':>7} {'SALT tot':>9} {'lin6 tot':>9} {'lin5 tot':>9} "
    f"{'SALT/lin6':>10} {'SALT/lin5':>10}   per-mode SALT/lin5"
)
rows = []
for j, mult in enumerate(x):
    if j == 0:
        continue  # six modes still lasing here
    I6, I5 = intensities(th, float(mult)), intensities(th5, float(mult))
    s = salt[:, j]
    st = s[keep].sum()
    t6 = sum(I6[order[i]] for i in range(6))
    t5 = sum(I5[order[i]] for i in keep)
    per = [100 * (s[i] / I5[order[i]] - 1) for i in keep if I5[order[i]] > 1e-12]
    rows.append((mult, st, t6, t5))
    print(
        f"{mult:7.4f} {st:9.2f} {t6:9.2f} {t5:9.2f} "
        f"{100 * (st / t6 - 1):+9.1f}% {100 * (st / t5 - 1):+9.1f}%   "
        f"{min(per):+6.1f}% .. {max(per):+6.1f}%"
    )
np.savez(
    "out/linear_same_modes.npz",
    mult=np.array([r[0] for r in rows]),
    salt_total=np.array([r[1] for r in rows]),
    lin6_total=np.array([r[2] for r in rows]),
    lin5_total=np.array([r[3] for r in rows]),
)
print("\nwrote out/linear_same_modes.npz")
print("DONE")

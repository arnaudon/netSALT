"""Does an extinguished mode come back at higher pump?

The mode full SALT switches off at 1.0846x is dark because the survivors have
burnt a hole where it lives. But the survivors' profiles keep deforming as the
pump rises, and nothing says the hole has to stay over that mode -- if the
deformation turns in its favour, its net gain climbs back through zero and it
re-ignites. The linear model cannot produce that at all: its competition matrix
is fixed, so once a mode is beaten it stays beaten.

`compute_modal_intensities_varying` already does the right continuation for
this: at every pump it drops members that no longer lase AND re-screens every
candidate -- including ones dropped earlier -- with `net_gain_alpha` on the
incumbents' saturated background, admitting any that have net gain. So the
answer is in the alphas it computes along the way; this just records them.

alpha < 0 means net gain (the mode must lase), alpha > 0 means below threshold.
Watching alpha of the extinguished mode across the pump range answers the
question directly: falling towards zero means re-ignition is coming.
"""

import os  # noqa: E402
from pathlib import Path  # noqa: E402

# The fixture these run on: the checked-in buffon over the production k window.
# Resolved from this file rather than hardcoded, so the script works from any
# working directory and any checkout. `bash run.sh` there builds out/ first.
os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition")


import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.simplefilter("ignore")
import netsalt.salt_varying as sv  # noqa: E402
from netsalt import pipeline as pl  # noqa: E402
from netsalt.config_loader import load_config  # noqa: E402
from netsalt.io import load_modes  # noqa: E402

top = float(sys.argv[1]) if len(sys.argv) > 1 else 1.30
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40

p = load_config("config.yaml")
p["out_folder"] = "out"
qg = pl.step_create_quantum_graph(p)
md = load_modes("out/passive_modes.h5")
pump = pl.step_create_pump_profile(p, qg, md, None)
qg = pl._attach_pump_to_graph(p, qg, pump)
qg.graph["params"]["intensity_varying_samples_per_wavelength"] = 5
tr = pl.step_compute_mode_trajectories(p, qg, md, pump, None)
th = pl.step_find_threshold_modes(p, qg, tr, pump, None)
thr = np.asarray(th["lasing_thresholds"]).ravel()
tlm = th["threshold_lasing_modes"].to_numpy()
thr0 = float(np.nanmin(thr))
kk = np.real(tlm)

print(
    f"grid {1.0:.3f}x .. {top:.3f}x in {steps} steps ({100 * (top - 1) / (steps - 1):.2f}% each)",
    flush=True,
)
print("candidate k:", " ".join(f"{float(k):.4f}" for k in kk), flush=True)
print("thresholds :", " ".join(f"{float(t) / thr0:.4f}" for t in thr), flush=True)

# record every admission test: (pump, candidate k, alpha)
real_alpha = sv.net_gain_alpha
log = []


def traced_alpha(graph, k0, profiles, **kw):
    out = real_alpha(graph, k0, profiles, **kw)
    log.append((float(traced_alpha.D0), float(k0), float(out[1])))
    np.save("reignite_alpha.npy", np.array(log))
    return out


traced_alpha.D0 = 0.0
sv.net_gain_alpha = traced_alpha

real_solve = sv.solve_salt_varying


def traced_solve(graph, ks, amps, D0, pmp, **kw):
    traced_alpha.D0 = D0 / thr0
    t = time.time()
    out = real_solve(graph, ks, amps, D0, pmp, **kw)
    print(
        f"    solve M={len(ks):2d} D0={D0 / thr0:6.4f}x conv={str(out.converged):>5} "
        f"[{time.time() - t:5.0f}s]",
        flush=True,
    )
    return out


sv.solve_salt_varying = traced_solve

t0 = time.time()
out = sv.compute_modal_intensities_varying(qg, th, thr0 * top, D0_steps=steps, n_steps=128)
print(f"TOTAL {time.time() - t0:.0f}s", flush=True)

cols = [c for c in out.columns if c[0] == "modal_intensities"]
salt = np.nan_to_num(out[cols].to_numpy(dtype=float))
d0s = np.array([float(c[1]) for c in cols])
np.savez("reignite.npz", salt=salt, d0=d0s, thr0=thr0, thr=thr, k=kk, alpha=np.array(log))
print(out.attrs["salt_varying_diagnostics"].to_string(), flush=True)
print("\nlasing count per pump:", (salt > 1e-4).sum(axis=0).tolist(), flush=True)
print("DONE", flush=True)

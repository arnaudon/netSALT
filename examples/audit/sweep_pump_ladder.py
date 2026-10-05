"""Buffon six-mode pump ladder, full SALT against the linear model.

Steps the pump upward in small increments, warm-starting each solve from the
previous one -- never a cold solve at a single pump. Seeds the first pump from
the linear model, which is within ~25 % there.

Writes ladder.npz after EVERY solve (the container is reclaimed roughly
hourly), carrying both the full-SALT amplitudes and the linear intensities on
the same grid so the comparison needs no second pass.
"""

import os  # noqa: E402
from pathlib import Path  # noqa: E402

# The fixture: the checked-in buffon over the production k window. Resolved from
# this file, so the script runs from any working directory. `bash run.sh` there
# builds out/ first.
os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition")


import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.simplefilter("ignore")
from netsalt import pipeline as pl  # noqa: E402
from netsalt.config_loader import load_config  # noqa: E402
from netsalt.io import load_modes  # noqa: E402
from netsalt.modes import compute_modal_intensities, compute_mode_competition_matrix  # noqa: E402
from netsalt.salt_varying import solve_salt_varying  # noqa: E402

start = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0769
step = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0077
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
outer = int(sys.argv[4]) if len(sys.argv) > 4 else 80
n_modes = int(sys.argv[5]) if len(sys.argv) > 5 else 6

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
order = list(np.argsort(thr))[:n_modes]
thr0 = float(np.nanmin(thr))
ks = [float(np.real(tlm[i])) for i in order]
all_slots = list(order)
all_k = list(ks)
T = compute_mode_competition_matrix(qg, th)


def linear_at(mult):
    o = compute_modal_intensities(th, thr0 * mult, T)
    cols = [c for c in o.columns if c[0] == "modal_intensities"]
    return np.nan_to_num(o[cols].to_numpy(dtype=float))[:, -1]


seed = sys.argv[6] if len(sys.argv) > 6 else ""
if seed:
    amps = [float(v) for v in seed.split(",")]
    if len(amps) != len(order):
        raise ValueError(f"seed has {len(amps)} amplitudes for {len(order)} modes")
else:
    amps = [max(float(linear_at(start)[i]), 1e-3) for i in order]
print(
    f"{n_modes} modes, {start:.4f}x .. {start + (n - 1) * step:.4f}x, "
    f"{100 * step:.2f}% steps, outer={outer}",
    flush=True,
)
print("k      :", " ".join(f"{k:9.5f}" for k in ks), flush=True)
print("seed   :", " ".join(f"{a:9.3f}" for a in amps), flush=True)

mults, salt, lin, conv, res, slots = [], [], [], [], [], []
for j in range(n):
    mult = start + j * step
    # the slots this solve is about to run with -- recorded before the active set
    # can shrink below, so the amplitude row and the slot row stay the same length
    slot_row = list(order)
    t = time.time()
    sol = solve_salt_varying(qg, ks, amps, thr0 * mult, pump, n_steps=128, outer=outer)
    if sol.converged:
        live = [j for j in range(len(sol.ks)) if float(sol.amplitudes[j]) > 1e-4]
        if len(live) < len(sol.ks):
            print(
                f"    dropping {len(sol.ks) - len(live)} extinguished mode(s): "
                + " ".join(
                    f"k={float(sol.ks[j]):.5f}" for j in range(len(sol.ks)) if j not in live
                ),
                flush=True,
            )
        ks = [float(sol.ks[j]) for j in live]
        amps = [float(sol.amplitudes[j]) for j in live]
        order = [order[j] for j in live]
    mults.append(mult)
    salt.append([float(v) for v in sol.amplitudes])
    live_res = [float(r) for j, r in enumerate(sol.residuals) if float(sol.amplitudes[j]) > 1e-4]
    print("    live residuals:", " ".join(f"{r:8.1e}" for r in live_res), flush=True)
    lin.append([float(linear_at(mult)[i]) for i in all_slots])
    slots.append(slot_row)
    conv.append(bool(sol.converged))
    res.append(float(max(live_res)) if live_res else float("nan"))
    n_all = len(all_slots)
    S = np.zeros((n_all, len(mults)))
    L = np.zeros((n_all, len(mults)))
    for col, (row_s, row_l, row_slot) in enumerate(zip(salt, lin, slots, strict=True)):
        for v_s, slot in zip(row_s, row_slot, strict=True):
            S[all_slots.index(slot), col] = v_s
        L[:, col] = row_l
    np.savez(
        "out/ladder.npz",
        mult=np.array(mults),
        salt=S,
        lin=L,
        conv=np.array(conv),
        res=np.array(res),
        k=np.array(all_k),
        thr0=thr0,
    )
    print(
        f"  D0={mult:.4f}x conv={str(sol.converged):>5} "
        f"res={max(live_res) if live_res else float('nan'):.1e} "
        f"it={sol.iterations:3d} [{time.time() - t:5.0f}s]  a="
        + " ".join(f"{v:8.3f}" for v in sol.amplitudes),
        flush=True,
    )
    if not sol.converged:
        print("  -> stopped: did not converge", flush=True)
        break
print("DONE", flush=True)

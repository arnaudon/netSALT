"""Sweep the pump DOWN through the switch, and look for hysteresis.

Upward, the laser holds the six-mode branch with the dying mode's intensity
still rising (3.445 -> 3.47 -> 3.50), then that branch stops existing between
1.0760x and 1.0765x and the state drops to five modes. probe_extinction /
fold testing shows both branches converging over ~1.073x-1.076x, with the sixth
mode genuinely lossy on the five-mode background there -- two stable states at
one pump.

If that is right the switch is first order and must show hysteresis: coming back
DOWN from a five-mode state the laser should stay on five modes well below the
pump where it jumped off six, until the five-mode state itself loses stability
(the sixth mode regains net gain) and it jumps back up.

That is an operational test which needs no continuation through the fold, so it
does not depend on reading a failed solve as a branch ending. It is also
something the linear competition matrix cannot produce at all: its solution is
unique by construction, so it has no branches to be trapped on and no history to
depend on.

Reports, at each pump going down: whether the five-mode state still converges,
and the sixth mode's alpha on it. alpha crossing back below zero marks the lower
edge of the bistable window.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition")

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.simplefilter("ignore")
from netsalt import pipeline as pl  # noqa: E402
from netsalt.config_loader import load_config  # noqa: E402
from netsalt.io import load_modes  # noqa: E402
from netsalt.salt_varying import (  # noqa: E402
    _resolved_n_steps,
    net_gain_alpha,
    net_gain_window,
    saturated_eps_profiles,
    solve_salt_varying,
)

TOP = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0800
BOT = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0650
STEP = (float(sys.argv[3]) if len(sys.argv) > 3 else 0.1) / 100.0

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
thr0 = float(np.nanmin(thr))
pump = np.asarray(pump, dtype=float)

# the five-mode state reached by sweeping UP through the switch
low = np.load("out/fine_low.npz")
row = int(np.argmin(np.abs(low["mult"] - TOP)))
ids = [int(i) for i in low["ids"][row] if i >= 0]
ks = [float(v) for v, i in zip(low["ks"][row], low["ids"][row], strict=True) if i >= 0]
amps = [float(v) for v, i in zip(low["amps"][row], low["ids"][row], strict=True) if i >= 0]
kk = np.real(low["k"])
DOOM_K = 10.680091
print(f"starting DOWN from the five-mode state at {float(low['mult'][row]):.4f}x", flush=True)
print(
    "  " + "  ".join(f"{float(kk[i]):.4f}:{a:.2f}" for i, a in zip(ids, amps, strict=True)),
    flush=True,
)
n_steps = _resolved_n_steps(qg, float(np.max(np.abs(ks))), 128, pump)

print(f"\n{'pump':>8} {'M':>3} {'conv':>6} {'alpha of the dropped mode':>26}  state", flush=True)
rows = []
for mult in np.arange(TOP, BOT - 0.5 * STEP, -STEP):
    D0 = thr0 * float(mult)
    t = time.time()
    sol = solve_salt_varying(qg, ks, amps, D0, pump, n_steps=128, outer=80)
    live = [j for j in range(len(ks)) if float(sol.amplitudes[j]) > 1e-4]
    if sol.converged and len(live) == len(ks):
        ks = [float(v) for v in sol.ks]
        amps = [float(v) for v in sol.amplitudes]
    alpha = float("nan")
    if sol.converged:
        prof = saturated_eps_profiles(
            qg, list(sol.ks), list(sol.amplitudes), list(sol.fields), D0, pump
        )
        _, alpha = net_gain_alpha(
            qg, DOOM_K, prof, n_steps=n_steps, k_window=net_gain_window(DOOM_K, ks)
        )
    tag = "NET GAIN -> must jump back up" if alpha < 0 else "lossy -> five modes hold"
    rows.append((float(mult), len(live), bool(sol.converged), float(alpha)))
    np.save("out/hysteresis.npy", np.array(rows))
    print(
        f"{mult:8.4f} {len(live):3d} {str(sol.converged):>6} {alpha:26.3e}  {tag} "
        f"[{time.time() - t:4.0f}s]",
        flush=True,
    )
print("DONE", flush=True)

"""Is a mode the sweep dropped really dark, or did the root walk onto a neighbour?

Usage: python probe_verified_drop.py <sweep.npz> <row> <next_mult> [fixture]

``solve_salt_varying`` declares a floored mode extinguished by rebuilding the
background from the survivors and asking :func:`net_gain_alpha` whether the
floored mode still has gain there. That check walks a root off the real axis
starting from the mode's own ``k``, and accepts the root it lands on as long as
it has not moved more than ``k_window`` in ``Re k`` -- whose default is 0.1.
On these fixtures 0.1 is wider than the gap to the next *candidate* (5e-03 on
the second graph), so in principle the probe could report a neighbouring mode's
loss as this mode's and retire a mode that is still lasing.

This tests that on a specific drop. It re-solves the set recorded at ``row`` of
the sweep at ``next_mult`` -- an independent solve, warm-started from the state
before the drop -- and if a mode does go to zero, re-runs the gain test at window
widths from 0.1 down to 2e-04. A drop is trustworthy when the root barely moves
and the verdict is the same at every width; it is an artefact when the root
travels a sizeable fraction of the window and the verdict flips as the window
closes.

Measured on the second graph's drop at 1.0100x (row 0, the state at 1.0050x):
the root moves +3.5e-04 and alpha = +2.0e-03 at every width from 1e-03 to 0.1,
so that drop is physics. (At 2e-04, tighter than the root's own displacement,
the probe correctly refuses to answer and returns +inf.)
"""

import os
import sys
from pathlib import Path

if len(sys.argv) < 4:
    raise SystemExit("usage: probe_verified_drop.py <sweep.npz> <row> <next_mult> [fixture]")
SWEEP = sys.argv[1]
ROW = int(sys.argv[2])
NEXT = float(sys.argv[3])
FIXTURE = sys.argv[4] if len(sys.argv) > 4 else "buffon_competition"
os.chdir(Path(__file__).resolve().parents[1] / "buffon" / FIXTURE)

import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.simplefilter("ignore")
from netsalt import pipeline as pl  # noqa: E402
from netsalt.config_loader import load_config  # noqa: E402
from netsalt.salt_varying import (  # noqa: E402
    SALT_VARYING_LASING_AMPLITUDE,
    _resolved_n_steps,
    net_gain_alpha,
    saturated_eps_profiles,
    solve_salt_varying,
)

WINDOWS = (0.1, 0.02, 5e-3, 1e-3, 2e-4)

p = load_config("config.yaml")
p["out_folder"] = "out"
qg = pl.step_create_quantum_graph(p)
md = pl.step_find_passive_modes(p, qg, None)
pump = pl.step_create_pump_profile(p, qg, md, None)
qg = pl._attach_pump_to_graph(p, qg, pump)
qg.graph["params"]["intensity_varying_samples_per_wavelength"] = 5
tr = pl.step_compute_mode_trajectories(p, qg, md, pump, None)
th = pl.step_find_threshold_modes(p, qg, tr, pump, None)
thr = np.asarray(th["lasing_thresholds"]).ravel()
thr0 = float(np.nanmin(thr))
pump = np.asarray(pump, dtype=float)

d = np.load(SWEEP)
ks = [float(k) for k in d["ks"][ROW] if k]
amps = [float(a) for k, a in zip(d["ks"][ROW], d["amps"][ROW], strict=True) if k]
print(
    f"{FIXTURE} {SWEEP} row {ROW} ({float(d['mult'][ROW]):.4f}x): "
    + " ".join(f"{k:.6f}:{a:.4f}" for k, a in zip(ks, amps, strict=True)),
    flush=True,
)
n_steps = _resolved_n_steps(qg, float(max(ks)), 128, pump)

D0 = thr0 * NEXT
sol = solve_salt_varying(qg, ks, amps, D0, pump, n_steps=128, outer=80)
print(
    f"\nindependent solve at {NEXT:.4f}x: converged={sol.converged} in {sol.iterations} "
    "outer iterations\n  "
    + " ".join(
        f"{float(k):.6f}:{float(a):.5f}" for k, a in zip(sol.ks, sol.amplitudes, strict=True)
    )
)
live = [j for j in range(len(sol.ks)) if float(sol.amplitudes[j]) > SALT_VARYING_LASING_AMPLITUDE]
if len(live) == len(sol.ks):
    print("\nno mode went dark here -- nothing to verify.")
    print("DONE")
    raise SystemExit

prof = saturated_eps_profiles(
    qg,
    [float(sol.ks[j]) for j in live],
    [float(sol.amplitudes[j]) for j in live],
    [sol.fields[j] for j in live],
    D0,
    pump,
)
for j in (j for j in range(len(sol.ks)) if j not in live):
    k_dead = float(sol.ks[j])
    print(f"\nmode k = {k_dead:.6f} went to zero. Net gain on the survivors' background:")
    print(f"  {'window':>9} {'root':>12} {'moved':>11} {'alpha':>13}  verdict")
    for w in WINDOWS:
        k_root, alpha = net_gain_alpha(qg, k_dead, prof, n_steps=n_steps, k_window=w)
        verdict = (
            "net gain" if alpha < 0 else ("no root found" if not np.isfinite(alpha) else "lossy")
        )
        print(f"  {w:9.1e} {k_root:12.6f} {k_root - k_dead:+11.2e} {alpha:+13.4e}  {verdict}")
print("DONE")

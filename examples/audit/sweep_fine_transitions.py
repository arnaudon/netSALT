"""Resolve the mode-count transitions with small pump steps.

The coarse sweep (probe_mode_reignition.py, 0.77 % steps) showed the co-lasing
count moving 6 -> 5 -> 9 and an extinguished mode's alpha crossing back through
zero near 1.150x. Both were read off steps too large to say whether the
transitions are sharp, and the count itself was reconstructed from solve
attempts rather than accepted sets -- the grow step accepts a set only if every
amplitude clears the lasing floor AND the residuals pass, so a solve can report
converged and still be rejected.

This fixes both. It records the ACCEPTED set explicitly at every pump, and it
steps finely, which it can afford because it starts from a state already known
rather than building the active set up from threshold: the converged five-mode
solution at 1.0846x (see AUDIT.md section 13).

Per pump it
  1. solves the current set -- solve_salt_varying drops a mode it has verified
     dark via net_gain_alpha and re-solves the remainder, so shrinking is
     automatic;
  2. screens every other candidate with net_gain_alpha on the survivors'
     saturated background, and tries admitting those with net gain, keeping each
     only if the enlarged set holds together;
  3. writes everything -- accepted ids, amplitudes, k, and the alpha of every
     candidate tested -- before moving on, since a sweep like this outlives
     several container restarts.

Usage: python sweep_fine_transitions.py [top] [step_pct] [out_name]
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
    SALT_VARYING_GAIN_MARGIN,
    SALT_VARYING_LASING_AMPLITUDE,
    _resolved_n_steps,
    net_gain_alpha,
    saturated_eps_profiles,
    solve_salt_varying,
)

TOP = float(sys.argv[1]) if len(sys.argv) > 1 else 1.16
STEP = (float(sys.argv[2]) if len(sys.argv) > 2 else 0.25) / 100.0
OUT = sys.argv[3] if len(sys.argv) > 3 else "out/fine_transitions.npz"
START = 1.0846
N_STEPS = 128

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
order = list(np.argsort(thr))
finite = [int(i) for i in np.where(np.isfinite(thr))[0]]
all_k = {int(i): float(np.real(tlm[i])) for i in finite}
pump = np.asarray(pump, dtype=float)
n_steps = _resolved_n_steps(qg, float(np.max(np.abs(list(all_k.values())))), N_STEPS, pump)

# the converged five-mode state at 1.0846x: the six lowest-threshold candidates
# minus the one extinguished there (k = 10.680091)
active = [order[i] for i in (0, 1, 2, 4, 5)]
ks = [all_k[i] for i in active]
amps = [2.253, 23.980, 9.699, 8.976, 9.726]

grid = np.arange(START, TOP + 0.5 * STEP, STEP)
print(f"{len(grid)} pumps, {START:.4f}x .. {grid[-1]:.4f}x, {100 * STEP:.2f}% steps", flush=True)
print("seed:", " ".join(f"{all_k[i]:.5f}" for i in active), flush=True)


def solve(set_ids, set_ks, set_amps, D0):
    sol = solve_salt_varying(qg, set_ks, set_amps, D0, pump, n_steps=N_STEPS, outer=80)
    live = [
        j for j in range(len(set_ids)) if float(sol.amplitudes[j]) > SALT_VARYING_LASING_AMPLITUDE
    ]
    ok = sol.converged and len(live) == len(set_ids)
    return sol, live, ok


records, alpha_log = [], []
for mult in grid:
    D0 = thr0 * float(mult)
    t0 = time.time()

    sol, live, ok = solve(active, ks, amps, D0)
    if not sol.converged and len(live) == len(active):
        print(f"  {mult:.4f}x  base solve did not converge; stopping", flush=True)
        break
    # shrink to whatever survived
    active = [active[j] for j in live]
    ks = [float(sol.ks[j]) for j in live]
    amps = [float(sol.amplitudes[j]) for j in live]

    # grow: screen every other candidate on the survivors' background
    if active:
        profiles = saturated_eps_profiles(qg, ks, amps, [sol.fields[j] for j in live], D0, pump)
        for cand in sorted(finite, key=lambda i: thr[i]):
            if cand in active or thr[cand] >= D0:
                continue
            gap = min(abs(all_k[cand] - k) for k in ks)
            window = float(np.clip(0.2 * gap, 1e-6, 0.3))
            _, alpha = net_gain_alpha(qg, all_k[cand], profiles, n_steps=n_steps, k_window=window)
            alpha_log.append((float(mult), float(all_k[cand]), float(alpha)))
            if alpha >= SALT_VARYING_GAIN_MARGIN:
                continue
            trial, tlive, tok = solve([*active, cand], [*ks, all_k[cand]], [*amps, 1e-3], D0)
            if tok:
                active = [*active, cand]
                ks = [float(v) for v in trial.ks]
                amps = [float(v) for v in trial.amplitudes]
                sol = trial
                profiles = saturated_eps_profiles(qg, ks, amps, list(trial.fields), D0, pump)

    records.append(
        {
            "mult": float(mult),
            "n": len(active),
            "ids": list(active),
            "ks": list(ks),
            "amps": list(amps),
        }
    )
    np.savez(
        OUT,
        mult=np.array([r["mult"] for r in records]),
        n_active=np.array([r["n"] for r in records]),
        ids=np.array(
            [np.pad(r["ids"], (0, 12 - len(r["ids"])), constant_values=-1) for r in records]
        ),
        amps=np.array([np.pad(r["amps"], (0, 12 - len(r["amps"]))) for r in records]),
        ks=np.array([np.pad(r["ks"], (0, 12 - len(r["ks"]))) for r in records]),
        alpha=np.array(alpha_log) if alpha_log else np.zeros((0, 3)),
        thr0=thr0,
        thr=thr,
        k=np.array([all_k[i] for i in finite]),
    )
    print(
        f"  {mult:.4f}x  M={len(active):2d}  [{time.time() - t0:5.0f}s]  "
        + " ".join(f"{all_k[i]:.4f}:{a:.2f}" for i, a in zip(active, amps, strict=True)),
        flush=True,
    )
print("DONE", flush=True)

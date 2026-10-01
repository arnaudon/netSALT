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
# Where to begin. The default starts from the converged five-mode state at
# 1.0846x, which is what makes a fine grid affordable. Passing a start at or
# below _FROM_THRESHOLD instead begins at the first lasing threshold with a
# single mode just above it, and lets the admit/drop logic build the whole set
# from nothing -- slower, but it draws the L-I curve from its foot.
START = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0846
_FROM_THRESHOLD = 1.02
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
if START <= _FROM_THRESHOLD:
    # from the foot of the curve: only the lowest-threshold mode lases, and
    # barely. Everything else the sweep admits on its own.
    active = [order[0]]
    ks = [all_k[order[0]]]
    amps = [1e-3]
else:
    active = [order[i] for i in (0, 1, 2, 4, 5)]
    ks = [all_k[i] for i in active]
    amps = [2.253, 23.980, 9.699, 8.976, 9.726]
records, alpha_log = [], []
start = START

# Resume, if this output already holds pumps. These sweeps run for hours and have
# now outlived three containers; restarting from the top would re-pay everything
# already on disk. The last recorded row is a converged state, which is exactly
# what the next pump wants as its seed.
if os.path.exists(OUT):
    prev = np.load(OUT)
    if len(prev["mult"]):
        for row in range(len(prev["mult"])):
            live = [j for j in range(12) if prev["ids"][row][j] >= 0]
            records.append(
                {
                    "mult": float(prev["mult"][row]),
                    "n": int(prev["n_active"][row]),
                    "ids": [int(prev["ids"][row][j]) for j in live],
                    "ks": [float(prev["ks"][row][j]) for j in live],
                    "amps": [float(prev["amps"][row][j]) for j in live],
                }
            )
        alpha_log = [tuple(map(float, r)) for r in prev["alpha"]]
        last = records[-1]
        active, ks, amps = last["ids"], last["ks"], last["amps"]
        start = last["mult"] + STEP
        print(
            f"resuming from {OUT}: {len(records)} pumps done, last {last['mult']:.4f}x "
            f"with M={last['n']}",
            flush=True,
        )

if start > TOP + 0.5 * STEP:
    print(f"nothing to do: {start:.4f}x is already past {TOP:.4f}x", flush=True)
    raise SystemExit

# Step control. A fixed grid is the wrong instrument near a mode switch: the
# state changes fast there, and a step sized for the smooth stretches overshoots
# what the continuation can track. Measured at the second switch -- a 0.50 %
# step from 1.4596x cost 8748 s and the next one 26001 s and never converged,
# while a 0.05 % step from the same state costs ~580 s and converges normally.
# The 26001 s solve also drove a mode to zero that a fine crossing shows still
# lasing at ~0.18, i.e. a failed solve with a plausible-looking answer.
#
# So halve on failure and retry rather than giving up, and creep back towards
# the requested step once things are smooth again. STEP is the ceiling, not the
# fixed value.
_STEP_SHRINK = 0.5
_STEP_GROW = 1.5
_STEP_MIN_FRACTION = 1 / 32  # below this, the failure is not about step size

# Shrinking only on failure means paying the whole failure first: a solve that
# will not converge still burns its full `outer` budget, which near this switch
# is ~90 minutes. The solver reports how many outer iterations it used, and a
# solve that needed most of its budget is already telling you the step is too
# long for the region. So shrink on strain, not just on failure, and only grow
# back after a solve that converged comfortably.
_STEP_STRAIN = 0.5  # fraction of the outer budget above which the step shrinks
_STEP_EASY = 0.25  # and below which it may grow again
print(f"{start:.4f}x .. {TOP:.4f}x, steps up to {100 * STEP:.2f}%", flush=True)
print("seed:", " ".join(f"{all_k[i]:.5f}" for i in active), flush=True)


def solve(set_ids, set_ks, set_amps, D0):
    sol = solve_salt_varying(qg, set_ks, set_amps, D0, pump, n_steps=N_STEPS, outer=80)
    live = [
        j for j in range(len(set_ids)) if float(sol.amplitudes[j]) > SALT_VARYING_LASING_AMPLITUDE
    ]
    ok = sol.converged and len(live) == len(set_ids)
    return sol, live, ok


step = STEP
mult = start
while mult <= TOP + 0.5 * step:
    D0 = thr0 * float(mult)
    t0 = time.time()

    sol, live, ok = solve(active, ks, amps, D0)
    if not sol.converged and step > STEP * _STEP_MIN_FRACTION:
        failed_at = mult
        step *= _STEP_SHRINK
        mult = records[-1]["mult"] + step if records else start
        print(
            f"  {failed_at:.4f}x did not converge; step -> {100 * step:.3f} %, "
            f"retrying at {mult:.4f}x",
            flush=True,
        )
        continue
    if not sol.converged:
        # Stop on ANY non-converged base solve. This used to stop only when the
        # solve also kept every mode, on the reasoning that a mode going dark is
        # the ordinary way a set shrinks -- but that let a FAILED solve through
        # whenever it happened to lose a mode too, and the failure and the loss
        # are exactly what happen together. Measured at 1.4646x: a 26001 s solve
        # dropped M from 11 to 10, rearranged the amplitudes wholesale (the
        # first mode 2.94 -> 8.67, the second 101.45 -> 45.58) and came out with
        # total output 7 % LOWER than the previous pump at 0.34 % less pump,
        # which no branch does. It was written to the npz as data.
        print(
            f"  {mult:.4f}x  base solve did not converge "
            f"(M {len(active)} -> {len(live)}); stopping",
            flush=True,
        )
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
        f"  {mult:.4f}x  M={len(active):2d}  [{time.time() - t0:5.0f}s]"
        f"{'' if step == STEP else f'  (step {100 * step:.3f} %)'}  "
        + " ".join(f"{all_k[i]:.4f}:{a:.2f}" for i, a in zip(active, amps, strict=True)),
        flush=True,
    )
    used = sol.iterations / 80.0
    if used > _STEP_STRAIN and step > STEP * _STEP_MIN_FRACTION:
        step *= _STEP_SHRINK
        print(
            f"      used {sol.iterations} outer iterations; step -> {100 * step:.3f} %",
            flush=True,
        )
    elif used < _STEP_EASY:
        step = min(step * _STEP_GROW, STEP)
    mult += step
print("DONE", flush=True)

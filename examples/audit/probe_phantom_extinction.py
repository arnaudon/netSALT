"""An extinction that is not one: the warm start loses a mode that is still lasing.

Usage: python probe_phantom_extinction.py

On the second graph the sweep records its first extinction at the second pump:

    1.0050x  M= 2  10.762463:1.6501 10.787597:0.0934
    1.0100x  M= 1  10.762463:3.3185

and the drop is "verified" -- ``solve_salt_varying`` floors the amplitude, and
:func:`net_gain_alpha` on the survivor's background reports the floored mode
lossy at ``alpha = +2.0e-03``. Three independent checks say the mode is still
lasing there, and this script runs all three.

1. **Its own root has net gain.** Continue the root in pump from its threshold
   at 1.0028x in 0.08 % steps, starting each solve from the previous root rather
   than from ``alpha = 0``. It tracks smoothly to 1.0204x and beyond: ``k``
   drifting -1.23e-06 per step, ``alpha`` going monotonically from 0 to
   -1.27e-04, ``|lambda_1| ~ 1.5e-10`` throughout. At 1.0100x, where the sweep
   killed it, ``alpha = -5.2e-05``.

2. **Gain competition is far too weak to have killed it.** The mean fractional
   gain depletion the survivor inflicts on it is 0.006 %, against its 0.718 %
   margin above its own threshold -- and it saturates itself 57x harder than the
   survivor saturates it. The two modes are spatially disjoint (pump-weighted
   overlap 0.0003).

3. **The two-mode solution exists.** Re-seed the same solve at 1.0100x from the
   linear model's amplitudes instead of from the previous pump, and it converges
   in 16 outer iterations to ``10.787598:0.304`` with residuals 1.0e-08 and
   7.0e-07 -- better converged than the one-mode answer it replaced, and within
   3 % of the linear model, which is what near-threshold agreement should look
   like.

So the extinction is an artefact, and the mechanism is specific. ``alpha = 0`` is
a bad place to start this root find: the trough is ~1e-05 wide in ``alpha`` while
``|lambda_1|`` at ``alpha = 0`` is already ~1.0, and a *second*, genuine root
sits 1.1e-03 away in ``Re k`` with ``alpha = +2.0e-03``. MINPACK, given 30
function evaluations from a cold start, converges to the neighbour -- which is a
real root, so checking ``|lambda_1|`` at the answer does not catch it. What
catches it is the distance travelled: the physical drift is 1.2e-06 per pump
step and ``net_gain_alpha``'s default ``k_window`` is 0.1, a thousand times
wider than the gap to the wrong root.

A earlier version of this check varied the window around the *drifted* ``k`` the
floored solve returned, found the same verdict at every width, and concluded the
drop was physical. That start was already off-branch, so every window agreed
about the wrong root. The window has to be judged against the physical drift,
not against whether the answer is stable.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition_b")

import warnings  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import root as scipy_root  # noqa: E402

warnings.simplefilter("ignore")
from netsalt import pipeline as pl  # noqa: E402
from netsalt.config_loader import load_config  # noqa: E402
from netsalt.physics import gamma  # noqa: E402
from netsalt.salt_varying import (  # noqa: E402
    _lam_varying,
    _resolved_n_steps,
    edge_field_profiles,
    node_solution_varying,
    saturated_eps_profiles,
    solve_salt_varying,
)

K_VICTIM = 10.787597  # the mode the sweep retires at 1.0100x
K_SURVIVOR = 10.762463  # the only other lasing mode there
STATE = [1.6501, 0.0934]  # the converged amplitudes at 1.0050x
A_SURVIVOR = 3.3185  # and what the survivor reaches at 1.0100x
LINEAR = [3.225, 0.294]  # what the linear competition matrix says at 1.0100x
TRACK_WINDOW = 2e-4  # ~160x the physical per-step drift, 1/5 of the gap to the wrong root

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
tlm = th["threshold_lasing_modes"].to_numpy()
thr0 = float(np.nanmin(thr))
pump = np.asarray(pump, dtype=float)
params = qg.graph["params"]
n_steps = _resolved_n_steps(qg, K_VICTIM, 128, pump)
i_victim = int(np.argmin(np.abs(np.real(tlm) - K_VICTIM)))
i_survivor = int(np.argmin(np.abs(np.real(tlm) - K_SURVIVOR)))

print("1. the victim's own root, continued in pump with a tight window")
print(f"   its stated threshold: {thr[i_victim] / thr0:.4f} x thr0")
print(f"   {'mult':>8} {'k':>14} {'dk':>11} {'alpha':>13} {'|lam|':>10}  verdict")


def track(profiles, k_start, alpha_start):
    """Root nearest ``(k_start, alpha_start)``, warm-started in BOTH unknowns."""

    def residual(x):
        value = _lam_varying(qg, complex(x[0], -x[1]), profiles, n_steps)
        return [value.real, value.imag]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = scipy_root(
            residual,
            np.array([float(k_start), float(alpha_start)]),
            method="hybr",
            tol=0,
            options={"maxfev": 400, "xtol": 1e-12},
        )
    k_root, alpha = float(out.x[0]), float(out.x[1])
    return k_root, alpha, abs(_lam_varying(qg, complex(k_root, -alpha), profiles, n_steps))


k, alpha = K_VICTIM, 0.0
for mult in np.arange(1.0028, 1.0205, 0.0008):
    profiles = saturated_eps_profiles(qg, [], [], [], thr0 * float(mult), pump)
    k_new, alpha_new, lam = track(profiles, k, alpha)
    on_branch = lam < 1e-5 and abs(k_new - k) < TRACK_WINDOW
    print(
        f"   {mult:8.4f} {k_new:14.8f} {k_new - k:+11.2e} {alpha_new:+13.4e} {lam:10.1e}"
        f"  {'NET GAIN' if alpha_new < 0 else 'lossy'}"
        f"{'' if on_branch else '   <- left the branch'}"
    )
    if not on_branch:
        break
    k, alpha = k_new, alpha_new

print("\n2. how much gain the survivor actually takes from it")
fields = {}
for i in (i_victim, i_survivor):
    k_i = float(np.real(tlm[i]))
    _, psi = node_solution_varying(k_i, qg, None, n_steps=n_steps)
    fields[i] = edge_field_profiles(k_i, qg, psi, None, n_steps=n_steps, pump=pump)


def weighted(a, b):
    return sum(
        float(np.sum(pump[e] * x * y))
        for e, (x, y) in enumerate(zip(fields[a], fields[b], strict=True))
    )


norm = sum(float(np.sum(pump[e] * x)) for e, x in enumerate(fields[i_victim]))
clamp = -np.imag(gamma(complex(float(np.real(tlm[i_survivor]))), params))
cross = clamp * A_SURVIVOR * weighted(i_survivor, i_victim) / norm
self_sat = clamp * STATE[1] * weighted(i_victim, i_victim) / norm
margin = 1.0100 / (thr[i_victim] / thr0) - 1.0
overlap = weighted(i_survivor, i_victim) / np.sqrt(
    weighted(i_survivor, i_survivor) * weighted(i_victim, i_victim)
)
print(f"   margin above its own threshold at 1.0100x : {100 * margin:7.3f} %")
print(f"   depleted by the survivor (a = {A_SURVIVOR:.2f})       : {100 * cross:7.3f} %")
print(f"   depleted by itself (a = {STATE[1]:.4f})          : {100 * self_sat:7.3f} %")
print(f"   their pump-weighted overlap                : {overlap:7.4f}")

print("\n3. the two-mode solution, re-seeded instead of warm-started")
for label, amps in (("previous pump (what the sweep used)", STATE), ("linear at 1.0100x", LINEAR)):
    sol = solve_salt_varying(
        qg, [K_SURVIVOR, K_VICTIM], list(amps), thr0 * 1.0100, pump, n_steps=128, outer=80
    )
    print(f"   seed = {label}: converged={sol.converged} in {sol.iterations} iterations")
    print(
        "      "
        + " ".join(
            f"{float(k):.6f}:{float(a):.5f}" for k, a in zip(sol.ks, sol.amplitudes, strict=True)
        )
        + "   residuals "
        + " ".join(f"{float(r):.1e}" for r in sol.residuals)
    )
print("DONE")

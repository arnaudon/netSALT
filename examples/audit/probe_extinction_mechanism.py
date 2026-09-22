"""Why does full SALT extinguish k=10.6801 when the linear model keeps it?

Three things differ between the two models, and they can be separated because
the admission test `net_gain_alpha` takes the saturated background as an
argument -- so the background can be built the way EITHER model would build it,
and the candidate's net gain read off each.

  * the mode PROFILES. Full SALT's modes are eigenvectors of the saturated
    operator, so |E(x)|^2 deforms as the hole is burnt. The linear competition
    matrix freezes every profile at its threshold shape.
  * the lasing FREQUENCIES. Full SALT pulls k with the pump; linear holds each
    k at threshold.
  * the SATURATION itself, 1/(1 + sum_nu Gamma_nu a_nu |E_nu|^2), which the
    linear model expands to first order.

Backgrounds built at the SAME five amplitudes, so only the named ingredient
changes:

  A  saturated shapes, pulled k   -- full SALT's own background
  B  threshold shapes, pulled k   -- A with the profile deformation removed
  C  threshold shapes, threshold k -- the background the linear model assumes

alpha < 0 means the candidate has net gain and must lase; alpha > 0 means it is
below threshold. If A says dark and C says lasing, the disagreement is real and
B says which ingredient does it.

Also reports how far each mode's profile actually moved, and the doomed mode's
spatial overlap with its near-degenerate partner in both backgrounds, since that
overlap is what sets how hard they compete.
"""

import os  # noqa: E402
from pathlib import Path  # noqa: E402

# The fixture these run on: the checked-in buffon over the production k window.
# Resolved from this file rather than hardcoded, so the script works from any
# working directory and any checkout. `bash run.sh` there builds out/ first.
os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition")


import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.simplefilter("ignore")
from netsalt import pipeline as pl  # noqa: E402
from netsalt.config_loader import load_config  # noqa: E402
from netsalt.io import load_modes  # noqa: E402
from netsalt.salt_varying import (  # noqa: E402
    _field_change,
    _resolved_n_steps,
    edge_field_profiles,
    net_gain_alpha,
    node_solution_varying,
    saturated_eps_profiles,
    solve_salt_varying,
)

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
order = list(np.argsort(thr))[:6]
thr0 = float(np.nanmin(thr))
all_k = [float(np.real(tlm[i])) for i in order]
pump = np.asarray(pump, dtype=float)

OUT_DATA = "out/extinction_mechanism.npz"
DOOMED = 3  # k = 10.680091, the mode full SALT extinguishes
PARTNER = 0  # k = 10.679331, 7.60e-04 away
keep = [0, 1, 2, 4, 5]
D0 = thr0 * 1.0846

t = time.time()
sol = solve_salt_varying(
    qg,
    [all_k[j] for j in keep],
    [2.253, 23.980, 9.699, 8.976, 9.726],
    D0,
    pump,
    n_steps=128,
    outer=80,
)
print(
    f"five-mode reference: conv={sol.converged} res={np.max(sol.residuals):.1e} "
    f"[{time.time() - t:.0f}s]",
    flush=True,
)
n_steps = _resolved_n_steps(qg, float(np.max(np.abs(sol.ks))), 128, pump)

# --- how far did the survivors' profiles and frequencies actually move? -------
print("\nwhat the pump did to the five survivors:", flush=True)
print(f"  {'k(thr)':>11} {'k(SALT)':>11} {'dk':>11} {'|dprofile|':>11}", flush=True)
frozen_shapes, pulled_frozen_shapes, dprofiles = [], [], []
for slot, j in enumerate(keep):
    k_thr, k_sat = all_k[j], float(sol.ks[slot])
    # threshold shape at the threshold k -- what the linear model uses
    _, psi = node_solution_varying(k_thr, qg, None, n_steps=n_steps)
    frozen_shapes.append(edge_field_profiles(k_thr, qg, psi, None, n_steps=n_steps, pump=pump))
    # threshold shape at the PULLED k -- isolates shape from frequency
    _, psi = node_solution_varying(k_sat, qg, None, n_steps=n_steps)
    pulled_frozen_shapes.append(
        edge_field_profiles(k_sat, qg, psi, None, n_steps=n_steps, pump=pump)
    )
    dprof = _field_change([pulled_frozen_shapes[-1]], [sol.fields[slot]])
    dprofiles.append(dprof)
    print(f"  {k_thr:11.6f} {k_sat:11.6f} {k_sat - k_thr:+11.2e} {dprof:11.3e}", flush=True)

# --- the candidate's net gain on each background -------------------------------
amps = list(sol.amplitudes)
cases = [
    ("A  saturated shapes, pulled k   (full SALT)", sol.fields, sol.ks),
    ("B  threshold shapes, pulled k   (no deformation)", pulled_frozen_shapes, sol.ks),
    (
        "C  threshold shapes, threshold k (linear's background)",
        frozen_shapes,
        [all_k[j] for j in keep],
    ),
]
print("\nnet gain of the candidate at k = 10.680091 on each background:", flush=True)
print(f"  {'background':<54} {'k root':>11} {'alpha':>12}  verdict", flush=True)
alphas = {}
for name, fields, ks_bg in cases:
    profiles = saturated_eps_profiles(qg, list(ks_bg), amps, fields, D0, pump)
    k_root, alpha = net_gain_alpha(qg, all_k[DOOMED], profiles, n_steps=n_steps)
    alphas[name[0]] = alpha
    verdict = "LASES" if alpha < 0 else "dark"
    print(f"  {name:<54} {k_root:11.6f} {alpha:12.4e}  {verdict}", flush=True)


# --- overlap with the near-degenerate partner, both backgrounds ----------------
def overlap(fa, fb):
    num = sum(float(np.sum(pump[e] * a * b)) for e, (a, b) in enumerate(zip(fa, fb, strict=True)))
    na = sum(float(np.sum(pump[e] * a * a)) for e, a in enumerate(fa))
    nb = sum(float(np.sum(pump[e] * b * b)) for e, b in enumerate(fb))
    return num / np.sqrt(na * nb)


print("\noverlap of the candidate with its partner (k = 10.679331), pump-weighted:", flush=True)
saved = {
    "alpha": np.array([alphas[key] for key in "ABC"]),
    "dk": np.array([float(sol.ks[i]) - all_k[j] for i, j in enumerate(keep)]),
    "k_keep": np.array([all_k[j] for j in keep]),
}
for name, fields, ks_bg in cases:
    profiles = saturated_eps_profiles(qg, list(ks_bg), amps, fields, D0, pump)
    k_root, _ = net_gain_alpha(qg, all_k[DOOMED], profiles, n_steps=n_steps)
    _, psi = node_solution_varying(k_root, qg, profiles, n_steps=n_steps)
    f_doomed = edge_field_profiles(k_root, qg, psi, profiles, n_steps=n_steps, pump=pump)
    f_partner = fields[keep.index(PARTNER)]
    print(f"  {name:<54} {overlap(f_doomed, f_partner):8.4f}", flush=True)
    # per-sample intensities of the pair, for the segregation scatter
    saved[f"overlap_{name[0]}"] = overlap(f_doomed, f_partner)
    saved[f"doomed_{name[0]}"] = np.concatenate([np.asarray(a) for a in f_doomed])
    saved[f"partner_{name[0]}"] = np.concatenate([np.asarray(a) for a in f_partner])
    saved[f"pumpmask_{name[0]}"] = np.concatenate(
        [np.full(len(a), float(pump[e])) for e, a in enumerate(f_doomed)]
    )

saved["dprofile"] = np.array(dprofiles)
np.savez(OUT_DATA, **saved)
print(f"\nwrote {OUT_DATA}", flush=True)
print("DONE", flush=True)

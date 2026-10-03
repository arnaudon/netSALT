"""Is a lasing state stable? Probe every candidate, not one.

`net_gain_alpha` reports what the root nearest a given k0 does on a given
saturated background. It is easy -- and wrong -- to read one such call as "this
state is stable". On this buffon the cluster near k = 10.6800 carries several
roots within 1e-4 of each other, and two probes started 1.5e-05 apart converge
to different ones: at 1.0730x, k0 = 10.680076 finds a root at 10.679988 with
alpha = +5.13e-05 (lossy) while k0 = 10.680091 finds 10.680069 with
alpha = -1.47e-04 (net gain). Both are correct answers about different modes.

A state is unstable if ANY root has net gain, so stability is the MINIMUM alpha
over all candidates, not the alpha of whichever one was asked about. This probes
every candidate not currently lasing, reports each root it actually lands on,
and takes the minimum.

Used here to settle whether the five-mode branch really is stable below the fold
at ~1.0763x -- i.e. whether the bistable window claimed from the hysteresis
sweep survives a proper stability test.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition")

import sys  # noqa: E402
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

PUMPS = (
    [float(x) for x in sys.argv[1].split(",")]
    if len(sys.argv) > 1
    else [1.0700, 1.0720, 1.0730, 1.0740, 1.0750, 1.0760]
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
thr0 = float(np.nanmin(thr))
pump = np.asarray(pump, dtype=float)
finite = [int(i) for i in np.where(np.isfinite(thr))[0]]
all_k = {int(i): float(np.real(tlm[i])) for i in finite}

z = np.load("out/fine_zoom.npz")
ids6 = [int(i) for i in z["ids"][-1] if i >= 0]
ks6 = [float(v) for v, i in zip(z["ks"][-1], z["ids"][-1], strict=True) if i >= 0]
a6 = [float(v) for v, i in zip(z["amps"][-1], z["ids"][-1], strict=True) if i >= 0]
kk = np.real(z["k"])
DOOM = int(np.argmin([abs(float(kk[i]) - 10.680091) for i in ids6]))
keep = [j for j in range(len(ids6)) if j != DOOM]
ks5 = [ks6[j] for j in keep]
a5 = [a6[j] for j in keep]
ids5 = [ids6[j] for j in keep]
n_steps = _resolved_n_steps(qg, float(np.max(np.abs(ks6))), 128, pump)

print("stability of the FIVE-mode state, every candidate probed", flush=True)
for mult in PUMPS:
    D0 = thr0 * float(mult)
    sol = solve_salt_varying(qg, list(ks5), list(a5), D0, pump, n_steps=128, outer=80)
    if not sol.converged:
        print(f"\n{mult:.4f}x  five-mode solve did not converge", flush=True)
        continue
    prof = saturated_eps_profiles(
        qg, list(sol.ks), list(sol.amplitudes), list(sol.fields), D0, pump
    )
    found = []
    for cand in finite:
        if cand in ids5:
            continue
        kr, al = net_gain_alpha(
            qg,
            all_k[cand],
            prof,
            n_steps=n_steps,
            k_window=net_gain_window(all_k[cand], [v for k, v in all_k.items() if k != cand]),
        )
        if np.isfinite(al):
            found.append((all_k[cand], kr, al))
    worst = min(found, key=lambda t: t[2]) if found else None
    verdict = "UNSTABLE, a mode must join" if worst and worst[2] < 0 else "stable"
    print(f"\n{mult:.4f}x  {verdict}", flush=True)
    for k0, kr, al in sorted(found, key=lambda t: t[2]):
        mark = "  <- net gain" if al < 0 else ""
        print(f"     k0={k0:.6f} -> root {kr:.6f}  alpha={al:+.4e}{mark}", flush=True)
print("\nDONE", flush=True)

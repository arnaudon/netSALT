"""The production buffon on the per-edge-DtN operator -- the check that closes #52.

Oversampling needs ~76600 nodes to reach lambda/12 on this graph; the default
node cap of 3000 gives 0.47 samples per wavelength, four times below Nyquist,
so the within-edge field is aliased rather than merely coarse. That is why full
SALT was out of reach here.

With the resolution moved into per-edge transfer matrices the matrix stays
208x208 at every resolution:

    n_steps   samples/wavelength    matrix    build       |lambda|
         64                  2.3   208x208    0.29s   1.73132837
        256                  9.2   208x208    0.95s   1.73129737
        512                 18.5   208x208    1.89s   1.73129710
       1024                 36.9   208x208    4.04s   1.73129709

18.5 samples per wavelength -- against 0.47 before -- converged to 6e-7, in a
matrix 368x smaller than the equivalent subdivision.

One trap worth stating, because it produced a perfect-looking result first:
running this with **zero amplitude** leaves the saturation denominator at 1, so
the profile is constant, the propagator is exact at any n_steps, and |lambda| is
identical to 1e-12 across the whole sweep. That exercises none of the varying
machinery. The sweep below uses a genuine saturated field.

Run from this directory::

    OMP_NUM_THREADS=1 python probe_buffon_varying.py
"""

import os
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
os.chdir("/home/user/netSALT/examples/buffon/buffon_uniform")
from netsalt.config_loader import load_config  # noqa: E402
from netsalt.pipeline import step_create_quantum_graph  # noqa: E402
from netsalt.salt_varying import (  # noqa: E402
    _lam_varying,
    node_solution_varying,
    saturated_eps_profiles,
)

p = load_config("config.yaml")
g = step_create_quantum_graph(p)
n = len(g)
lengths = np.asarray(g.graph["lengths"], float)
nmax = float(np.sqrt(max(abs(g[u][v].get("dielectric_constant", 1.0)) for u, v in g.edges)))
k = 0.5 * (p["k_min"] + p["k_max"])
lam = 2 * np.pi / (nmax * k)
osc = lengths / lam
print(f"buffon: {n} nodes / {len(g.edges)} edges")
print(f"lambda = {lam:.4f}; oscillations per edge: median {np.median(osc):.1f} max {osc.max():.1f}")
print(
    f"oversampling to lambda/12 would need {int(np.sum(np.maximum(lengths / (lam / 12), 1))):d} nodes\n"
)

pump = np.array([0.0 if not g[u][v]["inner"] else 1.0 for u, v in g.edges])
g.graph["params"]["pump"] = pump
# A *real* saturated field: zero amplitude leaves the denominator at 1, the
# profile constant, and the propagator exact at any n_steps -- which exercises
# none of the varying machinery and looks like perfect convergence.
from netsalt.salt_varying import edge_field_profiles  # noqa: E402

AMP = 0.5
print(f"{'n_steps':>8} {'samples/wavelength':>20} {'matrix':>9} {'build s':>9} {'|lambda|':>14}")
prev = None
for n_steps in (64, 128, 256, 512, 1024):
    _, psi0 = node_solution_varying(k, g, None, n_steps=n_steps)
    field = edge_field_profiles(k, g, psi0, None, n_steps=n_steps, pump=pump)
    prof = saturated_eps_profiles(g, [k], [AMP], [field], 0.005, pump)
    t0 = time.perf_counter()
    val = _lam_varying(g, k, prof, n_steps)
    dt = time.perf_counter() - t0
    spw = n_steps / np.median(osc)
    delta = "" if prev is None else f"  d={abs(val - prev):.2e}"
    print(f"{n_steps:>8} {spw:>20.1f} {n:>6}x{n:<3} {dt:>8.2f}s {abs(val):>14.8e}{delta}")
    prev = val

"""Do static two-mode measures predict which mode goes dark? Both graphs, both measures.

Usage: python probe_cross_saturation.py [fixture]

Section 8 ranks pairs by the *symmetric* pump-weighted overlap

    O_mn = int p |E_m|^2 |E_n|^2 / sqrt(int p |E_m|^4  int p |E_n|^4)

and finds the first graph's two switches at the top of it. The symmetric form is
the wrong physical object, though: what decides whether m can starve n is how
much of n's gain m burns *relative to what n burns itself*, which is asymmetric,

    S[m->n] = int p |E_m|^2 |E_n|^2 / int p |E_n|^4,

the normalised cross-saturation the linear competition matrix is built from. A
broad mode sitting on top of a narrow one gives a small O and a large S[broad ->
narrow], and it is the narrow mode that dies.

This computes both over every pair of threshold modes, ranks them, and locates
the pairs whose switches have actually been measured. Run it on both fixtures:
the measures agree with the first graph and both miss the second one, which is
the finding -- see README section 10.
"""

import os
import sys
from pathlib import Path

FIXTURE = sys.argv[1] if len(sys.argv) > 1 else "buffon_competition"
os.chdir(Path(__file__).resolve().parents[1] / "buffon" / FIXTURE)

import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.simplefilter("ignore")
from netsalt import pipeline as pl  # noqa: E402
from netsalt.config_loader import load_config  # noqa: E402
from netsalt.salt_varying import (  # noqa: E402
    _resolved_n_steps,
    edge_field_profiles,
    node_solution_varying,
)

# The measured events, as (saturator, victim) by threshold-mode k. "Saturator" is
# the mode that grows through the switch, "victim" the one that leaves the set.
MEASURED = {
    "buffon_competition": [
        (10.679331, 10.680091, "switch 1, the fold at 1.0750x"),
        (10.704320, 10.707383, "switch 2, the continuous one near 1.481x"),
    ],
    "buffon_competition_b": [
        (10.762463, 10.787597, "the only lasing mode kills it at 1.0100x"),
    ],
}.get(FIXTURE, [])

p = load_config("config.yaml")
p["out_folder"] = "out"
Path("out").mkdir(exist_ok=True)
qg = pl.step_create_quantum_graph(p)
md = pl.step_find_passive_modes(p, qg, None)
pump = pl.step_create_pump_profile(p, qg, md, None)
qg = pl._attach_pump_to_graph(p, qg, pump)
tr = pl.step_compute_mode_trajectories(p, qg, md, pump, None)
th = pl.step_find_threshold_modes(p, qg, tr, pump, None)
thr = np.asarray(th["lasing_thresholds"]).ravel()
tlm = th["threshold_lasing_modes"].to_numpy()
pump = np.asarray(pump, dtype=float)
finite = [int(i) for i in np.where(np.isfinite(thr))[0]]
ks = {i: float(np.real(tlm[i])) for i in finite}
n_steps = _resolved_n_steps(qg, float(max(ks.values())), 128, pump)

fields = {}
for i in finite:
    _, psi = node_solution_varying(ks[i], qg, None, n_steps=n_steps)
    fields[i] = edge_field_profiles(ks[i], qg, psi, None, n_steps=n_steps, pump=pump)


def moment(a, b):
    """int pump |E_a|^2 |E_b|^2."""
    return sum(
        float(np.sum(pump[e] * x * y))
        for e, (x, y) in enumerate(zip(fields[a], fields[b], strict=True))
    )


sym = {}
asym = {}
for m in finite:
    for n in finite:
        if m == n:
            continue
        cross = moment(m, n)
        asym[(m, n)] = cross / moment(n, n)
        if m < n:
            sym[(m, n)] = cross / np.sqrt(moment(m, m) * moment(n, n))

sym_rank = sorted(((v, m, n) for (m, n), v in sym.items()), reverse=True)
asym_rank = sorted(((v, m, n) for (m, n), v in asym.items()), reverse=True)
print(f"{FIXTURE}: {len(finite)} threshold modes")
print(f"\nsymmetric overlap, top 6 of {len(sym_rank)}")
for r, (v, m, n) in enumerate(sym_rank[:6], start=1):
    print(f"  {r:4d} {v:8.4f}  {ks[m]:11.6f} / {ks[n]:11.6f}")
print(f"\nasymmetric S[m->n], top 6 of {len(asym_rank)}")
for r, (v, m, n) in enumerate(asym_rank[:6], start=1):
    print(f"  {r:4d} {v:8.3f}  {ks[m]:11.6f} -> {ks[n]:11.6f}")


def locate(ranked, a, b, ordered):
    for r, (v, m, n) in enumerate(ranked, start=1):
        pair = (round(ks[m], 6), round(ks[n], 6))
        if pair == (a, b) or (not ordered and pair == (b, a)):
            return r, v, len(ranked)
    return None, None, len(ranked)


if MEASURED:
    print("\nwhere the measured events sit in each ranking")
    for a, b, label in MEASURED:
        rs, vs, ns = locate(sym_rank, a, b, ordered=False)
        ra, va, na = locate(asym_rank, a, b, ordered=True)
        print(f"  {a:.6f} -> {b:.6f}  ({label})")
        print(f"      symmetric O  = {vs:8.4f}   rank {rs} of {ns}")
        print(f"      asymmetric S = {va:8.3f}   rank {ra} of {na}")
print("DONE")

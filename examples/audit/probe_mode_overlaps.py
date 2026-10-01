"""Which mode pairs are at risk of the segregation that drives an extinction?

Usage: ``python probe_mode_overlaps.py [fixture]``, where *fixture* is a
directory under ``examples/buffon/`` and defaults to ``buffon_competition``.

Section 13 traced the extinction to two modes being spatially near-identical at
threshold and segregating under saturation, the winner reshaping to take the
pump-rich region while the loser starved. The obvious follow-up is whether other
pairs do the same higher up the pump.

k-spacing alone does not answer it. Among the first fixture's twelve candidates
there is exactly one near-degenerate cluster, a triplet at 10.679331 /
10.679976 / 10.680091 spanning 7.6e-04, while every other gap is at least
2.28e-03 -- and in any case gamma_perp = 0.5 puts even the widest gap 220x
inside the gain linewidth, so proximity in k is not what distinguishes them.

What distinguishes them is spatial overlap, which is what sets how hard two
modes compete for the same gain. This computes the pump-weighted overlap

    O_mn = int pump |E_m|^2 |E_n|^2 / sqrt(int pump |E_m|^4  int pump |E_n|^4)

over every pair of threshold modes, and ranks them. On ``buffon_competition``
the two measured switches sit at the top of that ranking (0.9995 and 0.9706)
with a cliff to 0.56 below them, which is the whole basis for reading overlap
as the predictor. Run it on ``buffon_competition_b`` -- an independent Buffon
realisation at identical parameters -- to get the same ranking on a graph the
claim was not derived from, and a registered prediction to sweep against.
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


def overlap(fa, fb):
    num = sum(float(np.sum(pump[e] * a * b)) for e, (a, b) in enumerate(zip(fa, fb, strict=True)))
    na = sum(float(np.sum(pump[e] * a * a)) for e, a in enumerate(fa))
    nb = sum(float(np.sum(pump[e] * b * b)) for e, b in enumerate(fb))
    return num / np.sqrt(na * nb) if na > 0 and nb > 0 else float("nan")


rows = []
for n, i in enumerate(finite):
    for j in finite[n + 1 :]:
        rows.append((overlap(fields[i], fields[j]), ks[i], ks[j], abs(ks[i] - ks[j])))
rows.sort(reverse=True)

# The two switches measured on buffon_competition, by the k of the mode that dies.
KNOWN = {"buffon_competition": [(10.679331, 10.680091), (10.704318, 10.707375)]}.get(FIXTURE, [])
known_keys = [frozenset((round(a, 4), round(b, 4))) for a, b in KNOWN]
print(f"{FIXTURE}: pump-weighted overlap of {len(finite)} threshold modes, {len(rows)} pairs")
print(f"{'rank':>4} {'overlap':>8} {'k_m':>11} {'k_n':>11} {'|dk|':>10} {'thr ratio':>10}")
thr_of = {ks[i]: float(thr[i]) for i in finite}
thr_min = min(thr_of.values())
for rank, (o, ka, kb, dk) in enumerate(rows[:14], start=1):
    mark = ""
    if frozenset((round(ka, 4), round(kb, 4))) in known_keys:
        mark = "   <- a measured switch"
    ratio = max(thr_of[ka], thr_of[kb]) / thr_min
    print(f"{rank:4d} {o:8.4f} {ka:11.6f} {kb:11.6f} {dk:10.2e} {ratio:10.4f}{mark}")

top = rows[0]
print(f"\ntop pair: {top[1]:.6f} / {top[2]:.6f} at overlap {top[0]:.4f}")
gap = [i for i in range(1, len(rows)) if rows[i - 1][0] - rows[i][0] > 0.1]
if gap:
    i = gap[0]
    print(
        f"largest early cliff: rank {i} ({rows[i - 1][0]:.4f}) -> rank {i + 1} ({rows[i][0]:.4f})"
    )
    print(f"pairs above the cliff: {i}")
print("DONE")

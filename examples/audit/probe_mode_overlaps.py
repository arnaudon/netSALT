"""Which mode pairs are at risk of the segregation that drives an extinction?

§13 traced the extinction to two modes being spatially near-identical at
threshold -- 0.68 pump-weighted overlap -- and segregating under saturation, the
winner reshaping to take the pump-rich region while the loser starved. The
obvious follow-up is whether other pairs do the same higher up the pump.

k-spacing alone does not answer it. Among this fixture's twelve candidates there
is exactly one near-degenerate cluster, a triplet at 10.679331 / 10.679976 /
10.680091 spanning 7.6e-04, while every other gap is at least 2.28e-03 -- and in
any case gamma_perp = 0.5 puts even the widest gap 220x inside the gain
linewidth, so proximity in k is not what distinguishes them.

What distinguishes them is spatial overlap, which is what sets how hard two
modes compete for the same gain. This computes the pump-weighted overlap

    O_mn = int pump |E_m|^2 |E_n|^2 / sqrt(int pump |E_m|^4  int pump |E_n|^4)

over every pair of threshold modes, and ranks them. A pair near the 0.68 of the
known case is a candidate for the same behaviour; a fixture whose pairs are all
well separated cannot show it, and testing the question properly then needs a
wider k window with more candidates rather than more pump on this one.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition")

import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.simplefilter("ignore")
from netsalt import pipeline as pl  # noqa: E402
from netsalt.config_loader import load_config  # noqa: E402
from netsalt.io import load_modes  # noqa: E402
from netsalt.salt_varying import (  # noqa: E402
    _resolved_n_steps,
    edge_field_profiles,
    node_solution_varying,
)

p = load_config("config.yaml")
p["out_folder"] = "out"
qg = pl.step_create_quantum_graph(p)
md = load_modes("out/passive_modes.h5")
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

KNOWN = (10.679331, 10.680091)
print(f"pump-weighted overlap of threshold modes, {len(rows)} pairs, most overlapping first")
print(f"{'overlap':>8} {'k_m':>11} {'k_n':>11} {'|dk|':>10}")
for o, ka, kb, dk in rows[:12]:
    mark = ""
    if {round(ka, 6), round(kb, 6)} == {round(KNOWN[0], 6), round(KNOWN[1], 6)}:
        mark = "   <- the pair that extinguishes (§13)"
    print(f"{o:8.4f} {ka:11.6f} {kb:11.6f} {dk:10.2e}{mark}")
known = [
    r for r in rows if {round(r[1], 6), round(r[2], 6)} == {round(KNOWN[0], 6), round(KNOWN[1], 6)}
]
if known:
    o = known[0][0]
    rank = rows.index(known[0]) + 1
    above = [r for r in rows if r[0] >= 0.5 * o and r is not known[0]]
    print(f"\nthe known pair sits at overlap {o:.4f}, rank {rank} of {len(rows)}")
    print(f"pairs within a factor 2 of it: {len(above)}")
    for r in above:
        print(f"   {r[0]:.4f}  {r[1]:.6f} / {r[2]:.6f}  |dk|={r[3]:.2e}")
print("DONE")

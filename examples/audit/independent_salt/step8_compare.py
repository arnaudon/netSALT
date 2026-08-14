"""Step 8: the comparison table.

Loads the independent finite-difference sweeps (several grids) and the netsalt
``solve_salt_fixed_set`` sweeps (several oversampling resolutions), Richardson-
extrapolates each side in its own discretisation parameter, and reports the
convention-free quantities:

  k_mu(D0),  I_mu(D0) = int_cavity |Psi_mu|^2 dx,
  inter-mode ratio I_1/I_2,  pump-to-pump ratio I_mu(D0)/I_mu(D0_ref).
"""

from __future__ import annotations

import json
import sys


def load(path):
    d = json.load(open(path))
    out = {}
    for r in d["records"]:
        out[round(r["D0"], 4)] = r
    return d, out


def richardson(v_coarse, v_fine, ratio):
    """h^2-extrapolation: ratio = (h_coarse/h_fine)^2."""
    return v_fine + (v_fine - v_coarse) / (ratio - 1.0)


def series(runs, key, ratios):
    """Richardson-extrapolated value + a genuine error estimate.

    With three grids the error bar is |Richardson(1,2) - Richardson(2,3)|, i.e.
    how much the extrapolated answer still moves; with two it falls back to the
    (much more conservative) distance from the finest value.
    """
    vals = [r.get(key) for _, r in runs]
    if any(v is None for v in vals):
        return None, None
    ex = richardson(vals[-2], vals[-1], ratios[-1])
    if len(vals) >= 3:
        ex_prev = richardson(vals[-3], vals[-2], ratios[-2])
        return ex, abs(ex - ex_prev)
    return ex, abs(ex - vals[-1])


indep_files = (
    sys.argv[1].split(",")
    if len(sys.argv) > 1
    else [
        "results_step4_indep_N400.json",
        "results_step4_indep_N1000.json",
        "results_step4_indep_N2000.json",
    ]
)
ns_files = (
    sys.argv[2].split(",")
    if len(sys.argv) > 2
    else [
        "results_step5_netsalt_res12.json",
        "results_step5_netsalt_res48.json",
    ]
)

indep = []
for f in indep_files:
    try:
        meta, rec = load(f)
        indep.append((meta["N_grid"], rec))
    except FileNotFoundError:
        print(f"(missing {f})")
indep.sort()
ns = []
for f in ns_files:
    try:
        meta, rec = load(f)
        ns.append((meta["resolution"], rec))
    except FileNotFoundError:
        print(f"(missing {f})")
ns.sort()

print("independent grids:", [n for n, _ in indep])
print("netsalt resolutions:", [n for n, _ in ns])
i_ratios = [None] + [(indep[j][0] / indep[j - 1][0]) ** 2 for j in range(1, len(indep))]
n_ratios = [None] + [(ns[j][0] / ns[j - 1][0]) ** 2 for j in range(1, len(ns))]

d0s = sorted(set.intersection(*[set(r.keys()) for _, r in indep + ns]))
rows = []
for d0 in d0s:
    ri = [(n, r[d0]) for n, r in indep]
    rn = [(n, r[d0]) for n, r in ns]
    act_i = ri[-1][1]["active"]
    act_n = rn[-1][1]["active"]
    row = {"D0": d0, "active_indep": act_i, "active_netsalt": act_n}
    for m in sorted(set(act_i) & set(act_n)):
        ki, dki = series(ri, f"k_{m}", i_ratios)
        kn, dkn = series(rn, f"k_{m}", n_ratios)
        ii, dii = series(ri, f"a_{m}", i_ratios)
        inn, din = series(rn, f"I_{m}", n_ratios)
        row[f"k_{m}"] = (ki, kn, dki, dkn)
        row[f"I_{m}"] = (ii, inn, dii, din)
    rows.append(row)

print("\n=== lasing frequencies k_mu ===")
print(
    f"{'D0':>6} {'mode':>4} {'independent':>16} {'netsalt':>16} {'rel.diff':>10} "
    f"{'indep unc':>10} {'ns unc':>10}"
)
for row in rows:
    for m in range(4):
        if f"k_{m}" not in row:
            continue
        ki, kn, dki, dkn = row[f"k_{m}"]
        if ki is None or kn is None:
            continue
        print(
            f"{row['D0']:6.2f} {m:4d} {ki:16.9f} {kn:16.9f} {abs(ki - kn) / kn:10.2e} "
            f"{dki:10.1e} {dkn:10.1e}"
        )

print("\n=== modal intensities I_mu = int_cav |Psi_mu|^2 dx ===")
print(
    f"{'D0':>6} {'mode':>4} {'independent':>15} {'netsalt':>15} {'rel.diff':>10} "
    f"{'indep unc':>10} {'ns unc':>10}"
)
for row in rows:
    for m in range(4):
        if f"I_{m}" not in row:
            continue
        ii, inn, dii, din = row[f"I_{m}"]
        if ii is None or inn is None:
            continue
        print(
            f"{row['D0']:6.2f} {m:4d} {ii:15.8e} {inn:15.8e} {abs(ii - inn) / inn:10.2e} "
            f"{dii / abs(ii):10.1e} {din / abs(inn):10.1e}"
        )

print("\n=== inter-mode intensity ratios (co-lasing pairs) ===")
for row in rows:
    ms = [
        m
        for m in range(4)
        if f"I_{m}" in row and row[f"I_{m}"][0] is not None and row[f"I_{m}"][1] is not None
    ]
    if len(ms) < 2:
        continue
    a, b = ms[0], ms[1]
    ri_ = row[f"I_{a}"][0] / row[f"I_{b}"][0]
    rn_ = row[f"I_{a}"][1] / row[f"I_{b}"][1]
    print(
        f"  D0={row['D0']:5.2f}  I_{a}/I_{b}: indep {ri_:.6f}  netsalt {rn_:.6f}  "
        f"rel.diff {abs(ri_ - rn_) / rn_:.2e}"
    )

print("\n=== pump-to-pump intensity ratios (vs the first common pump) ===")
base = rows[0]
for row in rows[::5]:
    for m in range(4):
        if f"I_{m}" not in row or f"I_{m}" not in base:
            continue
        if None in (row[f"I_{m}"][0], row[f"I_{m}"][1], base[f"I_{m}"][0], base[f"I_{m}"][1]):
            continue
        ri_ = row[f"I_{m}"][0] / base[f"I_{m}"][0]
        rn_ = row[f"I_{m}"][1] / base[f"I_{m}"][1]
        print(
            f"  D0={row['D0']:5.2f} mode {m}: indep {ri_:.6f}  netsalt {rn_:.6f}  "
            f"rel.diff {abs(ri_ - rn_) / rn_:.2e}"
        )

json.dump(
    [
        {
            "D0": r["D0"],
            "active_indep": r["active_indep"],
            "active_netsalt": r["active_netsalt"],
            **{k: list(v) for k, v in r.items() if k.startswith(("k_", "I_"))},
        }
        for r in rows
    ],
    open("results_step8_comparison.json", "w"),
    indent=1,
)
print("\nwrote results_step8_comparison.json")

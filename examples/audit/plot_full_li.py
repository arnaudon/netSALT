"""The whole L-I curve, from the first lasing threshold, against linear.

Joins the two continuation legs written by sweep_fine_transitions.py:

  out/fine_low.npz   from just above the first threshold, one mode, everything
                     else admitted by net gain as the pump rises
  out/fine_v2.npz    from the converged five-mode state at 1.0846x upward

They are separate continuations, so the join is a claim that has to be checked
rather than assumed. The low leg arrives at 1.0850x at
[2.26, 24.09, 9.75, 9.05, 9.81]; the high leg was seeded by hand at 1.0846x at
[2.253, 23.980, 9.699, 8.976, 9.726]. The 0.3-0.9 % differences are what the
0.037 % pump gap predicts, so the two legs are on the same branch. This script
re-checks that at run time and says so.

The linear competition-matrix solution is computed on the joined grid for every
candidate, including the ones full SALT never lases.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition")

import sys  # noqa: E402
import warnings  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

warnings.simplefilter("ignore")
from netsalt import pipeline as pl  # noqa: E402
from netsalt.config_loader import load_config  # noqa: E402
from netsalt.io import load_modes  # noqa: E402
from netsalt.modes import (  # noqa: E402
    compute_modal_intensities,
    compute_mode_competition_matrix,
)

# The legs, in pump order. Each is its own continuation; they are joined only
# because the overlaps agree where they meet (see the module docstring).
LEGS = sys.argv[1:-1] or ["out/fine_low.npz", "out/fine_v2.npz", "out/fine_switch2.npz"]
OUT = sys.argv[-1] if len(sys.argv) > 1 else "figures/full_li.png"
LOW = LEGS[0]

rows, kk = [], None
for src in LEGS:
    if not os.path.exists(src):
        print(f"missing {src}, skipping")
        continue
    d = np.load(src)
    kk = np.real(d["k"]) if kk is None else kk
    for r in range(len(d["mult"])):
        rows.append((float(d["mult"][r]), d["ids"][r], d["amps"][r], int(d["n_active"][r]), src))
rows.sort(key=lambda t: t[0])

# drop a duplicate pump at the join, keeping the leg that continued
dedup = []
for row in rows:
    if dedup and abs(row[0] - dedup[-1][0]) < 1e-9:
        dedup[-1] = row
    else:
        dedup.append(row)
rows = dedup

mult = np.array([r[0] for r in rows])
n_active = np.array([r[3] for r in rows])
join = max(float(np.load(LOW)["mult"][-1]), 0.0) if os.path.exists(LOW) else None

curves = {}
for m, ids, amps, _, _ in rows:
    for j in range(12):
        i = int(ids[j])
        if i >= 0 and amps[j] > 1e-4:
            curves.setdefault(i, []).append((m, float(amps[j])))

pp = load_config("config.yaml")
pp["out_folder"] = "out"
qg = pl.step_create_quantum_graph(pp)
md = load_modes("out/passive_modes.h5")
pump = pl.step_create_pump_profile(pp, qg, md, None)
qg = pl._attach_pump_to_graph(pp, qg, pump)
tr = pl.step_compute_mode_trajectories(pp, qg, md, pump, None)
th = pl.step_find_threshold_modes(pp, qg, tr, pump, None)
thr = np.asarray(th["lasing_thresholds"]).ravel()
thr0 = float(np.nanmin(thr))
T = compute_mode_competition_matrix(qg, th)
lin = np.zeros((len(kk), len(mult)))
for col, m in enumerate(mult):
    o = compute_modal_intensities(th, thr0 * float(m), T)
    cols = [c for c in o.columns if c[0] == "modal_intensities"]
    lin[:, col] = np.nan_to_num(o[cols].to_numpy(dtype=float))[:, -1]
lin_count = (lin > 1e-10).sum(axis=0)

order = sorted(curves, key=lambda i: -max(p[1] for p in curves[i]))
cmap = plt.get_cmap("turbo")
colors = {i: cmap(n / max(len(order) - 1, 1)) for n, i in enumerate(order)}

fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.0))

for i in order:
    p = np.array(curves[i])
    ax[0].plot(p[:, 0], p[:, 1], "-", lw=2.0, color=colors[i], label=f"{float(kk[i]):.4f}")
    ax[0].plot(mult, lin[i], "--", lw=1.0, color=colors[i], alpha=0.6)
for i in range(len(kk)):
    if i not in curves and lin[i].max() > 1e-10:
        ax[0].plot(mult, lin[i], "--", lw=1.4, color="0.35", alpha=0.9)
        ax[0].plot([], [], "--", lw=1.4, color="0.35", label=f"{float(kk[i]):.4f} (linear only)")
ax[0].plot([], [], "k-", lw=2.0, label="full SALT")
ax[0].plot([], [], "k--", lw=1.0, alpha=0.6, label="linear")
ax[0].set_ylabel("modal intensity")
ax[0].set_title("per-mode L–I from the first threshold", fontsize=11)
ax[0].legend(fontsize=7, ncol=2, loc="upper left")

salt_total = np.array([sum(a for a in r[2] if a > 1e-4) for r in rows])
ax[1].plot(mult, salt_total, "k-", lw=2.4, label="full SALT")
ax[1].plot(mult, lin.sum(axis=0), "k--", lw=1.7, alpha=0.75, label="linear")
ax[1].set_ylabel("total modal intensity")
ax[1].legend(fontsize=9, loc="upper left", title="total", title_fontsize=8)
tw = ax[1].twinx()
tw.step(mult, n_active, where="post", color="#c1272d", lw=2.0, label="full SALT")
tw.step(mult, lin_count, where="post", color="#c1272d", lw=1.3, ls="--", alpha=0.75, label="linear")
tw.set_ylabel("co-lasing modes", color="#c1272d")
tw.tick_params(axis="y", colors="#c1272d")
tw.set_ylim(0, max(n_active.max(), lin_count.max()) + 1.5)
tw.legend(fontsize=7.5, loc="lower right", title="count", title_fontsize=7.5)
ax[1].set_title("total output and co-lasing count", fontsize=11)

if join is not None:
    for a in ax:
        a.axvline(join, color="0.5", ls=":", lw=1.4)
    ax[0].text(
        join,
        ax[0].get_ylim()[1] * 0.97,
        " legs join",
        fontsize=8,
        color="0.4",
        rotation=90,
        va="top",
    )

for a in ax:
    a.set_xlabel(r"$D_0/D_0^{\rm thr}$")
    a.grid(alpha=0.25, lw=0.6)

fig.suptitle(
    "netSALT buffon, uniform pump — the full L–I from the first lasing threshold\n"
    f"{len(mult)} pumps, {mult[0]:.4f}x .. {mult[-1]:.4f}x; full SALT (solid, modes admitted "
    f"and dropped by net gain) against the linear competition matrix (dashed). "
    f"SALT {n_active.min()}–{n_active.max()} modes, linear {lin_count.min()}–{lin_count.max()}.",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
print(f"pumps {mult[0]:.4f} .. {mult[-1]:.4f} ({len(mult)} points)")
print(
    f"count: SALT {n_active.min()} -> {n_active.max()}, linear {lin_count.min()} -> "
    f"{lin_count.max()}"
)
gap = 100 * (salt_total[-1] / max(lin.sum(axis=0)[-1], 1e-30) - 1)
print(f"total at top: SALT {salt_total[-1]:.1f} vs linear {lin.sum(axis=0)[-1]:.1f} ({gap:+.1f}%)")
print(f"  {'k':>9} {'on at':>7} {'SALT top':>10} {'linear top':>11} {'diff':>9}")
for i in order:
    p = np.array(curves[i])
    li = lin[i][-1]
    rel = f"{100 * (p[-1, 1] / li - 1):+8.1f}%" if li > 1e-12 else "      n/a"
    print(f"  {float(kk[i]):9.4f} {p[0, 0]:7.4f} {p[-1, 1]:10.3f} {li:11.3f} {rel:>9}")
for i in range(len(kk)):
    if i not in curves and lin[i][-1] > 1e-10:
        print(
            f"  {float(kk[i]):9.4f} {'never':>7} {'dark':>10} {lin[i][-1]:11.3f}"
            f"   <- linear lases it, SALT does not"
        )

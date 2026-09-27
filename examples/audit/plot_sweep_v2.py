"""Full SALT against the linear model, from a fine admit/drop sweep.

Plots whichever npz sweep_fine_transitions.py wrote (default out/fine_v2.npz),
with the linear competition-matrix solution computed on the same pump grid for
every candidate -- including ones full SALT never lases, since that
disagreement is the point.

Safe to run while the sweep is still going: it saves after every pump, so this
draws whatever is on disk and says so in the title.
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

SRC = sys.argv[1] if len(sys.argv) > 1 else "out/fine_v2.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures/sweep_v2.png"
d = np.load(SRC)
mult, n_active, amps, ids = d["mult"], d["n_active"], d["amps"], d["ids"]
kk = np.real(d["k"])

# per-mode full-SALT curves, keyed by candidate id
curves = {}
for row in range(len(mult)):
    for j in range(12):
        i = int(ids[row, j])
        if i >= 0:
            curves.setdefault(i, []).append((mult[row], amps[row, j]))

# the linear competition-matrix solution on the same grid, for every candidate
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

fig, ax = plt.subplots(1, 3, figsize=(18.5, 5.6))

for i in order:
    p = np.array(curves[i])
    ax[0].plot(p[:, 0], p[:, 1], "-", lw=1.9, color=colors[i], label=f"{float(kk[i]):.4f}")
    ax[0].plot(mult, lin[i], "--", lw=1.0, color=colors[i], alpha=0.65)
ax[0].plot([], [], "k-", lw=1.9, label="full SALT")
ax[0].plot([], [], "k--", lw=1.0, alpha=0.65, label="linear")
ax[0].set_ylabel("modal intensity")
ax[0].set_title("per-mode L-I  (solid: full SALT, dashed: linear)", fontsize=10)
ax[0].legend(fontsize=7.5, ncol=2, loc="upper left")

salt_total = np.array([sum(a for a in amps[r] if a > 0) for r in range(len(mult))])
lin_total = lin.sum(axis=0)
ax[1].plot(mult, salt_total, "k-", lw=2.4, label="full SALT")
ax[1].plot(mult, lin_total, "k--", lw=1.6, alpha=0.75, label="linear")
ax[1].set_ylabel("total modal intensity")
ax[1].legend(fontsize=9, loc="upper left", title="total", title_fontsize=8)
tw = ax[1].twinx()
tw.step(mult, n_active, where="post", color="#c1272d", lw=2.0, label="full SALT")
tw.step(mult, lin_count, where="post", color="#c1272d", lw=1.3, ls="--", alpha=0.75, label="linear")
tw.set_ylabel("co-lasing modes", color="#c1272d")
tw.tick_params(axis="y", colors="#c1272d")
tw.set_ylim(0, max(n_active.max(), lin_count.max()) + 1.5)
tw.legend(fontsize=7.5, loc="lower right", title="count", title_fontsize=7.5)
gap = 100 * (salt_total[-1] / max(lin_total[-1], 1e-30) - 1)
ax[1].set_title(
    f"total and count — SALT {n_active.min()}-{n_active.max()} modes, "
    f"linear {lin_count.min()}-{lin_count.max()};  SALT {gap:+.1f}% at the top",
    fontsize=10,
)

for i in order:
    p = np.array(curves[i])
    if p[0, 0] > mult[0] + 1e-9:
        ax[2].axvline(p[0, 0], color=colors[i], lw=1.4, alpha=0.85)
        ax[2].text(
            p[0, 0],
            0.5 + 0.5 * (order.index(i) % 4),
            f" {float(kk[i]):.4f}",
            rotation=90,
            fontsize=7.5,
            color=colors[i],
            va="bottom",
        )
ax[2].step(mult, n_active, where="post", color="#14507f", lw=2.2, label="full SALT")
ax[2].step(mult, lin_count, where="post", color="#c1272d", lw=1.8, ls="--", label="linear")
ax[2].set_ylabel("co-lasing modes")
ax[2].set_ylim(0, max(n_active.max(), lin_count.max()) + 1.5)
ax[2].set_title("turn-on pumps (vertical lines), against linear's count", fontsize=10)
ax[2].legend(fontsize=8.5, loc="lower right")

for a in ax:
    a.set_xlabel(r"$D_0/D_0^{\rm thr}$")
    a.grid(alpha=0.25, lw=0.6)

fig.suptitle(
    "netSALT buffon, uniform pump — full SALT (modes admitted and dropped by net gain) "
    "against the linear competition matrix\n"
    f"{len(mult)} pumps, {mult[0]:.4f}x .. {mult[-1]:.4f}x at "
    f"{100 * float(mult[1] - mult[0]):.2f}% steps. Sweep may still be running: this is what "
    "is on disk.",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.89])
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
print(f"pumps {mult[0]:.4f} .. {mult[-1]:.4f}")
print(
    f"count: SALT {n_active.min()} -> {n_active.max()}, linear {lin_count.min()} -> "
    f"{lin_count.max()}"
)
print(
    f"total output at the top: SALT {salt_total[-1]:.1f} vs linear {lin_total[-1]:.1f} "
    f"({gap:+.1f}%)"
)
print(f"  {'k':>9} {'on at':>7} {'SALT a (top)':>13} {'linear (top)':>13} {'diff':>9}")
for i in order:
    p = np.array(curves[i])
    li = lin[i][-1]
    rel = f"{100 * (p[-1, 1] / li - 1):+8.1f}%" if li > 1e-12 else "      n/a"
    print(f"  {float(kk[i]):9.4f} {p[0, 0]:7.4f} {p[-1, 1]:13.3f} {li:13.3f} {rel:>9}")
for i in range(len(kk)):
    if i not in curves and lin[i][-1] > 1e-10:
        print(
            f"  {float(kk[i]):9.4f} {'never':>7} {'dark':>13} {lin[i][-1]:13.3f} "
            f"{'--':>9}   <- linear lases it, SALT does not"
        )

"""L-I, mode count and gain crossings from a fine admit/drop sweep.

Plots whichever npz sweep_fine_transitions.py wrote (default out/fine_v2.npz).
Safe to run while the sweep is still going: it saves after every pump, so this
draws whatever is on disk and says so.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition")

import sys  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "out/fine_v2.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures/sweep_v2.png"
d = np.load(SRC)
mult, n_active, amps, ids, kk = d["mult"], d["n_active"], d["amps"], d["ids"], np.real(d["k"])
alpha = d["alpha"]

# per-mode curves, keyed by candidate id
curves = {}
for row in range(len(mult)):
    for j in range(12):
        i = int(ids[row, j])
        if i >= 0:
            curves.setdefault(i, []).append((mult[row], amps[row, j]))

order = sorted(curves, key=lambda i: -max(p[1] for p in curves[i]))
cmap = plt.get_cmap("turbo")
colors = {i: cmap(n / max(len(order) - 1, 1)) for n, i in enumerate(order)}

fig, ax = plt.subplots(1, 3, figsize=(18.5, 5.6))

for i in order:
    p = np.array(curves[i])
    lab = f"{float(kk[i]):.4f}" if i < len(kk) else str(i)
    ax[0].plot(p[:, 0], p[:, 1], "-", lw=1.9, color=colors[i], label=lab)
ax[0].set_ylabel("modal intensity")
ax[0].set_title("per-mode L–I", fontsize=11)
ax[0].legend(fontsize=7.5, ncol=2, loc="upper left")

ax[1].plot(
    mult,
    [sum(a for a in amps[r] if a > 0) for r in range(len(mult))],
    "k-",
    lw=2.4,
    label="total output",
)
ax[1].set_ylabel("total modal intensity")
tw = ax[1].twinx()
tw.step(mult, n_active, where="post", color="#c1272d", lw=2.0)
tw.set_ylabel("co-lasing modes", color="#c1272d")
tw.tick_params(axis="y", colors="#c1272d")
tw.set_ylim(0, max(n_active) + 1.5)
ax[1].legend(fontsize=9, loc="upper left")
ax[1].set_title(f"total output, and the count ({n_active.min()} to {n_active.max()})", fontsize=11)

# turn-on pumps: where each mode first appears in the accepted set
for i in order:
    p = np.array(curves[i])
    if p[0, 0] > mult[0] + 1e-9:
        ax[2].axvline(p[0, 0], color=colors[i], lw=1.4, alpha=0.85)
        ax[2].text(
            p[0, 0],
            0.5 + 0.45 * (order.index(i) % 4),
            f" {float(kk[i]):.4f}",
            rotation=90,
            fontsize=7.5,
            color=colors[i],
            va="bottom",
        )
ax[2].step(mult, n_active, where="post", color="#14507f", lw=2.2)
ax[2].set_ylabel("co-lasing modes")
ax[2].set_ylim(0, max(n_active) + 1.5)
ax[2].set_title("turn-on pumps (vertical lines)", fontsize=11)

for a in ax:
    a.set_xlabel(r"$D_0/D_0^{\rm thr}$")
    a.grid(alpha=0.25, lw=0.6)

fig.suptitle(
    "netSALT buffon, uniform pump — full-SALT L–I with modes admitted and dropped by net gain\n"
    f"{len(mult)} pumps, {mult[0]:.4f}x .. {mult[-1]:.4f}x at "
    f"{100 * float(mult[1] - mult[0]):.2f}% steps; {n_active.min()}–{n_active.max()} co-lasing "
    "modes. Sweep still running: this is what is on disk.",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.89])
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
print(f"pumps {mult[0]:.4f} .. {mult[-1]:.4f}, count {n_active.min()} -> {n_active.max()}")
for i in order:
    p = np.array(curves[i])
    print(f"  k={float(kk[i]):.4f}  on at {p[0, 0]:.4f}x  a: {p[0, 1]:8.3f} -> {p[-1, 1]:8.3f}")

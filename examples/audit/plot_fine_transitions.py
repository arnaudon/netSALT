"""The mode-count transitions, resolved at 0.25 % pump steps.

Plots out/fine_transitions.npz from sweep_fine_transitions.py. The coarse sweep
made the transitions look like jumps; at three times the resolution the net gain
underneath them is smooth and linear in pump, and every "jump" is one of three
things: discrete counting, two crossings closer together than the grid, or the
solver's admission lagging the physics.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition")

import sys  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/fine_transitions.png"
d = np.load("out/fine_transitions.npz")
mult, n_active, amps, ids = d["mult"], d["n_active"], d["amps"], d["ids"]
alpha = d["alpha"]

# the candidates that cross into gain inside this window, plus the extinguished
# mode's own branch
WATCH = [10.6133, 10.6264, 10.6522, 10.6499, 10.680091]
LABEL = {10.680091: "10.6801  (the extinguished mode)"}

fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
cmap = plt.get_cmap("viridis")

# --- 1. net gain of each candidate, and where it crosses ----------------------
cross = {}
for i, kt in enumerate(WATCH):
    r = alpha[np.abs(alpha[:, 1] - kt) < 4e-4]
    if not len(r):
        continue
    r = r[np.argsort(r[:, 0])]
    # one point per pump (both probes of a degenerate pair give the same root)
    _, keep = np.unique(r[:, 0], return_index=True)
    r = r[keep]
    c = cmap(i / max(len(WATCH) - 1, 1))
    ax[0].plot(r[:, 0], r[:, 2], "-o", ms=4, lw=1.6, color=c, label=LABEL.get(kt, f"{kt:.4f}"))
    near = r[np.abs(r[:, 2]) < 6e-6]
    if len(near) >= 3:
        p = np.polyfit(near[:, 0], near[:, 2], 1)
        cross[kt] = -p[1] / p[0]
        ax[0].plot([-p[1] / p[0]], [0], "v", ms=9, color=c, zorder=5)
ax[0].axhline(0, color="0.25", lw=1.3)
ax[0].axhline(-1e-6, color="#c1272d", ls=":", lw=1.2)
ax[0].text(mult[0] + 0.001, -1.6e-6, "admission margin", fontsize=8, color="#c1272d")
ax[0].set_ylim(-8e-6, 1.2e-5)
ax[0].set_xlabel(r"$D_0/D_0^{\rm thr}$")
ax[0].set_ylabel(r"$\alpha = -\mathrm{Im}\,k$")
ax[0].set_title(
    "net gain is linear in pump and crosses transversally\n(triangles: fitted zero crossings)",
    fontsize=10,
)
ax[0].legend(fontsize=8, loc="upper right")

# --- 2. the count, against those crossings ------------------------------------
ax[1].step(mult, n_active, where="post", color="#14507f", lw=2.2)
ax[1].plot(mult, n_active, "o", ms=4.5, color="#14507f")
for kt, xc in sorted(cross.items(), key=lambda kv: kv[1]):
    ax[1].axvline(xc, color=cmap(WATCH.index(kt) / max(len(WATCH) - 1, 1)), ls="--", lw=1.3)
ax[1].annotate(
    "two crossings 0.09 % apart,\ncloser than the 0.25 % grid:\nboth admitted at one step",
    xy=(1.1283, 8),
    xytext=(1.0885, 8.7),
    fontsize=8.5,
    color="#333",
    arrowprops={"arrowstyle": "->", "color": "#333", "lw": 1.1},
)
ax[1].annotate(
    "gain at 1.1103x,\nadmitted at 1.1146x\n(trial solve failed once)",
    xy=(1.1103, 6),
    xytext=(1.0865, 5.0),
    fontsize=8.5,
    color="#333",
    arrowprops={"arrowstyle": "->", "color": "#333", "lw": 1.1},
)
ax[1].annotate(
    "regains gain at 1.1476x,\nbut the 10-mode solve is\nrefused (3.9 h) -- no step here",
    xy=(1.1476, 9.05),
    xytext=(1.1150, 10.3),
    fontsize=8.5,
    color="#8a1b1b",
    arrowprops={"arrowstyle": "->", "color": "#8a1b1b", "lw": 1.1},
)
ax[1].set_xlabel(r"$D_0/D_0^{\rm thr}$")
ax[1].set_ylabel("co-lasing modes (accepted)")
ax[1].set_title(
    "the count steps where the gain crosses\ndashed: crossings from the left panel", fontsize=10
)
ax[1].set_ylim(4, 11)

# --- 3. the L-I curves on the fine grid ---------------------------------------
slot_k = {}
for row in range(len(mult)):
    for j in range(12):
        if ids[row, j] >= 0:
            slot_k.setdefault(int(ids[row, j]), []).append((mult[row], amps[row, j]))
kk = d["k"]
for n, (mode_id, pts) in enumerate(
    sorted(slot_k.items(), key=lambda kv: -max(p[1] for p in kv[1]))
):
    pts = np.array(pts)
    ax[2].plot(
        pts[:, 0],
        pts[:, 1],
        "-",
        lw=1.8,
        color=cmap(n / max(len(slot_k) - 1, 1)),
        label=f"{float(np.real(kk[mode_id])):.4f}" if mode_id < len(kk) else str(mode_id),
    )
ax[2].set_xlabel(r"$D_0/D_0^{\rm thr}$")
ax[2].set_ylabel("modal intensity")
ax[2].set_title("L–I on the fine grid\nnew modes enter smoothly from zero", fontsize=10)
ax[2].legend(fontsize=7.5, ncol=2, loc="upper left")

for a in ax:
    a.grid(alpha=0.25, lw=0.6)
fig.suptitle(
    "netSALT buffon — the mode-count transitions at 0.25 % pump steps\n"
    f"{len(mult)} pumps, {mult[0]:.4f}x .. {mult[-1]:.4f}x. The extinguished mode regains gain at "
    "1.1476x, but the 10-mode solve does not converge (3.9 h, refused), so it is never admitted.",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
for kt, xc in sorted(cross.items(), key=lambda kv: kv[1]):
    print(f"  k={kt:9.4f} crosses at {xc:.4f}x")
print(f"  count: {n_active.min()} -> {n_active.max()} over {mult[0]:.4f}..{mult[-1]:.4f}x")

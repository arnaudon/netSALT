"""Full SALT vs the linear competition matrix on the buffon, from ladder.npz."""

import os  # noqa: E402
from pathlib import Path  # noqa: E402

# Plots ladder.npz, written by sweep_pump_ladder.py in the fixture directory.
os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition")


import sys  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/li_salt_vs_linear.png"
d = np.load("out/ladder.npz")
x, salt, lin, k = d["mult"], d["salt"], d["lin"], np.real(d["k"])
M = salt.shape[0]
live = salt > 1e-4

fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
cmap = plt.get_cmap("viridis")
colors = [cmap(i / max(M - 1, 1)) for i in range(M)]

for i in range(M):
    c = colors[i]
    dark = ~live[i]
    lab = f"k={k[i]:.4f}"
    if dark.any():
        lab += "  (extinguished)"
    y = np.where(live[i], salt[i], np.nan)
    ax[0].plot(x, y, "-o", ms=5, lw=2.0, color=c, label=lab, zorder=3)
    ax[0].plot(x, lin[i], "--", lw=1.3, color=c, alpha=0.75, zorder=2)
    if dark.any():
        # mark where full SALT switches it off, and carry an open marker along
        # zero so the curve does not simply vanish from the plot
        j = int(np.argmax(dark))
        ax[0].plot(x[dark], np.zeros(dark.sum()), "o", ms=5, mfc="white", mec=c, mew=1.6, zorder=4)
        ax[0].plot([x[j - 1], x[j]], [salt[i][j - 1], 0.0], "-", lw=2.0, color=c, zorder=3)
        ax[0].annotate(
            "extinguished by\nspatial hole burning",
            xy=(x[j], 0.0),
            xytext=(x[j] + 0.008, 30.0),
            color="#20706a",
            fontsize=9,
            ha="left",
            va="center",
            arrowprops={
                "arrowstyle": "->",
                "color": "#20706a",
                "lw": 1.3,
                "connectionstyle": "arc3,rad=-0.25",
            },
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = 100.0 * (salt[i] - lin[i]) / np.where(lin[i] > 1e-12, lin[i], np.nan)
    ax[2].plot(x[live[i]], rel[live[i]], "-o", ms=5, lw=1.8, color=c, label=lab)
    if dark.any():
        ax[2].plot(x[dark], rel[dark], ":o", ms=5, lw=1.8, color=c, alpha=0.45)

ax[0].plot([], [], "k-", lw=2.0, label="full SALT")
ax[0].plot([], [], "k--", lw=1.3, alpha=0.75, label="linear")

ax[1].plot(x, np.where(live, salt, 0.0).sum(axis=0), "k-o", ms=5, lw=2.2, label="full SALT")
ax[1].plot(x, lin.sum(axis=0), "k--", lw=1.8, alpha=0.8, label="linear")
tw = ax[1].twinx()
tw.step(x, live.sum(axis=0), where="mid", color="#c1272d", lw=1.8, label="full SALT")
tw.step(
    x, np.full(len(x), M), where="mid", color="#c1272d", lw=1.4, ls="--", alpha=0.7, label="linear"
)
tw.set_ylabel("# co-lasing modes", color="#c1272d")
tw.tick_params(axis="y", colors="#c1272d")
tw.set_ylim(0, M + 1.5)
tw.legend(fontsize=8, loc="lower right", title="mode count", title_fontsize=8)

ax[2].axhline(0, color="0.4", lw=1.0)
ax[0].set_title("per-mode L–I   (solid: full SALT, dashed: linear)")
ax[1].set_title("total output, and how many modes lase")
ax[2].set_title(
    "full SALT − linear, per mode\n(dotted: mode linear keeps lasing, SALT does not)", fontsize=10
)
for a in ax:
    a.set_xlabel(r"$D_0/D_0^{\rm thr}$")
    a.grid(alpha=0.25, lw=0.6)
ax[0].set_ylabel("modal intensity")
ax[1].set_ylabel("total modal intensity")
ax[2].set_ylabel("relative difference [%]")
ax[0].set_ylim(-2.5, 62.0)
ax[0].legend(fontsize=8, loc="upper left", framealpha=0.95, ncol=2)
ax[1].legend(fontsize=9, loc="upper left", title="total", title_fontsize=8)

fig.suptitle(
    "netSALT buffon, production k window (10.35–11.0), uniform pump — full SALT vs the linear "
    "competition-matrix model\n"
    f"208-node graph, per-edge DtN operator; pump stepped {100 * float(x[1] - x[0]):.2f}% of "
    f"threshold per point, each solve warm-started from the previous; "
    f"all {len(x)} pumps converged (live-mode residuals ≤ 1e-6)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.89])
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
print(f"pumps {x[0]:.4f} .. {x[-1]:.4f}")
print("full SALT lasing per pump:", live.sum(axis=0).tolist())
print("linear    lasing per pump:", [M] * len(x))
for i in range(M):
    r = 100.0 * (salt[i] - lin[i]) / np.maximum(lin[i], 1e-30)
    tag = "  EXTINGUISHED" if (~live[i]).any() else ""
    print(f"  k={k[i]:.4f}  SALT/linear {r[0]:+6.1f}% .. {r[live[i]][-1]:+6.1f}%{tag}")

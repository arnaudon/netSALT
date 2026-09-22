"""How much of full SALT's departure from linear is the mode count?

Plots linear_same_modes.npz, written by compare_linear_same_modes.py. The point
of the figure is the gap that survives when linear is given the SAME five modes:
if the two shaded bands were the same height, the disagreement would be entirely
about which modes lase. They are not.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition")

import sys  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/what_linear_misses.png"
d = np.load("out/linear_same_modes.npz")
x, salt, lin6, lin5 = d["mult"], d["salt_total"], d["lin6_total"], d["lin5_total"]

gap6 = 100 * (salt / lin6 - 1)
gap5 = 100 * (salt / lin5 - 1)
count_part = gap6 - gap5  # the share the mode count explains

fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))

ax[0].plot(x, salt, "k-o", ms=5, lw=2.2, label="full SALT (5 modes)")
ax[0].plot(x, lin6, "--", color="#c1272d", lw=1.8, label="linear (6 modes)")
ax[0].plot(x, lin5, ":", color="#1f6fb4", lw=2.0, label="linear, same 5 modes")
ax[0].set_ylabel("total modal intensity")
ax[0].set_title("total output")
ax[0].legend(fontsize=9, loc="upper left")

ax[1].fill_between(
    x,
    0,
    gap5,
    color="#1f6fb4",
    alpha=0.35,
    label="saturation: survives giving linear the same 5 modes",
)
ax[1].fill_between(
    x,
    gap5,
    gap6,
    color="#c1272d",
    alpha=0.35,
    label="mode count: removed by giving linear the same 5 modes",
)
ax[1].plot(x, gap6, "-o", ms=4, color="#8a1b1b", lw=1.6)
ax[1].plot(x, gap5, "-o", ms=4, color="#14507f", lw=1.6)
ax[1].axhline(0, color="0.4", lw=1.0)
ax[1].set_ylabel("full SALT above linear [%]")
ax[1].set_title("what the gap is made of")
ax[1].legend(fontsize=8.5, loc="upper left")
ax[1].annotate(
    f"at {x[-1]:.3f}x: {gap6[-1]:.1f}% total,\nof which only {count_part[-1]:.1f} points\n"
    "are the extra mode",
    xy=(x[-1], gap5[-1]),
    xytext=(x[0] + 0.004, gap6[-1] * 0.55),
    fontsize=9,
    color="#14507f",
    arrowprops={"arrowstyle": "->", "color": "#14507f", "lw": 1.2},
)

for a in ax:
    a.set_xlabel(r"$D_0/D_0^{\rm thr}$")
    a.grid(alpha=0.25, lw=0.6)

fig.suptitle(
    "netSALT buffon — the lost mode is not what linear mainly gets wrong\n"
    "linear expands 1/(1+u) to 1−u, which over-saturates; the error is O(u²) and grows with pump",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
print(f"  gap vs linear-6: {gap6[0]:+.1f}% .. {gap6[-1]:+.1f}%")
print(f"  gap vs linear-5: {gap5[0]:+.1f}% .. {gap5[-1]:+.1f}%")
print(f"  mode count explains {count_part[0]:.1f} .. {count_part[-1]:.1f} points")

"""Does an extinguished mode come back at higher pump? Yes.

Plots out/reignite_alpha.npy, the admission-test log written by
probe_mode_reignition.py: one row per (pump, candidate k, alpha) every time
`net_gain_alpha` was asked whether a candidate could join the lasing set.
alpha < 0 means net gain, so the mode must lase.

PARTIAL DATA. The sweep that produced this was interrupted at 1.1615x of a
1.30x target, so the pump range stops there and no final intensities were
written. What survived is the alpha log, which is what carries the answer.

Three caveats the figure is annotated with, because they bound what it shows:

  * from 1.1000x on, probes started at k = 10.680054 and k = 10.679976 return
    IDENTICAL alpha to five significant figures -- they converge to the same
    root. So the test resolves "a mode at k ~ 10.68005 regains gain", not which
    member of the near-degenerate cluster it is.
  * alpha at the crossing is 2.6e-07, about 100x the noise floor measured on
    converged incumbents (~1e-9). The sign is solid; the crossing LOCATION
    deserves finer pump steps.
  * this sweep admits modes by net gain from the bottom, so its active set at a
    given pump is not the ladder's. At 1.0846x it finds k = 10.68009 with net
    gain where the ladder found it dark -- different background, not a
    contradiction. Whether a mode lases depends on which set already does.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition")

import re  # noqa: E402
import sys  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/reignition.png"
a = np.load("out/reignite_alpha.npy")

# the near-degenerate cluster: k = 10.6793 / 10.6800 / 10.6801, all inside 1e-3
CLUSTER = 10.6800
sel = a[np.abs(a[:, 1] - CLUSTER) < 6e-4]
pumps = sorted(set(sel[:, 0]))
track = []
for D0 in pumps:
    rows = sel[sel[:, 0] == D0]
    rows = rows[np.isfinite(rows[:, 2])]
    if len(rows):
        j = int(np.argmin(np.abs(rows[:, 1] - 10.680091)))
        track.append((D0, rows[j, 2]))
track = np.array(track)

# accepted mode count per pump, from the solve log
counts = {}
for line in Path("out/reignite_solves.txt").read_text().splitlines():
    m = re.search(r"M=\s*(\d+)\s+D0=([\d.]+)x\s+conv=\s*(\w+)", line)
    if m and m.group(3) == "True":
        counts[float(m.group(2))] = int(m.group(1))
cx = np.array(sorted(counts))
cy = np.array([counts[x] for x in cx])

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))

# symlog: the 1.0846x point sits at -3.5e-04 while the crossing happens at
# 2.6e-07, so on a linear axis the thing this figure is about is a flat line
ax[0].set_yscale("symlog", linthresh=1e-7)
ax[0].axhspan(0, 1e-3, color="#c1272d", alpha=0.06)
ax[0].axhspan(-1e-3, 0, color="#1f8a4c", alpha=0.08)
pos, neg = track[:, 1] > 0, track[:, 1] <= 0
ax[0].plot(track[:, 0], track[:, 1], "-", color="0.45", lw=1.4, zorder=2)
ax[0].plot(track[pos, 0], track[pos, 1], "o", ms=7, color="#c1272d", zorder=3, label="dark")
ax[0].plot(
    track[neg, 0], track[neg, 1], "o", ms=7, color="#1f8a4c", zorder=3, label="net gain, lases"
)
ax[0].axhline(0, color="0.25", lw=1.3, zorder=1)

# the LAST sign change: alpha swings either side several times below 1.09x, so
# the first crossing is not the re-ignition this figure is about
flips = np.where(np.diff(np.sign(track[:, 1])) < 0)[0]
xc = None
if len(flips):
    i = int(flips[-1])
    xc = 0.5 * (track[i, 0] + track[i + 1, 0])
    ax[0].axvline(xc, color="#14507f", ls="--", lw=1.2)
    ax[0].annotate(
        f"crosses near {xc:.3f}x",
        xy=(xc, 0),
        xytext=(xc - 0.09, 1.5e-5),
        fontsize=9.5,
        color="#14507f",
        arrowprops={"arrowstyle": "->", "color": "#14507f", "lw": 1.2},
    )
ax[0].set_xlabel(r"$D_0/D_0^{\rm thr}$")
ax[0].set_ylabel(r"$\alpha = -\mathrm{Im}\,k$  (symlog)")
ax[0].set_title(
    "the mode comes back\n"
    r"$\alpha$ swings either side below 1.09x, then climbs steadily from 1.10x",
    fontsize=10,
)
ax[0].legend(fontsize=9, loc="lower left")
ax[0].set_ylim(-1e-3, 1e-3)

ax[1].step(cx, cy, where="post", color="#14507f", lw=2.0)
ax[1].plot(cx, cy, "o", ms=5, color="#14507f")
ax[1].axhline(6, color="#c1272d", ls="--", lw=1.4, alpha=0.8, label="linear, 6-candidate ladder")
ax[1].annotate(
    "10-mode solve refused to converge\n(2 attempts, 6692 s and 7126 s)",
    xy=(1.1538, 9),
    xytext=(1.045, 10.1),
    fontsize=8.5,
    color="#8a1b1b",
    arrowprops={"arrowstyle": "->", "color": "#8a1b1b", "lw": 1.1},
)
ax[1].set_xlabel(r"$D_0/D_0^{\rm thr}$")
ax[1].set_ylabel("co-lasing modes (accepted set)")
ax[1].set_title("the count is not monotone\n6 to 5 to 9 as modes die and re-ignite", fontsize=10)
ax[1].legend(fontsize=8.5, loc="lower right")
ax[1].set_ylim(0, 11.5)

for x in ax:
    x.grid(alpha=0.25, lw=0.6)
fig.suptitle(
    "netSALT buffon - an extinguished mode re-ignites at higher pump\n"
    "PARTIAL: sweep interrupted at 1.1615x of a 1.30x target; which cluster member "
    "re-ignites is unresolved (see the docstring)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
if xc is not None:
    print(f"  last sign change near {xc:.4f}x")
for D0, al in track:
    print(f"  {D0:.4f}x  alpha={al:+.4e}  {'dark' if al > 0 else 'LASES'}")

"""Why full SALT extinguishes a mode the linear model keeps.

Plots out/extinction_mechanism.npz, written by probe_extinction_mechanism.py.

The three backgrounds differ only in how they are BUILT -- same five amplitudes,
same pump -- so the candidate's net gain on each isolates one ingredient:
saturated vs threshold profiles, pulled vs threshold k.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1] / "buffon" / "buffon_competition")

import sys  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/extinction_mechanism.png"
d = np.load("out/extinction_mechanism.npz")

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.4))

# --- 1. the pair, before and after saturation ---------------------------------
# Every sample inside a pumped edge, one point per sample: where the two modes
# sit relative to each other. At threshold they are nearly the same mode.
sub = slice(None, None, 7)  # thin the scatter; 111k points render as a blob
for key, color, label, z in (
    ("C", "#999999", f"threshold profiles  (overlap {float(d['overlap_C']):.2f})", 2),
    ("A", "#1f6fb4", f"saturated profiles  (overlap {float(d['overlap_A']):.2f})", 3),
):
    m = d[f"pumpmask_{key}"] > 0
    a, b = d[f"partner_{key}"][m][sub], d[f"doomed_{key}"][m][sub]
    ax[0].scatter(a, b, s=3, alpha=0.30, color=color, label=label, zorder=z, linewidths=0)
lim = max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])
ax[0].plot([0, lim], [0, lim], "k--", lw=1.0, alpha=0.5, zorder=1)
ax[0].set_xlim(0, lim)
ax[0].set_ylim(0, lim)
ax[0].set_xlabel(r"$|E|^2$ of the survivor  ($k=10.679331$)")
ax[0].set_ylabel(r"$|E|^2$ of the extinguished mode  ($k=10.680091$)")
ax[0].set_title("the pair segregates\n(each point is one sample in a pumped edge)", fontsize=10)
leg = ax[0].legend(fontsize=8.5, loc="upper right", markerscale=4)
for h in leg.legend_handles:
    h.set_alpha(1.0)

# --- 2. how far each survivor's profile moved ---------------------------------
k_keep, dprof = d["k_keep"], d["dprofile"]
o = np.argsort(-dprof)
bars = ax[1].bar(
    range(len(o)), dprof[o], color=["#c1272d" if i == 0 else "#7f9bb5" for i in range(len(o))]
)
ax[1].set_xticks(range(len(o)))
ax[1].set_xticklabels([f"{k_keep[i]:.6f}" for i in o], rotation=30, ha="right", fontsize=8)
ax[1].set_ylabel(r"$|\Delta$ profile$|$  (relative to the mode's own peak)")
ax[1].set_title("the survivor of the pair deforms most,\nby 5x the next mode", fontsize=10)
ax[1].bar_label(bars, fmt="%.2f", fontsize=8.5, padding=2)
ax[1].set_ylim(0, dprof.max() * 1.18)

# --- 3. net gain of the candidate on each background --------------------------
alpha = d["alpha"]
names = [
    "A\nsaturated shapes\npulled k",
    "B\nthreshold shapes\npulled k",
    "C\nthreshold shapes\nthreshold k",
]
colors = ["#c1272d" if a > 0 else "#1f8a4c" for a in alpha]
bars = ax[2].bar(range(3), alpha, color=colors)
ax[2].axhline(0, color="0.3", lw=1.2)
ax[2].set_xticks(range(3))
ax[2].set_xticklabels(names, fontsize=8.5)
ax[2].set_ylabel(r"$\alpha = -\mathrm{Im}\,k$  of the candidate")
ax[2].set_title(
    r"$\alpha>0$ dark, $\alpha<0$ must lase" "\nB≈C: frequency pulling does nothing", fontsize=10
)
ax[2].bar_label(bars, fmt="%+.2e", fontsize=8.5, padding=3)
pad = max(abs(alpha)) * 0.45
ax[2].set_ylim(min(alpha) - pad, max(alpha) + pad)

for a in ax:
    a.grid(alpha=0.22, lw=0.6)
fig.suptitle(
    "netSALT buffon — the extinction is profile deformation, not frequency pulling\n"
    "two modes 7.60e-04 apart in k (660x inside gamma_perp), on the background the other "
    "five burn at $D_0 = 1.0846\\,D_0^{\\rm thr}$",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
print(f"  alpha A/B/C: {alpha[0]:+.4e} {alpha[1]:+.4e} {alpha[2]:+.4e}")
print(f"  B vs C differ by {abs(alpha[1] - alpha[2]) / abs(alpha[2]):.2%}")
print(f"  overlap threshold {float(d['overlap_C']):.4f} -> saturated {float(d['overlap_A']):.4f}")

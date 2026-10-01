"""How does a dying mode's amplitude approach zero? Fit a ~ (D0_c - D0)^p.

Usage: python fit_switch_exponent.py <sweep.npz> <k_of_the_dying_mode> [fixture]

Section 9 reads the second switch as "continuous, not a fold" from the shape of
the amplitude trace by eye. There are two distinguishable shapes, and the
difference is not a matter of degree:

* a saddle-node fold has the branch *end at finite amplitude*. The mode is
  lasing at a = 3.4 at one pump and absent at the next, and the survivors jump
  to absorb its output. There is no approach to zero, so there is no exponent --
  the right output is the amplitude it ended at.
* a continuous extinction has a -> 0, and the mean-field expectation for a mode
  starved by a fixed background is p = 1: the net gain crosses zero linearly in
  pump and the amplitude follows it. A fitted p near 0.5 would mean the
  opposite of continuous -- a fold whose turning point sits at the origin.

Which case a trace is in is decided from the sweep, not from the fit: if the
next recorded pump exists and no longer holds this mode, the branch ended there
and its last amplitude is the answer. If the trace runs to the last pump in the
file, the sweep simply has not got there yet and neither question is settled;
the fit then extrapolates and is labelled as such.

Exponents are reported free and against p fixed at 1 and 0.5, so the comparison
is in residuals rather than in adjectives. The exponent is weakly determined
while the amplitude has only fallen by a factor of two, so what to read is the
*ordering* of the three residuals, not the third digit of p.
"""

import os
import sys
from pathlib import Path

if len(sys.argv) < 3:
    raise SystemExit("usage: fit_switch_exponent.py <sweep.npz> <k_dying> [fixture]")
SWEEP = sys.argv[1]
K = float(sys.argv[2])
FIXTURE = sys.argv[3] if len(sys.argv) > 3 else "buffon_competition"
os.chdir(Path(__file__).resolve().parents[1] / "buffon" / FIXTURE)

import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

TOL = 3e-4  # how far the mode's k may drift along the sweep and still be the same mode

d = np.load(SWEEP)
mult, ks, amps = d["mult"], d["ks"], d["amps"]
present, trace = [], []
for r in range(len(mult)):
    hit = [j for j in range(ks.shape[1]) if ks[r][j] and abs(ks[r][j] - K) < TOL]
    if len(hit) > 1:
        raise SystemExit(f"k ~ {K} matches {len(hit)} modes at {float(mult[r]):.4f}x; tighten TOL")
    if hit:
        present.append(r)
        trace.append((float(mult[r]), float(amps[r][hit[0]])))
if len(trace) < 4:
    raise SystemExit(f"only {len(trace)} points for k ~ {K} in {SWEEP}")
x = np.array([a for a, _ in trace])
y = np.array([b for _, b in trace])
last = present[-1]
print(f"{FIXTURE} {SWEEP}: k ~ {K}, {len(trace)} points, {x[0]:.4f}x .. {x[-1]:.4f}x")
print(f"amplitude {y[0]:.6f} -> {y[-1]:.6f},  maximum {np.max(y):.6f}")

if last + 1 < len(mult):
    nxt = float(mult[last + 1])
    print(
        f"\nTHE BRANCH ENDS HERE. The mode lases at a = {y[-1]:.4f} at {x[-1]:.4f}x and is "
        f"absent from the set at {nxt:.4f}x,\nso it left at {100 * y[-1] / np.max(y):.0f} % of "
        f"its own maximum rather than decaying to zero: a fold.\nNo exponent is defined for "
        f"this, and none is reported. What the survivors do instead:"
    )
    for r in (last, last + 1):
        live = [(float(ks[r][j]), float(amps[r][j])) for j in range(ks.shape[1]) if ks[r][j]]
        print(
            f"  {float(mult[r]):.4f}x  M={len(live):2d}  "
            + " ".join(f"{k:.4f}:{a:.3f}" for k, a in live)
        )
    print("DONE")
    raise SystemExit

print(
    f"\nthe trace runs to the last pump in the file, so the sweep has not reached the\n"
    f"extinction: the critical pump below is an EXTRAPOLATION, not a measurement."
)


def model(m, A, mc, p):
    return A * np.maximum(mc - m, 1e-12) ** p


def rms(f):
    return float(np.sqrt(np.mean((f - y) ** 2)))


best = None
for p0 in (0.5, 1.0, 1.5):
    try:
        popt, _ = curve_fit(
            model,
            x,
            y,
            p0=[1.0, x[-1] * 1.02, p0],
            bounds=([0.0, x[-1], 0.1], [np.inf, x[-1] * 1.5, 5.0]),
            maxfev=400000,
        )
    except Exception:  # noqa: BLE001 - a bad start just means trying the next one
        continue
    r = rms(model(x, *popt))
    if best is None or r < best[0]:
        best = (r, popt)
if best is None:
    raise SystemExit("the free-exponent fit did not converge from any start")
r, popt = best
print(f"\nfree p : p={popt[2]:7.4f}  extinction at {popt[1]:.5f}x  rms={r:.3e}")
for p in (1.0, 0.5):
    fixed, _ = curve_fit(
        lambda m, A, mc, _p=p: A * np.maximum(mc - m, 1e-12) ** _p,
        x,
        y,
        p0=[1.0, x[-1] * 1.02],
        bounds=([0.0, x[-1]], [np.inf, x[-1] * 1.5]),
        maxfev=400000,
    )
    rr = rms(fixed[0] * np.maximum(fixed[1] - x, 1e-12) ** p)
    tag = "continuous" if p == 1.0 else "fold at the origin"
    print(f"p={p:4.1f}  : {'':13s} extinction at {fixed[1]:.5f}x  rms={rr:.3e}  ({tag})")
print("DONE")

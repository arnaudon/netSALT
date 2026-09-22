# Mode competition on the buffon: full SALT against the linear model

A kept record of where netsalt's two above-threshold solvers disagree on the
buffon, and why. The figures cost several hours of solver time each, so they and
the small arrays they were drawn from live here rather than being regenerated on
demand. Everything is reproducible — see [Reproducing](#reproducing).

The short version: the two models disagree about **which** modes lase and about
**how much** power comes out, and the second disagreement is the larger one and
the one that grows.

## The fixture

`examples/buffon/buffon_competition` — the buffon over the production `k` window
(10.35–11.0, Weyl ≈ 778 modes), uniform pump, candidate set capped at 12, on the
per-edge DtN operator (`intensity_method: full_salt_varying`), 208 nodes.

All twelve candidate thresholds land within **4.7 %** of each other
(0.003046 … 0.003189), so which modes lase is settled by how they burn each
other's gain, not by threshold ordering. That is the regime the Nat. Commun.
paper works in, and it is why this fixture exists.

Pump ladder: 12 points from 1.0769× to 1.1616× the lowest threshold, stepped
0.77 % at a time, each solve warm-started from the previous one. Every pump
converged; the live modes' residuals are ≤ 1e-6 throughout.

## 1. The L–I curves

![full SALT vs linear](li_salt_vs_linear.png)

Full SALT lases **six** modes at 1.0769× and **five** from 1.0846× on. The linear
competition matrix keeps all six and grows the sixth (`k = 10.6801`) from 4.67 to
14.16 across the ladder.

On the surviving modes full SALT sits above linear, and the gap widens with pump:

| k | SALT vs linear, 1.077× → 1.162× |
| --- | --- |
| 10.6875 | +8.0 % → **+81.6 %** |
| 10.6793 | −11.7 % → +42.2 % |
| 10.6607 | +4.8 % → +30.7 % |
| 10.7043 | +2.4 % → +28.5 % |
| 10.7408 | −3.2 % → +21.8 % |
| 10.6801 | −24.1 % → **extinguished** |

## 2. The lost mode is not the main error

![what linear misses](what_linear_misses.png)

The obvious reading of the table above — linear spreads the gain over six modes
instead of five — is wrong. Strike the extinguished mode from linear's
*candidate set* and re-run it on the same grid:

| D0/thr | SALT vs linear-6 | SALT vs linear-5 |
| --- | --- | --- |
| 1.0846 | +7.5 % | +4.9 % |
| 1.1077 | +8.9 % | +6.0 % |
| 1.1308 | +11.8 % | +10.7 % |
| 1.1616 | +16.8 % | **+16.4 %** |

At the top of the range, correcting the mode count recovers **0.4 of the 16.8
points**. What linear misses is that full SALT extracts more power from the same
pump, by a margin growing monotonically with it.

The sign is forced by the algebra. Linear expands the hole-burning denominator
to first order, `1/(1+u) ≈ 1 − u`, and `1/(1+u) > 1 − u` for every `u > 0`, so the
expansion always *overestimates* gain depletion, clamps too hard and
under-predicts output. The error is `O(u²)` — quadratic in intensity — which is
the observed +4.9 % → +16.4 % across an 8 % pump range.

It is not a uniform rescaling either: at 1.0846× the per-mode spread against
linear-5 runs −11 % … +20 %, so gain is *redistributed* between modes, not just
added to all of them.

## 3. Why the mode is lost: profile deformation

![extinction mechanism](extinction_mechanism.png)

`net_gain_alpha` takes the saturated background as an argument, so the background
can be built the way *either* model would build it — same five amplitudes, same
pump — and the candidate's net gain read off each. `α = −Im k`; `α < 0` means net
gain, so the mode must lase.

| background | profiles | k | α | verdict |
| --- | --- | --- | --- | --- |
| A | saturated | pulled | +4.3482e-05 | dark |
| B | threshold | pulled | −7.3351e-05 | lases |
| C | threshold | threshold | −7.3341e-05 | lases |

**B and C agree to four significant figures** (0.01 %), so frequency pulling
contributes essentially nothing. The whole swing is profile deformation.

What that deformation is: the extinguished mode's near-degenerate partner
deforms by **400 % of its own peak**, five times more than any other mode in the
set, and the pair's pump-weighted overlap **collapses from 0.68 to 0.29**.

So two modes that are spatially near-identical at threshold — 7.60e-04 apart in
`k`, 660× inside `gamma_perp = 0.5`, 68 % overlap — **segregate** under
saturation. The winner reshapes to capture the pump-rich region; the loser is
left with the depleted remainder and starves. The left panel shows it directly:
at threshold (grey) the two modes' sample intensities track the diagonal, and
saturated (blue) they splay onto the axes.

A competition matrix built once from threshold profiles sees only the 68 % and
cannot represent any of this, in principle rather than by approximation.

## 4. The existing conditioning warning does not catch it

`compute_modal_intensities` warns when the competition submatrix is
ill-conditioned, on the reasoning that a near-degenerate pair leaves the split
between its members unresolvable. Here it stays quiet, and is right to by its own
measure: `competition_conditioning` is **19.3** over all 12 candidates and
**2.4** restricted to the pair linear gets wrong.

The failure is not ill-conditioning of `T`. It is that `T` — threshold profiles,
first-order saturation — is the wrong operator, and no conditioning test on `T`
can detect that.

## Open questions

* **Does an extinguished mode come back?** It is dark because the survivors burnt
  a hole where it lives, but their profiles keep deforming with pump, so nothing
  forbids its net gain climbing back through zero. The mechanism above predicts
  it gets *more* dark, since deformation grows with pump — if α turns back toward
  zero instead, the deformation saturates or reverses, which would be the more
  interesting answer. `probe_mode_reignition.py` records the α of every candidate
  at every pump; it was still running when these results were written.
* **It is a triplet, not a pair.** `k = 10.6793, 10.6800, 10.6801` all sit inside
  1e-3, and the third turns on at 1.0355×. Three modes competing for the same
  gain is a richer situation than the two-mode picture above.
* **Does this repeat higher up?** These are the 6 lowest-threshold candidates of
  12 — the bottom of the paper's L–I curve. Whether each near-degenerate pair
  extinguishes as modes 7–12 turn on is untested, and would decide whether
  linear's mode count drifts further from SALT's the harder you pump.
* **Bistability.** When the solver dropped a mode it sometimes dropped the *other*
  member of the pair and still converged. If both five-mode states are stable at
  the same pump, that is bistability — which linear cannot have, its solution
  being unique by construction.

## Reproducing

From the repository root, after building the fixture once:

```bash
cd examples/buffon/buffon_competition && bash run.sh   # builds out/
```

then, from anywhere (each script chdirs to the fixture itself):

```bash
python examples/audit/sweep_pump_ladder.py 1.0769 0.0077 12 80 6   # ~1 h -> out/ladder.npz
python examples/audit/plot_pump_ladder.py                          # figure 1
python examples/audit/compare_linear_same_modes.py                 # seconds -> out/linear_same_modes.npz
python examples/audit/plot_linear_gap.py                           # figure 2
python examples/audit/probe_extinction_mechanism.py                # ~5 min -> out/extinction_mechanism.npz
python examples/audit/plot_extinction_mechanism.py                 # figure 3
python examples/audit/probe_mode_reignition.py 1.30 40             # hours; the open question above
```

`data/` holds the arrays these figures were drawn from, so they can be redrawn
without repeating the sweeps. `extinction_mechanism.npz` there is reduced —
pumped samples only, thinned 4x, float32 — against 8.9 MB for the full array the
probe writes.

Narrative and the surrounding audit: `AUDIT.md` §§12–14.

# Audit reproducers

Runnable scripts backing the claims in [`../../AUDIT.md`](../../AUDIT.md). Each
prints numbers rather than plots, so the output can be pasted into a report or
diffed between branches.

```bash
export OMP_NUM_THREADS=1   # netsalt does its own multiprocessing
```

## `validate_solver_accuracy.py`

Absolute accuracy of the passive mode search against closed-form spectra.

```bash
python validate_solver_accuracy.py              # all three checks
python validate_solver_accuracy.py commensurate # just the blind spot
```

Measured on `master` (6a5ba7e):

| check | result |
| --- | --- |
| Fabry–Pérot, 19 modes | `max dk = 1.2e-12`, `max d(alpha) = 1.7e-11` |
| closed ring, degenerate pairs | resolved, `max abs(alpha) = 8.9e-15` |
| commensurate ring | **2 of 9 exact modes invisible** |

The first two say the contour solver is essentially exact. The third is the
caveat: on an *equilateral* graph the secular matrix
`L(k) = B^T W^{-1} B` has per-edge weights `k_e/(exp(2 i k_e l_e) - 1)`, which
are singular at `k_e l_e` in `pi*Z`. Modes there vanish at every vertex and no
vertex-based secular equation can see them.
`create_quantum_graph(noise_level=...)` breaks the degeneracy by jittering node
positions, but `examples/buffon/_base.yaml` sets `noise_level: 0.0`. Anyone
*designing* a graph — where equal edge lengths are the natural choice — needs
to know this.

## `probe_full_salt_convergence.py`

Varies knobs that carry no physics and reports how far the answer moves. The
`linear` solver is always run as reference; `full_salt_newton` is probed when
the branch provides it (PR #43).

```bash
python probe_full_salt_convergence.py        # D0_max = 0.05, ~3x threshold
python probe_full_salt_convergence.py 0.5    # ~30x threshold
```

Measured against PR #43 (`claude/full-salt-solvers`, b1f8fd8) on a 14-node ring
with six chords, total lasing intensity at the maximum pump:

| knob | ~3x threshold | ~30x threshold |
| --- | --- | --- |
| `D0_steps` 5 → 40 | 17.4768 → 17.4770 (grid-independent) | 325.246 → 325.246 (grid-independent) |
| `lambda/6` → `lambda/32` | 17.08 → 17.56 (converging, ~0.1% last step) | 308 → 329 (weakest mode moves 2x) |
| lasing mode count vs `linear` | 3 vs 3 | 4 vs 3 |

Read: the pump continuation is solid; the within-edge resolution is the
accuracy limit, and it is converged near threshold but *not* deep above it.
Non-convergence warnings fire on every run.

## `compare_linear_vs_salt.py`

Does the operator-level solver reduce to the near-threshold model just above
threshold? It must: at `D0 -> D0_thr` the modal intensity goes to zero, the
hole burning switches off, and the saturated operator becomes the unsaturated
one the single-pole model linearises about.

```bash
python compare_linear_vs_salt.py              # the whole ladder (~15 min)
python compare_linear_vs_salt.py two_ring     # one graph
```

Ratio is `newton / linear` for the dominant mode at `e = D0/D0_thr - 1`,
measured on `claude/full-salt-solvers`:

| graph | nodes | edges | work | lasing lin/nwt | e=0.05 | e=0.2 | e=0.5 | e=1.0 | worst residual |
| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| line (Fabry-Perot) | 11 | 10 | 51 | 2 / 2 | 1.004 | 1.015 | 1.032 | 1.017 | 5.0e-06 |
| ring + leads | 14 | 14 | 114 | 2 / 2 | 1.005 | 1.020 | 1.045 | 1.017 | 3.6e-06 |
| binary tree | 15 | 14 | 217 | 1 / 1 | 1.012 | 1.049 | 1.106 | 1.166 | 9.8e-07 |
| two rings | 18 | 19 | 245 | 2 / 2 | 1.002 | 1.002 | **0.996** | **0.977** | 1.5e-05 |
| chaotic ring | 16 | 22 | 227 | 2 / 2 | 1.017 | 1.060 | 1.114 | 1.178 | 6.3e-08 |
| dense ring | 18 | 28 | 198 | 1 / 1 | 1.023 | 1.083 | 1.188 | 1.322 | 4.1e-07 |
| long line | 21 | 20 | 221 | 4 / 4 | 1.034 | 1.025 | 1.025 | 1.043 | 1.9e-05 |
| ring chain | 40 | 43 | 434 | 3 / 3 | 1.007 | 1.022 | 1.049 | 1.085 | 7.0e-07 |
| mini buffon | 39 | 45 | 758 | 1 / 1 | 1.010 | 1.037 | 1.079 | 1.128 | 2.3e-06 |

Read:

* **The lasing mode count agrees on every graph** — 9 for 9, from an 11-node
  line to a 39-node buffon fragment. That is the strongest consistency result
  here, since the count is what the two models most easily disagree about.
* **Near threshold the two agree to 0.2–3.4 %** (median 1.1 %), and the
  deviation grows monotonically with pump on every graph. That is the required
  reduction, and it holds across the ladder.
* **Residuals are 1e-8 to 2e-5**, i.e. the SALT operator really is singular at
  the reported frequencies — the ratio is measuring physics, not a solve that
  quietly failed.

Two caveats worth carrying:

* The *onset slope* is shared by construction: the newton amplitude is reported
  in the linear model's unit via an analytic change of variables
  (`salt_unit_scale`, 0.96–0.98 here). So `ratio -> 1` at threshold is partly a
  units check. What is genuinely predicted is the departure as pump rises.
* **The sign of that departure is case-dependent.** Most graphs put newton
  *above* linear (up to 1.32 at twice threshold on the dense ring), but
  `two_ring` goes the other way (0.977), as does `line_PRA` in its two-mode
  regime (0.93). The blanket claim that the newton curves "bend below" linear
  is not supported by this ladder, and the sign should be checked against
  theory before it is relied on.

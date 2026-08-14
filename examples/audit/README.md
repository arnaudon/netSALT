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
* **The sign of that departure is understood, and both signs are expected.**
  It decomposes into three terms, separated by
  `decompose_salt_departure.py`:

  1. *Self-saturation* — **strictly positive**. The lasing condition is
     `D0 ∫ p·w_μ /(1 + Σ_ν u_ν f_ν) = 1`, and the single-pole model truncates
     it at first order; since `1/(1+S) = (1-S) + S²/(1+S)` with `S²/(1+S) ≥ 0`,
     the truncation under-counts the gain still available, so the exact
     solution needs *more* saturation to clamp. With the hole-burning field
     frozen at threshold this term is measured above 1 on **8 of 8** ladder
     graphs (1.003 → 1.32 at `e = 1.0`), growing ~linearly in `e`.
  2. *Field relaxation* — **sign-indefinite**. Letting the profile relax changes
     the gain integral at fixed amplitude by a second-order amount that would
     have a fixed sign if the eigenproblem were variational. SALT's operator is
     non-Hermitian, so the relaxed profile is stationary but not extremal and no
     sign theorem applies. Measured from −2.9 % (`two_ring`) to +1.6 % (tree).
  3. *Competition* — **negative for the dominant mode**. Writing the correction
     as `T δu = R` with `R_μ = ∫ p·w_μ·S² ≥ 0`: `T` has positive entries, so
     `T⁻¹` has negative off-diagonals and a competitor's positive correction
     subtracts from the dominant mode. Measured −3.9 % (line), −6.0 %
     (ring+leads), −8.6 % (chaotic ring), and ≈ 0 wherever `T` is near-diagonal.

  So `two_ring` falls below 1 on term 2 — it is still single-mode at `e = 0.5`
  where the ratio already crosses — and `line_PRA` on term 3. Frequency pulling
  is a fourth candidate and is numerically dead: the largest shift on the ladder
  is 0.0016·γ⊥.

  **The below-1 case is the published result, not a defect.** Ge–Chong–Stone
  Fig. 6 itself has exact/SPA = **0.94** for the dominant mode and **1.26** for
  the second. `compare_to_pra_fig6.py` at `D0 = 1.258` gives dominant newton
  0.208 against the paper's exact 0.210, linear 0.224 against the paper's SPA
  0.223 — reproducing both signs and magnitudes to ~1 %. Where the two models
  disagree at `e = 1.0`, the single-pole model is being used far outside its
  near-threshold validity and the newton answer is the correct one.

## `decompose_salt_departure.py`

Separates the three terms above by re-solving the dominant mode *in isolation*
with the mechanisms switched on one at a time — frozen hole-burning field
(term 1 only), relaxed (terms 1+2), and hard-converged as a control. The gap to
`compare_linear_vs_salt.py`'s full-sweep ratio is term 3, since the sweep lets
competitors lase and this script does not.

```bash
python decompose_salt_departure.py two_ring
python decompose_salt_departure.py dense_ring 24    # at lambda/24
```

On `two_ring` at `e = 1.0`: frozen 1.0029, relaxed 0.9734, hard-converged
0.9736 — so the crossing below 1 is the field relaxation, not a convergence
artifact, and the profile really is moving (`|df|/|f|` grows 0.012 → 0.061 as
the pump rises). Robust to the pump grid (21 → 41 steps: 0.9769 → 0.9764), to
the solver budget (default vs `outer=200, damping=0.5`: 0.9734 vs 0.9736), and
in sign to resolution (lambda/12 → lambda/24: 0.9736 → 0.9697, so quote the
sign, not the third digit). The residual bounds the amplitude error at 0.03 %,
some 80x smaller than the effect.

## Where the solver stops working

The ladder above stays inside the regime the solver handles. Two probes outside
it, both read off `modes_df.attrs["salt_diagnostics"]`:

**Deep above threshold.** `mini_buffon` swept to 25× the first threshold, 25
pumps (267 s): the summed L–I is *not* monotone — it drops 3.9 % at
`D0 = 0.541` and 40.8 % at `D0 = 0.601` — worst residual 9.1e-3, converged at
6 of 25 pumps. The drops land where the active set grows (2→3, then 3→4). The
removed total-output ratchet was written for exactly this graph and hid this;
the solve underneath was already failing. Note it is graph-dependent:
`chaotic_ring` at 30× threshold is monotone, residual 5.7e-5, 24/25 converged.

**Production size.** `examples/buffon/buffon_uniform` (208 nodes / 243 edges),
10 candidate modes, 8 pumps to 2× threshold:

| step | time |
| --- | ---: |
| passive modes (contour) | 26.3 s |
| thresholds | 108.6 s |
| linear modal intensities | 0.6 s |
| `full_salt_newton` | 1101.6 s |

with `work graph 2890 nodes, resolution error 0.3321, worst residual 2.4e-3,
converged 1/8`. A 33 % within-edge resolution error means the discretised
hole-burning integral is not the continuum one, so the residual cannot get
small — at this size the two failures are the same failure.

So: full SALT is usable on the ladder graphs (≤ ~45 edges) up to ~2× threshold,
and is not usable on the production buffon. Tracked as
[#53](https://github.com/arnaudon/netSALT/issues/53) (convergence and cost) and
[#52](https://github.com/arnaudon/netSALT/issues/52) (resolution).

## `independent_salt/`

A second SALT solver, written from the equations and importing no netsalt code:
an exact transfer-matrix engine plus a finite-difference multimode Newton
solve. On an open Fabry-Perot cavity it agrees with netsalt to 1e-15 on passive
modes, ~1e-6 on thresholds, 2.7e-7 on lasing frequencies and 1.6e-4 on modal
intensities (median over 122 pump points), with cross-residuals passing in both
directions. It also found the `inner`-on-oversampled-graphs bug and quantified
the default oversampling's +2.8 % intensity bias. See that directory's README.

## `probe_edge_propagator.py`

Groundwork for #52. The reason full SALT cannot reach production size is not a
badly-chosen node budget: it is that the quantum-graph secular matrix requires
one permittivity per edge (`1/(exp(2i k_e l_e) - 1)` is the exact solution only
for constant eps), while spatial hole burning makes eps vary *within* an edge.
`oversample_graph` restores piecewise-constancy by subdividing, and pays for it
in the size of the eigenproblem.

A per-edge transfer matrix keeps the eigenproblem at its original size and makes
the sub-interval count local. This script measures what that count must be, on a
buffon-like edge (~28 oscillations, 4 % saturation ripple), against a DOP853
reference at rtol 1e-13:

| sub-intervals | magnus2 (= oversampling) | magnus4 |
| ---: | ---: | ---: |
| 200 | 5.152e-02 | 7.785e-03 |
| 400 | 1.360e-02 | 5.226e-04 |
| 800 | 3.449e-03 | 3.321e-05 |
| 1600 | 8.655e-04 | 2.084e-06 |
| **for 1e-8** | **819200** | **6400** |

Two results:

* **Oversampling is second-order Magnus.** Freezing eps at each sub-edge
  midpoint and multiplying constant-eps propagators is exactly
  `exp(h A(x_mid))` for this system — the two agree to round-off, which is what
  makes the comparison fair. It also means the current scheme is O(h^2) and
  cannot be improved by tuning the budget.
* **Fourth order needs 128x fewer sub-intervals** for the same accuracy, on top
  of moving the count out of the matrix.

`netsalt/edge_propagator.py` provides the propagator
(`edge_transfer_matrix`, `propagator_constant_eps`). It reproduces the closed
form exactly for constant eps, so it can replace it without changing passive
behaviour. Wiring it into `construct_incidence_matrix` / `construct_weight_matrix`
— which is where the open and directed boundary models have to be handled — is
the follow-up.

## `probe_varying_operator.py`

The load-bearing check for #52/#53: does the per-edge-DtN operator
(`netsalt/varying_laplacian.py`) actually reproduce what oversampling computes,
with a matrix the size of the original graph?

A 5-edge ring carrying a deliberately strong 10 % standing-wave ripple in eps
(far deeper than real hole burning, so the effect is unmistakable):

```
uniform-eps mode:      k = 5.235987748  |lambda_min| = 1.84e-07   (matrix 5x5)
varying, per-edge DtN: k = 5.415002428  |lambda_min| = 1.11e-13   (matrix 5x5)
  ripple shifts the mode by 1.790e-01
```

| sub-edges/edge | matrix | k | \|k − DtN\| | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 80 | 5.414378858 | 6.24e-04 | |
| 32 | 160 | 5.414841847 | 1.61e-04 | 3.88 |
| 64 | 320 | 5.414962004 | 4.04e-05 | 3.99 |

The oversampled answer converges **to** the DtN answer at a clean O(h²) — they
are solving the same problem — and the DtN operator arrives there directly, at
`|lambda_min| = 1e-13`, with a 5×5 matrix instead of 320×320. On the buffon the
equivalent subdivision is 76803 nodes against 243 edges.

Two traps this script now guards against, both of which produced convincing
nonsense first:

* **A minimum on the bracket boundary is not a mode.** An earlier version took a
  fixed window around a guess; every configuration returned its own endpoint and
  they "agreed" to 1e-13 without any of them having found a root. `find_mode`
  now requires an *interior* minimum and raises otherwise.
* **Sub-edge eps must be placed geometrically.** Accumulating sub-edge lengths in
  `work.edges` order assumes that order walks each parent edge end to end. It
  does not; doing so scrambles the profile and the reference lands on a
  different mode (6.42 instead of 5.42).

## `probe_varying_continuation.py`

Pump continuation through `netsalt.salt_varying.solve_salt_varying`, which never
oversamples. This is the direct answer to #53's complaint about the oversampled
solver: a non-monotone summed output (−40.8 % across one pump step on
`mini_buffon`) with residuals of 1e-2 deep above threshold.

Fabry-Perot fixture, 8-node matrix throughout, 15 pumps from 1.05x to 3.0x
threshold:

| D0/thr | k | a | residual | converged |
| ---: | ---: | ---: | ---: | :---: |
| 1.05 | 10.4410520 | 0.0372679 | 3.22e-07 | yes |
| 1.61 | 10.4411674 | 0.4053469 | 8.75e-07 | yes |
| 2.03 | 10.4411873 | 0.6906345 | 3.88e-07 | yes |
| 2.58 | 10.4411733 | 1.0778356 | 4.77e-07 | yes |
| 3.00 | 10.4411466 | 1.3715324 | 4.98e-07 | yes |

**15 of 15 pumps converged**, residuals 3e-7..9e-7, amplitude strictly
monotone, `k` stable to 1e-5 across the whole sweep, 11.4 s total.

The continuation is load-bearing. Starting each pump from a scan-derived guess
instead of from the previous solution makes the solver land on *different*
modes (`k` jumping 11.4 -> 10.4 -> 9.5) — each a genuine root at residual
< 1e-6, but not the same branch. Carrying the solution forward is what keeps it
on one.

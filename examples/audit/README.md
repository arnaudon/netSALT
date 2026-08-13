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

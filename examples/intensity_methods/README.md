# Modal-intensity approximations

A self-contained, runnable comparison of the four `intensity_method` solvers that
turn threshold modes into lasing L–I (intensity-vs-pump) curves. See
[`doc/source/lasing.rst`](../../doc/source/lasing.rst) and issue #42 for the
theory.

| method | what it relaxes | curve shape |
|---|---|---|
| `linear` | nothing (near-threshold competition matrix, linear solve) | piecewise-linear, kinks at activations |
| `self_consistent` | competition matrix rebuilt at the operating pump | piecewise-linear, shifted |
| `full_salt` | + per-edge spatial hole burning (surrogate) | bends over with saturation |
| `full_salt_newton` | operator-level nonlinear SALT | gain clamping → can lase **fewer** modes |

All four reduce to the same onset slope at threshold, so their curves share units
and can be overlaid directly — **except** `full_salt_newton`, which solves for an
amplitude in its own (`∫|Ê|²=1`) normalization that differs from the
competition-matrix modal-intensity unit by a graph-dependent constant. The script
rescales it onto the linear unit by matching the dominant mode's onset slope; on
the same unit it sits with the other nonlinear methods (its raw amplitude is
otherwise a few × larger and looks misleadingly different).

### What to expect

Near threshold all methods nearly coincide (the nonlinearity is small there). They
diverge only as the pump is pushed well above threshold: `self_consistent` and
`full_salt` saturate ~20–30 % below `linear`, and `full_salt_newton` adds
gain-clamping mode suppression (it lases fewer modes, so its surviving mode can
carry more). If the curves look very different, it is because the sweep reaches
~2–3× the lasing threshold — reduce `D0_MAX` to stay in the gentle regime.

## Run

```bash
OMP_NUM_THREADS=1 python compare_intensity_methods.py
# or
bash run.sh
```

The script builds three small **open** quantum graphs in memory — a 1D
Fabry–Pérot line, a ring resonator with two leads, and a binary-tree splitter
(the degree-1 lead nodes provide the radiative loss that sets a lasing
threshold) — runs the shared passive → pump → trajectories → threshold →
competition pipeline once per graph, then computes the L–I curves with every
method.

## Output

- `intensity_methods_comparison.pdf` — one panel per graph, the four total-L–I
  curves overlaid (the **various graphs × approximations** view).
- `intensity_methods_per_mode.pdf` — per-mode L–I on the line graph, one subplot
  per method. This makes the qualitative differences explicit: `full_salt` bends
  the individual curves over, and `full_salt_newton` suppresses modes the linear
  model lases (gain clamping).
- a summary table on stdout (`n_lasing`, `n_active@max`, `total@max` per method).

To compare methods on the *full* example configs instead, see
[`benchmark/bench_salt.py`](../../benchmark/bench_salt.py).

# Modal-intensity approximations

A self-contained, runnable comparison of the four `intensity_method` solvers that
turn threshold modes into lasing L–I (intensity-vs-pump) curves, with a worked
explanation of the underlying physics and of what each algorithm approximates.
See [`doc/source/lasing.rst`](../../doc/source/lasing.rst) and issue #42 for more.

## The physics in one paragraph

A network laser turns on when the pump `D₀` makes a mode's round-trip gain equal
its loss — its **lasing threshold** `D0_thr`. Above threshold the mode's field
grows until it **saturates** the gain it feeds on: the gain medium can only supply
so much, so each lasing mode **burns a spatial hole** in the gain along the edges
where its intensity `|E(x)|²` is large (the saturation term
`1 + Σ_ν Γ_ν a_ν |E_ν(x)|²`). Two consequences drive everything below:

- **Gain clamping.** Once a mode lases it pins the *saturated* gain at its own
  threshold level. Other modes then see a reduced, clamped gain.
- **Mode competition.** Whether a second mode can still lase depends on how much
  its field **overlaps** the first mode's spatial hole. Strong overlap (same
  region of the graph) → it is starved → *winner-take-all*. Weak overlap
  (different regions / distinct standing-wave patterns) → it finds untouched gain
  → *multimode* lasing. The overlap integrals are the **competition matrix** `T`.

## The four solvers (what each approximates)

| method | gain saturation | mode profiles | how intensities are found |
|---|---|---|---|
| `linear` | linearised (fixed `T`) | frozen at each mode's own threshold | one linear solve `T·I = …`, event sweep for activation |
| `self_consistent` | linearised (fixed `T` at each pump) | **followed** to the operating pump | same event sweep, `T` rebuilt per pump |
| `full_salt` | spatial hole burning, **per-edge-mean** surrogate | followed to the operating pump | saturate `T`'s rows in a fixed point, same sweep |
| `full_salt_newton` | spatial hole burning, **operator-level** | re-solved at every pump (mode-following) | solve the real nonlinear SALT eigenproblem `(kμ, aμ)` |

- **`linear`** — the original near-threshold SALT model. The competition matrix is
  built once with each mode at its own threshold and the intensities grow
  piecewise-linearly. It has **no gain clamping**: it never re-checks whether a
  lasing mode still has net gain once others saturate it.
- **`self_consistent`** — rebuilds `T` at the operating pump with every mode
  *followed* there (refined to the actual mode of the pumped operator). This
  relaxes the frozen-profile approximation but keeps the linear gain saturation.
- **`full_salt`** — additionally folds in spatial hole burning, but as a
  **per-edge-constant** factor (the mean `|E|²` on each edge) that inflates a
  mode's row of `T`; the curves then bend over. `intensity_oversample_size`
  subdivides edges to refine this toward the true within-edge field.
- **`full_salt_newton`** — abandons the competition matrix for the intensities and
  solves the **actual nonlinear SALT eigenproblem**: find `(kμ real, aμ ≥ 0)` so
  the shared *saturated operator* is singular at each real `kμ` simultaneously.
  The `aμ ≥ 0` bound enforces a sharp on/off: a mode that cannot satisfy its
  lasing condition with positive amplitude is driven to `aμ = 0` (suppressed).

All four are constructed to reduce to the same `1/(T_μμ·D0_thr)` onset slope at
threshold, so their curves share units and overlay directly (`full_salt_newton`
rescales its amplitude onto this unit internally).

## What the linear / surrogate models *miss* (why they over-count modes)

On the **narrow-gain** line, `linear`/`self_consistent`/`full_salt` lase **2**
modes but `full_salt_newton` lases **1**. The difference is exactly gain clamping:

- **`linear`** never imposes the saturated threshold condition. It lets the second
  mode lase as soon as its *interacting* threshold (a fixed-`T` extrapolation) is
  crossed, and never asks "given mode 0 lasing, does mode 1 still have net gain?"
  Here it does not — with mode 0 lasing, mode 1 sits at `α ≈ +0.046` (just *below*
  threshold), but `linear` cannot see that.
- **`full_salt`** *does* clamp, but through a **per-edge-mean** surrogate that
  smears `|E|²` over each edge; the effective hole burning is softer than reality,
  so it under-clamps and leaves the marginal second mode weakly on.
- **`full_salt_newton`** imposes the exact per-mode condition (operator singular at
  real `k` with `a ≥ 0`) and finds the second mode is sub-threshold → suppressed.

So the missing ingredient is the **self-consistent, operator-level gain clamping**:
the linear model omits it entirely; the surrogate approximates it too softly. The
truth here is a *marginal* call (`α` only `+0.046`), which is why the methods
disagree on this particular mode.

## How many modes does `full_salt_newton` lase?

`full_salt_newton` uses a **self-consistent active set**: at each pump it freezes
the saturated background field, solves all lasing `(k_μ, a_μ)` with one
trust-region step (clean residual → no chatter), refreshes the field, and adds a
candidate only when it has net gain on the current background. So it reports the
**physically-correct mode count**.

On these small, strongly-overlapping graphs (line, ring, tree) that count is
**one** — the dominant mode clamps the gain and genuinely holds the others below
threshold. This is exactly where `linear` / `full_salt` **over-count** (2–3
modes): they don't impose the self-consistent gain-clamping condition. Genuine
multimode under full SALT needs **spatially-distinct, low-overlap** modes
(disordered / multi-cavity graphs, e.g. buffon), where each mode burns its own
spatial hole; there the solver lases as many modes as the saturated gain truly
supports. `full_salt_newton` is more expensive than the matrix methods, so those
remain a good first pass for multimode L–I.

## Run

```bash
OMP_NUM_THREADS=1 python compare_intensity_methods.py
# or
bash run.sh
```

The script builds three small **open** quantum graphs in memory — a 1D
Fabry–Pérot line, a ring resonator with two leads, and a binary-tree splitter
(the degree-1 lead nodes provide the radiative loss that sets a lasing threshold)
— runs the shared passive → pump → trajectories → threshold → competition
pipeline once per graph, then computes the L–I curves with every method.

## Output

- `intensity_methods_comparison.pdf` — one panel per graph, the four total-L–I
  curves overlaid (the **graphs × approximations** view).
- `intensity_methods_per_mode.pdf` — per-mode L–I on the line. Dashed curves are
  the `linear` reference (colour-keyed by mode); a dashed curve with **no solid
  partner** is a mode `full_salt_newton` suppressed (gain clamping).
- a summary table on stdout (`n_lasing`, `n_active@max`, `total@max` per method).

To compare methods on the *full* example configs instead, see
[`benchmark/bench_salt.py`](../../benchmark/bench_salt.py).

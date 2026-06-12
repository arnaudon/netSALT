# Modal-intensity approximations

A self-contained, runnable comparison of the two `intensity_method` solvers that
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

## The two solvers (what each approximates)

| method | gain saturation | mode profiles | how intensities are found |
|---|---|---|---|
| `linear` | linearised (fixed `T`) | frozen at each mode's own threshold | one linear solve `T·I = …`, event sweep for activation |
| `full_salt_newton` | spatial hole burning, **operator-level** | re-solved at every pump (mode-following) | solve the real nonlinear SALT eigenproblem `(kμ, aμ)` |

- **`linear`** — the near-threshold SALT model. The competition matrix is
  built once with each mode at its own threshold and the intensities grow
  piecewise-linearly. It has **no gain clamping**: it never re-checks whether a
  lasing mode still has net gain once others saturate it. It is fast (one
  matrix build + a linear solve).
- **`full_salt_newton`** — abandons the competition matrix for the intensities and
  solves the **actual nonlinear SALT eigenproblem**: find `(kμ real, aμ ≥ 0)` so
  the shared *saturated operator* is singular at each real `kμ` simultaneously.
  The `aμ ≥ 0` bound enforces a sharp on/off: a mode that cannot satisfy its
  lasing condition with positive amplitude is driven to `aμ = 0` (suppressed).

Both are constructed to reduce to the same `1/(T_μμ·D0_thr)` onset slope at
threshold, so their curves share units and overlay directly (`full_salt_newton`
rescales its amplitude onto this unit internally).

Two intermediate solvers (`self_consistent` and `full_salt`, which rebuilt /
saturated the competition matrix at the operating pump) were removed: in the
strongly-multimode regime where they would have added value over `linear` their
per-pump matrix rebuild is ill-conditioned (non-deterministic run-to-run, modes
locking to equal intensities or collapsing to zero), and on weakly-competing
graphs they just track `linear`.

## How `full_salt_newton` relates to `linear` (and a validation against Ge)

`full_salt_newton` uses a **self-consistent active set**: at each pump it freezes
the saturated background field, solves all lasing `(k_μ, a_μ)` with one
trust-region step (clean residual → no chatter), refreshes the field, and adds a
candidate when it has net gain on the current background.

Done correctly it **reduces to `linear` near threshold** and agrees with it on the
lasing count; *above* threshold it gives the genuine full-SALT correction — the
L–I curves **bend over** and the secondary modes' intensities/onsets **shift** as
the spatial holes deepen (the linear model freezes the profiles and can't see
this). On the simple cavities (line/ring/tree) both lase the same count
(`simple_graphs_compare.py`).

**One subtlety that matters (and a validation).** The operator-level hole burning
samples `|E_ν(x)|²` per edge; with one sample per edge the per-edge *mean*
over-estimates the mode overlap (it washes out the standing-wave nodes) and
**over-clamps**, spuriously suppressing co-lasing modes. On the 1D `line_PRA`
cavity this made newton lase **one** mode where the competition matrix — and
**Ge–Chong–Stone, Phys. Rev. A 82, 063824 (2010), Eq. 28** — lase **two**.
Resolving the standing wave fixes it: `full_salt_newton` now auto-picks a
wavelength-resolving `oversample_size`, recovering the two-mode result (intensities
within a few % of Ge near threshold). The matrix methods remain a cheaper first
pass for the lasing count; full SALT adds the above-threshold saturation.

### Multimode demo: `two_ring_multimode.py`

Two **detuned** rings (radii 0.9 / 1.25) joined by a bridge, with a lead on each.
The size difference breaks the left/right symmetry and **localizes** each mode
onto one ring (identical rings would give symmetric/antisymmetric modes spread
over *both*, with high overlap). With a narrow gain on a cross-ring pair,
`full_salt_newton` lases **several modes** spread across the two rings (genuine
multimode). Run it with `OMP_NUM_THREADS=1 python two_ring_multimode.py`.

### Multimode demo: `chaotic_ring_multimode.py`

You don't need a multi-component graph for multimode lasing — a **single** 14-node
ring with **6 random chords** (extra edges across it) already does it, on a much
smaller graph. This is the buffon-network mechanism shrunk down: each chord closes
a new loop, so the cavity supports many interfering path lengths → a **dense,
irregular spectrum** of **spatially-distinct** modes (each concentrated on
different loops — see the per-mode *participation ratio* the script prints). They
burn their holes in different places and co-lase. With a narrow gain on a
four-mode cluster, `full_salt_newton` lases **4 modes**. The figure has three
panels: the **graph geometry** (ring edges, chords, leads), the **full-range L–I**,
and a **zoom on the onset** (`linear` dashed, `newton` solid, thresholds dotted).
The zoom shows the modes switching on in turn — and that newton turns the fourth
mode on *well above* its bare threshold: gain clamping delays it until enough pump
is present, exactly what the clamping-free `linear` model misses.

This deep-multimode regime (4–5 strongly-clustered thresholds) is exactly where
the solvers part ways — and where the cheap ones stop being reliable:

- `linear` is exact between events (clean straight lines) but has no gain clamping,
  so its count can be off either way (here it lases **3**, one fewer than newton,
  because its frozen-profile competition matrix over-estimates suppression of the
  fourth mode).
- `full_salt_newton` stays smooth and physical and imposes the exact
  self-consistent gain clamping → **4 modes**.

The chord layout is hard-coded (from a small seed scan), so the result is
reproducible regardless of the NumPy RNG. Run it with
`OMP_NUM_THREADS=1 python chaotic_ring_multimode.py`.

### Newton-vs-linear consistency: `dense_ring_compare.py`

A bigger, more strongly-competing graph — a **16-node ring with 10 chords** —
used to check *what should differ* between `full_salt_newton` and the
near-threshold `linear` model, and why. Both share the onset slope at threshold,
so they **coincide just above it**; above threshold they diverge because newton
re-solves each lasing mode's **profile** at the operating pump (so the spatial
holes, and hence the competition, shift), while `linear` freezes the profiles and
fixes `T`. The script prints a per-mode table (onset, near-threshold slope,
intensity at max) and overlays the curves (dashed linear / solid newton). The
signature: the **dominant** mode tracks linear closely (same onset, near-equal
initial slope, slope drifting as it saturates), while the **secondary** modes are
*reshuffled* — they switch on at different pumps and reach different intensities
than the frozen-profile model predicts (here one secondary lights up earlier and
stronger, another later and largely suppressed). Run it with
`OMP_NUM_THREADS=1 python dense_ring_compare.py`.

### The same picture on the simple cavities: `simple_graphs_compare.py`

The geometry + `linear` (dashed) vs `full_salt_newton` (solid) view applied to the
three textbook graphs from `compare_intensity_methods.py` — the **line**
(Fabry–Pérot), the **ring + leads**, and the **binary tree** splitter. With the
hole burning resolved, `full_salt_newton` **agrees with `linear` on the count**:
the line and ring lase **2** modes under both, the tree **1**. The solid (newton)
curves track the dashed (linear) ones near threshold and then bend below them above
threshold — the full-SALT gain saturation. Run it with
`OMP_NUM_THREADS=1 python simple_graphs_compare.py`.

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

- `intensity_methods_comparison.pdf` — one panel per graph, the two total-L–I
  curves overlaid (the **graphs × approximations** view).
- `intensity_methods_per_mode.pdf` — per-mode L–I on the line. Dashed curves are
  the `linear` reference (colour-keyed by mode); the solid `full_salt_newton`
  curves track them near threshold and bend below above it (full-SALT saturation).
- a summary table on stdout (`n_lasing`, `n_active@max`, `total@max` per method).

To compare methods on the *full* example configs instead, see
[`benchmark/bench_salt.py`](../../benchmark/bench_salt.py).

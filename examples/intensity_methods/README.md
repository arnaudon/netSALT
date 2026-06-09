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
multimode under full SALT needs **spatially-distinct, low-overlap** modes, where
each mode burns its own spatial hole — see **`two_ring_multimode.py`** and
**`chaotic_ring_multimode.py`** below.
`full_salt_newton` is more expensive than the matrix methods, so those remain a
good first pass for multimode L–I.

### Multimode demo: `two_ring_multimode.py`

Two **detuned** rings (radii 0.9 / 1.25) joined by a bridge, with a lead on each.
The size difference breaks the left/right symmetry and **localizes** each mode
onto one ring (identical rings would give symmetric/antisymmetric modes spread
over *both*, with high overlap). With a narrow gain on a cross-ring pair, all four
solvers — including `full_salt_newton` — lase **3 modes** (one in one ring, two in
the other). Run it with `OMP_NUM_THREADS=1 python two_ring_multimode.py`.

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
- `self_consistent` / `full_salt` — the event-driven sweep with a per-pump-rebuilt
  competition matrix becomes **numerically erratic** with this many competing
  modes (intensities go non-monotone; modes flick on/off). They are reliable near
  threshold and on weakly-multimode graphs, not here, so the script runs them
  (printing their unreliable endpoint counts) but does **not** plot them.
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
(Fabry–Pérot), the **ring + leads**, and the **binary tree** splitter. These are
the *opposite* regime to the chord rings: their modes overlap strongly, so they are
**single-mode under faithful SALT**. The plots show it directly — on the line and
the ring `linear` lases **2** modes but `full_salt_newton` lases **1**, the second
mode appearing as a dashed curve with no solid partner (gain clamping holds it
below threshold); the tree is single-mode for both. Run it with
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

- `intensity_methods_comparison.pdf` — one panel per graph, the four total-L–I
  curves overlaid (the **graphs × approximations** view).
- `intensity_methods_per_mode.pdf` — per-mode L–I on the line. Dashed curves are
  the `linear` reference (colour-keyed by mode); a dashed curve with **no solid
  partner** is a mode `full_salt_newton` suppressed (gain clamping).
- a summary table on stdout (`n_lasing`, `n_active@max`, `total@max` per method).

To compare methods on the *full* example configs instead, see
[`benchmark/bench_salt.py`](../../benchmark/bench_salt.py).

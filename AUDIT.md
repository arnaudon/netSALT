# netSALT audit — restarting research on network lasers

Scope: `master` at 6a5ba7e, plus PR #43 (`claude/full-salt-solvers`, b1f8fd8).
Goal: judge whether the code is *fast, flexible and accurate* enough to restart
research on how graph structure controls lasing, and lay out the intermediate
steps toward a trustworthy full-SALT solver.

Everything below is backed by a measurement. The reproducers live in
[`examples/audit/`](examples/audit/) and `benchmark/`.

Items marked **[fixed]** were repaired on this branch; the rest are the
roadmap in §6.

---

## 1. Verdict

**The passive layer is excellent. The near-threshold lasing layer is sound but
slow. The above-threshold (full-SALT) layer is not yet research-grade.**

| layer | what it computes | verdict |
| --- | --- | --- |
| secular matrix + contour mode search | passive modes `k` | **accurate to ~1e-12**; the *subdivision default* found zero modes on the flagship example **[fixed]**; one structural blind spot remains |
| pump trajectories + thresholds | `D0_thr`, threshold modes | sound physics; crashed on a failed refinement **[fixed]**; still slow |
| competition matrix `T` + linear L–I | near-threshold modal intensities | **physics is right**, validated against a published reference; kernel was ~100x slower than needed **[fixed]**; near-degenerate results now carry a conditioning flag **[fixed]** |
| `full_salt_newton` (PR #43) | above-threshold L–I | promising core, but wrapped in heuristics that make results unfalsifiable — the heuristics are now gone and the picture is sharper, see §8 |

Two headline points:

1. **The default pipeline on `examples/buffon` returned zero passive modes.**
   Not a slowdown — an empty spectrum, silently. §3.1.
2. **PR #43 fuses three separable things into one 430-line function, and
   patches the failures of the outermost layer inside the innermost one.**
   Splitting them is the intermediate step that makes full SALT usable — §5.

---

## 2. What is solid (with numbers)

### 2.1 The passive mode search is essentially exact

`examples/audit/validate_solver_accuracy.py` checks Beyn contour integration
against closed-form spectra:

```
Fabry-Perot (open line, n=1.5, L=1, 19 modes in k < 40)
    max |dk|        1.2e-12
    max |d alpha|   1.7e-11
closed ring, exactly degenerate pairs
    resolved cleanly, max |alpha| = 8.9e-15
```

Beyn's rank-revealing SVD handles exact degeneracies, which matters because
symmetric designed graphs produce them. This part of the code needs no work.

### 2.2 The near-threshold model is the right physics

`compute_mode_competition_matrix` + `compute_modal_intensities` implement the
single-pole SALT of Ge–Chong–Stone (PRA 82, 063824), i.e. the matrix equation
`D0/D0_thr - 1 = sum_nu Gamma_nu chi_munu I_nu`. The hole-burning factor used
throughout, `Gamma = -Im gamma(k)`, is correct: with
`gamma(k) = gamma_perp/(Re k - k_a + i gamma_perp)`,

```
-Im gamma(k) = gamma_perp^2 / ((Re k - k_a)^2 + gamma_perp^2) = |gamma(k)|^2
```

which is exactly the SALT Lorentzian saturation factor. This is the layer to
build the "graph structure vs mode competition" research on — it is trustworthy
at and near threshold.

### 2.3 Pump-continuation in PR #43 is grid-independent

The steady state must not depend on how many pumps you visited on the way.
Measured with `examples/audit/probe_full_salt_convergence.py` against PR #43
(14-node ring with six chords, total lasing intensity at max pump):

```
D0_steps    5      10      20      40
~3x  thr    17.4768 17.4771 17.4768 17.4770
~30x thr    325.246 325.246 325.246 325.246
```

Grid-independent to 5–6 digits. The continuation machinery itself is fine.

---

## 3. Accuracy and robustness problems

### 3.1 The contour subdivision default returned an empty spectrum  **[fixed]**

A single Beyn contour resolves at most `probe_dim` modes — the SVD of `A_0` has
that many non-zero singular values — and past capacity the extraction collapses,
usually to *nothing*. `find_passive_modes` sized the subdivision as
`n_k = max(round(k_max - k_min), 1)`, which has nothing to do with how many
modes the window holds. On `examples/buffon` (`k` in `[10.35, 11.0]`) that is
`n_k = 1`, and the default pipeline found **zero** modes. There was also no
config key to override it.

`n_k` is now sized from the Weyl law, `N ~ L_opt (k_max - k_min) / pi` with
`L_opt = sum_e sqrt(eps_e) l_e` the optical length:

```
buffon: 208 nodes, 243 edges, optical length 3758
  Weyl estimate  778 modes
  old default    n_k = 1   ->    0 modes
  new default    n_k = 78  ->  454 modes in 27 s
```

454 is exactly the count `netsalt/contour.py`'s own docstring records as ground
truth for that workload. `contour_n_k` / `contour_n_alpha` / `contour_n_quad` /
`contour_probe_dim` are now config keys, and an empty result raises.

### 3.2 A failed refinement crashed the pump sweep  **[fixed]**

`refine_mode` returns `None` when it cannot converge. Both tracking loops
mishandled it:

* `pump_trajectories` substituted the stale previous position and kept feeding
  it to `pump_linear` at the *new* pump, where `mode_on_nodes` correctly
  rejected it. The run then died one iteration later with `"Not a mode, as
  quality is too high"` naming a mode that was never the problem.
* `find_threshold_lasing_modes` assigned the `None` into a float array (opaque
  numpy `ValueError`); its `is None` recovery check was unreachable dead code,
  since a row of a float array is never `None`.

Reproducer: on the shipped `examples/line_PRA` config, changing only
`gamma_perp` from 3.0 to 1.5 crashes the run at `D0 = 1.655`. Unrefinable modes
are now frozen and reported — a `tracking_lost_at_D0` column plus a warning —
so the failure is visible in the data instead of killing the run.

### 3.3 Editing a config silently reused stale results  **[fixed]**

Every `step_*` short-circuits on `out.exists() and not force`, keyed on the
*filename*. Measured: on `examples/line_PRA`, changing `k_a` 15.0 -> 16.5 and
`gamma_perp` 3.0 -> 1.5 and re-running produced a **byte-identical**
`modal_intensities.h5`. For a parameter sweep run in place this is the worst
possible failure mode. A per-key config fingerprint is now written next to the
outputs, and a changed physics key raises and names itself.

### 3.4 Modes are silently lost on equilateral graphs  *(tracked as issue #45)*

The secular matrix `L(k) = B^T W^{-1} B` uses per-edge weights
`k_e / (exp(2 i k_e l_e) - 1)`, singular when `k_e l_e` is a multiple of `pi`.
On an equilateral graph *every* edge hits that point at the same `k`, and the
corresponding eigenfunctions vanish at every vertex — invisible to any
vertex-based secular equation.

Measured on a closed 8-node equilateral ring (`L = 1`, `n = 1`, `l = 0.125`):

```
k          k*l/pi   |lambda_1(L(k))|   found?
25.1327    1.000    1.250e+00          NO
50.2655    2.000    1.250e+00          NO
(all other exact modes: |lambda_1| < 3e-14, found)
```

Two of nine exact modes are missing. `create_quantum_graph(noise_level=0.001)`
dodges this by jittering node positions when >20% of edges share a length — but
`examples/buffon/_base.yaml` sets `noise_level: 0.0`, disabling it, and random
graphs never trigger it anyway.

Scope: this is demonstrated for *simultaneous* commensurability (all edges at
once). A single commensurate edge among many appears to be harmless; the
partial case was not measured and should not be assumed either way.

**Why this matters for the research plan.** "Design graphs to achieve some
property" means building *regular* graphs — equal edges, symmetric layouts —
which is precisely the case that breaks. Any structure/spectrum study on
designed graphs must either use the noise dodge (and pay a geometry error) or
switch to a pole-free secular equation (§6, item 8). Tracked as issue #45,
deliberately deferred: the fix is a real project and the practically important
question — how wide the danger zone is around the exact blind spot — is still
being measured.

### 3.5 Near-degenerate modes had no error bar  **[fixed]**

`compute_modal_intensities` inverts the competition submatrix with
`np.linalg.pinv` and reports intensities with no conditioning diagnostic. When
two modes are closer than the gain linewidth, the competition rows become
nearly parallel and the *split* of intensity between them is not resolvable —
only their sum is. PR #43 independently rediscovers this (its "twin takeover"
guard exists because a near-degenerate pair's amplitude split is
ill-conditioned).

The conditioning is now measured over the *candidate* (finite-threshold) set as
well as the active sets actually solved, and reported in `modes_df.attrs` with a
warning past `1e8`. The distinction matters: a near-degenerate pair is typically
never *co*-active — the sweep picks one and suppresses the other — so watching
only the active set misses the pathology entirely. What is unresolved there is
*which* of them won, and `_find_next_lasing_mode` picks the winner from the same
inverse.

### 3.6 The above-threshold within-edge resolution is not converged

The spatial hole burning samples `|E(x)|^2` per edge, so the graph is
oversampled to resolve the standing wave. Measured spread of the total lasing
intensity as that resolution is refined:

```
                  lambda/6   lambda/12   lambda/20   lambda/32
~3x  threshold      17.08      17.48       17.54       17.56    (converging)
~30x threshold     308.19     325.25      328.78         --    (weakest mode moves 2x)
```

Near threshold `lambda/12` (the default) is ~0.5% off the refined value — fine.
Deep above threshold the weakest mode's intensity changes by a factor of two
between `lambda/6` and `lambda/20`. The docstring's claim of ~1% convergence at
`lambda/12` holds on the 1D `line_PRA` cavity but **not** on a graph with a
dense spectrum.

---

## 4. Speed

Per-step wall time, measured end to end (4 cores, `OMP_NUM_THREADS=1`) on
`line_PRA` (11 nodes, 6 modes, 100x30 scan grid) and a mid-size buffon analogue
(61 nodes / 73 edges, 33 modes, 400x50 grid, `n_workers=4`):

| step | line_PRA | mid |
| --- | ---: | ---: |
| `step_scan_frequencies` | 2.87 s (25.8%) | 21.99 s (47.8%) |
| `step_compute_mode_trajectories` | 3.35 s (30.1%) | 12.77 s (27.8%) |
| `step_find_threshold_modes` | 2.44 s (21.9%) | 7.27 s (15.8%) |
| `step_find_passive_modes` (contour) | 0.63 s (5.6%) | 0.63 s (1.4%) |
| `step_compute_mode_competition_matrix` | 0.08 s | 0.49 s |
| all plots | 1.74 s | 2.80 s |
| **total** | **11.13 s** | **46.01 s** |

Three steps are 96% of every run.

### 4.1 The grid scan was pure waste on the default path  **[fixed]**

`compute_lasing_modes` always ran the dense `k_n x alpha_n` quality grid, but
the default contour search never reads it — verified by feeding
`find_passive_modes` the real grid, `None`, and an all-zeros grid: identical
modes to 5e-12 in all three cases. It only fed two figures.

```
                 before   after   speedup
line_PRA         11.13 s  8.12 s   1.37x
mid              46.01 s 26.61 s   1.73x
```

At production buffon scale (8000 x 500 = 4M eigensolves at 3.5 ms) that is
**3.9 CPU-hours discarded per run**. The scan is now gated on something needing
it, with a `with_scan` key to force it back on.

### 4.2 The competition-matrix kernel was the scaling wall  **[fixed]**

`_compute_mode_competition_element` was a pure-Python loop over edges, called
`M^2` times through a pool — ~14.5 microseconds per edge, and `M^2 E` of them.
The whole contraction factorises: every transcendental in the `(mu, nu, edge)`
tensor splits into a mu-only times a nu-only exponential, so the batched form
has no transcendentals left in the hot tensor at all, and the E/F terms collapse
to a single matrix product. Measured (`benchmark/bench_competition.py`):

| `E` | `M` | loop (serial) | batched | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 50 | 6.70 s | 0.064 s | 104x |
| 500 | 100 | 52.8 s | 0.765 s | 69x |
| 2500 | 100 | 272.9 s | 2.89 s | 94x |

Agreement with the scalar loop on real pipeline data: **5.8e-16** relative. The
original loop is kept as `_compute_mode_competition_element_reference`, used as
the permanent test oracle. Extrapolating, the research-scale case (`M=400`,
`E=2500`) drops from ~73 min of serial CPU to ~46 s.

This matters because the matrix is the central object for the mode-competition
question: an ensemble sweep parallelises over *graphs*, so each graph gets one
core and pays the serial number.

### 4.3 Pool churn, a duplicated solve, and idle workers  **[fixed]**

Three separate wastes, all behaviour-preserving to fix:

* **A fresh `multiprocessing.Pool` per D0 iteration** — and `find_threshold_lasing_modes`
  created *two*. Measured fork + teardown is ~15 ms + ~8 ms at 4 workers and
  ~690 ms at 80, and the tail iterations carry one or two modes each while
  paying it in full. Both loops now hold one pool for the whole sweep. Graph
  pickling to the workers, by contrast, is *not* a problem (<1 ms per dump even
  at buffon size) and `chunksize` makes no measurable difference — worth
  recording, because both were the obvious suspects and both are innocent.
* **`pump_trajectories` ran `pump_linear` serially in the parent** while the
  pool sat idle: 30% of the step. It now goes through the pool.
* **`_get_new_D0` computed the same overlap factor twice** with byte-identical
  arguments — 52% of the function. It is now computed once and passed to both
  consumers.

Measured on `line_PRA` (`n_workers=1`, an 11-node graph — the case where pool
overhead is *smallest*): trajectories + thresholds 8.49 s → 6.27 s.

### 4.4 The quantum matrices rebuilt their sparsity pattern every call  **[fixed]**

`construct_laplacian` was 38% of profiled compute, and most of that was scipy
bookkeeping rather than arithmetic — building a `csr_matrix` from COO triplets
re-sorts the indices and re-runs the index-dtype and format checks each time.
The tell is that the cost is nearly flat in graph size:

```
                       total    incidence   weight   BT*W*B
n_edges=30            0.778 ms    0.268      0.183    0.145
n_edges=400           0.959 ms    0.347      0.245    0.200
```

The quantum incidence matrices have exactly one entry per (bond, node) pair, so
the COO path is a pure reordering with nothing to sum. Caching the permutation
next to the existing `_incidence_topology` makes the result **bit-identical**
(`max|diff| = 0`, not merely close) at 1.4–1.7x the speed. `construct_weight_matrix`
likewise built a DIA matrix and converted it on every call, where a diagonal is
trivially its own CSC pattern.

### 4.5 Cumulative

`examples/line_PRA` end to end, `master` (6a5ba7e) versus this branch:

```
18.40 s  ->  7.89 s     2.33x
```

on `n_workers=1` and the smallest shipped example — the configuration where
every fix above is at its *least* effective. The test suite itself drops
192 s → 141 s. At production buffon scale the scan gating alone is 3.9 CPU-hours
and the pool churn 105–207 s per run.

### 4.6 Still on the table

* **`mode_quality` is 83% of what remains**, and `eigs(sigma=0)` is a 31x more
  expensive way to test singularity than the LU it already computes (2.395 ms
  vs 0.077 ms for `splu` + `logdet(U)` on a 61-node graph). Swapping the scan to
  `quality_method="determinant"` cut it 21.99 s → 8.39 s. **Not a drop-in**: the
  determinant field has a different scale, and
  `find_rough_modes_from_scan(threshold_abs=0.1)` returned 0 candidates on it
  versus 31 on the eigenvalue field, so the peak detection has to be
  recalibrated alongside. This is the one remaining big win and the one that
  needs care.

---

## 5. PR #43 (full SALT): assessment

The physics kernel is right and the pump continuation is grid-independent
(§2.3). The problem is architectural.

`_full_salt_newton_impl` fuses three separable concerns:

1. **the saturated operator** — `dispersion_relation_pump_saturated` with a
   per-edge `D0_eff = D0*pump / (1 + sum_nu Gamma_nu a_nu |E_nu|^2)`. Small,
   correct, independently testable.
2. **the spatial resolution** of `|E(x)|^2` (oversampling). A discretisation
   with a measurable convergence rate.
3. **the nonlinear solve**, itself two things: (3a) for a *given* active set,
   root-find `(k_mu, a_mu)` so `L_sat` is singular at every real `k_mu`; and
   (3b) *discover* the active set by pump continuation.

3a is a clean, well-posed `2N`-equation root-find that needs no heuristics.
3b is genuinely hard. PR #43 solves 3b with a stack of hand-tuned guards —
`veteran_snapshot`, `killed_lower`, `wrong_basin`, `rejected_adds`,
`bootstrapped_off`, `pending_add`, an amplitude cap of `1e-2 * max(a)`, a
`0.2 * min_gap` k-window — and then patches 3b's failures *inside* 3a's loop.

Two of those guards are disqualifying for research use as written:

* **The total-output ratchet.** When the summed intensity drops with pump, the
  solver *holds collapsing modes at their previous value* to keep the L–I curve
  monotone. This imposes the expected physics on the numerics. A non-monotone
  L–I is exactly the kind of thing one would want to investigate; here it can
  no longer be observed, only inferred from a warning.
* **`_newton_onset_unit_scale`.** The reported amplitude is rescaled by an
  analytic factor derived to match the linear solver's onset slope. The
  derivation is sound and the factor comes out near 1, but it means the
  *absolute* intensity scale is anchored to the near-threshold model rather
  than predicted independently. Agreement with `linear` near threshold is then
  a weaker piece of evidence than the PR presents it as.

### 5.1 It gets roughly the right answer while reporting non-convergence

The PR ships a real validation asset: `examples/line_PRA` is the
Ge–Chong–Stone 1D slab (PRA 82, 063824, Figs. 3/5/6) with the paper's Fig. 6
digitized, and `compare_to_pra_fig6.py` overlays both solvers on it. Running
it (D0 = 1.258, the figure edge):

```
             paper exact   newton   |   paper SPA   linear
dominant        0.210      0.208    |     0.223      0.224
second          0.110      0.096    |     0.087      0.080
first threshold: netsalt 0.6107 (paper ~0.61)
```

This is genuinely good, and it is the strongest evidence in the PR:

* `linear` reproduces the paper's **single-pole approximation** to 0.4% on the
  dominant mode — confirming §2.2 independently.
* `full_salt_newton` reproduces the paper's **exact SALT** to ~1% on the
  dominant mode, and reproduces the qualitative full-SALT correction (dominant
  *below* the SPA, second mode *above* it) with the right sign.
* The second mode is ~13% off, traceable in the PR's own README to a +0.3%
  offset in its noninteracting threshold, amplified by gain-clamping proximity.

But `full_salt_newton field loop did not fully converge` fires at **every**
pump step from D0 ≈ 0.83 to 1.27 — i.e. across the entire two-mode regime that
the validation is about. The headline result is obtained while the solver is
telling you it did not converge. That is the concrete sense in which the PR
"does not completely work": not that the physics is wrong, but that there is
no working convergence criterion, so a correct-looking answer and a wrong one
are indistinguishable from the output.

On the chaotic-ring probe at ~3x threshold the total intensity is 17.48 versus
`linear`'s 13.77 — about 23% *above* the linear model, while the PR documents
the newton curves as bending *below* linear. That may be genuine (3x threshold
is a real nonlinear regime) but it is not what the docs say, and the same
non-convergence warnings fire.

**What is worth keeping and merging now, independently of the solver:**

* `dispersion_relation_pump_saturated` (physics.py, +33 lines) — small,
  correct, no behaviour change at `D0_eff = D0*pump`.
* the `DENSE_EIG_MAX` dense eigensolve fast path in `laplacian_quality`
  (quantum_graph.py, +30 lines) — faster *and* deterministic for small graphs.
* `check_quality` flags on `mode_on_nodes` / `flux_on_edges` /
  `mean_mode_on_edges` — needed to evaluate a profile above its threshold.
* `_mode_competition_matrix_block` / `_scatter_competition_block` — a clean
  refactor of the competition matrix that makes it reusable at an arbitrary
  operating pump.
* **`examples/line_PRA` and its digitized Fig. 6 data.** This is the most
  valuable artefact in the PR and is independent of the solver — it validates
  the *linear* path against a published reference. It should be merged on its
  own.

That is roughly 150 lines of code plus the validation example, out of 4981
changed lines. The rest is the solver and its demos.

---

## 6. Roadmap

Ordered so each rung is independently verifiable and immediately useful.
Items 1-4 landed on this branch; 5 onward is the work ahead.

### Stage 0 — foundations  **[done on this branch]**

1. **Size the contour subdivision from the expected mode count** (§3.1) — the
   flagship example went from 0 to 454 modes.
2. **Report unrefinable modes instead of crashing** (§3.2).
3. **Refuse to reuse cached results from a different config** (§3.3).
4. **Report when the competition result is numerically unresolved** (§3.5).
5. **Performance** (§4.1–4.5): skip the grid scan when nothing reads it,
   vectorise the competition-matrix kernel, reuse worker pools, parallelise
   `pump_linear`, drop the duplicated overlap solve, cache the CSR sparsity
   patterns, and stop using `svds(which="SM")`. 2.33x end to end on the
   smallest example, ~100x on the competition kernel.

### Stage 1 — the remaining performance item

6. **Replace `eigs(sigma=0)` in `laplacian_quality` with an LU-based residual**
   (§4.6). The big one — 83% of what is left — and the riskiest:
   `refine_mode_root` needs a signed complex residual and the peak-detection
   threshold must be recalibrated. Do it behind the existing `quality_method`
   switch with the current path kept as the reference, and gate it on the
   analytic validation in `examples/audit/`.

### Stage 2 — make the answers falsifiable

7. **A SALT residual checker.** Given any candidate `(k_mu, a_mu)` set and a
   pump, build `L_sat` and report `|lambda_1(k_mu)|` and `Im k_mu`. A dozen
   lines, and it turns "does the solver work?" into a number independent of how
   the solution was obtained. This should be the acceptance test for every
   solver, including `linear`. It is the single highest-value item on this list:
   §5.1 shows PR #43 producing a roughly-correct answer while reporting
   non-convergence, and there is currently no way to tell those cases apart.
8. **A pole-free secular equation** (§3.4) — the Kottos-Smilansky bond
    scattering form `det(I - S_B(k)) = 0` is entire in `k`, so it has neither
    the equilateral blind spot nor the conditioning problem near it. This is a
    real project (netsalt's open/directed models and complex per-edge dielectric
    all have to be re-expressed), so it is worth doing only if designed
    equilateral graphs become central to the research.

### Stage 3 — split the full-SALT solver

9. **Expose the fixed-active-set solve on its own:**
    `solve_salt_fixed_set(graph, modes, D0)` — given which modes lase, solve
    `(k, a)`. No active-set discovery, no ratchet, no continuity guards; returns
    the solution *and* its residual (item 7). Directly useful on its own: a
    researcher usually knows from `linear` which modes are candidates and wants
    the above-threshold correction.
10. **Rebuild active-set discovery on top of it** as a separate, testable
    continuation layer whose failures are reported, not patched. Every heuristic
    in §5 becomes a flag on the output row.
11. **Validate at each step against `line_PRA` / Ge Fig. 6**, using the residual
    as the primary criterion rather than agreement with `linear`.

### Stage 4 — the research questions

12. **Observables module** for the properties to classify: effective number of
    lasing modes vs pump, `T` asymmetry and off-diagonal strength (the low/high
    mode-competition axis), mode localisation on the graph (`compute_IPRs`
    exists; add edge participation and a graph-distance localisation length),
    and pump-region overlap.
13. **An ensemble/sweep layer.** Given a graph family and a parameter grid,
    produce one tidy dataframe of per-mode descriptors, cached by content hash.
    This is what turns "which structures give low mode competition?" into a
    query rather than a project. Stage 0 item 5 is the prerequisite — without it
    a 500-graph sweep is weeks of CPU.
14. **Inverse design.** `pump.py` already optimises the *pump* profile; graph
    *structure* optimisation is new work and should wait until 12-13 make the
    forward model cheap and the objective well-defined.

### Deliberately not on the list

The CF-state route sketched in PR #43's `doc/cf_states_design.md` is real
research — the note honestly records that both attempted basis finders failed —
and it is not needed for feasibility, since bounded oversampling already runs on
buffon. Revisit only if Stage 3 shows the oversampled operator is the accuracy
limit.

---

## 7. Reproducing these numbers

| claim | how |
| --- | --- |
| passive-solver accuracy, equilateral blind spot | `examples/audit/validate_solver_accuracy.py` |
| full-SALT grid independence and within-edge convergence | `examples/audit/probe_full_salt_convergence.py` (run against PR #43) |
| competition-matrix speedup and agreement | `benchmark/bench_competition.py` |
| Ge-Chong-Stone Fig. 6 validation | `examples/line_PRA/compare_to_pra_fig6.py` on PR #43 |
| stale-cache hazard, gamma_perp crash | run `examples/line_PRA` twice, editing `gamma_perp` between runs |
| full-SALT reduces to the linear model near threshold (9-graph ladder) | `examples/audit/compare_linear_vs_salt.py` |
| where full SALT stops converging (deep pump, production size) | `examples/audit/README.md`, "Where the solver stops working" |
| independent full-SALT cross-check (passive, thresholds, above threshold) | `examples/audit/independent_salt/` |
| the sign of the newton-vs-linear departure, term by term | `examples/audit/decompose_salt_departure.py` |

---

## 8. Update — after the §5 split landed

§5 recommended splitting `_full_salt_newton_impl` and replacing its guards with
diagnostics. That landed on `claude/full-salt-solvers` (issue #51): the
implementation is ~180 lines instead of 432, the fixed-active-set solve is
exposed as `solve_salt_fixed_set`, the acceptance test as `salt_residuals`, and
the total-output ratchet and wrong-basin guard are **removed** rather than
retuned. Every solve now records `modes_df.attrs["salt_diagnostics"]` — per
pump: `D0`, `n_active`, `active`, `added`, `dropped`, `converged`,
`max_residual`, `iterations` — persisted alongside the HDF5 output so a cached
step keeps its report.

That makes §5's central complaint testable, and the answer is mixed.

**The solver reduces to the near-threshold model where it must.** Over a
nine-graph ladder (11 → 39 nodes; `examples/audit/compare_linear_vs_salt.py`),
the lasing mode count agrees 9 for 9, and the dominant mode's intensity agrees
to 0.2–3.4 % just above threshold, with the deviation growing monotonically
with pump on every graph. Residuals over the ladder are 1e-8 to 2e-5, so the
agreement is between two converged answers, not two failures. This is the
consistency check §6 asked for, and it passes.

**It stops working in two regimes, and now says so.**

* *Deep above threshold.* On `mini_buffon` at 25× threshold the summed L–I
  drops 40.8 % across one pump step, worst residual 9.1e-3, converged 6/25.
  This is the case the removed ratchet was written for: the guard was hiding a
  genuine non-convergence, exactly as §5 suspected. It is graph-dependent —
  `chaotic_ring` at 30× is monotone with residual 5.7e-5. Issue #53.
* *Production size.* On `examples/buffon/buffon_uniform` (208 nodes / 243
  edges), 10 modes, 8 pumps: 1101.6 s against 0.6 s for the linear model,
  converged 1/8, and a within-edge resolution error of **33 %**. PR #43's
  "~12 s on the production buffon" does not survive the contour fix and the
  restructure. Issues #53 and #52.

**The YAML/CLI path works, and the diagnostics survive caching.** Checked
end-to-end on `examples/line_PRA` with `intensity_method: full_salt_newton`:
`python -m netsalt lasing config.yaml` runs the whole flow, writes
`modal_intensities_1.h5` plus its `_attrs.json` sidecar, and a second
invocation returns the cached result byte-identically in 2.3 s with the
diagnostics intact. That graph converges at 7 of 8 pumps (worst residual
2.2e-6, at the first lasing pump) with a within-edge resolution error of
0.76 % — and its `salt_unit_scale` is 0.99–1.008 rather than the ladder's
0.96–0.98, which is the independent confirmation that the unit-scale deficit
is the discretisation error and not a convention mismatch.

**An independent solver now agrees with it.** §5 closed by noting that no
ground-truth full-SALT solver lived in-repo, so accuracy rested on indirect
evidence. `examples/audit/independent_salt/` is that solver: a transfer-matrix
engine (exact for piecewise-constant media, verified against the closed form to
4e-15) plus a finite-difference multimode SALT Newton solve (verified O(h²),
successive-ratio 4.00 from N=500 to N=16000), sharing no code with netsalt. On
an open Fabry-Perot cavity, over 122 pump points and three configurations
(uniform pump, `gamma_perp` doubled, and a partial pump):

| quantity | median rel. diff | max |
| --- | ---: | ---: |
| passive modes `k` | 6e-16 | 1e-15 |
| thresholds `D0_thr` | 7e-7 | 2e-6 |
| lasing frequency `k_mu` | 2.7e-7 | 2.1e-6 |
| modal intensity `I_mu` | 1.6e-4 | 8.1e-3 |
| inter-mode ratio `I_1/I_2` | 2.6e-4 | 8.6e-3 |

with cross-residuals passing in both directions (netsalt's solution against the
exact transfer-matrix secular function: equivalent `dk` ≤ 2e-5; an external
solution through netsalt's own `salt_residuals`: falling with netsalt's
resolution, 5e-3 at λ/12 → 1e-4 at λ/96). The absolute intensities match, not
only the ratios. The remaining differences are the size of the two
discretisations, not a bias.

Two defects fell out of that comparison and are fixed: `oversample_graph`
re-derived `inner` from node degree, relabelling most of every vacuum lead as
cavity (5–117 % over-count across the ladder); and the default λ/12
oversampling carries a **+2.8 %** intensity bias where the docstring claimed
~1 %. The ≥3-mode regime remains unvalidated — a uniformly-pumped Fabry-Perot
is nearly rank-1 in competition and gain-clamps to two modes, which *both*
solvers agree on to 0.25 % on the third mode's sub-threshold gain.

**The sign of the newton-vs-linear departure is resolved.** The ladder found
it case-dependent — mostly newton above linear, but `two_ring` (0.977) and
`line_PRA` (0.93) below — and §8 previously flagged it as unverified. It is
expected physics, decomposing into three terms with different sign rules:
self-saturation (strictly positive, measured above 1 on 8/8 graphs with the
hole-burning field frozen), field relaxation (sign-indefinite, because SALT's
operator is non-Hermitian so the relaxed profile is stationary but not
extremal), and competition (negative for the dominant mode, since `T` has
positive entries and hence `T⁻¹` negative off-diagonals). `two_ring` crosses
below on the second term, `line_PRA` on the third. Decisively, the below-1 case
is the *published* result: Ge-Chong-Stone Fig. 6 has exact/SPA = 0.94 for the
dominant mode and 1.26 for the second, and `compare_to_pra_fig6.py` reproduces
both to ~1% (newton 0.208 vs paper 0.210; linear 0.224 vs paper SPA 0.223).
Reproducer: `examples/audit/decompose_salt_departure.py`.

**Revised verdict for the above-threshold layer:** research-grade on graphs up
to ~45 edges at up to ~2× threshold, where it is validated against both the
linear model and Ge–Chong–Stone Fig. 6. Not yet usable at production size or
far above threshold. The blocking item is the within-edge resolution (#52) —
until the oversampling is set from `k_max` and edge length rather than a flat
node budget, the residual on a large graph cannot get small and the cost
figures cannot be re-measured meaningfully.


---

## 9. Update — the within-edge resolution wall (issues #52, #53)

§8 concluded that full SALT was unusable at production size, and attributed it to
a flat oversampling node budget. That diagnosis was **wrong**, and the correction
matters because it changes what the fix is.

`construct_weight_matrix` uses one `k_e` per edge, and
`1/(exp(2i k_e l_e) - 1)` is the exact edge solution *only for constant epsilon*
— it is what makes the quantum-graph secular matrix exact. Spatial hole burning
makes epsilon vary *within* an edge, breaking that assumption outright.
`oversample_graph` restores piecewise-constancy by subdividing, and pays for it
in the size of the **eigenproblem**. On the buffon that is ~76600 nodes against
243 edges, so the default cap leaves 0.47 samples per wavelength: aliased, not
coarse. No budget tuning fixes an O(h^2) scheme sampling below Nyquist.

**The fix is to stop discretising the graph.** `L` is a sum of per-edge 2x2
Dirichlet-to-Neumann blocks, and the DtN map of a varying-epsilon edge follows
from its transfer matrix, `D = (1/M12) [[M11, -1], [-1, M22]]`. The matrix then
stays one node per vertex while the resolution lives in per-edge transfer
matrices. Landed as `netsalt/edge_propagator.py`,
`netsalt/varying_laplacian.py` and `netsalt/salt_varying.py`, with:

* exact reduction to `construct_laplacian` (1e-12) on the closed and open models,
  so it can replace the closed form without touching passive behaviour;
* oversampling converging **to** the DtN answer at a clean O(h^2) (ratios 3.88,
  3.99), i.e. the two solve the same problem;
* fourth-order Magnus needing 6400 sub-intervals for 1e-8 where the
  piecewise-constant scheme — which *is* second-order Magnus — needs 819200;
* the production buffon resolved at **18.5 samples per wavelength** (against
  0.47) in a 208x208 matrix, ~2 s per operator build;
* pump continuation converging at **15 of 15** pumps to 3x threshold, residuals
  3e-7..9e-7, amplitude strictly monotone — against the oversampled solver's
  non-monotone output and 1e-2 residuals.

**Remaining:** the new path is a public API but is not yet what
`intensity_method: full_salt_newton` runs, and it has not been cross-checked
against `examples/audit/independent_salt/`. Until both land, #52 and #53 stay
open — what has changed is that the wall is understood and demonstrably passable,
not that the shipped default has moved.

## 10. Update — which solution the varying solver lands on (issues #52, #53)

§9 established that the per-edge-DtN operator removes the resolution wall. On
the **production buffon** the solver then landed on the wrong solution branch
above ~1.05x threshold. This section records the diagnosis and the fix.

Reproducer: `examples/audit/probe_varying_amplitude_branch.py`, on
`examples/buffon/buffon_narrow` (208 nodes, 243 edges, `n_steps = 512`).

| D0/D0_thr | branch | solver, before | solver, now | k − k0 now |
| --- | --- | --- | --- | --- |
| 1.02 | 1.431 | 1.419 | 1.419 | +1.59e-05 |
| 1.05 | 3.092 | 3.063 | 3.063 | +4.76e-05 |
| 1.10 | 5.623 | **22.73, converged** | 5.576 | +7.32e-05 |
| 1.26 | 15.558 | 85.67, failed | 15.388 | +2.82e-05 |
| 2.09 | 68.077 | 412.2, failed | **still fails** | +2.91e-05 |

**The equations were never the problem.** Fixing `a` and solving for `(k, D0)` —
the same two equations, the same `_lam_varying` residual, a different pair of
unknowns held — traces a `D0(a)` that is **strictly increasing** at all 56
continuation points from `a = 0.2` to `70`. Monotone means bijective: the target
pump has exactly one amplitude on the branch, and the continuation places the
old solver's `a = 22.73` at `D0 = 1.38x`, not the `1.10x` requested. It was
returning the amplitude of a *different pump*, at a residual of 6.8e-07. **A
small residual is not evidence for this class of failure.**

**What that branch column is worth.** It is not an independent solver — it runs
through netsalt's own varying operator, so it validates *branch selection*, not
the model. Its independent anchors are the linear competition matrix's onset
slope near threshold (`1/T00 = 74.01` against a measured 75.5, a different code
path) and, for the model itself, `independent_salt/` — see §11.

**The fix** (landed): continue in the *total* amplitude and solve for the pump.
Unknowns become `(k_0..k_{M-1}, u_1..u_{M-1}, D0)` with `a_mu = s·u_mu/Σu` and
`u_g ≡ 1`, so `Σa ≡ s` by construction — no penalty term to weight against the
residuals. `s` is the continuation parameter, `D0` is solved for, and a secant
on `s` walks the achieved pump onto the requested one. Only the total scale is
gauge-fixed: pinning a *single* mode's amplitude is the seemingly equivalent
thing and is not — it left `line_PRA`'s first five pumps byte-identical, then
collapsed mode 1 to zero and invented a third mode. `outer` was raised 25 → 40
because the continuation needs more field refreshes and the 1.05x pump was
consuming exactly 25.

Ruled out before that, each by measurement: the frozen field misplacing the root
(its residual minimum sits exactly on the truth), the initial guess being in the
wrong basin (seeded *at* 5.80 against a true 5.623, the old solve still left for
52.7), and amplitude runaway alone (capping it moved 1.10x from 52.7 to 22.7 and
no further — every failing solve came back with `k` pinned on its own cap).

**Still open.** The `2.09x` pump is a 4.4x amplitude step, too large for the
secant to bridge: the solve holds at the incoming value and reports residual 1.2
with `converged = False` — honest failure, not a wrong answer — after exhausting
40 outer iterations in 38 minutes. The working range is now ~1.26x threshold,
against ~1.05x before. Cost is ~2 min per pump against ~1 min.

Two traps for anyone probing this, both of which produced confident wrong
numbers here: a probe that lets `k` travel silently measures the **neighbouring
mode** (a genuine root sits at `k − k0 = +6.5e-04`, three times the cap); and
scanning `|lambda|` at real `k` cannot answer a branch question, because for an
`a` that is not a solution there is no real-k root and the number reported is
the bound `k` ran into.

`line_PRA` is unchanged to every printed digit throughout — two lasing modes,
8/8 pumps converged, worst residual 9.16e-07.

## 11. The varying operator, against an independent solver

§10 leaned on a reference curve traced through netsalt's *own* varying operator,
which checks branch selection and not the model. This closes that gap:
`examples/audit/independent_salt/` is a SALT solver written from the equations,
sharing **no code** with netsalt (numpy and scipy only), verified to 0–4e-15
against the closed-form Fabry–Pérot spectrum and O(h²) with successive-ratio
4.00. It had only ever been run against netsalt's *oversampled* path. The
varying path has now been run through the same comparison.

Reproducers: `independent_salt/step5v_netsalt_varying_run.py` (the varying sweep,
emitting step 5's schema) and `step8v_compare_varying.py` (the comparison — no
Richardson stage, since the varying path converges in `n_steps` rather than an
oversampling resolution).

Fabry–Pérot, `n_steps = 64`, D0 = 0.58 … 1.40, **two co-lasing modes**, 72
(pump, mode) comparisons of the convention-free quantities:

| quantity | median rel. diff | max rel. diff |
| --- | ---: | ---: |
| lasing frequency `k_μ` | **4.4e-06** | 5.7e-06 |
| modal intensity `I_μ` | **1.1e-04** | 7.8e-03 |

These are **absolute** intensities, not ratios: `∫|Ψ_μ|² dx` on both sides, so
SALT's denominator fixes the scale and there is no free normalisation. At
D0 = 0.70 the varying path gives 7.804615e-2 against the independent solver's
7.804230e-2.

For comparison, the oversampled path's published figures on the same case are
median 1.6e-4 / max 8.1e-3 in intensity — so the varying operator matches
independent truth at least as well, on a matrix that never grows. The worst
point (7.8e-3, D0 = 0.82, mode 2) sits in the pump steps straddling the second
mode's turn-on, where its intensity is near zero — the same place the
oversampled path's 8.1e-3 outliers sit, i.e. the discretisations rather than a
bias.

**What this does and does not license.** It validates the varying *model* —
`construct_laplacian_varying`, `saturated_eps_profiles`, the per-edge
propagators — above threshold and multimode, on a 1D Fabry–Pérot. It does not
independently validate the buffon, whose geometry is far harsher (dense
spectrum, tens of wavelengths per edge). What it does mean is that §10's
continuation reference is no longer resting only on self-consistency: the
operator underneath it now has independent backing on a case where truth exists.

Not covered: a single `n_steps` with no Richardson extrapolation, so the `k`
agreement (4.4e-06 against the oversampled path's Richardson-extrapolated
2.7e-07) is limited by discretisation on both sides rather than by either
solver.

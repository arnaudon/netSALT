# netSALT audit — restarting research on network lasers

Scope: `master` at 6a5ba7e, plus PR #43 (`claude/full-salt-solvers`, b1f8fd8).
Goal: judge whether the code is *fast, flexible and accurate* enough to restart
research on how graph structure controls lasing, and lay out the intermediate
steps toward a trustworthy full-SALT solver.

Everything below is backed by a measurement. The reproducers live in
[`examples/audit/`](examples/audit/).

---

## 1. Verdict

**The passive layer is excellent. The near-threshold lasing layer is sound but
slow. The above-threshold (full-SALT) layer is not yet research-grade.**

| layer | what it computes | verdict |
| --- | --- | --- |
| secular matrix + contour mode search | passive modes `k` | **accurate to ~1e-12**; one structural blind spot |
| pump trajectories + thresholds | `D0_thr`, threshold modes | sound; slow, no diagnostics |
| competition matrix `T` + linear L–I | near-threshold modal intensities | **physics is right**; the kernel is ~100x slower than it needs to be |
| `full_salt_newton` (PR #43) | above-threshold L–I | promising core, but wrapped in heuristics that make results unfalsifiable |

The single most important structural point: **PR #43 fuses three separable
things into one 430-line function, and patches the failures of the outermost
layer inside the innermost one.** Splitting them is the intermediate step that
makes full SALT usable — see §5.

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

## 3. Accuracy problems

### 3.1 Modes are silently lost on commensurate-length graphs

The secular matrix `L(k) = B^T W^{-1} B` uses per-edge weights
`k_e / (exp(2 i k_e l_e) - 1)`, singular when `k_e l_e` is a multiple of `pi`.
On an equilateral graph every edge hits that point at the same `k`, and the
corresponding eigenfunctions vanish at every vertex — invisible to *any*
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

**Why this matters for the research plan.** "Design graphs to achieve some
property" means building *regular* graphs — equal edges, symmetric layouts —
which is precisely the case that breaks. Any structure/spectrum study on
designed graphs must either use the noise dodge (and pay a geometry error) or
switch to a pole-free secular equation.

### 3.2 Near-degenerate modes have no error bar

`compute_modal_intensities` inverts the competition submatrix with
`np.linalg.pinv` and reports intensities with no conditioning diagnostic. When
two modes are closer than the gain linewidth, the competition rows become
nearly parallel and the *split* of intensity between them is not resolvable —
only their sum is. PR #43 independently rediscovers this (its "twin takeover"
guard exists because a near-degenerate pair's amplitude split is
ill-conditioned). Today the user gets numbers with no signal that they are
unresolved.

### 3.3 The above-threshold within-edge resolution is not converged

The spatial hole burning samples `|E(x)|^2` per edge, so the graph is
oversampled to resolve the standing wave. Measured spread of the total lasing
intensity as that resolution is refined:

```
                  lambda/6   lambda/12   lambda/20   lambda/32
~3x  threshold      17.08      17.48       17.54       17.56    (converging)
~30x threshold     308.19     325.25      328.78         --     (weakest mode moves 2x)
```

Near threshold `lambda/12` (the default) is ~0.5% off the refined value — fine.
Deep above threshold the weakest mode's intensity changes by a factor of two
between `lambda/6` and `lambda/20`. The docstring's claim of ~1% convergence at
`lambda/12` holds on the 1D `line_PRA` cavity but **not** on a graph with a
dense spectrum.

---

## 4. Speed

### 4.1 The competition-matrix kernel is the scaling wall

`_compute_mode_competition_element` is a pure-Python loop over edges, called
`M^2` times. Measured cost, ~14.5 microseconds per edge:

| edges `E` | per element | `M=100` | `M=400` |
| --- | --- | --- | --- |
| 250 | 3.5 ms | 35 s | 9.4 min |
| 500 | 7.5 ms | 74 s | 20 min |
| 2500 | 36 ms | 6.0 min | **98 min** |

(serial CPU; the pool divides it by `n_workers`). This matrix *is* the central
object for "how does graph structure influence lasing via mode competition" —
it will be recomputed for every graph in an ensemble. The whole contraction is
a batched tensor operation; the Python loop is pure overhead.

### 4.2 Other measured costs

See §7 for the profiling results.

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

### Stage 0 — foundations (small, unblocks everything)

1. **Vectorise the competition-matrix kernel.** Batched tensor contraction over
   `(mu, nu, edge)` instead of an `M^2` Python fan-out. Verified against the
   scalar loop as a permanent test oracle.
2. **Skip the grid scan when the contour solver is in use.** See §7.
3. **Warn on commensurate edge lengths.** A cheap check at
   `create_quantum_graph` time: if many edges share a length and
   `noise_level == 0`, say which modes will be invisible.
4. **Cherry-pick the four safe pieces of PR #43 listed in §5.**

### Stage 1 — make the answers falsifiable

5. **A SALT residual checker.** Given any candidate `(k_mu, a_mu)` set and a
   pump, build `L_sat` and report `|lambda_1(k_mu)|` and `Im k_mu`. This is a
   dozen lines and turns "does the solver work?" into a number, independent of
   how the solution was obtained. It should be the acceptance test for every
   solver, including `linear`.
6. **Conditioning diagnostics.** Report `cond(T_active)` alongside the modal
   intensities, and flag mode pairs whose split is unresolvable.
7. **Convergence reporting instead of silent guards.** Every heuristic in
   §5 becomes a *flag on the output row* rather than a silent correction.

### Stage 2 — split the full-SALT solver

8. **Expose 3a on its own:** `solve_salt_fixed_set(graph, modes, D0)` — given
   which modes lase, solve `(k, a)`. No active-set discovery, no ratchet, no
   continuity guards; returns the solution *and* its residual. This is
   directly useful: a researcher usually knows (from `linear`) which modes are
   candidates, and wants the above-threshold correction.
9. **Rebuild 3b on top of it** as a separate, testable continuation layer whose
   failures are reported, not patched.
10. **Validate against `line_PRA` / Ge Fig. 6** at each step, with the residual
    checker as the primary criterion rather than agreement with `linear`.

### Stage 3 — the research questions

11. **Observables module** for the properties you want to classify: effective
    number of lasing modes vs pump, `T` asymmetry and off-diagonal strength
    (low/high competition), mode localisation on the graph (`compute_IPRs`
    exists; add edge participation and a graph-distance localisation length),
    and pump-region overlap.
12. **An ensemble/sweep layer.** Given a graph family and a parameter grid,
    produce one tidy dataframe of per-mode descriptors, cached by content hash.
    This is what makes "which structures give low mode competition?" a query
    rather than a project. Stage 0.1 is the prerequisite — without it a
    500-graph sweep is weeks of CPU.
13. **Inverse design.** `pump.py` already optimises the *pump* profile; graph
    *structure* optimisation is new work and should wait until 11–12 make the
    forward model cheap and the objective well-defined.

### Deliberately not on the list yet

The CF-state route sketched in `doc/cf_states_design.md` is real research (the
note honestly records that both attempted basis finders failed) and it is not
needed for feasibility — bounded oversampling already runs on buffon. Revisit
only if Stage 2 shows the oversampled operator is the accuracy limit.

---

## 7. Profiling

_Filled in below from the profiling pass._

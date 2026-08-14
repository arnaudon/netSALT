# An independent SALT solver, and what it says about netsalt

`AUDIT.md` used to close with "no ground-truth full-SALT solver lives in-repo;
accuracy is validated by reduction-to-linear, the Ge–Chong–Stone Fig. 6 result,
physical direction, determinism, and path-independence." Those are all indirect.
This directory closes that gap: a second SALT solver, written from the equations
rather than from netsalt, sharing **no code** with it — `indep_salt.py` imports
only numpy and scipy.

## The two engines

**Transfer matrix / shooting** — exact for piecewise-constant media, so the
passive modes and the unsaturated thresholds carry no discretisation error at
all beyond the root finder. Checked against the closed form
`exp(2 i n k L) = ((n+1)/(n−1))²`: agrees to **0–4e-15**.

**Finite differences + damped Newton** — the full multimode SALT solve.
Unknowns per mode are `(Re φ, Im φ, a_μ, k_μ)` with `∫|φ|² dx = 1`, so
`Ψ = √a·φ` and `a_μ = ∫|Ψ_μ|² dx`. Making the amplitude an explicit unknown and
adding the normalisation row removes the trivial `φ = 0` root that a bare
`A(k,|Ψ|²)Ψ = 0` always has, and leaves a square system that plain Newton
solves. Second-order Robin boundary via ghost points. Verified O(h²):
successive-ratio **4.00 exactly** at every refinement from N=500 to N=16000.

## The test case

A 1D open Fabry–Pérot cavity: `0 < x < L`, `L = 0.5`, `ε = 9` (n = 3), uniform
pump, vacuum outgoing at both ends, `k_a = 15`, `γ⊥ = 3`. On the netsalt side,
a path graph of 11 equal edges — 9 inner plus 2 vacuum leads, since a lead
terminated by netsalt's "open" BC is reflectionless, making the effective
condition at the cavity end exactly `Ψ' = ∓ i k Ψ`.

`noise_level=0` (the shipped examples jitter the geometry, which would make an
exact comparison impossible), and **9** inner edges — chosen so no lasing mode
sits on the secular matrix's pole at `q_e l_e ∈ πZ`. The modes have `qL = πm`
with m = 6, 7, 8; the pole for P equal inner edges needs `P | m`, and P = 9
avoids all three. This is issue #45 in miniature, and it is why the edge count
is not arbitrary.

Two replicate configurations run through the same scripts via environment
variables: **B** = `NS_GAMMA_PERP=6`, **C** = `NS_PUMP_EDGES=0,1,2,3,4` (pump on
the left 5 of 9 inner edges, exercising the non-uniform pump-overlap path).

## Results

**Passive modes agree to ~1e-15 relative** — both `Re k` and
`α = ln(4)/(2nL) = 0.462098120373297`, at machine precision, for m = 6, 7, 8.

**Thresholds agree to 2e-7 … 2e-6 relative** across all three configurations
(and to ≤3.4e-8 in `k_thr`). The residual is netsalt's own
`find_threshold_lasing_modes` stopping tolerance, not a physics difference.

**Cross-residuals pass in both directions.** Forward: take netsalt's
`(k_μ, a_μ, f_μ)`, rebuild the saturated permittivity, and evaluate the *exact*
transfer-matrix secular function at netsalt's `k_μ` — equivalent `dk ≲ 2e-5`
over the whole sweep. Reverse: hand externally computed `|Ψ_μ(x)|²` to
netsalt's public `salt_residuals` — the residual falls with netsalt's
resolution (dk 5e-3 at res 12 → 1e-4 at res 96), as it must if the only
difference is discretisation.

**Above threshold, 1 and 2 co-lasing modes**, both sides Richardson-extrapolated
in their own parameter, 122 pump points over `D0 = 0.58 … 3.0`:

| quantity | n | median rel. diff | max rel. diff |
| --- | ---: | ---: | ---: |
| lasing frequency `k_μ` | 231 | **2.7e-7** | 2.1e-6 |
| modal intensity `I_μ` | 231 | **1.6e-4** | 8.1e-3 |
| inter-mode ratio `I_1/I_2` | 109 | **2.6e-4** | 8.6e-3 |
| pump-to-pump ratio | 121 | **1.2e-4** | 4.7e-4 |

The 8e-3 outliers occur only in the two pump steps straddling the second mode's
turn-on, where the intensity is near zero. Turn-on agrees to one 0.02 pump step
in every configuration. Both extrapolation uncertainties bracket the
differences, so this is agreement at the level of the discretisations rather
than a residual bias.

Note the **absolute** intensities match, not only ratios — SALT's denominator
fixes the scale, so netsalt's `∫ a·f dx` and this solver's `∫|Ψ|² dx` are the
same physical number: 7.8058e-2 vs 7.80462e-2 at `D0 = 0.70`, 1.6e-4.

## Two findings that came out of this

1. **`inner` was wrong on oversampled graphs** — fixed in
   `oversample_graph`; sub-edges now inherit the parent's flag instead of it
   being re-derived from node degree, which relabelled most of every vacuum
   lead as inner. See the commit for the measured over-count (5–117 % across
   the example ladder).

2. **The default `λ/12` oversampling carries a +2.8 % bias** in the single-mode
   intensity (res 12/24/48/96/192 → 8.0216e-2, 7.8594e-2, 7.8182e-2, 7.8089e-2,
   7.8066e-2, clean O(h²) toward 7.8058e-2), and ~7 % in the sub-threshold gain
   of the next mode. The `_auto_oversample_size` docstring claimed ~1 %;
   corrected. Issue #52.

## Not established

**Three co-lasing modes were never reached**, in any configuration — and both
solvers agree on why. In a uniform-index, uniformly-pumped Fabry–Pérot all
modes' `|E|²` share a large constant component, so the competition is nearly
rank-1 and gain clamping locks out everything past the second. The two solvers
agree to **0.25 %** on how far below threshold the third mode sits (α = 0.1188
vs 0.1185 at `D0 = 3.0`), on a quantity neither is solving for. netsalt's own
sweep confirms only 2 modes out to `D0 = 8.98`. Validating the ≥3-mode regime
needs a different geometry — a longer cavity with many modes under the gain
curve, or a two-piece index.

## Running it

```bash
cd examples/audit/independent_salt
python step1_passive_indep.py     # the independent solver's self-verification
python step2_netsalt_passive.py   # (a) passive modes
python step3_thresholds.py        # (b) thresholds
python step4_indep_salt_run.py    # independent above-threshold sweep
python step5_netsalt_salt_run.py  # netsalt above-threshold sweep
python step6_cross_residual.py    # netsalt solution -> exact TM secular
python step7_reverse_residual.py  # independent solution -> netsalt.salt_residuals
python step8_compare.py           # Richardson extrapolation + comparison tables
python step9_third_mode.py        # sub-threshold gain of the non-lasing third mode
```

Each writes a `results_*.json` next to itself (gitignored); steps 4–8 read the
earlier ones. Configuration is by environment variable: `NS_K_A`,
`NS_GAMMA_PERP`, `NS_PUMP_EDGES`, `NS_D0_MAX`, `NS_K_MIN`, `NS_K_MAX`.

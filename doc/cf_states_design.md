# Constant-flux (CF) state SALT solver — design and validated foundation

Status: **foundation validated, full solver is a multi-PR effort.** This note
records the formulation, the validated starting point, and the concrete plan,
so the work is grounded and resumable.

## Why CF states (what the existing solvers cannot do)

`full_salt_newton` resolves the within-edge hole burning by **oversampling** the
graph to ~λ/12 sub-edges. On a buffon network (`inner_total_length = 2500`,
k ≈ 10.7) that is a ~77 000-node operator — infeasible. CLAUDE.md therefore
scopes newton to small sparse cavities and `linear` to buffon scale.

Two cheaper routes were tried and **rejected**, each for a concrete measured
reason:

* **Analytic per-edge gain (oversample-free).** Replace oversampling with the
  closed-form within-edge integral `⟨1/(1+Γa|E(x)|²)⟩` by Gauss-Legendre
  quadrature of the exact plane-wave field. *Validated correct for a single
  mode* (matches oversample on line_PRA in the single-mode regime, 5.8× faster).
  But it **cannot discriminate co-lasing modes**: on line_PRA at D0 = 1.27 it
  dumped all intensity on the dominant mode (0.30) and suppressed the second
  (0.0) where oversample lases two (0.21 / 0.099). Folding the gain into one
  *constant per edge* — even the exact edge-average — washes out the within-edge
  standing-wave structure that lets two modes share the gain (each saturates it
  where *its* field peaks). That structure lives in the operator; a per-edge
  constant cannot carry it. *Series expansion of `1/(1+s)` also fails* — it
  diverges (s > 1 on antinode edges even at moderate amplitude).
* **Coarser oversampling.** Discriminating modes that vary on scale λ needs the
  gain resolved on scale λ *in the operator*, so the node count stays O(L/λ)
  regardless of how exactly `|E|²` is computed. No real win.

The CF basis is the principled fix: it represents the field in **exact graph
eigenfunctions** (plane waves per edge), so (i) overlap integrals are analytic
(the competition-matrix machinery), **no oversampling**, and (ii) the basis
carries the full within-edge spatial structure, so it **resolves multimode**.

## The quantum-graph CF eigenproblem

Ge-Chong-Stone (PRA 82, 063824) Eq. (17) defines the threshold CF (TCF) states
at a fixed real frequency k:

    [∇² + ε(r) k² + ηₙ(k) F(r) k²] uₙ(r, k) = 0,   outgoing BC,

where `F` is the pump profile and `ηₙ` (complex, `Im ηₙ < 0`) is the **CF
eigenvalue** — the scale of amplifying dielectric needed for that state to reach
threshold at k. On a quantum graph this means the per-edge wavenumber is

    k_edge = k · √(ε_e + ηₙ · pump_e),

i.e. the secular matrix `L(k, η)` built with the per-edge dielectric
`ε + η·pump` (plain dispersion) is **singular** for the CF eigenvalues `ηₙ(k)`.
This is a *nonlinear eigenproblem in η* at fixed real k — the same shape as the
passive mode search (nonlinear in k), so the existing Beyn machinery applies in
the η-plane.

### Validated foundation (line_PRA)

The **threshold** CF eigenvalue must equal the threshold gain `γ(k_thr)·D0_thr`,
because at threshold the amplifying dielectric *is* the actual gain and the CF
state *is* the threshold lasing mode (the paper's Fig. 3). Measured on
line_PRA's dominant mode (`/tmp` prototype, reproducible):

    k_thr = 15.44376,  D0_thr = 0.61069,  γ(k_thr) = 0.14475 − 0.97859j
    expected η = γ·D0_thr        = 0.08840 − 0.59762j
    scanned argmin |λ₁(η)| at η  = 0.08840 − 0.59762j   (|λ₁| = 6e-7)
    agreement: 0.00%

So the operator `construct_laplacian(k_thr, graph_with dielectric ε+η·pump)` is
singular *exactly* at `η = γ·D0_thr` — the CF formulation is correct and
computable on the original 11-node graph, no oversampling.

## Plan (the multi-PR work ahead)

1. **CF basis finder — the hard step.** Two approaches were tried and **both
   fail**, which is the central open problem:

   * *Beyn in the η-plane* — run the `contour.py` Cauchy-moment extraction on
     `L(η) = construct_laplacian(k, dielectric = ε + η·pump)` instead of `L(k)`.
     Tested on line_PRA: it returns a wrong root (Im η > 0, absorbing) and
     **misses the validated threshold root even with a tight contour around
     it** (found 0). Diagnosis: on a graph the secular matrix depends on η
     through `√(ε + η·pump)` in the per-edge wavenumber, so `L(η)` is **not
     analytic** in η (branch points at `η = −ε_e/pump_e`) and the
     near-null behaviour around a CF eigenvalue is *soft*, not a clean simple
     pole — Beyn's contour integral, which assumes a meromorphic `L(η)⁻¹`,
     does not extract it. (This is the key difference from the *continuous* CF
     operator of Eq. 17, which is **linear** in η.)
   * *Coarse grid scan of `|λ₁(η)|`* — the η-roots are isolated points; a 61×61
     box found only the threshold one (which was centred), none of the basis.

   So the basis finder is genuine research, not plumbing. Candidate routes:
   (a) **nonlinear root-finding in η** (Newton / secant on `λ₁(η)` or
   `det L(η)`) seeded from physical guesses (passive modes near k, UCF
   estimates `ηₙ = c(Kₙ²/k² − 1)` from Eq. 21) — handles non-analyticity since
   it never integrates around the root; (b) a **linear-in-η reformulation** —
   the edge ODE `u'' + k²(ε+η pump)u = 0` *is* linear in η, so a spatial
   discretisation gives a standard generalised eigenproblem
   `(K − k²M_ε)u = k²η M_pump u` solving all CF states at once, but that
   reintroduces a mesh (the cost we are trying to avoid); (c) a **hybrid** —
   coarse mesh only to seed guesses for the exact nonlinear η-solve.
   Route (a) is the most promising and is the next concrete experiment.
2. **Self-orthogonality + overlaps.** CF states obey the paper's self-orthogonality
   relation (Eq. 20); the hole-burning overlaps `∫ uₙ uₘ* / (1 + Σ |Ψ|²)` over
   each edge are products of plane waves → closed-form exponential integrals
   (extend `_compute_mode_competition_element`). No oversampling.
3. **SALT in the CF basis (Eq. 28).** Expand each lasing mode `Ψμ = Σₙ aₙᵘ uₙ(kμ)`;
   the SALT condition becomes the nonlinear fixed-point map `Tₙₙ'(k) a = a`
   (Eq. 28) in the CF coefficients, with the gain-clamping nonlinearity in the
   overlaps. Solve `(kμ, aₙᵘ)` self-consistently — the lasing map's fixed points,
   no spatial discretization.
4. **Validation milestones.** (a) threshold CF = threshold mode (✓ done);
   (b) CF basis reproduces a passive mode by expansion; (c) single-mode L–I
   matches oversample on line_PRA; (d) **two-mode** L–I reproduces Ge Fig. 6
   (the test the per-edge-average failed); (e) run the real buffon at its native
   node count.

The payoff: steps 1–3 keep the operator at the original node count, so the real
buffon runs at 96 nodes instead of 77 000 — and unlike the per-edge-average
shortcut, the CF basis resolves co-lasing modes.

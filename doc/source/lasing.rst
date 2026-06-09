Multimode lasing and mode competition
======================================

This page documents how netSALT turns the passive modes of a quantum graph into
**lasing amplitude curves** — the modal intensities as a function of pump
strength — within the SALT (steady-state ab-initio laser theory) approximation.
It is the physics behind :func:`~netsalt.modes.find_threshold_lasing_modes`,
:func:`~netsalt.modes.compute_mode_competition_matrix` and
:func:`~netsalt.modes.compute_modal_intensities`. See :doc:`theory` for the
underlying quantum-graph construction and the :math:`k = \mathrm{Re}\,k - i\alpha`
convention used throughout.

Overview
--------

Above threshold the laser settles into a **stationary multimode** state: each
mode :math:`\mu` lases at a real wavenumber with a modal intensity
:math:`I_\mu`, and the modes compete for a shared gain medium through spatial
hole burning. netSALT implements the *linearized, near-threshold* SALT
expansion: the field profiles are frozen at their threshold shape, and the
nonlinear gain saturation is condensed into a single **mode-competition matrix**
:math:`T` (called the ``T`` or ``M`` matrix in the code) followed by a linear
solve for the intensities. The flow is

.. math::

   \text{passive modes}
   \;\xrightarrow{\text{raise } D_0}\;
   \text{thresholds } D_0^{\mathrm{thr}}_\mu
   \;\xrightarrow{T}\;
   \text{intensities } I_\mu(D_0).

The gain curve
--------------

The two-level gain medium enters through the Lorentzian
:func:`~netsalt.physics.gamma`,

.. math::

   \gamma(k) = \frac{\gamma_\perp}{\mathrm{Re}\,k - k_a + i\gamma_\perp},

centred on the atomic transition wavenumber ``k_a`` with half-width
``gamma_perp``. The physically active quantity is the **normalised gain**

.. math::

   -\mathrm{Im}\,\gamma(k)
   = \frac{\gamma_\perp^2}{(\mathrm{Re}\,k - k_a)^2 + \gamma_\perp^2} \in (0, 1],

a Lorentzian peaking at :math:`1` at line centre. This is the gain curve that
weights every threshold and competition term below.

Single-mode threshold: gain versus loss
----------------------------------------

A mode lases, in isolation, when its gain balances its loss. The
non-interacting threshold pump is
(:func:`~netsalt.modes.lasing_threshold_linear`)

.. math::

   D_0^{\mathrm{thr}}_\mu
   = \frac{1}{Q_\mu \,\big(-\mathrm{Im}\,\gamma(k_\mu)\big)\,\mathrm{Re}\,\Gamma_\mu},

with three competing factors:

* :math:`Q_\mu = \mathrm{Re}\,k / (2\alpha)` — the passive quality factor
  (:func:`~netsalt.physics.q_value`); a low-loss mode is cheap to pump.
* :math:`-\mathrm{Im}\,\gamma(k_\mu)` — the gain at the mode frequency; modes
  near the gain centre :math:`k_a` reach threshold first.
* :math:`\Gamma_\mu` — the **pump overlap factor**
  (:func:`~netsalt.modes.compute_overlapping_factor`), the fraction of the
  mode's energy living on the pumped edges,

  .. math::

     \Gamma_\mu = \frac{\int_{\text{pump}} |E_\mu|^2}{\int_{\text{inner}} \varepsilon\,|E_\mu|^2}.

  The overlap integrals are evaluated in closed form on the graph edges via
  :func:`~netsalt.modes.compute_z_matrix` and the quadratic form
  ``_graph_norm``.

The product :math:`Q_\mu\,(-\mathrm{Im}\,\gamma)` is exposed directly as
:func:`~netsalt.modes.gamma_q_value`.

Pulling modes to threshold
--------------------------

Raising the pump makes the effective dielectric
:math:`\varepsilon + \gamma(k)\,D_0\,\delta_\mathrm{pump}` (see
:func:`~netsalt.physics.dispersion_relation_pump`), which drags each modal
wavenumber. The linear approximation is
(:func:`~netsalt.modes.pump_linear`)

.. math::

   k(D_0^{(1)}) = k(D_0^{(0)})
   \sqrt{\frac{1 + \gamma\,\Gamma\,D_0^{(0)}}{1 + \gamma\,\Gamma\,D_0^{(1)}}}.

Gain pushes the modal decay rate :math:`\alpha = -\mathrm{Im}\,k` down; the mode
reaches **threshold** when :math:`\alpha \to 0`, i.e. leakage is exactly
balanced by gain. :func:`~netsalt.modes.find_threshold_lasing_modes` tracks
every passive mode up in :math:`D_0` to that crossing, returning the
``threshold_lasing_modes`` (now at real :math:`k`) and the per-mode
``lasing_thresholds`` :math:`D_0^{\mathrm{thr}}_\mu`.

The mode-competition matrix :math:`T`
-------------------------------------

Once lasing, mode :math:`\nu` burns a spatial hole in the gain that raises the
threshold of every other mode :math:`\mu`. This coupling is the
mode-competition matrix
(:func:`~netsalt.modes.compute_mode_competition_matrix`),

.. math::

   T_{\mu\nu}
   = -\mathrm{Im}\,\gamma(k_\nu)
     \int_{\substack{\text{pumped}\\\text{inner edges}}}
     |E_\nu(x)|^2 \, E_\mu^2(x)\, dx.

Each mode is evaluated at its **own** threshold pump
(``_precomputations_mode_competition``), with the edge flux normalised by the
pump norm. The edge integral is done analytically per edge
(``_compute_mode_competition_element``): the :math:`4\times4` inner matrix
(terms A–F) collects the closed-form integrals of products of the two
counter-propagating plane waves on the edge; the left vector carries
:math:`|E_\nu|^2` (mode :math:`\nu`'s **intensity** — the hole it burns) and the
right vector :math:`E_\mu^2` (mode :math:`\mu`'s field it depletes), weighted by
mode :math:`\nu`'s gain.

Interpreting :math:`T`: the diagonal :math:`T_{\mu\mu}` is **self-saturation**
(a mode depleting its own gain), the off-diagonal :math:`T_{\mu\nu}` is
**cross-saturation** — how strongly mode :math:`\nu` suppresses mode
:math:`\mu`. Strong overlap in the pumped region means strong competition.

Lasing-amplitude (L–I) curves
-----------------------------

The steady-state modal intensities solve the linear SALT system over the
currently active set of modes:

.. math::

   \sum_\nu T_{\mu\nu}\, I_\nu = \frac{D_0}{D_0^{\mathrm{thr}}_\mu} - 1,
   \qquad
   I_\mu(D_0) = \mathrm{slope}_\mu\, D_0 - \mathrm{shift}_\mu,

with :math:`\mathrm{slope} = T^{-1}\,(1/D_0^{\mathrm{thr}})` and
:math:`\mathrm{shift} = T^{-1}\,\mathbf{1}` (the row sums of
:math:`T^{-1}`). Within a fixed active set the intensities are therefore
**linear in pump** — the laser L–I curve, whose slope is set by
:math:`T^{-1}`. :func:`~netsalt.modes.compute_modal_intensities` builds the full
curves by stepping :math:`D_0` from the first threshold up to
``max_pump_intensity``, re-solving the linear system at each event. The result
is **piecewise linear** with kinks at two kinds of event:

* **Activation** (``_find_next_lasing_mode``): a dark mode turns on when its
  *interacting* threshold — its bare :math:`D_0^{\mathrm{thr}}` raised by
  competition from the active set (a Schur-complement-style correction with
  :math:`T`) — is reached.
* **Vanishing**: an active mode whose intensity slope has gone negative is
  **switched off** when :math:`I_\mu` reaches :math:`0` (the
  ``shift/slope`` crossing). Competition can extinguish a previously-lasing
  mode.

The output ``modes_df["modal_intensities"]`` holds :math:`I_\mu` versus
:math:`D_0` for every mode; the number of modes with positive final intensity is
the number of lasing modes. Optimising *which* modes lase — by shaping the pump
profile :math:`\delta_\mathrm{pump}` — is the job of :mod:`netsalt.pump`.

Approximations and validity
---------------------------

* **Single-pole / frozen-profile SALT.** :math:`T` is assembled once from the
  threshold field profiles and the gain-saturation denominator is linearised.
  This is accurate **near threshold** and degrades far above it, where a full
  self-consistent SALT iteration of the field profiles would be required.
* **Well-separated modal frequencies.** The coherent :math:`E_\mu^2` (rather
  than :math:`|E_\mu|^2`) in the competition integral is the leading-order
  hole-burning term; it neglects four-wave / phase-locking contributions that
  matter only for near-degenerate modes.
* **Sign conventions.** Every gain term carries the
  :math:`-\mathrm{Im}\,\gamma` / :math:`\alpha = -\mathrm{Im}\,k` convention used
  across the package (loss = positive imaginary part); the threshold and
  competition expressions are consistent with it.

Relation to full SALT
---------------------

The model above is the *linearized* (near-threshold) limit of SALT. The full
SALT equation for each lasing mode :math:`\Psi_\mu` at real frequency
:math:`k_\mu` is

.. math::

   \left[\nabla^2 + \left(\varepsilon(x)
   + \frac{\gamma(k_\mu)\,D_0\,\delta_\mathrm{pump}(x)}
          {1 + \sum_\nu \Gamma_\nu\,|\Psi_\nu(x)|^2}\right) k_\mu^2\right]
   \Psi_\mu(x) = 0,
   \qquad
   \Gamma_\nu = \frac{\gamma_\perp^2}{(k_\nu - k_a)^2 + \gamma_\perp^2},

a set of coupled nonlinear equations whose unknowns — the number of lasing
modes, their frequencies :math:`k_\mu`, and their profiles/intensities
:math:`\Psi_\mu` — are all linked through the **spatial-hole-burning
denominator** :math:`1 + \sum_\nu \Gamma_\nu |\Psi_\nu|^2`.

netSALT already builds almost this operator:
:func:`~netsalt.physics.dispersion_relation_pump` uses
:math:`\varepsilon_\mathrm{eff} = \varepsilon + \gamma(k)\,D_0\,\delta_\mathrm{pump}`,
i.e. the same active gain **with the denominator set to 1** (unsaturated gain).
Everything specific to multimode SALT lives in that denominator, and the two
approximations on this page are exactly its two halves:

.. math::

   \frac{1}{1 + \sum_\nu \Gamma_\nu |\Psi_\nu|^2}
   \;\approx\;
   1 - \sum_\nu \Gamma_\nu\,\big|\Psi_\nu^{\mathrm{(thr)}}(x)\big|^2\, I_\nu .

* dropping all but the **first-order** term is the pump-independent / linear
  :math:`T` (the L–I curves are straight segments);
* freezing :math:`\Psi_\nu` at its **threshold** profile
  :math:`\Psi_\nu^{\mathrm{(thr)}}` rather than the profile at the operating
  pump.

Both come from not solving the denominator self-consistently, so they cannot be
improved independently. Relaxing them defines a hierarchy:

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - Level
     - What is solved
     - Captures
   * - **Linearized SALT** (current)
     - one :math:`T`, linear solve
       :math:`\sum_\nu T_{\mu\nu} I_\nu = D_0/D_0^{\mathrm{thr}}_\mu - 1`
     - competition to first order; exact at threshold
   * - **Self-consistent linearized** (relax the frozen profile only)
     - same event-driven sweep, but :math:`T` is rebuilt from the profiles at
       each operating :math:`D_0` instead of held fixed at threshold. With linear
       saturation :math:`T` depends only on the pump, so no inner fixed point is
       needed.
     - profile deformation, gain guiding, frequency pulling; saturation still
       linear
   * - **Full SALT** (relax both)
     - the nonlinear eigenproblem above, Newton on
       :math:`(I_\mu, k_\mu, \Psi_\mu)` at each :math:`D_0` with the real
       denominator
     - sub-linear (clamping) L–I slopes, frequency shifts, quantitatively
       correct competition far above threshold

Quantum graphs are a favourable setting for the full solve: the per-edge field
is two analytic plane waves, the secular matrix
:math:`L(k) = B^{\mathsf T} W^{-1} B` is already assembled, and the saturated-gain
integrals are the same closed-form edge integrals used in the competition
matrix (``_compute_mode_competition_element``).

How many modes lase? Gain clamping vs. competition
--------------------------------------------------

The most visible difference between the solvers is **how many modes they lase**,
and it comes straight from the hole-burning denominator. Once mode :math:`\mu`
lases it **clamps** the saturated gain at its own threshold level. A second mode
:math:`\nu` keeps lasing only if it still has net gain *after* that clamping,
which depends on how much its intensity :math:`|E_\nu|^2` **overlaps** mode
:math:`\mu`'s spatial hole — exactly the off-diagonal :math:`T_{\mu\nu}` relative
to the self-saturation :math:`T_{\mu\mu}`:

* **Strong overlap** (modes share the same region of the graph, e.g. a short
  cavity with a narrow gain line) — the second mode is starved → *winner-take-all*
  single-mode lasing.
* **Weak overlap** (modes occupy different regions / have distinct standing-wave
  patterns, e.g. a broad gain line exciting well-separated modes) — the second
  mode finds gain the first did not burn → *multimode* lasing.

The solvers treat this clamping at different levels of fidelity:

* ``linear`` switches a mode on when its fixed-:math:`T` *interacting threshold* is
  crossed and never re-tests it. It captures the near-threshold competition exactly
  (that is what :math:`T` is) but freezes the mode profiles, so above threshold it
  misses how the deepening holes reshape the competition.
* ``self_consistent`` and ``full_salt`` re-solve the saturated competition matrix
  at the operating pump, so the L–I curves **bend over** (``full_salt``) and the
  secondary modes' intensities/onsets shift. These are the single-pole-approximation
  (SPA) intensities ``D0/D0_thr - 1 = Σ_ν Γ_ν χ_μν I_ν`` — **the way Ge–Chong–Stone
  obtain modal intensities** — and are the methods to trust for quantitative L–I.
* ``full_salt_newton`` imposes the exact operator condition (the saturated operator
  singular at real :math:`k` with :math:`a\ge 0`) — the *operator-level* SALT, rather
  than the SPA matrix equation. It contributes a self-consistent gain-clamping
  **active set** and the lasing **frequencies** :math:`k_\mu` that the
  competition-matrix solvers cannot: on ``line_PRA`` it lases the **two** modes of
  Ge–Chong–Stone (PRA 82, 063824, Eq. 28) where ``self_consistent`` over-suppresses
  to one. Its amplitude is put in the linear unit by an onset scale that
  **measures** the operator's onset slope (an isolated-mode solve at
  :math:`1.2\,D_0^{\rm thr}`, with a Hellmann–Feynman analytic fallback), so it
  **reduces to** ``linear`` at threshold.

.. note::

   **Validated against Ge–Chong–Stone Fig. 6.** Above threshold ``full_salt_newton``
   gives the genuine full-SALT correction beyond the SPA: when a second mode turns
   on, the dominant mode picks up a **negative kink** and is suppressed *below* the
   SPA (its gain is stolen), while the second mode sits *above* the SPA, the two
   nearly cancelling in the total. On ``line_PRA`` the per-mode intensities track the
   digitized exact-SALT curves of Fig. 6 to a few percent (dominant 0.21 vs 0.205,
   second 0.10 vs 0.108 at :math:`D_0 = 1.27`), and the single-mode regime reduces to
   the SPA. This requires *measuring* the onset slope: an earlier analytic-only scale
   over-shot it by ~10–30 %, lifting the whole curve above the exact result. The
   competition-matrix solvers remain the cheaper first pass; ``full_salt_newton``
   adds the operator-level above-threshold correction.

.. warning::

   The operator-level hole burning samples :math:`|E_\nu(x)|^2` per edge. With the
   bare edges (one sample per edge) the per-edge **mean** over-estimates the mode
   overlap — it washes out the standing-wave nodes/antinodes where the coherent
   competition is weak — and **over-clamps**, spuriously suppressing co-lasing
   modes. On ``line_PRA`` this made ``full_salt_newton`` lase one mode where Ge
   Eq. 28 and the competition matrix lase two. ``oversample_size=None`` therefore
   auto-picks a wavelength-resolving sub-edge size
   (:func:`~netsalt.modes._auto_oversample_size`); resolving the standing wave
   removes the over-clamping and recovers the correct count. The oversampled graph
   is larger, but the eigensolve stays cheap because it runs through ARPACK
   shift-invert (which is ~flat in the node count on the banded quantum-graph
   laplacian; see ``DENSE_EIG_MAX``), so the solver scales to large graphs. It is
   still the heaviest of the four, so the competition-matrix methods remain the
   cheaper first pass for the lasing count.

.. note::

   Multimode lasing is demonstrated in
   ``examples/intensity_methods/two_ring_multimode.py``: two **detuned** rings
   (different sizes) joined by a bridge. The detuning localises each mode onto
   one ring -- identical rings would give symmetric/antisymmetric modes spread
   over both, with high overlap -- so with a narrow gain the modes barely
   compete and ``full_salt_newton`` lases several at once (spread across the two
   rings). ``examples/intensity_methods/chaotic_ring_multimode.py`` shows
   the same effect on a *single* small graph: one 14-node ring with six random
   chords (the buffon mechanism shrunk down). The chords close extra loops, so
   the spectrum is dense and the modes localise on different loops; with a
   narrow gain on a four-mode cluster ``full_salt_newton`` lases four. Getting
   there did need a tight ``k``-window (a loose one let the trust region collapse
   the multimode set to one mode by drifting a mode's ``k`` to a spurious
   ``a = 0`` root). Multimode remains the more delicate path, so treat it as
   experimental and sanity-check the mode count.

Selecting a solver
^^^^^^^^^^^^^^^^^^

Four solvers are available and chosen with the ``intensity_method`` config
key (default ``"linear"``), dispatched by
:func:`~netsalt.pipeline.step_compute_modal_intensities`:

``"linear"``
    :func:`~netsalt.modes.compute_modal_intensities` — the original
    near-threshold model described above. Unchanged; this is the default and the
    other two reduce to it at threshold.
``"self_consistent"``
    :func:`~netsalt.modes.compute_modal_intensities_self_consistent` — rebuilds
    the competition matrix at the operating pump
    (:func:`~netsalt.modes.compute_mode_competition_matrix_at_pump`) while reusing
    the same event-driven activation / vanishing sweep
    (``_modal_intensity_sweep``). Relaxes the frozen-profile approximation; keeps
    the linear saturation.
``"full_salt"``
    :func:`~netsalt.modes.compute_modal_intensities_full_salt` —
    *experimental, opt-in.* Folds the per-edge hole-burning denominator in via
    :func:`~netsalt.physics.dispersion_relation_pump_saturated` and a damped
    fixed point in the modal intensities at each pump, on top of the same sweep,
    so the gain clamps and the L–I curves bend over. Validated on the small
    ``line_PRA`` example; on large graphs treat it as exploratory.
    ``intensity_oversample_size`` (forwarded to
    :func:`~netsalt.quantum_graph.oversample_graph`) refines the
    per-edge-constant saturation toward the true within-edge field.
``"full_salt_newton"``
    :func:`~netsalt.modes.compute_modal_intensities_full_salt_newton` —
    *experimental, operator-level.* Rather than saturating the competition
    matrix, it solves the real nonlinear SALT eigenproblem: at each pump it finds,
    for every lasing mode, ``(k_μ, a_μ)`` so the shared saturated operator
    ``L_sat`` (:func:`~netsalt.physics.dispersion_relation_pump_saturated`) is
    singular at each real ``k_μ``. Two ingredients make it robust:

    * **Frozen-field trust-region solve** (:func:`~netsalt.modes._solve_active_set`)
      -- for a fixed active set the saturated background fields are frozen while a
      bounded trust-region least-squares solves all ``(k_μ, a_μ)``; the fields are
      then refreshed and the step repeated. Freezing the field makes each residual a
      single clean eigensolve (no inner fixed point), so the Jacobian is noise-free
      -- the earlier decoupled solve, whose residual re-ran an inner fixed point,
      chattered for several co-lasing modes.
    * **Self-consistent active-set continuation** -- the pump is stepped up; the
      confirmed lasing set is solved, modes whose amplitude vanishes are dropped,
      and a candidate is added when it has net gain (``α < 0``) on the current
      saturated background. The active set is found from the saturated operator, not
      borrowed from the linear model. This is only faithful when the within-edge
      hole burning is **resolved**: ``oversample_size=None`` auto-picks a
      wavelength-resolving sub-edge size, without which the per-edge mean
      over-clamps and spuriously drops co-lasing modes (see the warning above).

    Its amplitude is put in the **linear modal-intensity unit** by an onset scale
    that *measures* the operator's onset slope (an isolated-mode solve at
    ``1.2·D0_thr``, Hellmann–Feynman analytic fallback), so it **reduces to linear at
    threshold** and agrees on the lasing count -- validated against Ge–Chong–Stone
    (PRA 82, 063824) on ``line_PRA`` (both lase two modes, where ``self_consistent``
    over-suppresses to one). Above threshold it is the *exact-spatial* SALT: the
    dominant mode gets a negative kink (suppressed below the SPA when the second mode
    steals gain) and the second mode sits above the SPA, tracking the **exact-SALT
    data of Fig. 6 to a few percent** (a measured onset slope is required; an
    analytic-only scale over-shot the magnitudes by ~10-30 %). The competition-matrix
    solvers remain the cheaper first pass. It is deterministic, path-independent,
    never raises, and *expensive* (a nested per-pump solve on the oversampled graph),
    so use a modest
    ``salt_D0_steps``.

``benchmark/bench_salt.py`` compares the solvers on speed and accuracy: it runs
the shared pipeline once, swaps only the intensity step, writes overlaid L–I
curves and a within-edge (oversample) convergence study, and contrasts the
operator-level Newton solver with the linear model (onset slope + which modes
lase). All relaxations are exact at threshold, so the linear model remains the
threshold-limit check.

``examples/intensity_methods/compare_intensity_methods.py`` is a self-contained
worked example (with a physics walkthrough in its ``README``): it builds several
small open graphs -- a Fabry–Pérot line, a ring resonator, a tree splitter -- and
overlays the four methods' L–I curves, with a per-mode breakdown that makes the
above-threshold bend-over explicit.


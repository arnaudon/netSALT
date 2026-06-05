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

Relation to full SALT (future work)
-----------------------------------

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
     - fixed-point loop: solve :math:`I` → rebuild :math:`T` from the profiles
       at the *current* :math:`D_0` → repeat
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
matrix (``_compute_mode_competition_element``). A full-SALT solver would fold the
hole-burning denominator into the gain term of
:func:`~netsalt.quantum_graph.construct_laplacian` and Newton-solve over the
modal amplitudes and frequencies at each pump, reusing those overlaps, with the
linearized model as the threshold-limit check. This is tracked as future work.

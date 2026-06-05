Physics background
==================

This page sketches the physics *netSALT* implements, so the API reference reads
against a model rather than in isolation. For the full derivation and the
experimental context see Saxena *et al.*, *Nat. Commun.* **13**, 6573 (2022)
(https://doi.org/10.1038/s41467-022-34073-3).

Quantum graphs and the secular matrix
-------------------------------------

A *quantum graph* is a network whose edges carry a one-dimensional wave. On each
edge :math:`e` of length :math:`\ell_e` the field obeys the Helmholtz equation

.. math::

   \left(\partial_x^2 + k^2 n_e^2\right)\psi_e(x) = 0,

where :math:`k` is the (complex) wavenumber and :math:`n_e^2 = \varepsilon_e` is
the edge dielectric constant. The general solution on an edge is a pair of
counter-propagating plane waves; the unknowns are their amplitudes. Imposing
continuity of the field and a current-conservation (Kirchhoff) condition at every
node couples those amplitudes into a single linear system

.. math::

   L(k)\,\phi = 0, \qquad L(k) = B^{\mathsf T}(k)\,W^{-1}(k)\,B(k),

the **quantum Laplacian** (or secular matrix) assembled by
:func:`~netsalt.quantum_graph.construct_laplacian` from the incidence matrix
:func:`~netsalt.quantum_graph.construct_incidence_matrix` and the weight matrix
:func:`~netsalt.quantum_graph.construct_weight_matrix`. The boundary conditions
at the network's outer nodes are selected by ``params["open_model"]``
(``"open"`` radiating leads, ``"closed"``, ``"directed"``).

A **mode** is a non-trivial field that satisfies the system, i.e. a wavenumber
:math:`k` at which :math:`L(k)` is singular:

.. math::

   \det L(k) = 0 \quad\Longleftrightarrow\quad \lambda_1\!\big(L(k)\big) = 0,

where :math:`\lambda_1` is the eigenvalue of smallest magnitude.
:func:`~netsalt.quantum_graph.mode_quality` returns :math:`|\lambda_1(L(k))|`,
so locating modes means driving that quality to zero — either by rooting it on a
grid (:func:`~netsalt.modes.scan_frequencies` + refinement, see
:mod:`netsalt.algorithm`) or, by default, with the contour-integration search in
:mod:`netsalt.contour`.

The complex-wavenumber convention
---------------------------------

Modes live at complex :math:`k`. netSALT stores a mode as the real pair
:math:`(\,\mathrm{Re}\,k,\ \alpha\,)` with

.. math::

   k = \mathrm{Re}\,k - i\,\alpha, \qquad \alpha = -\,\mathrm{Im}\,k \ge 0,

so :math:`\alpha` is the modal **decay rate / leakage** through the open leads
(a passive, lossy resonance has :math:`\alpha > 0`). The minus sign is baked into
:func:`~netsalt.utils.to_complex` / :func:`~netsalt.utils.from_complex` and must
be respected everywhere. The scan rectangle is therefore
``[k_min, k_max] × [alpha_min, alpha_max]``.

Gain medium and the SALT pump
-----------------------------

To make the network *lase*, the inner edges are filled with a two-level gain
medium. Within the SALT (steady-state ab-initio laser theory) approximation the
gain enters the dielectric through a Lorentzian lineshape
:func:`~netsalt.physics.gamma`,

.. math::

   \gamma(k) = \frac{\gamma_\perp}{\mathrm{Re}\,k - k_a + i\,\gamma_\perp},

centred on the **atomic transition wavenumber** ``k_a`` with **linewidth**
``gamma_perp`` (the transverse relaxation rate, the half-width of the gain
curve). The pumped dispersion relation
:func:`~netsalt.physics.dispersion_relation_pump` then reads

.. math::

   k(\omega) = \omega\,\sqrt{\varepsilon + \gamma(\omega)\,D_0\,\delta_\mathrm{pump}},

where ``D0`` is the **pump strength** and :math:`\delta_\mathrm{pump}`
(``params["pump"]``) is the per-edge pump profile (which edges are pumped).
With ``D0 = 0`` this collapses to the passive
:func:`~netsalt.physics.dispersion_relation_dielectric`.

Passive modes, thresholds and competition
------------------------------------------

The pipeline (:mod:`netsalt.pipeline`) proceeds in physically meaningful stages:

* **Passive modes** — solve :math:`\det L(k) = 0` at ``D0 = 0``
  (:func:`~netsalt.modes.find_passive_modes`). Every mode has :math:`\alpha > 0`.
* **Lasing threshold** — raise the pump and track each mode
  (:func:`~netsalt.modes.pump_trajectories`). Gain pushes :math:`\alpha`
  downward; a mode reaches **threshold** when :math:`\alpha \to 0`, i.e. its
  leakage is exactly balanced by gain
  (:func:`~netsalt.modes.find_threshold_lasing_modes`). The modal quality factor
  is :math:`\mathcal{Q} = \mathrm{Re}\,k / (2\,\mathrm{Im}\,k)`
  (:func:`~netsalt.physics.q_value`).
* **Modal intensities and competition** — above threshold, modes share a common
  gain medium and compete: the spatial overlap of their intensities sets a
  competition matrix (:func:`~netsalt.modes.compute_mode_competition_matrix`)
  whose solution gives the steady-state modal intensities
  (:func:`~netsalt.modes.compute_modal_intensities`).
* **Spectral control** — :mod:`netsalt.pump` optimises the pump profile
  :math:`\delta_\mathrm{pump}` to select which modes lase, the central result of
  the accompanying paper.

Key physical parameters
-----------------------

The physics knobs in :class:`~netsalt.params.NetSaltParams`:

============== ====================================================================
``k_a``        atomic transition wavenumber — centre of the gain curve
``gamma_perp`` gain linewidth (transverse relaxation rate), half-width of
               :math:`\gamma(k)`
``D0``         pump strength (gain amplitude); ``D0 = 0`` is the passive graph
``pump``       per-edge pump profile :math:`\delta_\mathrm{pump}` (which edges are
               pumped)
``c``          wavespeed entering the dispersion relations
``open_model`` outer-node boundary condition (``open`` / ``closed`` / ``directed``)
============== ====================================================================

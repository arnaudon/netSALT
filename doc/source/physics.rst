The physics module
=========================

The physical model on top of the quantum-graph construction: the dispersion
relations :math:`k(\omega)` that map a frequency to a wavenumber on each edge,
and the SALT gain lineshape that turns the network into a laser. See
:doc:`theory` for how these fit together.

The dispersion relations range from the trivial linear law
(:func:`~netsalt.physics.dispersion_relation_linear`, :math:`k = \omega/c`)
through the dielectric law
(:func:`~netsalt.physics.dispersion_relation_dielectric`,
:math:`k = \omega\sqrt{\varepsilon}/c`) to the pumped law
(:func:`~netsalt.physics.dispersion_relation_pump`), which adds the gain term
:math:`\gamma(k)\,D_0\,\delta_\mathrm{pump}` under the square root. The gain
itself is the Lorentzian :func:`~netsalt.physics.gamma`, centred on the atomic
transition wavenumber ``k_a`` with linewidth ``gamma_perp``; ``D0`` is the pump
strength and ``params["pump"]`` the per-edge pump profile.
:func:`~netsalt.physics.set_dielectric_constant` assigns the per-edge
:math:`\varepsilon`, and :func:`~netsalt.physics.q_value` returns the modal
quality factor :math:`\mathcal{Q} = \mathrm{Re}\,k / (2\,\mathrm{Im}\,k)`.

.. automodule:: netsalt.physics
   :members:

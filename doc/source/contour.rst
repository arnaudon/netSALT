The contour module
=========================

Contour-integration mode search (Beyn's method). This is the **default**
mode-search backend (``params["mode_search_method"] = "contour"``): rather than
evaluating ``|λ₁(L(k))|`` on a dense Cartesian grid and refining each local
minimum, it locates every root of ``det(L(k)) = 0`` inside a closed contour in
the complex plane in ``O(N_quad · L²)`` work.

:func:`~netsalt.contour.find_modes_contour` is the production entry point.
:func:`~netsalt.contour.find_modes_contour_adaptive` does saturation-driven
recursion for parameter discovery when the mode count is unknown (coverage is
not guaranteed at deep recursion), and
:func:`~netsalt.contour.tune_contour_parameters` bridges the two: run adaptive
once, get an ``n_k`` sized for a reliable production
:func:`~netsalt.contour.find_modes_contour` call. See ``README.md`` and
``benchmark/README.md`` for the empirical sizing study.

.. automodule:: netsalt.contour
   :members:

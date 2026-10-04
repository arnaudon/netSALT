r"""Edge propagators for permittivity that varies *within* an edge.

The quantum-graph secular matrix is exact because each edge carries a single
permittivity: the 1D Helmholtz equation :math:`\psi'' + k^2 \epsilon \psi = 0`
then has the closed-form solutions :math:`e^{\pm i q x}`, which is where
:func:`~netsalt.quantum_graph.construct_weight_matrix`'s
:math:`1/(e^{2 i k_e l_e} - 1)` and
:func:`~netsalt.quantum_graph.construct_incidence_matrix`'s :math:`e^{i k_e l_e}`
come from.

Spatial hole burning breaks that assumption. The saturated permittivity

.. math:: \epsilon_{\rm eff}(x) = \epsilon
          + \gamma(k)\,\frac{D_0\,p(x)}{1 + \sum_\nu \Gamma_\nu a_\nu |E_\nu(x)|^2}

varies continuously along an edge, and the only remedy currently available is
``oversample_graph``: subdivide until :math:`\epsilon` is nearly constant on
each sub-edge. That works, but it pays by growing the *eigenproblem* -- on the
production buffon, resolving to :math:`\lambda/12` needs 76803 nodes against
243 edges, which is why full SALT is out of reach there (issues #52, #53).

An edge with varying :math:`\epsilon` still has an exact :math:`2\times 2`
transfer matrix taking :math:`(\psi, \psi')` from one end to the other; it is
simply not available in closed form. Computing it *per edge* keeps the
eigenproblem at its original size and moves the within-edge resolution into
independent, cheap, parallel work.

This module provides that transfer matrix. :func:`edge_transfer_matrix` is the
entry point; ``method="magnus4"`` is the default and the reason this is
worthwhile:

============================  =======================
scheme                        sub-intervals for 1e-8
============================  =======================
piecewise-constant (midpoint)                 819 200
Magnus, 2nd order                             819 200
Magnus, 4th order                             **6 400**
============================  =======================

measured on a buffon-like edge (length 11, :math:`k = 10.7`, :math:`n = 1.5`,
so ~28 oscillations, with a 4 % saturation ripple) --
``examples/audit/probe_edge_propagator.py``. Piecewise-constant and 2nd-order
Magnus coincide *exactly*, which is not a coincidence: for the constant-:math:`A`
system here, :math:`\exp(h A(x_{\rm mid}))` **is** the closed-form
constant-:math:`\epsilon` propagator. Oversampling is second-order Magnus, and
going to fourth order buys 128x fewer intervals for the same accuracy on top of
taking the count out of the matrix.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = [
    "edge_field_samples",
    "edge_step_propagators",
    "edge_transfer_matrices",
    "edge_transfer_matrix",
    "propagator_constant_eps",
]

#: Gauss-Legendre abscissae on [0, 1] for the two-point rule, used by the
#: fourth-order Magnus commutator term.
_C1 = 0.5 - np.sqrt(3.0) / 6.0
_C2 = 0.5 + np.sqrt(3.0) / 6.0


def propagator_constant_eps(q: complex, length: float) -> np.ndarray:
    r"""Exact :math:`(\psi, \psi')` propagator over a constant-:math:`\epsilon` step.

    This is the closed form the quantum-graph method already relies on, written
    in transfer-matrix form:

    .. math::

        \begin{pmatrix}\cos q h & \sin(q h)/q \\ -q \sin q h & \cos q h\end{pmatrix}

    Args:
        q: local wavenumber :math:`k\sqrt{\epsilon}` (may be complex).
        length: step length :math:`h`.

    Returns:
        The 2x2 propagator. Reduces to the free-propagation limit
        :math:`\bigl(\begin{smallmatrix}1&h\\0&1\end{smallmatrix}\bigr)` as
        :math:`q h \to 0`, which is handled explicitly rather than by
        cancellation.
    """
    qh = q * length
    if abs(qh) < 1e-14:
        return np.array([[1.0 + 0j, length + 0j], [0.0 + 0j, 1.0 + 0j]])
    cos, sin = np.cos(qh), np.sin(qh)
    return np.array([[cos, sin / q], [-q * sin, cos]], dtype=complex)


def _exp_traceless_2x2(matrix: np.ndarray) -> np.ndarray:
    r"""``expm`` of a traceless 2x2, in closed form.

    For traceless :math:`M`, Cayley-Hamilton gives :math:`M^2 = -\det(M)\,I`, so
    the series collapses to :math:`\cos w\,I + \frac{\sin w}{w} M` with
    :math:`w = \sqrt{\det M}`. Every matrix this module exponentiates is
    traceless (the Helmholtz system matrix is
    :math:`\bigl(\begin{smallmatrix}0&1\\-k^2\epsilon&0\end{smallmatrix}\bigr)`
    and so is its Magnus commutator), so this replaces a general ``expm`` call
    per step -- worth it when there are thousands of steps per edge.

    This scalar form is the *definition*; the propagators actually call
    :func:`_exp_traceless_batch`, which is this evaluated over a whole stack in
    the same order and hence to the same bits.
    """
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    w = np.sqrt(complex(determinant))
    if abs(w) < 1e-14:
        return np.eye(2, dtype=complex) + matrix
    return np.cos(w) * np.eye(2, dtype=complex) + (np.sin(w) / w) * matrix


def _system_matrix(k: complex, eps: complex) -> np.ndarray:
    r"""The Helmholtz system matrix :math:`A` in :math:`y' = A y`, :math:`y = (\psi, \psi')`."""
    return np.array([[0.0, 1.0], [-(k**2) * eps, 0.0]], dtype=complex)


#: ``sqrt(3)/12``, the fourth-order Magnus commutator coefficient.
_MAGNUS4_COMMUTATOR = np.sqrt(3.0) / 12.0


def _exp_traceless_batch(matrices: np.ndarray) -> np.ndarray:
    """:func:`_exp_traceless_2x2` over a whole ``(..., 2, 2)`` stack.

    Same closed form, evaluated with array ufuncs so a per-sub-interval Python
    loop is not paid. Agrees with :func:`_exp_traceless_2x2` elementwise to
    machine precision (measured max relative deviation 2.2e-15), not bit for
    bit: ``cos w`` and ``sinc w`` are obtained from a single complex
    exponential rather than from :func:`numpy.sin` and :func:`numpy.cos`.
    ``cos`` and ``sin`` of a *complex* argument each decompose into several real
    transcendentals, while ``exp(1j w)`` gives both from one -- measured on the
    production buffon's 217 x 408 sample grid, 9.85 ms -> 4.23 ms, which is
    ~10 % of a whole secular-matrix rebuild.
    """
    determinant = (
        matrices[..., 0, 0] * matrices[..., 1, 1] - matrices[..., 0, 1] * matrices[..., 1, 0]
    )
    w = np.sqrt(determinant.astype(complex))
    degenerate = np.abs(w) < 1e-14
    # np.where evaluates both branches, so keep the division away from w = 0.
    safe = np.where(degenerate, 1.0, w)
    phase = np.exp(1j * safe)
    inverse = 1.0 / phase
    scale = np.where(degenerate, 1.0, (-0.5j * (phase - inverse)) / safe)
    cosine = np.where(degenerate, 1.0, 0.5 * (phase + inverse))
    out = scale[..., None, None] * matrices
    out[..., 0, 0] += cosine
    out[..., 1, 1] += cosine
    return out


def _magnus_step_matrices(k, h, eps: tuple[np.ndarray, ...], method: str) -> np.ndarray:
    r"""Per-sub-interval propagators of the Helmholtz system, as one array.

    ``h`` and the ``eps`` samples broadcast against each other, so this serves a
    single edge (``eps`` shaped ``(n_steps,)``) and a whole batch of edges
    (``(n_edges, n_steps)`` against ``h`` shaped ``(n_edges, 1)``) with the same
    code.

    The Magnus generator is written out in closed form instead of being
    assembled from :func:`_system_matrix` and a matrix commutator. With
    :math:`A_i = \bigl(\begin{smallmatrix}0&1\\a_i&0\end{smallmatrix}\bigr)` and
    :math:`a_i = -k^2\epsilon_i`, the commutator is diagonal,
    :math:`[A_1, A_2] = (a_2 - a_1)\,\mathrm{diag}(1, -1)`, so

    .. math::

        \Omega = \begin{pmatrix} -c & h \\ \tfrac{h}{2}(a_1 + a_2) & c\end{pmatrix},
        \qquad c = \tfrac{\sqrt 3}{12} h^2 (a_2 - a_1),

    which is what the loop it replaces computed, in the same order.
    """
    if method == "magnus2":
        (eps_mid,) = eps
        a_mid = -(k**2) * eps_mid
        omega = np.zeros(np.shape(a_mid) + (2, 2), dtype=complex)
        omega[..., 0, 1] = h
        omega[..., 1, 0] = h * a_mid
        return _exp_traceless_batch(omega)

    eps_1, eps_2 = eps
    a_1 = -(k**2) * eps_1
    a_2 = -(k**2) * eps_2
    omega = np.zeros(np.broadcast_shapes(np.shape(a_1), np.shape(a_2)) + (2, 2), dtype=complex)
    commutator = _MAGNUS4_COMMUTATOR * h * h * (a_2 - a_1)
    omega[..., 0, 0] = -commutator
    omega[..., 1, 1] = commutator
    omega[..., 0, 1] = h
    omega[..., 1, 0] = 0.5 * h * (a_1 + a_2)
    return _exp_traceless_batch(omega)


def _ordered_product(steps: np.ndarray) -> np.ndarray:
    """``steps[..., -1] @ ... @ steps[..., 0]``, reducing over the second-to-last axis.

    The loop is sequential -- the ordered product is -- but each iteration
    multiplies *every* edge's sub-interval at once, so the Python overhead is
    paid ``n_steps`` times for the whole graph rather than ``n_edges * n_steps``
    times.

    The four entries are carried as separate arrays and the 2x2 product is
    written out, rather than calling ``@`` on a ``(n_edges, 2, 2)`` stack. That
    is not a micro-optimisation: at 2x2 the FLOPs are trivial and numpy's
    matmul gufunc dominates, 76 us per call against the ~5 us the arithmetic
    needs. Measured on the production buffon (217 edges, ``n_steps = 408``),
    25.38 ms -> 3.82 ms, a **6.6x** on what profiling showed to be 55 % of a
    secular-matrix rebuild. The association order is unchanged, so this agrees
    with the matmul form to machine precision (max relative deviation 1.3e-15).

    A log-depth pairwise tree would replace the ``n_steps`` iterations with
    ``log2(n_steps)`` and was measured instead of assumed, twice. Against
    matmul it is 2x on a 4-edge graph, where numpy call overhead is what is
    being paid, and 1.0x / 1.03x at 243 edges, where the strided gathers cost
    what the saved calls buy. Against the explicit form below it is *slower*
    still (5.70 ms against 3.82 ms), for the same reason. Not worth it at the
    scale that matters.
    """
    a = np.array(steps[..., 0, 0, 0], dtype=complex)
    b = np.array(steps[..., 0, 0, 1], dtype=complex)
    c = np.array(steps[..., 0, 1, 0], dtype=complex)
    d = np.array(steps[..., 0, 1, 1], dtype=complex)
    for i in range(1, steps.shape[-3]):
        p, q = steps[..., i, 0, 0], steps[..., i, 0, 1]
        r, s = steps[..., i, 1, 0], steps[..., i, 1, 1]
        a, b, c, d = p * a + q * c, p * b + q * d, r * a + s * c, r * b + s * d
    total = np.empty(steps.shape[:-3] + (2, 2), dtype=complex)
    total[..., 0, 0] = a
    total[..., 0, 1] = b
    total[..., 1, 0] = c
    total[..., 1, 1] = d
    return total


#: ``(length, n_steps, method)`` -> ``(h, points)``. The abscissae depend only on
#: an edge's geometry and the scheme, so they are the same at every ``k``, every
#: pump and every mode -- yet they were rebuilt on each of the ~1600
#: secular-matrix rebuilds a single solve performs, once per edge. Measured on
#: the production buffon, that is 0.80 ms of the 5.11 ms sampling loop, and the
#: loop is 28 % of a rebuild. Bounded by the number of distinct edge lengths,
#: which is bounded by the graph.
_SAMPLE_POINT_CACHE: dict[tuple[float, int, str], tuple[float, tuple[np.ndarray, ...]]] = {}


def _magnus_sample_points(length: float, n_steps: int, method: str):
    """Positions at which ``eps`` is sampled, and the sub-interval width.

    Cached on ``(length, n_steps, method)``; the arrays are shared with every
    caller and must not be mutated.
    """
    key = (float(length), int(n_steps), method)
    cached = _SAMPLE_POINT_CACHE.get(key)
    if cached is not None:
        return cached
    h = length / n_steps
    starts = np.arange(n_steps) * h
    points = (starts + 0.5 * h,) if method == "magnus2" else (starts + _C1 * h, starts + _C2 * h)
    for point in points:
        point.flags.writeable = False
    _SAMPLE_POINT_CACHE[key] = (h, points)
    return h, points


def edge_transfer_matrix(
    k: complex,
    length: float,
    eps: Callable[[np.ndarray], np.ndarray] | complex,
    n_steps: int = 64,
    method: str = "magnus4",
) -> np.ndarray:
    r"""Transfer matrix of one edge whose permittivity may vary along it.

    Args:
        k: vacuum wavenumber.
        length: edge length.
        eps: either a constant permittivity, or a callable mapping an array of
            positions in ``[0, length]`` to permittivities. A constant (or a
            callable that is constant) is propagated exactly in one step
            regardless of ``n_steps``.
        n_steps: sub-intervals along the edge. This does **not** enter the
            eigenproblem -- it is local to this edge, which is the entire point
            of computing a transfer matrix instead of oversampling the graph.
        method: ``"magnus4"`` (default, fourth order) or ``"magnus2"``
            (second order, identical to freezing eps at each sub-interval
            midpoint -- i.e. exactly what ``oversample_graph`` does).

    Returns:
        The 2x2 matrix mapping :math:`(\psi, \psi')` at ``x = 0`` to its value
        at ``x = length``.

    Raises:
        ValueError: if ``method`` is not recognised or ``n_steps < 1``.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    if method not in ("magnus2", "magnus4"):
        raise ValueError(f"Unknown method {method!r}; expected 'magnus2' or 'magnus4'.")

    if not callable(eps):
        # Constant permittivity: the closed form is exact, so do not discretise.
        return propagator_constant_eps(k * np.sqrt(complex(eps)), length)

    h, points = _magnus_sample_points(length, n_steps, method)
    samples = tuple(np.asarray(eps(point), dtype=complex) for point in points)
    return _ordered_product(_magnus_step_matrices(k, h, samples, method))


def sample_eps_profiles(lengths, eps_profiles, n_steps: int = 64, method: str = "magnus4"):
    r"""Sample every edge's permittivity on the propagator's own abscissae.

    Split out of :func:`edge_transfer_matrices` because the result does not
    depend on ``k``: the profiles and the geometry fix it. A SALT residual
    evaluation asks for :math:`\lambda_1(k_\mu)` at ``M`` different ``k`` on the
    *same* saturated profiles, so sampling inside the transfer-matrix build
    repeated identical work ``M`` times. On the production buffon the sampling
    loop is 5.11 ms of an 18.25 ms ``lambda_1`` evaluation, so hoisting it is
    1.34x at ``M = 11``.

    Args:
        lengths: one length per edge.
        eps_profiles: one callable per edge, as :func:`edge_transfer_matrices`.
        n_steps, method: as :func:`edge_transfer_matrix`.

    Returns:
        ``(h, eps_samples)`` to hand back as ``samples=``: the per-edge
        sub-interval widths, and one ``(n_edges, n_steps)`` array per Magnus
        abscissa.
    """
    lengths = np.asarray(lengths, dtype=float)
    n_edges = len(lengths)
    n_points = 1 if method == "magnus2" else 2
    h = np.empty(n_edges)
    eps_samples = tuple(np.empty((n_edges, n_steps), dtype=complex) for _ in range(n_points))
    for edge_index, (length, profile) in enumerate(zip(lengths, eps_profiles, strict=True)):
        h[edge_index], points = _magnus_sample_points(float(length), n_steps, method)
        for sample, point in zip(eps_samples, points, strict=True):
            sample[edge_index] = profile(point)
    return h, eps_samples


def edge_transfer_matrices(
    k: complex,
    lengths,
    eps_profiles,
    n_steps: int = 64,
    method: str = "magnus4",
    samples=None,
) -> np.ndarray:
    """:func:`edge_transfer_matrix` for many edges at once.

    Every edge is propagated with the same ``n_steps``, so the sub-interval
    matrices form one ``(n_edges, n_steps, 2, 2)`` array and the ordered product
    is ``n_steps`` batched matmuls for the whole set instead of
    ``n_edges * n_steps`` scalar ones. That is where the time goes when the
    saturated operator is rebuilt at every residual evaluation: the eigenproblem
    is small, the per-edge propagation is not.

    Args:
        k: vacuum wavenumber, shared by all edges.
        lengths: one length per edge.
        eps_profiles: one callable per edge. Constant permittivities do not
            belong here -- they have a closed form; use
            :func:`propagator_constant_eps`.
        n_steps, method: as :func:`edge_transfer_matrix`.
        samples: the output of :func:`sample_eps_profiles` for these profiles,
            when the caller has already computed it (it does not depend on
            ``k``, so a residual evaluation over ``M`` modes should compute it
            once). ``None`` samples them here.

    Returns:
        ``(n_edges, 2, 2)`` transfer matrices, agreeing with
        :func:`edge_transfer_matrix` per edge to machine precision.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    if method not in ("magnus2", "magnus4"):
        raise ValueError(f"Unknown method {method!r}; expected 'magnus2' or 'magnus4'.")

    lengths = np.asarray(lengths, dtype=float)
    n_edges = len(lengths)
    if n_edges != len(eps_profiles):
        raise ValueError(f"{len(eps_profiles)} profiles for {n_edges} lengths.")
    if n_edges == 0:
        return np.empty((0, 2, 2), dtype=complex)

    if samples is None:
        samples = sample_eps_profiles(lengths, eps_profiles, n_steps, method)
    h, eps_samples = samples
    return _ordered_product(_magnus_step_matrices(k, h[:, None], eps_samples, method))


def edge_step_propagators(
    k: complex,
    length: float,
    eps: Callable[[np.ndarray], np.ndarray] | complex,
    n_steps: int = 64,
    method: str = "magnus4",
) -> list[np.ndarray]:
    """Per-sub-interval propagators, in order along the edge.

    :func:`edge_transfer_matrix` is their ordered product. They are kept
    separately here because the SALT iteration needs the field *inside* the
    edge -- the hole-burning profile is built from ``|E(x)|**2`` -- and
    re-propagating from scratch to get it would double the work.

    A constant ``eps`` still yields ``n_steps`` equal factors rather than one,
    so that :func:`edge_field_samples` returns a usable grid either way.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    if method not in ("magnus2", "magnus4"):
        raise ValueError(f"Unknown method {method!r}; expected 'magnus2' or 'magnus4'.")

    if not callable(eps):
        q = k * np.sqrt(complex(eps))
        return [propagator_constant_eps(q, length / n_steps)] * n_steps

    h, points = _magnus_sample_points(length, n_steps, method)
    samples = tuple(np.asarray(eps(point), dtype=complex) for point in points)
    return list(_magnus_step_matrices(k, h, samples, method))


def edge_field_samples(
    k: complex,
    length: float,
    eps: Callable[[np.ndarray], np.ndarray] | complex,
    psi_start: complex,
    dpsi_start: complex,
    n_steps: int = 64,
    method: str = "magnus4",
) -> tuple[np.ndarray, np.ndarray]:
    r"""Field along one edge, given :math:`(\psi, \psi')` at its start.

    Args:
        k, length, eps, n_steps, method: as :func:`edge_transfer_matrix`.
        psi_start: :math:`\psi(0)`.
        dpsi_start: :math:`\psi'(0)`.

    Returns:
        ``(positions, psi)``, both of length ``n_steps + 1``, sampled at the
        sub-interval boundaries including both endpoints. This is what the
        saturated permittivity is evaluated from, so its resolution is the
        within-edge resolution -- and, unlike ``oversample_graph``, it costs
        nothing in the size of the eigenproblem.
    """
    steps = edge_step_propagators(k, length, eps, n_steps=n_steps, method=method)
    state = np.array([complex(psi_start), complex(dpsi_start)], dtype=complex)
    psi = np.empty(len(steps) + 1, dtype=complex)
    psi[0] = state[0]
    for i, step in enumerate(steps):
        state = step @ state
        psi[i + 1] = state[0]
    return np.linspace(0.0, length, len(steps) + 1), psi

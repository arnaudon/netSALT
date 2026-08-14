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
    """
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    w = np.sqrt(complex(determinant))
    if abs(w) < 1e-14:
        return np.eye(2, dtype=complex) + matrix
    return np.cos(w) * np.eye(2, dtype=complex) + (np.sin(w) / w) * matrix


def _system_matrix(k: complex, eps: complex) -> np.ndarray:
    r"""The Helmholtz system matrix :math:`A` in :math:`y' = A y`, :math:`y = (\psi, \psi')`."""
    return np.array([[0.0, 1.0], [-(k**2) * eps, 0.0]], dtype=complex)


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

    h = length / n_steps
    starts = np.arange(n_steps) * h

    if method == "magnus2":
        eps_mid = np.asarray(eps(starts + 0.5 * h), dtype=complex)
        total = np.eye(2, dtype=complex)
        for value in eps_mid:
            total = _exp_traceless_2x2(h * _system_matrix(k, value)) @ total
        return total

    eps_1 = np.asarray(eps(starts + _C1 * h), dtype=complex)
    eps_2 = np.asarray(eps(starts + _C2 * h), dtype=complex)
    total = np.eye(2, dtype=complex)
    for value_1, value_2 in zip(eps_1, eps_2, strict=True):
        a_1 = _system_matrix(k, value_1)
        a_2 = _system_matrix(k, value_2)
        # Omega_2 = h/2 (A1 + A2) - sqrt(3)/12 h^2 [A1, A2]
        commutator = a_1 @ a_2 - a_2 @ a_1
        omega = 0.5 * h * (a_1 + a_2) - (np.sqrt(3.0) / 12.0) * h * h * commutator
        total = _exp_traceless_2x2(omega) @ total
    return total


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

    h = length / n_steps
    starts = np.arange(n_steps) * h
    if not callable(eps):
        q = k * np.sqrt(complex(eps))
        return [propagator_constant_eps(q, h)] * n_steps

    if method == "magnus2":
        values = np.asarray(eps(starts + 0.5 * h), dtype=complex)
        return [_exp_traceless_2x2(h * _system_matrix(k, value)) for value in values]

    eps_1 = np.asarray(eps(starts + _C1 * h), dtype=complex)
    eps_2 = np.asarray(eps(starts + _C2 * h), dtype=complex)
    out = []
    for value_1, value_2 in zip(eps_1, eps_2, strict=True):
        a_1 = _system_matrix(k, value_1)
        a_2 = _system_matrix(k, value_2)
        omega = 0.5 * h * (a_1 + a_2) - (np.sqrt(3.0) / 12.0) * h * h * (a_1 @ a_2 - a_2 @ a_1)
        out.append(_exp_traceless_2x2(omega))
    return out


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

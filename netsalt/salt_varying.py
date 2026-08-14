r"""SALT on the per-edge-DtN operator, without oversampling the graph.

:mod:`netsalt.modes`' full-SALT path represents the spatial hole burning as a
per-edge **scalar** ``D0_eff`` and reaches within-edge resolution by
``oversample_graph``, which grows the eigenproblem (issues #52, #53). This module
is the same physics carried on
:func:`~netsalt.varying_laplacian.construct_laplacian_varying`: the hole-burning
denominator stays a *function of position along each edge*, the matrix stays one
node per vertex, and the resolution lives in per-edge transfer matrices.

The pieces, in the order the solver needs them:

1. :func:`node_solution_varying` -- the null vector of the varying operator at a
   real ``k``, i.e. the field's values at the vertices.
2. :func:`edge_field_profiles` -- :math:`|E(x)|^2` sampled *inside* every edge,
   obtained by propagating :math:`(\psi, \psi')` from each edge's start. The
   starting derivative comes from the same transfer matrix that built the
   operator, :math:`\psi'_u = (\psi_v - M_{11}\psi_u)/M_{12}`.
3. :func:`saturated_eps_profiles` -- the saturated permittivity per edge,
   :math:`\epsilon + \gamma(k)\,D_0\,p/(1 + \sum_\nu \Gamma_\nu a_\nu |E_\nu|^2)`,
   as callables built from those samples.
4. :func:`salt_residuals_varying` -- the acceptance test, the direct analogue of
   :func:`~netsalt.modes.salt_residuals`.

Normalisation matches ``modes._single_mode_field_intensity``: the profiles are
divided by the pump-region norm, so the amplitudes they multiply are in the same
``∫_pump |Ê|^2 = 1`` convention the rest of the solver uses and the two paths are
directly comparable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import scipy as sc
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline

from .edge_propagator import edge_field_samples, edge_transfer_matrix, propagator_constant_eps
from .physics import gamma
from .varying_laplacian import construct_laplacian_varying

__all__ = [
    "SaltVaryingSolution",
    "edge_field_profiles",
    "node_solution_varying",
    "salt_residuals_varying",
    "saturated_eps_profiles",
    "solve_salt_varying",
]

#: Above this many nodes the null vector is found with ARPACK rather than a
#: dense SVD. Mirrors ``quantum_graph.DENSE_SVD_MAX``'s role on the other path.
DENSE_NULL_MAX = 400


def _smallest_eigenpair(matrix):
    """Smallest-magnitude eigenvalue of ``matrix`` and its eigenvector."""
    size = matrix.shape[0]
    if size <= DENSE_NULL_MAX:
        dense = np.asarray(matrix.todense()) if sc.sparse.issparse(matrix) else np.asarray(matrix)
        values, vectors = np.linalg.eig(dense)
        i = int(np.argmin(np.abs(values)))
        return values[i], vectors[:, i]
    values, vectors = sc.sparse.linalg.eigs(
        matrix.asfptype(), k=1, sigma=0, which="LM", return_eigenvectors=True
    )
    return values[0], vectors[:, 0]


def node_solution_varying(
    wavenumber: complex,
    graph,
    eps_profiles=None,
    n_steps: int = 64,
) -> tuple[complex, np.ndarray]:
    r"""Null eigenpair of the varying-permittivity operator at ``wavenumber``.

    Returns:
        ``(lambda_1, psi_nodes)`` -- the smallest-magnitude eigenvalue, which is
        the SALT residual at this ``k``, and its eigenvector, the field at the
        vertices. A genuine lasing solution has ``|lambda_1| ~ 0`` at real ``k``.
    """
    laplacian = construct_laplacian_varying(wavenumber, graph, eps_profiles, n_steps=n_steps)
    return _smallest_eigenpair(laplacian)


def _edge_transfer(graph, k_over_c, edge_index, u, v, profile, length, n_steps, method):
    """Transfer matrix of one edge, constant or varying."""
    if profile is None:
        return propagator_constant_eps(np.asarray(graph.graph["ks"])[edge_index], length)
    return edge_transfer_matrix(k_over_c, length, profile, n_steps=n_steps, method=method)


def edge_field_profiles(
    wavenumber: complex,
    graph,
    node_solution: np.ndarray,
    eps_profiles=None,
    n_steps: int = 64,
    method: str = "magnus4",
    normalise: bool = True,
    pump: np.ndarray | None = None,
) -> list[np.ndarray]:
    r"""Sampled :math:`|E(x)|^2` inside every edge.

    Args:
        wavenumber: real lasing frequency.
        graph: quantum graph (**not** oversampled).
        node_solution: field at the vertices, from :func:`node_solution_varying`.
        eps_profiles: per-edge permittivity profiles, ``None`` where constant.
        n_steps: samples per edge, minus one. Local to the edge.
        method: propagator order.
        normalise: divide by the pump-region norm, matching
            ``modes._single_mode_field_intensity`` so amplitudes share its
            convention. Note that norm is **unconjugated**: ``modes._graph_norm``
            contracts with ``node_solution.T`` (not ``.conj().T``) against a
            ``compute_z_matrix`` built from ``exp(2i l k)/(2i k)``, i.e. it
            integrates :math:`\psi^2`, while the per-edge intensity it divides is
            :math:`|E|^2` (``_mean_intensity_from_flux`` uses ``k + conj(k)``).
            That asymmetry is the SALT biorthogonal convention; matching it is
            what makes an amplitude ``a`` mean the same thing on both paths.
            Using :math:`\int|\psi|^2` here instead leaves a constant few-percent
            offset -- 2.2 % on the shipped Fabry-Perot fixture.
        pump: per-edge pump, required when ``normalise`` is True.

    Returns:
        One array of ``n_steps + 1`` samples per edge, in ``graph.edges`` order.
    """
    params = graph.graph["params"]
    k_over_c = wavenumber / float(params.get("c", 1.0) or 1.0)
    lengths = np.asarray(graph.graph["lengths"], dtype=float)
    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    if eps_profiles is None:
        eps_profiles = [None] * len(graph.edges)

    profiles: list[np.ndarray] = []
    raw_fields: list[np.ndarray] = []
    for edge_index, (u, v) in enumerate(graph.edges):
        length = float(lengths[edge_index])
        profile = eps_profiles[edge_index]
        transfer = _edge_transfer(
            graph, k_over_c, edge_index, u, v, profile, length, n_steps, method
        )
        psi_u = complex(node_solution[index[u]])
        psi_v = complex(node_solution[index[v]])
        # psi'_u from the same transfer matrix that built the operator's block
        dpsi_u = (psi_v - transfer[0, 0] * psi_u) / transfer[0, 1]
        eps_for_samples = (
            profile
            if profile is not None
            else complex(np.asarray(params["dielectric_constant"])[edge_index])
        )
        _, psi = edge_field_samples(
            k_over_c, length, eps_for_samples, psi_u, dpsi_u, n_steps=n_steps, method=method
        )
        profiles.append(np.abs(psi) ** 2)
        raw_fields.append(psi)

    if not normalise:
        return profiles
    if pump is None:
        raise ValueError("normalise=True requires the per-edge pump profile")
    # Pump-region norm: integral of psi**2 -- *unconjugated*, matching
    # modes._graph_norm (see the `normalise` note in this docstring).
    #
    # Simpson, not trapezoid: the propagator is fourth-order, so an O(h^2)
    # quadrature here would set the accuracy of the whole profile and silently
    # throw away what the Magnus step bought. With trapezoid the per-edge means
    # converge as h^2 (4.4e-5 -> 1.1e-5 -> 2.8e-6 over doublings on the shipped
    # Fabry-Perot fixture); Simpson matches the propagator's order.
    pump = np.asarray(pump, dtype=float)
    total = 0.0 + 0.0j
    for edge_index, psi in enumerate(raw_fields):
        if pump[edge_index] <= 0.0:
            continue
        dx = float(lengths[edge_index]) / (len(psi) - 1)
        total += float(pump[edge_index]) * simpson(psi**2, dx=dx)
    if abs(total) == 0.0:
        return profiles
    return [samples / abs(total) for samples in profiles]


def saturated_eps_profiles(
    graph,
    ks: np.ndarray,
    amplitudes: np.ndarray,
    field_profiles: list[list[np.ndarray]],
    D0: float,
    pump: np.ndarray,
) -> list[Callable[[np.ndarray], np.ndarray] | None]:
    r"""Per-edge saturated permittivity, as callables.

    .. math::

        \epsilon_{\rm eff}(x) = \epsilon_e
        + \gamma(k)\,\frac{D_0\,p_e}{1 + \sum_\nu \Gamma_\nu a_\nu |E_\nu(x)|^2}

    evaluated on each edge's sample grid and interpolated with a cubic spline --
    cubic so the interpolation error stays at the fourth-order propagator's
    level rather than dominating it.

    Args:
        graph: quantum graph.
        ks: real lasing frequencies, one per mode.
        amplitudes: modal amplitudes.
        field_profiles: ``field_profiles[nu][edge]`` -- the sampled
            :math:`|E_\nu(x)|^2` of every lasing mode.
        D0: pump strength.
        pump: per-edge pump.

    Returns:
        One entry per edge: ``None`` where the edge is unpumped (its permittivity
        is exactly constant, so the closed form stays exact and no discretisation
        is spent), otherwise a callable.
    """
    params = graph.graph["params"]
    dielectric = np.asarray(params["dielectric_constant"])
    lengths = np.asarray(graph.graph["lengths"], dtype=float)
    pump = np.asarray(pump, dtype=float)
    ks = np.asarray(ks, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    clamps = [-np.imag(gamma(complex(k), params)) for k in ks]
    # Gain is evaluated at each mode's own k in the denominator, but the operator
    # is built at one k; the prefactor gamma(k) is applied by the caller's choice
    # of `wavenumber`, so use the first (the mode being tested) here.
    out: list[Callable[[np.ndarray], np.ndarray] | None] = []
    for edge_index in range(len(graph.edges)):
        if pump[edge_index] <= 0.0 or D0 == 0.0:
            out.append(None)
            continue
        n_samples = len(field_profiles[0][edge_index]) if field_profiles else 2
        grid = np.linspace(0.0, float(lengths[edge_index]), n_samples)
        denom = np.ones(n_samples)
        for clamp, a_nu, per_edge in zip(clamps, amplitudes, field_profiles, strict=True):
            denom = denom + clamp * float(a_nu) * np.asarray(per_edge[edge_index], dtype=float)
        d0_eff = D0 * float(pump[edge_index]) / denom
        out.append(_spline_profile(grid, d0_eff, dielectric[edge_index], params))
    return out


def _spline_profile(grid, d0_eff, eps_edge, params):
    """Build eps(x) = eps_edge + gamma(k) D0_eff(x); gamma is bound at call time."""
    spline = CubicSpline(grid, d0_eff)
    lo, hi = grid[0], grid[-1]

    def profile(x, _spline=spline, _eps=eps_edge, _lo=lo, _hi=hi):
        clipped = np.clip(np.asarray(x, dtype=float), _lo, _hi)
        return _eps + profile.gain * _spline(clipped)

    profile.gain = 0.0 + 0.0j  # set by salt_residuals_varying, which knows k
    return profile


def salt_residuals_varying(
    graph,
    ks,
    amplitudes,
    field_profiles,
    D0: float,
    pump,
    n_steps: int = 64,
) -> np.ndarray:
    r"""SALT residual per mode, on the varying-permittivity operator.

    The direct analogue of :func:`~netsalt.modes.salt_residuals`: it reports
    ``|lambda_1|`` of the shared saturated operator at each mode's real ``k``,
    and a genuine solution has all of them at solver tolerance. The difference is
    that the hole burning is resolved *within* each edge instead of by
    subdividing the graph, so the matrix is ``len(graph)`` square regardless of
    the resolution.

    Args:
        graph: quantum graph, **not** oversampled.
        ks: real lasing frequencies.
        amplitudes: modal amplitudes.
        field_profiles: ``[nu][edge]`` sampled ``|E_nu(x)|^2``.
        D0: pump strength.
        pump: per-edge pump.
        n_steps: sub-intervals per varying edge.

    Returns:
        ``np.ndarray`` of ``|lambda_1|``, one per mode.
    """
    ks = np.asarray(ks, dtype=float)
    params = graph.graph["params"]
    profiles = saturated_eps_profiles(graph, ks, amplitudes, field_profiles, D0, pump)
    residuals = []
    for k in ks:
        gain = gamma(complex(k), params)
        for profile in profiles:
            if profile is not None:
                profile.gain = gain
        value, _ = node_solution_varying(float(k), graph, profiles, n_steps=n_steps)
        residuals.append(abs(value))
    return np.array(residuals)


#: Residual the fixed-set solve drives ``|lambda_1|`` towards before declaring
#: convergence. Matches ``modes.SALT_RESIDUAL_TARGET`` so the two paths report
#: convergence on the same footing.
SALT_VARYING_RESIDUAL_TARGET = 1e-6

#: Amplitudes below this are treated as "not lasing" when refreshing the fields,
#: mirroring ``modes.SALT_LASING_AMPLITUDE``.
SALT_VARYING_LASING_AMPLITUDE = 1e-4


class SaltVaryingSolution(NamedTuple):
    """Result of :func:`solve_salt_varying`.

    ``ks`` and ``amplitudes`` are the solved lasing frequencies and amplitudes;
    ``fields`` the ``|E(x)|^2`` profiles they were solved against; ``residuals``
    the per-mode ``|lambda_1|`` at the returned state -- the acceptance test, not
    a proxy for it. ``converged`` says whether every lasing mode reached
    :data:`SALT_VARYING_RESIDUAL_TARGET`; a solve can be un-converged with a
    small residual (it ran out of iterations just after arriving) or converged
    with a large one only if the target itself is loose, so read both.
    """

    ks: np.ndarray
    amplitudes: np.ndarray
    fields: list
    residuals: np.ndarray
    converged: bool
    iterations: int


def _lam_varying(graph, k, profiles, n_steps):
    """Complex ``lambda_1`` at real ``k``, with the gain bound at this ``k``."""
    gain = gamma(complex(k), graph.graph["params"])
    for profile in profiles:
        if profile is not None:
            profile.gain = gain
    value, _ = node_solution_varying(float(k), graph, profiles, n_steps=n_steps)
    return value


def solve_salt_varying(
    graph,
    ks,
    amplitudes,
    D0: float,
    pump,
    *,
    n_steps: int = 64,
    outer: int = 25,
    damping: float = 0.7,
    residual_tol: float = SALT_VARYING_RESIDUAL_TARGET,
    max_nfev: int = 60,
) -> SaltVaryingSolution:
    r"""Solve SALT for a given set of lasing modes, without oversampling.

    The varying-operator counterpart of
    :func:`~netsalt.modes.solve_salt_fixed_set`. For each mode the unknowns are
    :math:`(k_\mu, a_\mu)` and the equations are
    :math:`\mathrm{Re}\,\lambda_1(k_\mu) = \mathrm{Im}\,\lambda_1(k_\mu) = 0` --
    two real equations for two real unknowns, so the system is square. The
    hole-burning fields are frozen during each least-squares solve and refreshed
    afterwards, which is what makes each residual a single clean eigensolve.

    Args:
        graph: quantum graph, **not** oversampled.
        ks: initial real frequencies, one per lasing mode.
        amplitudes: initial amplitudes.
        D0: pump strength.
        pump: per-edge pump.
        n_steps: sub-intervals per varying edge.
        outer: field-refresh iterations.
        damping: mixing applied when refreshing the fields.
        residual_tol: convergence target on ``|lambda_1|``.
        max_nfev: budget for each frozen-field least-squares solve.

    Returns:
        :class:`SaltVaryingSolution`.
    """
    from scipy.optimize import least_squares

    ks = np.asarray(ks, dtype=float).copy()
    amplitudes = np.asarray(amplitudes, dtype=float).copy()
    n_modes = len(ks)
    pump = np.asarray(pump, dtype=float)

    # initial fields, from the unsaturated operator
    fields = []
    for k in ks:
        _, psi = node_solution_varying(float(k), graph, None, n_steps=n_steps)
        fields.append(edge_field_profiles(float(k), graph, psi, None, n_steps=n_steps, pump=pump))

    converged = False
    iterations = 0
    for _outer_step in range(outer):
        iterations += 1
        frozen = [list(f) for f in fields]

        def residual(x, _frozen=frozen):
            local_ks = x[0::2]
            local_a = np.clip(x[1::2], 0.0, None)
            profiles = saturated_eps_profiles(graph, local_ks, local_a, _frozen, D0, pump)
            out = []
            for k in local_ks:
                value = _lam_varying(graph, float(k), profiles, n_steps)
                out.extend((value.real, value.imag))
            return np.asarray(out, dtype=float)

        x0 = np.empty(2 * n_modes)
        x0[0::2] = ks
        x0[1::2] = amplitudes
        result = least_squares(residual, x0, method="lm", max_nfev=max_nfev, xtol=1e-12)
        ks = result.x[0::2]
        amplitudes = np.clip(result.x[1::2], 0.0, None)

        # refresh the fields against the solved state
        profiles = saturated_eps_profiles(graph, ks, amplitudes, frozen, D0, pump)
        refreshed = []
        for k in ks:
            gain = gamma(complex(k), graph.graph["params"])
            for profile in profiles:
                if profile is not None:
                    profile.gain = gain
            _, psi = node_solution_varying(float(k), graph, profiles, n_steps=n_steps)
            refreshed.append(
                edge_field_profiles(float(k), graph, psi, profiles, n_steps=n_steps, pump=pump)
            )
        fields = [
            [
                (1.0 - damping) * old + damping * new
                for old, new in zip(per_mode_old, per_mode_new, strict=True)
            ]
            for per_mode_old, per_mode_new in zip(fields, refreshed, strict=True)
        ]

        lasing = [i for i in range(n_modes) if amplitudes[i] > SALT_VARYING_LASING_AMPLITUDE]
        residuals = salt_residuals_varying(graph, ks, amplitudes, fields, D0, pump, n_steps=n_steps)
        if lasing and all(residuals[i] <= residual_tol for i in lasing):
            converged = True
            break

    residuals = salt_residuals_varying(graph, ks, amplitudes, fields, D0, pump, n_steps=n_steps)
    return SaltVaryingSolution(ks, amplitudes, fields, residuals, converged, iterations)

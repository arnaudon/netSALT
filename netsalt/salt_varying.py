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

import warnings
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import scipy as sc
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline

from .contour import optical_length
from .edge_propagator import edge_field_samples, edge_transfer_matrix, propagator_constant_eps
from .physics import gamma
from .varying_laplacian import construct_laplacian_varying

__all__ = [
    "SaltVaryingSolution",
    "compute_modal_intensities_varying",
    "edge_field_profiles",
    "net_gain_alpha",
    "node_solution_varying",
    "salt_residuals_varying",
    "saturated_eps_profiles",
    "solve_salt_varying",
]

#: Above this many nodes the null vector is found by ARPACK shift-invert rather
#: than a dense ``eig``. Mirrors ``quantum_graph.DENSE_EIG_MAX``'s role on the
#: other path, and like it the value is the *measured* crossover rather than a
#: comfortable-looking round number: only the smallest-magnitude eigenpair is
#: wanted, and a dense ``eig`` computes all ``n`` of them in ``O(n^3)`` while
#: shift-invert on the banded secular matrix is nearly flat in ``n``. On the
#: saturated varying operator of a line graph (median of repeated solves,
#: agreeing to 1e-12):
#:
#: =====  =========  ==========
#: ``n``  dense      shift-inv.
#: =====  =========  ==========
#: 33      0.68 ms     0.78 ms
#: 65      3.47 ms     0.91 ms
#: 129    22.6  ms     0.98 ms
#: 257    97.4  ms     1.30 ms
#: 1025    2.69 s      2.30 ms
#: =====  =========  ==========
#:
#: -- so the two are level around 40 nodes, not 400. The production buffon
#: (208 nodes) spent two thirds of every operator rebuild in ``eig`` at the old
#: threshold.
DENSE_NULL_MAX = 40

#: ARPACK start vector seed. Fixed so a residual is reproducible: without it the
#: sparse branch draws a fresh random start and reports slightly different last
#: digits on every call, which the finite-difference Jacobian of the frozen-field
#: least-squares reads as noise.
DENSE_NULL_SEED = 42

#: Field samples per wavelength :func:`_resolved_n_steps` targets *inside* a
#: pumped edge.
#:
#: ``n_steps`` is a per-edge sample count, not a resolution, so what it buys
#: depends entirely on how many wavelengths fit in an edge -- and that varies by
#: two orders of magnitude between the graphs this solver is run on. On
#: ``line_PRA`` an edge is 0.92 wavelengths long, so the shipped default of 64
#: is 69 samples per wavelength and everything is converged. On the production
#: buffon the longest pumped edge is 50.9 wavelengths, so the same 64 is
#: **1.3 samples per wavelength**: the :math:`|E(x)|^2` that drives the hole
#: burning is aliased, and with it the pump-region norm that sets the amplitude
#: scale. The error this makes is not subtle, and it is measurable without a
#: reference solver -- SALT must reproduce the linear competition matrix's
#: onset slope :math:`1/T_{\mu\mu}` as :math:`D_0 \to D_0^{thr}`. On the
#: buffon's lowest mode (:math:`1/T_{\mu\mu} = 74.01`), solving at
#: :math:`D_0 = 1.005\,D_0^{thr}` and reading :math:`a/(D_0/D_0^{thr} - 1)`:
#:
#: ========= ================== ======= =======
#: n_steps   samples/wavelength  slope   ratio
#: ========= ================== ======= =======
#: 64         1.3                108.70  1.469
#: 128        2.5                 87.40  1.181
#: 256        5.0                 74.19  1.002
#: 512       10.1                 73.63  0.995
#: ========= ================== ======= =======
#:
#: -- a 47 % amplitude error at the default, gone once the edge is resolved,
#: and converged to <1 % by 5 samples per wavelength. 8 is that with margin.
#: The count is quoted per wavelength of the *field*; the intensity being
#: integrated has half the period, so it is 4 samples per intensity fringe.
#: On ``line_PRA`` the requirement is 8 steps, far below the default, so the
#: shipped fixture is untouched.
SALT_VARYING_SAMPLES_PER_WAVELENGTH = 8

#: Ceiling on the auto-raised ``n_steps``. The propagator cost is linear in it,
#: so this bounds the per-solve cost the way ``modes.oversample_node_cap``
#: bounds the oversampled path's. A graph that hits the cap is under-resolved
#: exactly as before -- raise it (or ``SALT_VARYING_SAMPLES_PER_WAVELENGTH``)
#: when the absolute amplitudes matter.
SALT_VARYING_N_STEPS_CAP = 4096


def _resolved_n_steps(graph, wavenumber, n_steps: int, pump=None) -> int:
    r"""``n_steps`` raised until a pumped edge is resolved at this ``wavenumber``.

    The requested value is treated as a *floor*: it is raised, never lowered, so
    a caller asking for more resolution than the wavelength needs still gets it.
    See :data:`SALT_VARYING_SAMPLES_PER_WAVELENGTH` for why the floor alone is
    not enough.

    Only pumped edges are measured: they are the ones that carry a varying
    profile, whose :math:`|E(x)|^2` enters the hole burning and whose
    :math:`\psi^2` enters the norm. Unpumped edges keep the exact closed form.
    """
    params = graph.graph["params"]
    spw = float(params.get("intensity_varying_samples_per_wavelength") or 0.0)
    if spw <= 0.0:
        spw = float(SALT_VARYING_SAMPLES_PER_WAVELENGTH)
    lengths = np.asarray(graph.graph["lengths"], dtype=float)
    index = np.sqrt(np.abs(np.asarray(params["dielectric_constant"])))
    c = float(params.get("c", 1.0) or 1.0)
    k = abs(float(np.real(wavenumber)))
    waves = lengths * index * k / (2.0 * np.pi * c)
    if pump is not None:
        pumped = np.asarray(pump, dtype=float) > 0.0
        if not pumped.any():
            return int(n_steps)
        waves = waves[pumped]
    needed = int(np.ceil(spw * float(np.max(waves)))) if len(waves) else 0
    return int(min(max(int(n_steps), needed), SALT_VARYING_N_STEPS_CAP))


def _smallest_eigenpair(matrix):
    """Smallest-magnitude eigenvalue of ``matrix`` and its eigenvector."""
    size = matrix.shape[0]
    if size > DENSE_NULL_MAX:
        try:
            values, vectors = sc.sparse.linalg.eigs(
                matrix.asfptype(),
                k=1,
                sigma=0,
                which="LM",
                return_eigenvectors=True,
                v0=np.random.default_rng(DENSE_NULL_SEED).normal(size=size),
            )
            return values[0], vectors[:, 0]
        except (sc.sparse.linalg.ArpackError, RuntimeError):
            # Shift-invert factorises the operator itself, which is exactly
            # singular at a converged lasing solution; fall back rather than
            # fail on the answer we were looking for.
            pass
    dense = np.asarray(matrix.todense()) if sc.sparse.issparse(matrix) else np.asarray(matrix)
    values, vectors = np.linalg.eig(dense)
    i = int(np.argmin(np.abs(values)))
    return values[i], vectors[:, i]


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


#: Fraction of the distance to the nearest candidate that a mode's ``k`` may
#: travel. This bounds the search to one *analytic branch* of the eigenvalue, not
#: merely to "nearer this mode than the next": ``_smallest_eigenpair`` returns
#: ``argmin |lambda|``, a min over branches, so past the point where two branches
#: cross, the eigenvalue being root-found belongs to the *neighbouring* mode and
#: is discontinuous there. On the production buffon that crossing was measured at
#: 0.395 of the spacing, so a 0.4 bound put it 3.1e-5 *inside* the box: the solve
#: slid onto the neighbour (eigenvector overlap 0.89 with the wrong mode), pinned
#: at the bound, and drove the amplitude to zero. 0.25 keeps a margin.
_BRANCH_SAFETY = 0.25

#: Bytes of resampling matrices :func:`_cubic_resampling_weights` may hold.
#: Once it is full nothing is evicted and nothing more is built -- the profiles
#: that missed keep using ``CubicSpline`` directly. Deterministic admission
#: rather than an LRU is the point: every residual evaluation sweeps *all* the
#: pumped edges in the same order, so an LRU that cannot hold them all would
#: evict exactly the entry needed next and rebuild every matrix every time.
#: A matrix is ``n_query * n_samples * 8`` bytes, so at the default
#: ``n_steps = 64`` this holds ~1000 edges' worth.
CUBIC_RESAMPLE_CACHE_BYTES = 64 * 1024 * 1024

_RESAMPLE_CACHE: dict[tuple[int, float, float, bytes], np.ndarray] = {}
_RESAMPLE_CACHE_BYTES = 0


def _cubic_resampling_weights(n_samples: int, lo: float, hi: float, query: bytes):
    """``W`` with ``W @ y == CubicSpline(linspace(lo, hi, n_samples), y)(x)``.

    Cubic-spline interpolation is *linear* in the sampled values, so for a fixed
    grid and fixed query points it is one matrix. Both are fixed here -- the
    grid is the edge's sample grid, and the query points are the Magnus
    abscissae, which depend only on the edge's length and ``n_steps`` -- while
    the values change at every residual evaluation. Rebuilding a ``CubicSpline``
    for each of those was the largest cost after the propagator itself.

    Feeding the spline the identity gets every column in one construction, so a
    cache miss costs one spline build rather than ``n_samples`` of them.

    Returns ``None`` when the matrix will not pay for itself -- it is
    ``O(n_samples^2)`` in both memory and apply cost, against ``O(n_samples)``
    for the spline, so past a few hundred samples per edge the caller is better
    off with the spline and is told so.
    """
    global _RESAMPLE_CACHE_BYTES

    key = (n_samples, lo, hi, query)
    cached = _RESAMPLE_CACHE.get(key)
    if cached is not None:
        return cached
    n_query = len(query) // 8
    size = n_query * n_samples * 8
    if _RESAMPLE_CACHE_BYTES + size > CUBIC_RESAMPLE_CACHE_BYTES:
        return None
    grid = np.linspace(lo, hi, n_samples)
    weights = np.asarray(
        CubicSpline(grid, np.eye(n_samples), axis=0)(np.frombuffer(query, dtype=float))
    )
    _RESAMPLE_CACHE[key] = weights
    _RESAMPLE_CACHE_BYTES += weights.nbytes
    return weights


class _SaturatedEdgeProfile:
    r"""``eps(x) = eps_edge + gamma(k) * D0_eff(x)`` on one edge.

    ``D0_eff`` is known on the edge's uniform sample grid and cubic-interpolated
    in between; ``gain`` (:math:`\gamma(k)`) is rebound by the caller, which is
    the only thing that changes as the solver moves ``k`` at fixed fields.
    """

    __slots__ = ("_d0_eff", "_eps", "_hi", "_lo", "_n_samples", "gain")

    def __init__(self, grid, d0_eff, eps_edge):
        self._d0_eff = np.asarray(d0_eff, dtype=float)
        self._eps = eps_edge
        self._lo, self._hi = float(grid[0]), float(grid[-1])
        self._n_samples = len(grid)
        self.gain = 0.0 + 0.0j  # set by the caller, which knows k

    def __call__(self, x):
        clipped = np.clip(np.asarray(x, dtype=float), self._lo, self._hi)
        flat = np.ascontiguousarray(np.ravel(clipped))
        weights = _cubic_resampling_weights(self._n_samples, self._lo, self._hi, flat.tobytes())
        if weights is None:
            grid = np.linspace(self._lo, self._hi, self._n_samples)
            return self._eps + self.gain * CubicSpline(grid, self._d0_eff)(clipped)
        return self._eps + self.gain * (weights @ self._d0_eff).reshape(np.shape(clipped))


def _spline_profile(grid, d0_eff, eps_edge, params):
    """Build eps(x) = eps_edge + gamma(k) D0_eff(x); gamma is bound at call time."""
    del params  # kept for signature stability; gamma is bound via `.gain`
    return _SaturatedEdgeProfile(grid, d0_eff, eps_edge)


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
    n_steps = _resolved_n_steps(graph, float(np.max(np.abs(ks))), n_steps, pump)
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
    """Complex ``lambda_1`` at ``k``, with the gain bound at this ``k``.

    ``k`` may be complex: the frozen-field solve only ever asks for real ``k``
    (a lasing mode is a root *on* the real axis), but the net-gain probe of
    :func:`net_gain_alpha` walks off it to read the candidate's ``alpha``.
    """
    gain = gamma(complex(k), graph.graph["params"])
    for profile in profiles:
        if profile is not None:
            profile.gain = gain
    value, _ = node_solution_varying(complex(k), graph, profiles, n_steps=n_steps)
    return value


#: Net-gain margin for admitting a candidate to the active set, mirroring
#: :data:`~netsalt.modes.SALT_GAIN_MARGIN`. Saturation pins an established
#: mode's alpha at ~0-, so the bar sits a hair below zero rather than at it.
SALT_VARYING_GAIN_MARGIN = -1e-6


def net_gain_alpha(
    graph,
    k0: float,
    profiles,
    n_steps: int = 64,
    *,
    k_window: float = 0.1,
    max_steps: int = 30,
) -> tuple[float, float]:
    r"""``(k, alpha)`` of the root nearest ``k0`` on a *given* saturated background.

    The cheap admission test. A candidate lases on top of the modes already
    active when the operator saturated by them still has a root with net gain,
    :math:`\alpha = -\mathrm{Im}\,k < 0`; this walks the candidate's root off the
    real axis and reports where it sits. It is the varying-operator counterpart
    of what :func:`~netsalt.modes._full_salt_newton_impl` does with
    ``_refine_local`` on a ``_saturated_graph_multi`` background.

    It replaces asking the same question by *solving* the enlarged set and
    seeing whether it holds together -- same answer, one local two-variable root
    find (tens of eigensolves) instead of a complete nonlinear solve (hundreds).

    Args:
        graph: quantum graph.
        k0: the candidate's current real frequency, and the root find's start.
        profiles: saturated per-edge permittivity callables, i.e. the background
            the incumbents have burnt. Their ``gain`` attribute is rebound at
            every probed ``k``, so pass the profiles, not a frozen operator.
        n_steps: sub-intervals per varying edge.
        k_window: how far the root may travel in ``Re k`` before the probe is
            judged to have left the candidate's branch, in which case ``k0`` is
            returned with ``alpha = +inf`` -- "no evidence of gain here" rather
            than a neighbouring mode's gain misattributed to this one.
        max_steps: MINPACK function-evaluation budget.

    Returns:
        ``(k, alpha)``. Admit when ``alpha`` is below
        :data:`SALT_VARYING_GAIN_MARGIN`.
    """
    from scipy.optimize import root

    def residual(x):
        # to_complex's convention: alpha is stored as -imag(k).
        value = _lam_varying(graph, complex(x[0], -x[1]), profiles, n_steps)
        return [value.real, value.imag]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = root(
            residual,
            np.array([float(k0), 0.0]),
            method="hybr",
            tol=0,
            options={"maxfev": int(max_steps), "xtol": 1e-9},
        )
    k, alpha = float(result.x[0]), float(result.x[1])
    if not np.isfinite(k) or not np.isfinite(alpha) or abs(k - k0) > k_window:
        return float(k0), np.inf
    return k, alpha


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
    k_window_cap: float | None = None,
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
        n_steps: sub-intervals per varying edge. A **floor**: it is raised to
            whatever resolves the within-edge wavelength (see
            :data:`SALT_VARYING_SAMPLES_PER_WAVELENGTH`), because the same
            per-edge count means 69 samples per wavelength on ``line_PRA`` and
            1.3 on the buffon.
        outer: field-refresh iterations.
        damping: mixing applied when refreshing the fields.
        residual_tol: convergence target on ``|lambda_1|``.
        max_nfev: budget for each frozen-field least-squares solve.
        k_window_cap: how far each ``k`` may move from its starting value.
            May be a scalar, or one value per mode -- a caller that knows the
            *whole* candidate set can bound each mode by its own nearest
            neighbour, which a solve seeing only the active set cannot do.
            ``None`` derives it from the mode spacing, the gain linewidth
            ``gamma_perp``, and a fraction of ``k`` itself. This is not
            cosmetic, and both failure modes it prevents report *success*:
            two active modes drifting onto the same ``k`` leave the system
            degenerate, so any split of intensity between them satisfies the
            equations and the residual converges while the per-mode amplitudes
            are arbitrary; and an unbounded single-mode solve walks to
            ``k ~ 0``, where the operator is degenerate and a 1e-8 residual is
            reported for something that is not a lasing mode.

    Returns:
        :class:`SaltVaryingSolution`.
    """
    from scipy.optimize import least_squares

    ks = np.asarray(ks, dtype=float).copy()
    amplitudes = np.asarray(amplitudes, dtype=float).copy()
    n_modes = len(ks)
    pump = np.asarray(pump, dtype=float)
    # ``n_steps`` is a per-edge sample count, so on a graph whose edges span many
    # wavelengths the requested value can be far below what the within-edge field
    # needs; raise it to the wavelength (see SALT_VARYING_SAMPLES_PER_WAVELENGTH).
    n_steps = _resolved_n_steps(graph, float(np.max(np.abs(ks))), n_steps, pump)

    # Bound how far each k may travel. Without this two active modes can drift
    # onto the *same* k, where the system is degenerate -- any split of intensity
    # between them satisfies lambda_1(k) = 0 for both -- so the residual converges
    # while the per-mode amplitudes are arbitrary, and consecutive pumps report
    # wildly different splits of the same total. The cap is a fraction of the
    # spacing between the modes being solved, which keeps each on its own branch.
    if k_window_cap is not None and np.ndim(k_window_cap) > 0:
        # per-mode caps, supplied by a caller that knows the whole candidate set
        caps = np.asarray(k_window_cap, dtype=float)
        if len(caps) != n_modes:
            raise ValueError(f"k_window_cap has {len(caps)} entries for {n_modes} modes.")
        k_lo, k_hi = ks - caps, ks + caps
        k_window_cap = float(np.min(caps))
    elif k_window_cap is None:
        cap = np.inf
        if n_modes > 1:
            cap = 0.2 * float(np.min(np.diff(np.sort(ks))))
        # Frequency pulling is bounded by the gain linewidth, so gamma_perp is a
        # physical ceiling on how far a lasing k can travel.
        gamma_perp = graph.graph["params"].get("gamma_perp")
        if gamma_perp:
            cap = min(cap, float(gamma_perp))
        # And never let k move by a large fraction of itself. Unbounded, a
        # single-mode solve walks to k ~ 0, where the operator is degenerate:
        # it reports a 1e-8 residual and "converged" for a root that is not a
        # lasing mode at all.
        cap = min(cap, 0.1 * float(np.min(np.abs(ks))))
        k_window_cap = float(np.clip(cap, 1e-6, np.inf))
        k_lo, k_hi = ks - k_window_cap, ks + k_window_cap
    else:
        k_lo, k_hi = ks - k_window_cap, ks + k_window_cap

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
        x0[0::2] = np.clip(ks, k_lo, k_hi)
        x0[1::2] = np.maximum(amplitudes, 0.0)
        lower = np.empty(2 * n_modes)
        upper = np.empty(2 * n_modes)
        lower[0::2], upper[0::2] = k_lo, k_hi
        # a >= 0 as a real bound rather than a clip after the fact: clipping lets
        # the solver explore negative amplitudes and converge to a state that is
        # then silently altered.
        lower[1::2], upper[1::2] = 0.0, np.inf
        result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            method="trf",
            max_nfev=max_nfev,
            xtol=1e-12,
            # x_scale="jac" is load-bearing, not a tuning knob. The residual's
            # sensitivity to k and to a differ by ~6e5 on the production buffon
            # (|dlam/dk| = 8.2e2 against |dlam/da| = 1.3e-3), because the
            # amplitude's unit is set by the pump-region norm: mean |E|^2 is
            # 5.4e-4 over 2500 units of pumped length there against 1.04 over 1.0
            # on line_PRA, so the natural amplitude is ~1900x larger. With the
            # default isotropic x_scale=1.0 the trust region takes steps sized
            # for k, which are useless for a, and the amplitude never leaves its
            # initial guess -- the buffon converged to a = 6.29 where the root is
            # at a = 343, and looked like a physically impossible answer rather
            # than an unmoved one.
            x_scale="jac",
        )
        ks = result.x[0::2]
        amplitudes = np.maximum(result.x[1::2], 0.0)

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


def compute_modal_intensities_varying(
    graph,
    modes_df,
    max_pump_intensity: float,
    D0_steps: int = 10,
    *,
    n_steps: int = 64,
    outer: int = 25,
    residual_tol: float = SALT_VARYING_RESIDUAL_TARGET,
):
    r"""Above-threshold L--I curves on the un-oversampled operator.

    The counterpart of
    :func:`~netsalt.modes.compute_modal_intensities_full_salt_newton`, carried on
    :func:`~netsalt.varying_laplacian.construct_laplacian_varying` so the graph
    is never subdivided: the matrix stays ``len(graph)`` square while the
    within-edge hole burning is resolved by per-edge transfer matrices. On the
    production buffon that is a 208x208 matrix at 18.5 samples per wavelength,
    against ~76600 nodes and 0.47 samples per wavelength for the oversampled
    path (issues #52, #53).

    The pump is stepped from the first threshold to ``max_pump_intensity``, and
    each solve starts from the previous one. That continuation is load-bearing:
    solving each pump from a fresh guess makes the solver land on *different*
    modes, each a genuine root but not the same branch.

    Args:
        graph: quantum graph, **not** oversampled.
        modes_df: must carry ``threshold_lasing_modes`` and
            ``lasing_thresholds``; modified in place and returned.
        max_pump_intensity: largest ``D0``.
        D0_steps: pump grid points.
        n_steps: sub-intervals per varying edge; a floor, raised to resolve the
            within-edge wavelength (:data:`SALT_VARYING_SAMPLES_PER_WAVELENGTH`).
            The value actually used is reported in
            ``attrs["salt_varying_n_steps"]``.
        outer: field-refresh iterations per solve.
        residual_tol: convergence target on ``|lambda_1|``.

    Returns:
        ``modes_df`` with ``("modal_intensities", D0)`` columns, plus
        ``attrs["salt_varying_diagnostics"]`` -- per pump: ``D0``, ``n_active``,
        ``converged``, ``max_residual``, ``iterations`` -- so a run reports what
        it actually achieved rather than only its answer.

    Note:
        Two things make the multimode sweep correct, and both were found by
        watching it be wrong first.

        *The active set is self-consistent.* A set is accepted only when every
        member both lases (amplitude above the floor) and satisfies its own
        ``lambda_1(k) = 0``; failing members are dropped, and candidates are
        admitted one at a time and kept only if the enlarged set still holds.
        Admitting on the *linear* threshold instead -- the obvious thing --
        hands the solver modes that cannot lase and forces an impossible
        equation on them: residuals near 1.7 at every pump instead of 1e-7.

        Candidates are screened by :func:`net_gain_alpha` before that solve:
        a mode the incumbents have burnt below threshold has ``alpha > 0`` on
        their saturated background, and would come back from the enlarged solve
        with ``a ~ 0`` and be rejected. Reading its ``alpha`` costs one local
        root find instead of a whole nonlinear solve, and that is the difference
        between an ``O(candidates)`` sequence of full solves per pump and a
        single one. On ``line_PRA`` it takes the sweep from 35 solves to 8, with
        a bit-identical answer -- on the converged ``D0 = 4`` background the two
        lasing modes sit at ``alpha = +5e-9`` and ``+2e-9`` (gain clamped, as
        SALT requires) while the four non-lasing candidates sit at ``+1e-2`` to
        ``+2e-1``, so the screen is nowhere near a close call.

        *Each mode's ``k`` is bounded* (see ``k_window_cap`` in
        :func:`solve_salt_varying`). Without it two active modes drift onto the
        same ``k``, where the system is degenerate: any split of intensity
        between them satisfies the equations, so the residual converges to 1e-7
        while the per-mode amplitudes are arbitrary and consecutive pumps report
        different splits of the same total. That failure is invisible in the
        residual and in the summed output -- only the per-mode curve shows it.

        With both in place, ``line_PRA`` gives two lasing modes with strictly
        monotone L--I curves and residuals of 3e-7..8e-7 at every pump -- the
        two-mode count being the published Ge-Chong-Stone result.
    """
    import pandas as pd

    pump = np.asarray(graph.graph["params"]["pump"], dtype=float)
    thresholds = np.asarray(modes_df["lasing_thresholds"]).ravel()
    threshold_modes = modes_df["threshold_lasing_modes"].to_numpy()
    finite = np.where(np.isfinite(thresholds))[0]
    if not len(finite):
        raise ValueError("no mode has a finite lasing threshold")
    first = float(np.min(thresholds[finite]))

    # Bound each candidate by its own nearest neighbour among *all* candidates.
    # solve_salt_varying only sees the active set, so with one mode active it has
    # no spacing to work from and falls back to gamma_perp -- on a dense spectrum
    # that is enormous: the buffon's candidates sit 7.6e-4 apart against a 0.5
    # fallback, 658x the spacing, and a single active mode wanders across every
    # candidate (modes appear to switch *off* as the pump rises).
    #
    # A single global cap from the *minimum* spacing is the obvious fix and is
    # also wrong: it punishes well-separated modes for a near-degenerate pair
    # elsewhere in the window. On the buffon two candidates are 7.6e-4 apart
    # while the others are ~20x further out, and the global cap left nothing able
    # to lase at all. Per-mode is what the geometry actually calls for.
    cand_ks = np.array([float(np.real(threshold_modes[i])) for i in finite])
    n_steps = _resolved_n_steps(graph, float(np.max(np.abs(cand_ks))), n_steps, pump)
    # The spacing must come from the *Weyl law*, not from the candidate list.
    # That list is only as complete as the mode search behind it, and on the
    # production buffon it is not remotely complete: Weyl gives ~120 modes in
    # k = [10.63, 10.73] (mean spacing pi / L_opt = 8.4e-4) where the fixture
    # carries 4. Keying the bound off those 4 gives ~3e-4 -- a third of the true
    # spacing -- so the box still holds whole *unlisted* modes, and the solver
    # converges to genuine 1e-7 roots belonging to a different mode. A residual
    # pins k to 2.5e-10 on this graph and says nothing about being on the
    # intended branch. Taking the min of the two keeps the bound honest whether
    # or not the list is complete.
    mean_spacing = np.pi / max(optical_length(graph), 1e-12)
    separation = np.full(len(cand_ks), mean_spacing)
    if len(cand_ks) > 1:
        pairwise = np.abs(cand_ks[:, None] - cand_ks[None, :])
        np.fill_diagonal(pairwise, np.inf)
        separation = np.minimum(separation, pairwise.min(axis=1))
    gamma_perp = graph.graph["params"].get("gamma_perp")
    ceiling = float(gamma_perp) if gamma_perp else np.inf
    caps = {
        int(i): float(np.clip(_BRANCH_SAFETY * d, 1e-9, ceiling))
        for i, d in zip(finite, separation, strict=True)
    }

    grid = np.linspace(first, float(max_pump_intensity), int(D0_steps))
    intensities = pd.DataFrame(index=modes_df.index)
    state: dict[int, tuple[float, float]] = {}
    active: list[int] = []
    rows = []

    def attempt(candidate_set, D0):
        """Solve for a candidate active set; return ``(solution, ok)``.

        A set is accepted only when every member actually lases -- amplitude
        above the lasing floor *and* its own residual at target. That is what
        makes the set self-consistent: a mode that cannot satisfy
        ``lambda_1(k) = 0`` is not a lasing mode, and leaving it in forces an
        impossible equation on the least-squares solve, which is what made the
        linear-threshold active set thrash.

        A rejected solve is returned rather than discarded: its amplitudes are
        exactly what the shrink step needs to name the weakest member, and
        re-running the identical (deterministic) solve to find that out was
        doubling the cost of every drop.
        """
        if not candidate_set:
            return None, False
        ks0, a0 = [], []
        for i in candidate_set:
            k_prev, a_prev = state.get(i, (float(np.real(threshold_modes[i])), 1e-2))
            ks0.append(k_prev)
            a0.append(max(a_prev, 1e-3))
        solution = solve_salt_varying(
            graph,
            ks0,
            a0,
            float(D0),
            pump,
            n_steps=n_steps,
            outer=outer,
            residual_tol=residual_tol,
            k_window_cap=(
                [caps[i] for i in candidate_set] if all(i in caps for i in candidate_set) else None
            ),
        )
        ok = all(
            solution.amplitudes[slot] > SALT_VARYING_LASING_AMPLITUDE
            and solution.residuals[slot] <= residual_tol
            for slot in range(len(candidate_set))
        )
        return solution, ok

    def has_net_gain(cand, current, solution, D0):
        """Does ``cand`` still see gain on what ``current`` has already burnt?

        The admission test. Asking it by *solving* the enlarged set costs a
        complete nonlinear solve per candidate per pump, which is what made this
        sweep unaffordable; the same question is answered by one local root find
        on the incumbents' saturated background (:func:`net_gain_alpha`).
        A candidate the background has driven below threshold converges to
        ``a ~ 0`` in the enlarged solve, i.e. it is rejected -- so skipping it is
        skipping a solve whose outcome is already known.
        """
        if solution is None or not current:
            return True  # nothing has saturated anything yet; there is no background
        profiles = saturated_eps_profiles(
            graph, solution.ks, solution.amplitudes, solution.fields, D0, pump
        )
        k_cand = state.get(cand, (float(np.real(threshold_modes[cand])), 0.0))[0]
        gap = min(abs(k_cand - float(k)) for k in solution.ks)
        window = float(np.clip(0.2 * gap, 1e-6, 0.3))
        _, alpha = net_gain_alpha(graph, k_cand, profiles, n_steps=n_steps, k_window=window)
        return alpha < SALT_VARYING_GAIN_MARGIN

    for D0 in grid:
        # 1. shrink: drop members that no longer lase at this pump
        current, solution = list(active), None
        while current:
            solution, ok = attempt(current, D0)
            if ok:
                break
            # the set does not hold together: drop its weakest member and retry
            current.pop(int(np.argmin(solution.amplitudes)))
            solution = None

        # 2. grow: admit candidates one at a time, keeping only those that hold up
        for cand in sorted(finite, key=lambda i: thresholds[i]):
            if cand in current or thresholds[cand] >= D0:
                continue
            if not has_net_gain(cand, current, solution, D0):
                continue
            trial, ok = attempt([*current, cand], D0)
            if ok:
                current = [*current, cand]
                solution = trial

        active = current
        column = np.zeros(len(modes_df.index))
        if solution is not None and active:
            for slot, i in enumerate(active):
                state[i] = (float(solution.ks[slot]), float(solution.amplitudes[slot]))
                column[i] = float(solution.amplitudes[slot])
        intensities[D0] = column
        rows.append(
            {
                "D0": float(D0),
                "n_active": len(active),
                "converged": bool(solution.converged) if solution is not None else True,
                "max_residual": (
                    float(np.max(solution.residuals)) if solution is not None else 0.0
                ),
                "iterations": int(solution.iterations) if solution is not None else 0,
            }
        )

    diagnostics = pd.DataFrame(rows)
    unconverged = diagnostics[~diagnostics["converged"] & (diagnostics["n_active"] > 0)]
    if len(unconverged):
        warnings.warn(
            f"compute_modal_intensities_varying: {len(unconverged)} of "
            f"{int((diagnostics['n_active'] > 0).sum())} pumps did not converge "
            f"(worst residual {unconverged['max_residual'].max():.3g} against a "
            f"{residual_tol:g} target). The intensities at those pumps are not "
            "solutions of the SALT equations -- see attrs['salt_varying_diagnostics'] "
            "and the function's docstring.",
            stacklevel=2,
        )

    if "modal_intensities" in modes_df:
        del modes_df["modal_intensities"]
    for D0 in sorted(intensities.columns):
        modes_df["modal_intensities", np.around(D0, 8)] = intensities[D0]
    modes_df.attrs["salt_varying_diagnostics"] = diagnostics
    modes_df.attrs["salt_varying_n_steps"] = int(n_steps)
    return modes_df

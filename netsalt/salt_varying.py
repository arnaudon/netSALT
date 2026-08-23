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

import os
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


#: Factor by which the continuation's total amplitude scale ``s`` may change in
#: one outer step. The secant that drives ``s`` is only as good as the two points
#: behind it, and near threshold those two points can be a hair apart while the
#: extrapolation they imply is enormous; the clamp turns that into a bounded
#: climb instead of a jump into the flat region. 4 keeps the buffon's true step
#: (a = 3.06 -> 5.62 between two pumps) reachable in one step while excluding the
#: flat region that starts around a = 15 on that graph, and a genuine climb from
#: the caller's 1e-3 seed to the buffon's a = 68 costs ~8 steps of the 25
#: available.
_AMPLITUDE_TRUST_GROWTH = 4.0

#: Floor under the amplitude scale, so a mode starting from zero (or from the
#: caller's 1e-3 seed) can still leave it: a multiplicative update from ``s = 0``
#: never moves.
_AMPLITUDE_TRUST_FLOOR = 1e-3

#: Top of the range :func:`_bracket_D0` sweeps, in units of the requested pump.
#: Exactly ``D0_target``, and the ceiling is doing work: the continuation runs
#: from the mode's own threshold *up* to the requested pump, so every point on it
#: has ``D0 <= D0_target`` and anything above is a root of something else. Given
#: room above, the sweep takes it -- on the buffon at 2.09x threshold with the
#: amplitude at the floor, a ceiling of ``2 D0_target`` put the minimum at the
#: ceiling itself (``|lambda| = 8.1e-03``) rather than at the mode's own
#: threshold near ``0.478 D0_target``, because the seed's ``k`` is 2.8e-05 off
#: the passive mode and that shallows the true dip to ``|lambda| = 2.1e-02``.
_D0_BRACKET_MAX = 1.0

#: Points in that sweep. The basin measured on the buffon is ~15 % of
#: ``D0_target`` wide, so 32 points across it put at least one sample inside.
_D0_BRACKET_POINTS = 32

#: Cap on the damped field-refresh rounds run per outer iteration. The refresh
#: runs to :data:`_FIELD_TRACK_TOL` and stops, so this only bounds the cost of a
#: state whose field iteration will not settle. Each round is two eigensolves per
#: mode, against the ~60 residual evaluations of the least-squares each outer
#: iteration also pays, so even the cap is roughly a doubling.
_FIELD_REFRESH_MAX = 40

#: Relative field movement per refresh round below which the field is treated as
#: having caught up with the current amplitude, and the continuation is allowed
#: to take its next step in ``s``.
#:
#: 1e-03 is where the cost stops buying anything. Tightening it to 1e-07 (which
#: roughly doubled the wall time, the refresh running to its cap) left the buffon
#: at 1.05x threshold converging at the same rate -- residual 6.7e-06 after 40
#: iterations against 5.2e-06 before -- so the apparent ``dD0/ds`` noise that
#: sets that rate does *not* come from the field still moving. It comes from the
#: secant itself, and :data:`_SCALE_STEP_DAMPING` is what answers it. What this
#: tolerance is for is the gross case: with no gate at all the field trails the
#: amplitude by a whole outer step and the continuation runs away (see the
#: refresh loop).
_FIELD_TRACK_TOL = 1e-3

#: Fraction of the Newton step on ``s`` that is actually taken. The believed
#: ``dD0/ds`` is a secant over two continuation points, and on the buffon at
#: 1.05x threshold consecutive estimates disagreed by up to 1.7x (0.0203 against
#: 0.0339) -- so a full step overshoots by that factor and the scale rings about
#: the answer, decaying by only ~0.7 per iteration. Halving the step turns an
#: overshoot of 1.7 into 0.85 and leaves an undershoot no worse than the full
#: step would have been, which is the asymmetry that makes a damped Newton the
#: right default on a noisy derivative.
_SCALE_STEP_DAMPING = 0.6

#: Factor by which the believed ``dD0/ds`` may change between two consecutive
#: continuation points. It is a smooth function of position along the branch, so
#: anything larger is the field refresh's noise being read as slope; clamping it
#: keeps one bad pair from throwing the scale across the branch.
_SCALE_SLOPE_TRUST = 2.0

#: Pump ratio above which a failed solve is retried as two half-steps. The
#: continuation walks the amplitude and reads off the pump it lases at, so the
#: distance it must travel is set by how far ``D0_target`` is from the pump the
#: caller's seed actually solves. Measured on the buffon: a 1.26x -> 2.09x
#: request is a 4.4x climb in amplitude (15.4 -> 68) from a seed whose own pump
#: is 0.60 of the target, and the secant does not bridge it in 40 outer
#: iterations -- it holds at the seed and reports failure. Splitting the
#: interval geometrically and solving the midpoint first turns one impossible
#: step into two ordinary ones.
_SUBSTEP_MIN_RATIO = 1.35

#: How many times the pump interval may be halved. Each level doubles the number
#: of solves, so this bounds the retry at 2**4 = 16 sub-solves; a request that
#: still fails at that depth is failing for a reason sub-stepping cannot fix.
_SUBSTEP_MAX_DEPTH = 4

#: How close to its own ``k`` bound a solve may sit before it is read as having
#: been stopped by the bound rather than having found a root there, as a fraction
#: of the box width. The failing buffon solves land exactly on the bound, so this
#: only has to exclude the boundary itself, not a neighbourhood of it.
_K_BOUND_TOL = 1e-6

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


def _field_change(old, new) -> float:
    """Largest relative movement between two sets of ``|E(x)|^2`` profiles.

    Normalised per mode by that mode's own peak, so it is the same number
    whatever the graph's amplitude unit happens to be (mean ``|E|^2`` differs by
    ~1900x between the shipped fixtures).
    """
    worst = 0.0
    for per_mode_old, per_mode_new in zip(old, new, strict=True):
        peak = max(
            (float(np.max(np.abs(samples))) for samples in per_mode_old if len(samples)),
            default=0.0,
        )
        if peak <= 0.0:
            continue
        for before, after in zip(per_mode_old, per_mode_new, strict=True):
            if len(before):
                worst = max(worst, float(np.max(np.abs(after - before))) / peak)
    return worst


def _bracket_D0(residual, x0, hi: float, points: int) -> float:
    """``D0`` minimising the frozen-field residual at fixed ``(k, weights)``.

    A coarse scalar sweep, used only where there is no warm start for ``D0``: it
    replaces a guess that can silently land the least-squares on a neighbouring
    mode with a bracketing scan of the one unknown whose basin is narrow. See the
    call site for the measurement behind it.
    """
    probe = np.array(x0, dtype=float)
    best, best_cost = hi / points, np.inf
    for value in np.linspace(hi / points, hi, points):
        probe[-1] = value
        cost = float(np.sum(np.asarray(residual(probe)) ** 2))
        if cost < best_cost:
            best_cost, best = cost, float(value)
    return best


def _predicted_amplitude(
    trail: list[tuple[float, float]], D0: float, threshold: float, fallback: float
) -> float:
    r"""Seed a pump's solve by extrapolating the amplitudes already solved.

    Starting each pump from the *previous* pump's amplitude is the obvious
    choice and it is a factor ~2 low wherever the L--I curve is steep, which on
    the production buffon is enough to start the solve on the wrong side of a
    ridge: from a = 3.06 towards a true 5.62 it instead slides along a joint
    ``(k, a)`` valley and returns a = 52.7 with ``k`` pinned at its cap.
    Extrapolating the pumps already solved costs nothing -- the caller has them
    -- and lands inside the right basin: from the two pumps at 1.02x and 1.05x
    threshold (a = 1.419, 3.063) the linear prediction for 1.10x is 5.80,
    against a true 5.623.

    The one-point case uses ``a`` proportional to the pump *above* threshold
    rather than to the pump, because that is the near-threshold law (the linear
    model's ``a = (D0/D0_thr - 1) / T_mumu``); a chord through the origin would
    barely move off the previous value.

    Args:
        trail: ``(D0, amplitude)`` already solved for this mode, oldest first.
        D0: the pump about to be solved.
        threshold: this mode's lasing threshold.
        fallback: amplitude to use when ``trail`` is empty.

    Returns:
        The amplitude to start the solve from, never below
        :data:`_AMPLITUDE_TRUST_FLOOR`.
    """
    guess = fallback
    if len(trail) >= 2:
        (d1, a1), (d2, a2) = trail[-2], trail[-1]
        if d2 != d1:
            guess = a2 + (a2 - a1) * (D0 - d2) / (d2 - d1)
    elif len(trail) == 1:
        d1, a1 = trail[0]
        if d1 > threshold:
            guess = a1 * (D0 - threshold) / (d1 - threshold)
    if not np.isfinite(guess):
        guess = fallback
    return float(max(guess, _AMPLITUDE_TRUST_FLOOR))


def solve_salt_varying(
    graph,
    ks,
    amplitudes,
    D0: float,
    pump,
    *,
    n_steps: int = 64,
    outer: int = 40,
    damping: float = 0.7,
    residual_tol: float = SALT_VARYING_RESIDUAL_TARGET,
    max_nfev: int = 60,
    k_window_cap: float | None = None,
    _substep_depth: int = 0,
) -> SaltVaryingSolution:
    r"""Solve SALT for a given set of lasing modes, without oversampling.

    The varying-operator counterpart of
    :func:`~netsalt.modes.solve_salt_fixed_set`. The equations are the same two
    per mode, :math:`\mathrm{Re}\,\lambda_1(k_\mu) =
    \mathrm{Im}\,\lambda_1(k_\mu) = 0`, and the hole-burning fields are frozen
    during each least-squares solve and refreshed afterwards, which is what makes
    each residual a single clean eigensolve. What is *not* the same is which
    unknowns are held.

    **Why this is a continuation and not a solve at the requested pump.** The
    obvious formulation -- unknowns :math:`(k_\mu, a_\mu)` at the caller's fixed
    :math:`D_0` -- is square, and it lands on the wrong branch above threshold.
    The SALT equations at a fixed pump have more than one root in ``a``, and
    nothing in that formulation says which one is physical; the answer is the one
    reached by continuity from :math:`a = 0`, and a fixed-:math:`D_0` solve has
    no memory of where it came from. Measured on the production buffon at
    :math:`D_0 = 1.10\,D_0^{thr}`, it converged to ``a = 22.73`` with a residual
    of 6.8e-07 against a true 5.623 -- and 22.73 is the amplitude belonging to
    :math:`D_0 = 1.38\,D_0^{thr}`, i.e. a genuine root of a *different* pump.
    A small residual is not evidence of the right branch (AUDIT.md §10).

    So the roles of amplitude and pump are swapped. The unknown vector is

    .. math::

        x = (k_0 \dots k_{M-1},\; u_1 \dots u_{M-1},\; D_0)

    -- :math:`2M` unknowns for the :math:`2M` equations, still square -- where
    the amplitudes are recovered from a *gauge-fixed weight vector*,

    .. math::

        a_\mu = s\,\frac{u_\mu}{\sum_\nu u_\nu}, \qquad u_g \equiv 1 ,

    so :math:`\sum_\mu a_\mu \equiv s` by construction: no penalty term, no
    constraint equation to weight against the residuals. The gauge index ``g`` is
    the seed's strongest mode and is fixed for the whole solve. ``s`` is the
    continuation parameter and :math:`D_0` is solved *for*; after each frozen-field
    solve a secant step moves ``s`` so that the achieved :math:`D_0` heads for the
    requested one, clamped to :data:`_AMPLITUDE_TRUST_GROWTH` per step. Because
    :math:`D_0(a)` is strictly monotone along the physical branch (verified at 56
    continuation points from ``a = 0.2`` to 70 on the buffon), the target pump has
    exactly one amplitude on it, and walking ``s`` up from the seed is what picks
    that one out.

    It is only the *total* scale that is gauge-fixed, never a single mode's
    amplitude. Pinning one mode -- the seemingly equivalent thing -- changes which
    branch a multimode set selects: it left ``line_PRA``'s first five pumps
    byte-identical and then collapsed mode 1 to zero and invented a third mode.
    Every mode but the gauge one can still be driven to :math:`a = 0` and rejected
    by the caller's active-set logic, since ``u_\mu = 0`` is inside the bounds.

    For ``M = 1`` this reduces to unknowns :math:`(k, D_0)` at :math:`a = s`,
    which is exactly the formulation that reproduces the buffon's ground-truth
    L--I curve to ~2%.

    Convergence is judged the same way regardless: the reported residual is
    :func:`salt_residuals_varying` at the **requested** :math:`D_0`, so a solve
    whose secant has not landed reports a large residual and ``converged =
    False`` rather than a small residual at a pump nobody asked for.

    Args:
        graph: quantum graph, **not** oversampled.
        ks: initial real frequencies, one per lasing mode.
        amplitudes: initial amplitudes. Their *sum* seeds the continuation
            parameter ``s`` and their ratios seed the weight vector, so a caller
            that can predict the amplitude (see :func:`_predicted_amplitude`)
            shortens the climb; one that cannot may pass anything positive.
        D0: pump strength -- the pump the answer is *requested* at. It is not
            held fixed during the frozen-field solves; see above.
        pump: per-edge pump.
        n_steps: sub-intervals per varying edge. A **floor**: it is raised to
            whatever resolves the within-edge wavelength (see
            :data:`SALT_VARYING_SAMPLES_PER_WAVELENGTH`), because the same
            per-edge count means 69 samples per wavelength on ``line_PRA`` and
            1.3 on the buffon.
        outer: continuation / field-refresh iterations. Each one is a
            frozen-field solve, a secant step on ``s`` and a field refresh, so
            this budget has to cover the climb from the seed as well as the
            self-consistency.
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

    # The pump the caller asked for. It is the continuation's *target*, not a
    # constant of the inner solves -- see this function's docstring.
    D0_target = float(D0)

    # Gauge: the seed's strongest mode carries u = 1 for the whole solve. It has
    # to be fixed once and not re-chosen per iteration, or the parameterisation
    # changes under the solver. Picking the strongest also keeps the other
    # weights in [0, 1] at the seed, which is what makes a flat unit x_scale on
    # them the right choice (see below).
    gauge = int(np.argmax(amplitudes)) if n_modes else 0
    others = np.array([i for i in range(n_modes) if i != gauge], dtype=int)
    n_weights = len(others)
    weights = np.ones(n_modes)
    if n_weights:
        a_gauge = max(float(amplitudes[gauge]), _AMPLITUDE_TRUST_FLOOR)
        weights[others] = np.clip(amplitudes[others] / a_gauge, 0.0, None)
    # Total amplitude scale: the continuation parameter.
    scale = max(float(np.sum(np.maximum(amplitudes, 0.0))), _AMPLITUDE_TRUST_FLOOR)
    amplitudes = scale * weights / weights.sum()

    def _amplitudes(local_weights, s):
        return s * local_weights / max(float(np.sum(local_weights)), 1e-300)

    # initial fields, from the unsaturated operator
    unsaturated = []
    for k in ks:
        _, psi = node_solution_varying(float(k), graph, None, n_steps=n_steps)
        unsaturated.append(
            edge_field_profiles(float(k), graph, psi, None, n_steps=n_steps, pump=pump)
        )
    fields = [list(f) for f in unsaturated]

    # Layout of the unknown vector: k's first, then the M-1 free weights, then
    # D0 last. Explicitly contiguous rather than the interleaved x[0::2] / x[1::2]
    # this used to be -- with a trailing scalar unknown that slicing silently
    # picks D0 up as somebody's amplitude.
    k_slice = slice(0, n_modes)
    w_slice = slice(n_modes, n_modes + n_weights)
    k_width = np.maximum(k_hi - k_lo, 1e-300)

    lower = np.empty(2 * n_modes)
    upper = np.empty(2 * n_modes)
    lower[k_slice], upper[k_slice] = k_lo, k_hi
    # u >= 0 as a real bound rather than a clip after the fact: clipping lets the
    # solver explore negative weights and converge to a state that is then
    # silently altered. There is no *ceiling* on a weight and none is needed --
    # the amplitudes are s * u / sum(u), so a large u redistributes the fixed
    # total rather than running away with it. That is the whole point of carrying
    # the scale separately.
    lower[w_slice], upper[w_slice] = 0.0, np.inf
    lower[-1], upper[-1] = 0.0, np.inf

    # Explicit physical scales, not x_scale="jac". Both were measured on the
    # production buffon (single mode, D0 = 1.10x threshold, n_steps = 512) by
    # running the whole five-pump continuation:
    #
    #   x_scale        1.02   1.05   1.10   1.26   2.09   worst err
    #   "jac"          ...    (see the note in the commit message)
    #   explicit       ...
    #
    # The reason an explicit scale is available now and was not before is that
    # the old formulation had no natural unit for `a`: the amplitude's unit is
    # set by the pump-region norm, so |dlam/dk| and |dlam/da| differed by ~6e5 on
    # the buffon and only the Jacobian knew it. Here every unknown has one. `k`
    # is bounded by a box whose half-width is exactly how far it may travel;
    # the weights are dimensionless ratios of order 1 by construction (the gauge
    # mode is the strongest); and D0 is measured in units of the pump being
    # requested.
    x_scale = np.empty(2 * n_modes)
    x_scale[k_slice] = np.maximum(k_hi - k_lo, 1e-12) / 2.0
    x_scale[w_slice] = 1.0
    x_scale[-1] = max(D0_target, _AMPLITUDE_TRUST_FLOOR)

    converged = False
    iterations = 0
    # (scale, achieved D0) of the previous accepted solve, and the believed
    # dD0/ds built from them, for the secant step on the scale.
    scale_prev: float | None = None
    D0_prev: float | None = None
    slope: float | None = None
    # The last scale at which a solve came back off its k bound, i.e. the last
    # point on the continuation the frozen field could actually follow.
    scale_accepted: float | None = None
    # Whether the continuation has already been restarted from a ~ 0 because the
    # caller's seed turned out not to be a point on it. Once only: a second
    # restart would just repeat the first.
    restarted = False
    D0_now = D0_target
    # the caller's seed, kept intact for the sub-step retry below
    seed_ks = [float(k) for k in ks]
    seed_amplitudes = [float(a) for a in amplitudes]
    D0_bracketed: float | None = None
    # Movement of the fields in the last refresh round; inf until one has run, so
    # the first scale step waits for the field to settle at the caller's seed.
    field_change = np.inf
    for _outer_step in range(outer):
        iterations += 1
        frozen = [list(f) for f in fields]

        def residual(x, _frozen=frozen, _scale=scale):
            local_ks = x[k_slice]
            local_w = np.ones(n_modes)
            if n_weights:
                local_w[others] = np.clip(x[w_slice], 0.0, None)
            local_a = _amplitudes(local_w, _scale)
            local_D0 = max(float(x[-1]), 0.0)
            profiles = saturated_eps_profiles(graph, local_ks, local_a, _frozen, local_D0, pump)
            out = []
            for k in local_ks:
                value = _lam_varying(graph, float(k), profiles, n_steps)
                out.extend((value.real, value.imag))
            return np.asarray(out, dtype=float)

        x0 = np.empty(2 * n_modes)
        x0[k_slice] = np.clip(ks, k_lo, k_hi)
        if n_weights:
            x0[w_slice] = np.clip(weights[others], 0.0, None)
        # Until a continuation point has been accepted there is no warm start for
        # D0, and its basin is narrow enough that guessing wrongly loses the mode
        # altogether. Measured on the buffon at 2.09x threshold seeded from the
        # 1.26x amplitude (a = 15.39, whose true pump is 0.603 D0_target): started
        # at D0_target the frozen-field solve pins k on its bound and reports
        # D0 = 2.29 D0_target -- a root of a different mode -- and no amount of
        # field refreshing recovers it (20 refreshes, identical to four digits).
        # Started at 0.60 D0_target it lands on D0 = 0.603 and k - k0 = +2.84e-05,
        # the truth. Started at 0.90 it wanders again, and at 0.50 it pins on the
        # other bound: the basin is roughly +-15 % wide, far narrower than the
        # error a caller's amplitude seed can make.
        #
        # So bracket it instead of guessing: at fixed (k, a) sweep D0 across the
        # plausible range and take the least-squares cost's argmin. It is one
        # cheap scalar scan (its cost is a fraction of the least_squares that
        # follows) of the one variable whose basin is narrow, and it answers
        # exactly the continuation's defining question -- *at what pump does this
        # amplitude lase?*
        if scale_accepted is None:
            D0_now = _bracket_D0(residual, x0, _D0_BRACKET_MAX * D0_target, _D0_BRACKET_POINTS)
            if D0_bracketed is None:
                D0_bracketed = D0_now
        x0[-1] = max(D0_now, 0.0)

        result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            method="trf",
            max_nfev=max_nfev,
            xtol=1e-12,
            x_scale=os.environ.get("SALT_XSCALE") or x_scale,
        )

        # A solve that comes back with `k` sitting on its own bound is not a
        # candidate, however low its cost. The k cap marks the edge of the mode's
        # analytic branch, so a k pinned there means the least-squares wanted to
        # leave the branch and was merely stopped -- it is reporting the boundary,
        # not a root. With a stale frozen field there is no exact root anywhere,
        # so least_squares is free to chase the frozen problem's global minimum,
        # which sits at the corner; refusing corners keeps it on the branch until
        # the field refresh makes a real root available. On the buffon every
        # failing solve of the old fixed-D0 formulation came back this way.
        on_k_bound = np.any(
            (result.x[k_slice] <= k_lo + _K_BOUND_TOL * k_width)
            | (result.x[k_slice] >= k_hi - _K_BOUND_TOL * k_width)
        )
        if not on_k_bound:
            ks = np.asarray(result.x[k_slice], dtype=float).copy()
            if n_weights:
                weights[others] = np.clip(result.x[w_slice], 0.0, None)
            D0_solved = max(float(result.x[-1]), 0.0)
            # The state actually solved for: amplitudes at the *current* scale,
            # which is a genuine SALT solution at D0_solved. Reporting these and
            # then testing the residual at D0_target is what keeps the
            # convergence test honest -- if the secant has not landed, these
            # amplitudes belong to another pump and the residual says so.
            amplitudes = _amplitudes(weights, scale)

            # Secant on the scale, so that the achieved D0 heads for the
            # requested one. D0(a) is strictly monotone along the physical
            # branch, so this is a scalar root find on a monotone function and
            # the only question is the step size.
            #
            # It only steps once the field has caught up with the scale it is
            # already at. A frozen-field solve reports D0(s) for the field it was
            # given, so while that field is still moving the number the secant is
            # fed is not D0(s) at all, and stepping on it walks off the branch.
            # Measured on the buffon at 2.09x threshold seeded from the 1.26x
            # amplitude: the initial (unsaturated) field makes a = 15.39 look
            # unsaturated, so the first solve reports 0.48 D0_target where the
            # truth is 0.60; the secant read that as "nowhere near saturated" and
            # pushed, and with the field never allowed to settle the scale ran to
            # a = 928 against a true 68.08 while D0 crawled from 0.48 to 0.91.
            # Holding the scale until the field settles turns that same
            # sub-problem into an ordinary fixed-point iteration on (field, D0) at
            # fixed amplitude, which is exactly what the seed's operating point
            # is defined by.
            if field_change > _FIELD_TRACK_TOL:
                pass
            elif np.isfinite(D0_solved) and D0_solved > 0.0:
                # The slope is only *believed* when it is positive. Two
                # consecutive solves differ by more than their scales: the field
                # refresh in between moves D0 too, and near the answer it moves
                # it further than the scale step does. Measured on the buffon at
                # 1.05x threshold, iteration 6 went s 3.432 -> 3.363 while
                # D0/D0_target went 1.00116 -> 1.00363 -- a *negative* apparent
                # slope, from which the raw secant produced a step of +0.10 in s,
                # i.e. uphill, away from the target it had already overshot. That
                # is what set up the oscillation that stalled the residual at
                # ~1e-4. Monotonicity of D0(a) is not a heuristic here, it is the
                # measured property of the branch (56 continuation points), so a
                # non-positive apparent slope is noise by definition and the last
                # believed slope is kept instead.
                if scale_prev is not None and D0_prev is not None and scale != scale_prev:
                    candidate = (D0_solved - D0_prev) / (scale - scale_prev)
                    if np.isfinite(candidate) and candidate > 0.0:
                        # ... and it may not jump by more than a factor of
                        # _SCALE_SLOPE_TRUST against the slope already believed.
                        # dD0/da varies smoothly along the branch, so a large
                        # jump between two neighbouring continuation points is
                        # the same field-refresh noise seen from the other side:
                        # on the buffon at 1.26x, two solves 0.3 % apart in scale
                        # differed by 1.7e-06 in D0 -- at the noise floor -- and
                        # the resulting near-zero slope threw the scale from 15.5
                        # back to 13.3 before it recovered.
                        if slope is not None:
                            candidate = float(
                                np.clip(
                                    candidate,
                                    slope / _SCALE_SLOPE_TRUST,
                                    slope * _SCALE_SLOPE_TRUST,
                                )
                            )
                        slope = candidate
                if slope is not None:
                    scale_next = scale + _SCALE_STEP_DAMPING * (D0_target - D0_solved) / slope
                else:
                    # No believed slope yet. The chord through the origin is the
                    # safe first move: D0(s) = D0_thr + c s has D0/s decreasing
                    # in s, so s * D0_target / D0_solved *under*-shoots the true
                    # scale whenever the target is above the achieved pump, and
                    # approaching from below is the direction that stays on the
                    # branch. It is a poor step near threshold (D0_thr dominates,
                    # so it barely moves) -- that is what the slope is for.
                    scale_next = scale * D0_target / D0_solved
                if not np.isfinite(scale_next):
                    scale_next = scale
                scale_next = float(
                    np.clip(
                        scale_next,
                        scale / _AMPLITUDE_TRUST_GROWTH,
                        scale * _AMPLITUDE_TRUST_GROWTH,
                    )
                )
                scale_prev, D0_prev = scale, D0_solved
                scale_accepted = scale
                scale = max(scale_next, _AMPLITUDE_TRUST_FLOOR)
            D0_now = D0_solved
        else:
            # The solve wanted to leave the branch: at this scale, and with this
            # frozen field, there is no root inside the mode's own k box. Do not
            # step onto the boundary -- that is what put the buffon at a = 52.7
            # with a residual of 0.43. Retreat the continuation instead.
            #
            # This is not optional bookkeeping. Holding the state and waiting for
            # the field refresh to rescue it deadlocks: the refresh is handed
            # exactly the state it was handed last time, so it converges to its
            # own fixed point and every later iteration repeats the same solve to
            # the last digit -- measured on the buffon at 1.26x threshold,
            # iterations 8 through 25 (cost 1.846e-02, residual 0.712) after the
            # scale overshot from 6.84 to 10.82.
            if scale_accepted is not None and scale != scale_accepted:
                # There is a point behind us the field could follow: the step was
                # simply too long. Halve it -- ordinary continuation step control.
                scale = max(0.5 * (scale + scale_accepted), _AMPLITUDE_TRUST_FLOOR)
                # Keep the amplitudes on the scale about to be tried, so the
                # field refresh below is consistent with it rather than with a
                # scale the continuation has already abandoned.
                amplitudes = _amplitudes(weights, scale)
            elif not restarted:
                # Nothing behind us: the caller's *seed* is not a usable
                # continuation point -- at that amplitude, against the field it
                # comes with, no root exists inside the mode's k box. Start the
                # continuation where one certainly does, at a ~ 0, where the
                # operator is unsaturated and the mode lases at its own
                # threshold, and climb from there. That is the continuity from
                # a = 0 the whole formulation rests on; beginning at the caller's
                # seed is only a shortcut, and this is where the shortcut fails.
                #
                # The fields are reset with the scale. Leaving them is worse than
                # useless: measured on the buffon at 2.09x threshold seeded from
                # the 1.26x amplitude, the fields refreshed against the
                # unusable seed drove the following solves to pumps of 2.4x,
                # 5.5x and 7.3x the requested one -- genuine roots, of other
                # modes -- and the scale collapsed to the floor chasing them.
                restarted = True
                scale = _AMPLITUDE_TRUST_FLOOR
                amplitudes = _amplitudes(weights, scale)
                fields = [list(f) for f in unsaturated]
                slope = None
                scale_prev = D0_prev = None
            # else: already restarted and still on the bound. Hold, and let the
            # field refresh below run at the bracketed D0 -- the bracket is what
            # makes those refreshes converge on the state's own operating point
            # instead of on a pump it cannot support.

        # Refresh the fields against the solved state, to self-consistency. The
        # profiles use the D0 that state actually solves -- it is a
        # self-consistent SALT solution there, and at D0_target it is not one yet.
        #
        # Iterated, not applied once. A single damped mix leaves the field
        # trailing the scale by roughly a factor (1 - damping) per outer step,
        # which is invisible while the continuation creeps and fatal once it
        # strides: on the buffon at 2.09x threshold the scale grew ~4x per
        # accepted step and the lagging field under-reported the hole burning, so
        # every solve came back with a D0 far below the requested one, the secant
        # read that as "not saturated enough yet" and pushed the scale further --
        # to a = 928 against a true 68. The extra rounds are nearly free: each is
        # two eigensolves per mode against the ~60 residual evaluations of the
        # least-squares they precede.
        for _ in range(_FIELD_REFRESH_MAX):
            profiles = saturated_eps_profiles(graph, ks, amplitudes, fields, D0_now, pump)
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
            mixed = [
                [
                    (1.0 - damping) * old + damping * new
                    for old, new in zip(per_mode_old, per_mode_new, strict=True)
                ]
                for per_mode_old, per_mode_new in zip(fields, refreshed, strict=True)
            ]
            field_change = _field_change(fields, mixed)
            fields = mixed
            if field_change <= _FIELD_TRACK_TOL:
                break

        lasing = [i for i in range(n_modes) if amplitudes[i] > SALT_VARYING_LASING_AMPLITUDE]
        residuals = salt_residuals_varying(
            graph, ks, amplitudes, fields, D0_target, pump, n_steps=n_steps
        )
        import os as _os

        if _os.environ.get("SALT_DEBUG"):
            print(
                f"    it={iterations:3d} s={scale:12.6g} a={amplitudes} "
                f"D0/tgt={D0_now / D0_target if D0_target else 0:14.10f} "
                f"kb={int(on_k_bound)} res={np.max(residuals):9.3e} cost={result.cost:9.3e}",
                flush=True,
            )
        if lasing and all(residuals[i] <= residual_tol for i in lasing):
            converged = True
            break

    residuals = salt_residuals_varying(
        graph, ks, amplitudes, fields, D0_target, pump, n_steps=n_steps
    )
    if converged or _substep_depth >= _SUBSTEP_MAX_DEPTH:
        return SaltVaryingSolution(ks, amplitudes, fields, residuals, converged, iterations)

    # Sub-step. The continuation failed to walk from the seed's own pump to the
    # requested one in a single solve; if the two are far apart that is a step
    # length problem rather than a solvability one, so bridge it. The midpoint is
    # geometric because the amplitude climbs multiplicatively -- the buffon's
    # 1.26x -> 2.09x request is a 4.4x climb, and its geometric midpoint splits
    # that into two of 2.1x.
    start = D0_bracketed if D0_bracketed and np.isfinite(D0_bracketed) else None
    if start is None or start <= 0.0:
        return SaltVaryingSolution(ks, amplitudes, fields, residuals, converged, iterations)
    ratio = D0_target / start
    if not np.isfinite(ratio) or ratio < _SUBSTEP_MIN_RATIO:
        return SaltVaryingSolution(ks, amplitudes, fields, residuals, converged, iterations)

    kwargs = {
        "n_steps": n_steps,
        "outer": outer,
        "damping": damping,
        "residual_tol": residual_tol,
        "max_nfev": max_nfev,
        "k_window_cap": k_window_cap,
        "_substep_depth": _substep_depth + 1,
    }
    middle = solve_salt_varying(
        graph, seed_ks, seed_amplitudes, float(start * np.sqrt(ratio)), pump, **kwargs
    )
    if not middle.converged:
        return SaltVaryingSolution(ks, amplitudes, fields, residuals, converged, iterations)
    final = solve_salt_varying(
        graph, middle.ks, middle.amplitudes, float(D0_target), pump, **kwargs
    )
    total = iterations + middle.iterations + final.iterations
    if not final.converged and float(np.max(residuals)) <= float(np.max(final.residuals)):
        # the sub-stepped attempt is no better than what we already had
        return SaltVaryingSolution(ks, amplitudes, fields, residuals, converged, total)
    return SaltVaryingSolution(
        final.ks, final.amplitudes, final.fields, final.residuals, final.converged, total
    )


def compute_modal_intensities_varying(
    graph,
    modes_df,
    max_pump_intensity: float,
    D0_steps: int = 10,
    *,
    n_steps: int = 64,
    outer: int = 40,
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
    # Per mode, the (D0, amplitude) pairs accepted so far. Only this loop knows
    # how a mode's amplitude has moved across pumps, and that trail is what puts
    # the next solve in the right basin (see :func:`_predicted_amplitude`).
    trail: dict[int, list[tuple[float, float]]] = {}
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
            a0.append(
                _predicted_amplitude(
                    trail.get(i, []), float(D0), float(thresholds[i]), max(a_prev, 1e-3)
                )
            )
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
                trail.setdefault(i, []).append((float(D0), float(solution.amplitudes[slot])))
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

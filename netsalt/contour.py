"""Contour-integration mode search (Beyn's method).

Alternative to the ``scan_frequencies`` + ``find_modes`` grid-search
pipeline. Rather than evaluating ``|λ₁(L(k))|`` on a dense Cartesian
grid (millions of eigensolves) and refining each local minimum, Beyn's
method locates **all** roots of ``det(L(k)) = 0`` inside a closed
contour in the complex plane with O(N_quad · L²) work, where N_quad is
the number of quadrature nodes on the contour (typically 50–200,
independent of the number of modes).

For a buffon-scale simulation (8 000 × 500 = 4 M grid points, ~100
modes), this is a ~10 000× reduction in ARPACK calls — see the
comparison in :func:`_benchmark_repr` and the notes in PR #38.

Observations on the buffon_uniform workload (96-node graph,
``k ∈ [10.35, 11.0]`` × ``α ∈ [0.006, 0.015]``):

* True mode count in the region is ~410 (agrees between a fine
  ``peak_local_max`` grid and Beyn with tuned subdivision).
* Grid scan at 2000×250 finds 414 peaks in ~25 min serial.
* Beyn subdivided (``n_k=130, n_α=2, probe_dim=30, n_quad=80``) finds
  **406 real modes in 20 s** — ≥ 98 % coverage, ~75× faster than the
  grid scan, and the returned modes already sit at
  ``|λ₁| ≲ 1e-10`` so no refinement step is needed.

References
----------
Beyn, "An integral method for solving nonlinear eigenvalue problems",
Linear Algebra Appl. 436 (10), 2012.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy as sc

from .quantum_graph import construct_laplacian

L = logging.getLogger(__name__)


def _elliptical_contour(
    k_min: float, k_max: float, alpha_min: float, alpha_max: float, n_quad: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return complex quadrature nodes ``z_j`` and weighted derivatives ``w_j``
    on an ellipse encircling the scan rectangle.

    The netsalt sign convention stores a mode as ``[Re(k), -Im(k)]`` (i.e.
    ``k_complex = Re(k) - j·alpha``), so the contour lives in the lower
    half-plane when ``alpha > 0``. The ellipse is inflated by a small
    buffer factor so quadrature nodes don't sit exactly on a mode.
    """
    pad = 1.02  # 2 % buffer so nodes stay off any mode on the contour edge
    cx = 0.5 * (k_min + k_max)
    cy = -0.5 * (alpha_max + alpha_min)  # note the minus sign
    rx = pad * 0.5 * (k_max - k_min)
    ry = pad * 0.5 * (alpha_max - alpha_min)
    theta = np.linspace(0.0, 2.0 * np.pi, n_quad, endpoint=False)
    z = cx + rx * np.cos(theta) + 1.0j * (cy + ry * np.sin(theta))
    # dz/dθ · dθ for trapezoidal quadrature on a periodic integrand.
    dz_dtheta = -rx * np.sin(theta) + 1.0j * ry * np.cos(theta)
    # Integration weight includes the 1/(2πj) factor from Cauchy.
    weights = dz_dtheta * (2.0 * np.pi / n_quad) / (2.0j * np.pi)
    return z, weights


def _inside_contour(k: complex, contour_bounds: tuple[float, float, float, float]) -> bool:
    """True if a complex ``k`` lies inside the search ellipse. The ellipse
    itself is slightly inflated in :func:`_elliptical_contour`, so here we
    test against the original rectangle-inscribed ellipse — modes exactly
    on the buffer ring are counted as inside."""
    k_min, k_max, alpha_min, alpha_max = contour_bounds
    cx = 0.5 * (k_min + k_max)
    cy = -0.5 * (alpha_max + alpha_min)
    rx = 0.5 * (k_max - k_min)
    ry = 0.5 * (alpha_max - alpha_min)
    dx = k.real - cx
    dy = k.imag - cy
    return (dx / rx) ** 2 + (dy / ry) ** 2 <= 1.0


def find_modes_contour(
    graph: Any,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    n_quad: int = 80,
    probe_dim: int | None = None,
    svd_tol: float = 1e-10,
    quality_filter: float | None = 1e-3,
    rng: np.random.Generator | None = None,
):
    """Find modes via Beyn's contour-integration method.

    Given a contour ``Γ`` encircling the scan rectangle, compute the first
    two Cauchy moments

    .. math::

        A_0 = \\frac{1}{2\\pi j} \\oint_Γ L(k)^{-1} V \\, dk, \\qquad
        A_1 = \\frac{1}{2\\pi j} \\oint_Γ k \\cdot L(k)^{-1} V \\, dk

    with ``V`` a random ``(L, ℓ)`` probe matrix. The SVD of ``A_0`` gives
    the number of modes ``r`` inside ``Γ``; the reduced matrix
    ``B = U_r^* A_1 W_r Σ_r^{-1}`` is ``r × r`` and its eigenvalues are
    exactly the wavenumbers ``k`` for which ``det L(k) = 0`` — no
    refinement step needed.

    Args:
        graph: a fully-configured netsalt quantum graph (``k`` through
            :func:`construct_laplacian` must already give a well-defined
            laplacian; in particular, ``params["D0"]`` and
            ``dispersion_relation`` must be set).
        bounds: ``(k_min, k_max, alpha_min, alpha_max)`` rectangle to
            search. If None, taken from ``graph.graph["params"]``.
        n_quad: number of quadrature nodes on the contour. More nodes →
            more accuracy, linearly more cost. 80 is a good default for
            graphs with up to ~100 modes inside the contour.
        probe_dim: columns of the random probe matrix ``V``. Must exceed
            the expected number of modes; defaults to ``min(2·ℓ_est, L)``
            where ``ℓ_est = n_quad / 2``.
        svd_tol: relative threshold for keeping singular values of
            ``A_0``. Singular values below ``svd_tol · σ_max`` are
            discarded (they correspond to numerically-zero moments).
        quality_filter: if not None, each extracted eigenvalue is
            re-evaluated with :func:`mode_quality`; candidates whose
            ``|λ₁|`` exceeds this threshold are discarded as spurious
            outputs of the SVD/eig extraction. Set to None to return
            the raw Beyn output (useful when chaining into a refiner).
            Default is ``1e-3``, loose enough to pass all true roots
            (which are typically ``≤ 1e-6``) while rejecting the false
            positives Beyn can produce when ``probe_dim`` exceeds the
            actual mode count.
        rng: numpy ``Generator`` for the probe matrix. If None, fresh
            entropy.

    Returns:
        numpy array of shape ``(n_modes, 2)``, each row the netsalt
        ``[Re(k), -Im(k)]`` pair. Modes are ordered by ``|Re(k)|``
        ascending.
    """
    if bounds is None:
        params = graph.graph["params"]
        bounds = (
            params["k_min"],
            params["k_max"],
            params["alpha_min"],
            params["alpha_max"],
        )
    if rng is None:
        rng = np.random.default_rng()

    n_nodes = len(graph)
    if probe_dim is None:
        probe_dim = min(max(n_quad // 2, 4), n_nodes)
    probe_dim = min(probe_dim, n_nodes)

    V = rng.standard_normal((n_nodes, probe_dim)) + 1j * rng.standard_normal((n_nodes, probe_dim))

    z, weights = _elliptical_contour(*bounds, n_quad=n_quad)

    A0 = np.zeros((n_nodes, probe_dim), dtype=np.complex128)
    A1 = np.zeros((n_nodes, probe_dim), dtype=np.complex128)
    for zi, wi in zip(z, weights, strict=True):
        laplacian = construct_laplacian(zi, graph).tocsc()
        try:
            solver = sc.sparse.linalg.splu(laplacian)
        except RuntimeError:
            L.info("Singular laplacian on contour, skipping node k=%s", zi)
            continue
        Y = solver.solve(V)
        A0 += wi * Y
        A1 += wi * zi * Y

    # Rank-revealing SVD of A_0 — the rank is the number of modes in Γ.
    U, S, Wh = np.linalg.svd(A0, full_matrices=False)
    if S.size == 0 or S[0] == 0.0:
        return np.empty((0, 2))
    rank = int(np.sum(S > svd_tol * S[0]))
    if rank == 0:
        return np.empty((0, 2))

    U_r = U[:, :rank]
    S_r = S[:rank]
    W_r = Wh[:rank, :].conj().T

    # Reduced problem ``B = U_r^* A_1 W_r Σ_r^{-1}``; eigenvalues of B are the modes.
    B = (U_r.conj().T @ A1 @ W_r) / S_r
    eigenvalues = np.linalg.eigvals(B)

    from .quantum_graph import mode_quality

    modes = []
    for k in eigenvalues:
        if not _inside_contour(k, bounds):
            continue
        # netsalt convention: [Re(k), -Im(k)]. alpha = -imag(k) ≥ 0 physically.
        candidate = [k.real, -k.imag]
        if quality_filter is not None:
            q = mode_quality(candidate, graph)
            if q > quality_filter:
                # Spurious output of the SVD extraction — drop.
                continue
        modes.append(candidate)
    if not modes:
        return np.empty((0, 2))
    modes_arr = np.asarray(modes)
    # Sort by real part for stable output.
    order = np.argsort(modes_arr[:, 0])
    return modes_arr[order]


def find_modes_contour_subdivided(
    graph: Any,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    n_k: int = 1,
    n_alpha: int = 1,
    n_quad: int = 80,
    probe_dim: int | None = None,
    svd_tol: float = 1e-10,
    quality_filter: float | None = 1e-3,
    dedup_rtol: float = 1e-5,
    rng: np.random.Generator | None = None,
):
    """Run :func:`find_modes_contour` on a grid of sub-contours.

    Beyn's method requires the probe-matrix column count to exceed the
    number of modes inside the contour. For dense-mode regions (a
    buffon-scale simulation has ~800 modes in its scan rectangle),
    partition the rectangle into ``n_k × n_alpha`` sub-cells and collect
    modes from each. Duplicates along shared cell boundaries are removed
    via a distance-based dedup.

    Args are otherwise the same as :func:`find_modes_contour`; ``n_k`` and
    ``n_alpha`` set the grid of sub-contours in the ``Re(k)`` and
    ``alpha`` directions respectively.
    """
    if bounds is None:
        params = graph.graph["params"]
        bounds = (
            params["k_min"],
            params["k_max"],
            params["alpha_min"],
            params["alpha_max"],
        )
    k_min, k_max, a_min, a_max = bounds
    if rng is None:
        rng = np.random.default_rng()

    k_edges = np.linspace(k_min, k_max, n_k + 1)
    a_edges = np.linspace(a_min, a_max, n_alpha + 1)

    collected = []
    for i in range(n_k):
        for j in range(n_alpha):
            cell = (k_edges[i], k_edges[i + 1], a_edges[j], a_edges[j + 1])
            sub = find_modes_contour(
                graph,
                bounds=cell,
                n_quad=n_quad,
                probe_dim=probe_dim,
                svd_tol=svd_tol,
                quality_filter=quality_filter,
                rng=rng,
            )
            if len(sub):
                collected.append(sub)
    if not collected:
        return np.empty((0, 2))
    all_modes = np.concatenate(collected, axis=0)

    # Dedup: cells overlap at boundaries, so two adjacent cells can each
    # claim a mode that sits on their shared edge.
    scale_k = max(k_max - k_min, 1e-9)
    scale_a = max(a_max - a_min, 1e-9)
    kept = []
    for mode in all_modes:
        duplicate = False
        for other in kept:
            if (
                abs(mode[0] - other[0]) / scale_k < dedup_rtol
                and abs(mode[1] - other[1]) / scale_a < dedup_rtol
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(mode)
    kept_arr = np.asarray(kept)
    order = np.argsort(kept_arr[:, 0])
    return kept_arr[order]

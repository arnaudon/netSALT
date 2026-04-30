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

* True mode count in the region is ~454 (found by Beyn with tuned
  subdivision; the grid scan at 2000×250 finds only 414 because the
  ``peak_local_max`` filter merges modes within ``min_distance=2``
  pixels — it undercounts).
* Grid scan at 2000×250: 414 peaks in ~25 min serial.
* Beyn subdivided (``n_k=130, n_α=2, probe_dim=30, n_quad=80``) finds
  **454 real modes in ~30 s**, 100 % at ``|λ₁| ≤ 1e-7``. No refinement
  step is needed.

Close-mode separation
---------------------
Beyn's SVD resolves close modes cleanly as long as ``probe_dim`` exceeds
the number of modes inside the contour. On a dense line graph with
mode spacing ``π/20 ≈ 0.157``, seven consecutive modes in a single
contour come out at quality ``1e-12``–``1e-14`` each — see
``test_contour_separates_closely_spaced_modes``. When ``probe_dim``
approaches the mode count, precision degrades (1e-5 instead of 1e-12)
but the modes stay separated; subdivide the contour to recover
precision.

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
    on an ellipse that **circumscribes** the scan rectangle.

    The netsalt sign convention stores a mode as ``[Re(k), -Im(k)]`` (i.e.
    ``k_complex = Re(k) - j·alpha``), so the contour lives in the lower
    half-plane when ``alpha > 0``. The ellipse half-axes are ``√2`` times
    the rectangle half-widths — that's the smallest axis-aligned ellipse
    that contains every corner of the rectangle. Without this, modes at
    the rectangle's horizontal extremes sit outside the inscribed
    ellipse and get missed.
    """
    cx = 0.5 * (k_min + k_max)
    cy = -0.5 * (alpha_max + alpha_min)  # note the minus sign
    # ``sqrt(2) · half-width`` circumscribes the rectangle; a further 1.02
    # factor keeps quadrature nodes off any mode sitting exactly on the
    # rectangle boundary.
    pad = 1.02 * np.sqrt(2.0)
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
    """True if a complex ``k`` lies inside the *rectangle* the caller
    asked about. The contour itself is a circumscribed ellipse, so it can
    legally pick up modes outside the rectangle but still inside the
    ellipse; filter those out here so the function's semantics match the
    ``bounds`` argument.
    """
    k_min, k_max, alpha_min, alpha_max = contour_bounds
    return k_min <= k.real <= k_max and -alpha_max <= k.imag <= -alpha_min


def _find_modes_in_cell(
    graph: Any,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    n_quad: int = 80,
    probe_dim: int | None = None,
    svd_tol: float = 1e-10,
    quality_filter: float | None = 1e-3,
    rng: np.random.Generator | None = None,
):
    """Internal: run the basic Beyn algorithm on a single cell.

    The public :func:`find_modes_contour` is the subdivided wrapper —
    a single contour caps at ``probe_dim`` modes, so production
    callers always go through subdivision. This helper exists only so
    the subdivided and adaptive variants share the per-cell extraction.

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


def find_modes_contour(
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
    """Find modes via Beyn's contour-integration method on an
    ``n_k × n_alpha`` grid of sub-contours.

    Each sub-contour caps at ``probe_dim`` modes (the basic Beyn
    algorithm's SVD-extraction step has at most ``probe_dim`` non-zero
    singular values). For production-scale rectangles you almost
    always need ``n_k > 1`` so each cell fits under that ceiling;
    duplicates along shared cell boundaries are removed via a
    distance-based dedup at ``dedup_rtol``.

    The trivial ``n_k=1, n_alpha=1`` case reduces to a single
    contour — useful only when the rectangle has comfortably fewer
    modes than ``probe_dim``. ``n_k`` defaults to 1 to keep the
    function callable with no parameters; pick it via the rule
    ``ceil(expected_modes / (0.65 · probe_dim))`` or use
    :func:`find_modes_contour_adaptive` /
    :func:`tune_contour_parameters` to size it automatically.

    Args:
        graph: a fully-configured netsalt quantum graph.
        bounds: ``(k_min, k_max, α_min, α_max)`` rectangle. If None,
            taken from ``graph.graph["params"]``.
        n_k, n_alpha: grid of sub-contours in ``Re(k)`` / ``α``.
        n_quad: trapezoidal-quadrature node count per contour.
        probe_dim: random-probe column count (default
            ``min(40, n_nodes)``; clamped to ``n_nodes``).
        svd_tol: relative threshold for keeping singular values of
            ``A_0``.
        quality_filter: ``|λ₁|`` threshold for spurious-mode rejection.
        dedup_rtol: relative tolerance for boundary-mode dedup.
        rng: optional ``numpy.random.Generator``.

    Returns:
        ``(n_modes, 2)`` array of ``[Re(k), -Im(k)]`` rows, sorted by
        ``Re(k)``.
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
            sub = _find_modes_in_cell(
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
    return _concatenate_and_dedup(collected, bounds, dedup_rtol)


def _concatenate_and_dedup(collected, bounds, dedup_rtol):
    """Stack mode arrays from multiple sub-contours and remove
    boundary duplicates. Returns ``(0, 2)`` empty array if no modes."""
    if not collected:
        return np.empty((0, 2))
    k_min, k_max, a_min, a_max = bounds
    all_modes = np.concatenate(collected, axis=0)
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


def _split_cell(cell, axis="k"):
    """Bisect a 4-tuple ``(k_min, k_max, α_min, α_max)`` into two halves
    along the named axis.

    Default is ``"k"`` because netsalt workloads concentrate modes
    near a single ``α`` (loss of the open boundary) and scan over a
    wide range of ``k``; splitting ``α`` puts every mode in one half
    of every cell and wastes the other half. Pass ``axis="alpha"``
    or ``axis="auto"`` (split the longer side) to override.
    """
    k_min, k_max, a_min, a_max = cell
    if axis == "auto":
        axis = "k" if (k_max - k_min) >= (a_max - a_min) else "alpha"
    if axis == "k":
        mid = 0.5 * (k_min + k_max)
        return [(k_min, mid, a_min, a_max), (mid, k_max, a_min, a_max)]
    if axis == "alpha":
        mid = 0.5 * (a_min + a_max)
        return [(k_min, k_max, a_min, mid), (k_min, k_max, mid, a_max)]
    raise ValueError(f"Unknown split axis {axis!r}; expected 'k', 'alpha', or 'auto'")


def tune_contour_parameters(
    graph: Any,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    probe_dim: int | None = None,
    n_quad: int = 200,
    safety_factor: float = 1.5,
    saturation_factor: float = 0.7,
    max_depth: int = 6,
    split_axis: str = "k",
    rng: np.random.Generator | None = None,
):
    """Run :func:`find_modes_contour_adaptive` once to discover the mode
    count on a representative graph, then return parameters suitable
    for the cheaper :func:`find_modes_contour` on similar instances.

    Use case: you have many graphs of the same topology / density (say,
    different random seeds for a buffon planar graph) and want to skip
    the adaptive overhead on every call. Tune once on one
    representative — or a few, taking the max — and reuse the
    parameters across the batch.

    Args:
        graph: the representative graph to tune on.
        bounds: ``(k_min, k_max, α_min, α_max)`` rectangle. If None,
            taken from ``graph.graph["params"]``.
        probe_dim: probe dimension to use for both the adaptive
            discovery pass and the recommended output. Default
            ``min(40, n_nodes)``; clamped to the node count by
            ``find_modes_contour``.
        n_quad: quadrature node count, fixed across the recursion
            and reused in the output. 200 is a safe baseline for
            netsalt workloads with up to ~50 modes per cell.
        safety_factor: target ``modes_per_cell ≤ 0.65 · probe_dim /
            safety_factor`` when sizing ``n_k``. ``1.5`` gives 50%
            headroom — useful when the batch's randomized instances
            may have slightly more modes than the representative
            graph used for tuning.
        saturation_factor / max_depth / split_axis / rng: forwarded
            to :func:`find_modes_contour_adaptive`.

    Returns:
        Tuple ``(params, info)``:

        * ``params`` — dict ready to splat into
          :func:`find_modes_contour`:
          ``{"n_k", "n_alpha", "n_quad", "probe_dim"}``.
        * ``info`` — dict with ``discovered_modes`` (mode count from
          the adaptive run), ``modes`` (the actual mode positions),
          and ``modes_per_cell`` (the design target after applying
          ``safety_factor``).
    """
    n_nodes = len(graph)
    if probe_dim is None:
        probe_dim = min(40, n_nodes)
    probe_dim = min(probe_dim, n_nodes)

    modes = find_modes_contour_adaptive(
        graph,
        bounds=bounds,
        n_quad=n_quad,
        probe_dim=probe_dim,
        saturation_factor=saturation_factor,
        max_depth=max_depth,
        split_axis=split_axis,
        rng=rng,
    )
    n_modes = len(modes)

    # Pick ``n_k`` so each cell has comfortably fewer than
    # ``probe_dim`` modes, with ``safety_factor`` headroom for
    # batch-instance variation.
    target_modes_per_cell = max(1.0, 0.65 * probe_dim / safety_factor)
    if split_axis == "k":
        n_k = max(1, int(np.ceil(n_modes / target_modes_per_cell)))
        n_alpha = 1
    elif split_axis == "alpha":
        n_k = 1
        n_alpha = max(1, int(np.ceil(n_modes / target_modes_per_cell)))
    else:  # split_axis == "auto"
        # Distribute the splits roughly along whichever side is longer.
        n_total_cells = max(1, int(np.ceil(n_modes / target_modes_per_cell)))
        n_k = n_alpha = max(1, int(np.ceil(np.sqrt(n_total_cells))))

    params = {
        "n_k": n_k,
        "n_alpha": n_alpha,
        "n_quad": n_quad,
        "probe_dim": probe_dim,
    }
    info = {
        "discovered_modes": n_modes,
        "modes": modes,
        "target_modes_per_cell": target_modes_per_cell,
    }
    return params, info


def find_modes_contour_adaptive(
    graph: Any,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    n_quad: int = 80,
    probe_dim: int | None = None,
    saturation_factor: float = 0.7,
    max_depth: int = 6,
    split_axis: str = "k",
    svd_tol: float = 1e-10,
    quality_filter: float | None = 1e-6,
    dedup_rtol: float = 1e-5,
    rng: np.random.Generator | None = None,
):
    """Adaptive contour search: subdivide cells that saturate ``probe_dim``.

    .. warning::

        Use this for *parameter discovery*, not as the production
        search. The saturation heuristic and boundary-dedup at deep
        recursion can drop a small fraction of modes near cell
        edges; benchmarks typically see 1-5% under-counting on
        300-500 mode workloads (see ``benchmark/results_500_modes.md``
        and ``benchmark/results_stress.md``). For runs where every
        mode matters, call :func:`find_modes_contour` with an
        explicit ``n_k`` — :func:`tune_contour_parameters` is the
        recommended bridge: it runs adaptive once and gives you the
        ``n_k`` to use, then a second non-adaptive call recovers any
        modes adaptive missed.

    The basic Beyn algorithm caps at ``probe_dim`` modes per contour
    (the SVD of ``A_0`` has at most ``probe_dim`` non-zero singular
    values). When the rectangle has more modes than that ceiling, the
    SVD's tail singular values get cut by ``svd_tol`` and the
    extraction collapses — empirically returning either zero modes
    or a partial set.

    This routine treats *saturation* as a signal that a cell needs
    to be split. After running the basic Beyn extraction on a cell:

    * If the cell returns ``≥ saturation_factor · probe_dim`` modes,
      it is suspected of under-counting (the basic algorithm is at
      its capacity ceiling). The cell is bisected along its longer
      side and each half is processed recursively.
    * If the cell returns **zero** modes, it is *ambiguous* — the cell
      either has no modes in it (correct answer) or is so over-capacity
      that the SVD's tail got cut and the extraction collapsed
      (incorrect answer, common signal observed in
      ``benchmark/bench_stress.py``). Until ``max_depth`` is hit, an
      empty cell is split too. At ``max_depth`` an empty result is
      finally trusted.
    * Otherwise the cell's modes are accepted as-is.

    The recursion terminates at ``max_depth`` (so worst case is
    ``2^max_depth`` cells along the longer axis). Modes from all leaf
    cells are concatenated and de-duplicated by relative position
    (``dedup_rtol``).

    Args:
        graph: a fully-configured netsalt quantum graph.
        bounds: ``(k_min, k_max, α_min, α_max)`` rectangle. If None,
            taken from ``graph.graph["params"]``.
        n_quad: trapezoidal-quadrature node count per contour. Stays
            fixed across the recursion — accuracy comes from making
            cells smaller, not from beefing up ``n_quad`` per cell.
        probe_dim: random-probe dimension. Default
            ``min(40, n_nodes)``; clamped to the node count.
        saturation_factor: split a cell when its returned mode count
            is at or above ``saturation_factor · probe_dim``. ``0.7``
            is the empirical sweet spot from
            ``benchmark/bench_stress.py``.
        max_depth: maximum recursion depth. ``2^max_depth`` is the
            worst-case cell count along the chosen split axis.
        split_axis: ``"k"`` (default), ``"alpha"``, or ``"auto"``.
            Default ``"k"`` works for the netsalt-conventional case
            where every mode sits at the same ``α`` (loss of the
            open boundary) and the scan covers a wide ``k`` range —
            splitting ``α`` would put all the modes in one half and
            none in the other. Use ``"auto"`` if the modes are
            spread in both directions; rarely needed.
        svd_tol: forwarded to the per-cell Beyn extraction.
        quality_filter: ``|λ₁|`` threshold for spurious-mode rejection
            on each cell. Default ``1e-6`` is tighter than
            :func:`find_modes_contour`'s default since the adaptive
            search trades cell size for residual quality.
        dedup_rtol: relative tolerance for mode-position dedup at
            cell boundaries.
        rng: optional ``numpy.random.Generator``.

    Returns:
        ``(n_modes, 2)`` array of ``[Re(k), -Im(k)]`` rows, sorted by
        ``Re(k)``.
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
        probe_dim = min(40, n_nodes)
    probe_dim = min(probe_dim, n_nodes)
    saturation_threshold = max(1, int(np.ceil(saturation_factor * probe_dim)))

    stack = [(bounds, 0)]
    collected: list[np.ndarray] = []

    while stack:
        cell, depth = stack.pop()
        modes = _find_modes_in_cell(
            graph,
            bounds=cell,
            n_quad=n_quad,
            probe_dim=probe_dim,
            svd_tol=svd_tol,
            quality_filter=quality_filter,
            rng=rng,
        )
        # Either of two signals triggers a split: a saturated cell
        # (≥ saturation_threshold modes returned, basic algorithm at
        # capacity ceiling) OR an empty cell (0 modes returned, which
        # could equally well mean "no modes here" or "so over capacity
        # that the SVD collapsed and lost everything"). The latter is
        # the dominant failure mode in the stress benchmark.
        n_found = len(modes)
        ambiguous_empty = n_found == 0
        saturated = n_found >= saturation_threshold
        if (saturated or ambiguous_empty) and depth < max_depth:
            for sub in _split_cell(cell, axis=split_axis):
                stack.append((sub, depth + 1))
            continue
        if n_found:
            collected.append(modes)

    return _concatenate_and_dedup(collected, bounds, dedup_rtol)

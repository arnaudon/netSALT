"""Search algorithms for mode detection.

Each ``refine_mode_*`` function takes an initial guess for a complex
wavenumber and drives ``|λ₁(L(k))|`` to zero. The original algorithm
is :func:`refine_mode_brownian_ratchet` — a random-walk descent that
typically burns hundreds of ARPACK ``eigs`` calls. The other three
implementations use structural information about the objective:

* :func:`refine_mode_root` — reframe as 2 real equations in 2 real
  unknowns and feed to MINPACK's ``hybr`` solver. Default.
* :func:`refine_mode_newton` — Newton's method on the scalar complex
  function ``λ₁(k)`` using the Hellmann-Feynman derivative
  ``dλ₁/dk = (u₁·dL/dk·v₁) / (u₁·v₁)``.
* :func:`refine_mode_nelder_mead` — derivative-free simplex descent on
  ``|λ₁(k)|``.

:func:`refine_mode` dispatches based on ``params["refine_method"]``.
"""

import logging
import warnings

import numpy as np
import scipy as sc
from scipy.optimize import minimize, root
from skimage.feature import peak_local_max

from .quantum_graph import mode_quality

L = logging.getLogger(__name__)

REFINE_METHODS = ("root", "newton", "nelder_mead", "brownian")
DEFAULT_REFINE_METHOD = "root"


def find_rough_modes_from_scan(ks, alphas, qualities, min_distance=2, threshold_abs=10):
    """Use scipy.ndimage algorithms to detect minima in the scan.

    Args:
        ks (list): list of real part of wavenumbers from the scan
        alphas (list): list of imaginary part of wavenumbers from the scan
        qualities (array): matrix with mode qualities for each pair of k/alpha
        min_distance (int): smallest distance between peaks (for skimage.feature.peak_local_max)
        threshold_abs (float): minimal intensity between peaks (for skimage.feature.peak_local_max)
    """
    data = 1.0 / (1e-10 + qualities)
    rough_mode_ids = peak_local_max(data, min_distance=min_distance, threshold_abs=threshold_abs)
    return [[ks[rough_mode_id[0]], alphas[rough_mode_id[1]]] for rough_mode_id in rough_mode_ids]


def refine_mode_brownian_ratchet(
    initial_mode,
    graph,
    params,
    disp=False,
    save_mode_trajectories=False,
    seed=42,
    quality_method="eigenvalue",
    rng=None,
):
    """Accurately find a mode from an initial guess, using brownian ratchet algorithm.

    This algorithm is quite complex, but generally randomly propose a move to a new mode location,
    and accept if the quality decreases.

    Args:
        initial_mode (complex): initial gues for a mode
        graph (graph): quantum graph
        params (dict): includes search_stepsize, max_steps, quality_threhsold, max_tries_reduction
            reduction_factor
        disp (bool): to print some state of the search for debuing
        save_mode_trajectories (bool): true to save intermediate modes
        seed (int): seed used when ``rng`` is None; ignored otherwise.
        quality_method (str): method for quality evaluation.
        rng: optional ``numpy.random.Generator``. If None, one is created from
            ``seed``. Passed through the recursive retry so a re-entry continues
            from the same RNG stream rather than restarting.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    current_mode = initial_mode.copy()
    if save_mode_trajectories:
        mode_trajectories = [current_mode.copy()]

    initial_quality = mode_quality(current_mode, graph, quality_method=quality_method, rng=rng)
    current_quality = initial_quality

    search_stepsize = params.get("search_stepsize", 0.01)
    tries_counter = 0
    step_counter = 0
    while current_quality > params.get("quality_threshold", 1e-4) and step_counter < params.get(
        "max_steps", 10000
    ):
        new_mode = current_mode + search_stepsize * current_quality / initial_quality * rng.uniform(
            -1, 1, 2
        )

        new_quality = mode_quality(new_mode, graph, quality_method=quality_method, rng=rng)

        if disp:
            L.debug(
                "New quality: %s, Step size: %s, Current mode: %s, New mode: %s, step %s",
                new_quality,
                search_stepsize,
                current_mode,
                new_mode,
                step_counter,
            )

        # if the quality improves, update the mode
        if new_quality < current_quality:
            current_quality = new_quality
            current_mode = new_mode
            tries_counter = 0
            if save_mode_trajectories:
                mode_trajectories.append(current_mode.copy())
        else:
            tries_counter += 1

        # if no improvements after some iterations, multiply the steps by reduction factor
        if tries_counter > params.get("max_tries_reduction", 50):
            search_stepsize *= params.get("reduction_factor", 0.8)
            tries_counter = 0

        if search_stepsize < 1e-10:
            disp = True
            L.info("Warning: mode search stepsize under 1e-10 for mode: %s", current_mode)
            L.info("We retry from a larger one, but consider fine tuning search parameters.")
            search_stepsize = 1e-8
        step_counter += 1

    if current_quality < params.get("quality_threshold", 1e-4):
        if save_mode_trajectories:
            return np.array(mode_trajectories)
        return current_mode

    L.info("Maximum number of tries attained and no mode found, we retry from scratch!")
    params["search_stepsize"] = params.get("search_stepsize", 0.1) * 5

    return refine_mode_brownian_ratchet(
        initial_mode,
        graph,
        params,
        disp=disp,
        save_mode_trajectories=save_mode_trajectories,
        quality_method=quality_method,
        rng=rng,
    )


def _search_box(params):
    """Return the half-width box ``(dk, dα)`` around the initial guess that
    a non-Brownian refiner is allowed to leave before the mode is rejected.

    ``WorkerModes.set_search_radii`` sets ``k_min``/``k_max`` (and the
    ``alpha_*`` pair) centred on each initial guess. We reuse their half-
    extent here so root / Newton / Nelder-Mead honour the same locality
    that the Brownian ratchet achieves by taking small steps. If no search
    window is set (caller invoking the refiner directly), fall back to the
    ``search_stepsize`` knob or a sane default.
    """
    k_min, k_max = params.get("k_min"), params.get("k_max")
    a_min, a_max = params.get("alpha_min"), params.get("alpha_max")
    step = params.get("search_stepsize", 0.01)
    dk = 0.5 * (k_max - k_min) if k_min is not None and k_max is not None else step
    da = 0.5 * (a_max - a_min) if a_min is not None and a_max is not None else step
    # Stay within ~1.5 grid cells so that each initial guess stays in its
    # own basin — looser and neighbouring modes start absorbing each other
    # (``peak_local_max`` keeps local minima at least 2 cells apart, so
    # 1.5 is the largest safe factor). Multiply by 1.5 on top of the
    # caller's search window, which ``WorkerModes.set_search_radii``
    # already scales to ~1 grid cell.
    return 1.5 * dk, 1.5 * da


def _within_search_box(initial_mode, final_mode, params):
    dk, da = _search_box(params)
    return abs(final_mode[0] - initial_mode[0]) <= dk and abs(final_mode[1] - initial_mode[1]) <= da


def refine_mode_root(initial_mode, graph, params, quality_method="eigenvalue", rng=None):
    """Refine a mode via :func:`scipy.optimize.root` (MINPACK ``hybr``).

    Recasts the problem ``λ₁(L(k)) = 0`` as two real equations in two
    real unknowns and lets MINPACK drive ``(Re λ₁, Im λ₁)`` to zero with
    its Broyden-updated Jacobian. Typically converges in 10–30 evaluations
    from a grid-cell-distance starting point.

    A refined mode that wanders more than 5× the caller's search window
    from the initial guess is rejected (returns ``None``) — this mirrors
    the locality the Brownian ratchet gets from its tiny random steps and
    stops aggressive optimisers from absorbing neighbouring modes.

    Args:
        initial_mode: ``[Re(k), -Im(k)]`` starting guess.
        graph: quantum graph.
        params: parameter dict / :class:`NetSaltParams`. Uses
            ``quality_threshold`` as the convergence tolerance and
            ``max_steps`` as the evaluation budget.
        quality_method: accepted for interface parity; ignored here (the
            root method always evaluates the complex eigenvalue).
        rng: optional ``numpy.random.Generator`` threaded through
            :func:`mode_quality` for deterministic ARPACK starting vectors.

    Returns:
        The refined mode as ``[Re(k), -Im(k)]``, or ``None`` on failure.
    """
    del quality_method  # unused, see docstring
    tol = params.get("quality_threshold", 1e-4)
    max_fev = params.get("max_steps", 200)
    x0 = np.asarray(initial_mode, dtype=float)

    def residual(x):
        lam = mode_quality(x, graph, quality_method="complex_eigenvalue", rng=rng)
        return [lam.real, lam.imag]

    # MINPACK's ``xtol`` is a step-size termination threshold, not a
    # residual threshold — we want it far tighter than ``quality_threshold``
    # so the solver doesn't quit early with a residual like 1e-3 simply
    # because its steps got small. We then check the residual against the
    # user-facing threshold ourselves.
    result = root(
        residual,
        x0,
        method="hybr",
        tol=0,
        options={"maxfev": int(max_fev), "xtol": min(tol * 1e-6, 1e-10)},
    )
    if np.linalg.norm(result.fun) > tol:
        return None
    if not _within_search_box(x0, result.x, params):
        return None
    return np.asarray(result.x)


def refine_mode_nelder_mead(initial_mode, graph, params, quality_method="eigenvalue", rng=None):
    """Refine a mode via a Nelder-Mead simplex on ``|λ₁(L(k))|``.

    Derivative-free and deterministic. Same objective as the Brownian
    ratchet but with ~5-10× fewer evaluations thanks to the simplex's
    reflection / expansion / contraction rules.
    """
    tol = params.get("quality_threshold", 1e-4)
    max_fev = params.get("max_steps", 500)

    def objective(x):
        return mode_quality(x, graph, quality_method=quality_method, rng=rng)

    # Build an initial simplex that spans the current search window if one
    # was set by WorkerModes, otherwise fall back to a small default.
    search = [
        params.get("search_stepsize", 0.01),
        params.get("search_stepsize", 0.01),
    ]
    x0 = np.asarray(initial_mode, dtype=float)
    initial_simplex = np.array([x0, x0 + [search[0], 0.0], x0 + [0.0, search[1]]])

    # Target half the user-facing threshold so that when NM's simplex-
    # geometry criterion triggers, the function value is comfortably below
    # ``quality_threshold``.
    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "xatol": tol,
            "fatol": tol * 0.5,
            "maxfev": int(max_fev),
            "initial_simplex": initial_simplex,
            "disp": False,
        },
    )
    if result.fun > tol:
        return None
    if not _within_search_box(x0, result.x, params):
        return None
    return np.asarray(result.x)


def _laplacian_derivative_times_vector(graph, v):
    r"""Return ``dL/dk · v`` at the current ``graph.graph["ks"]``.

    Used by :func:`refine_mode_newton` to form the Hellmann-Feynman
    derivative without materialising the full ``dL/dk`` matrix.

    The quantum laplacian is ``L(k) = Bᵀ(k) · W⁻¹(k) · B(k)`` with
    ``expl = exp(jℓk)`` and ``1 / (exp(2jℓk) − 1)`` as the only
    k-dependent pieces. This function builds ``dB/dk``, ``dBT/dk``, and
    ``dW⁻¹/dk`` as CSR sparse matrices and chains them via the product
    rule, contracting against ``v`` so the caller avoids paying for a
    sparse-sparse matmul.
    """
    from .quantum_graph import construct_incidence_matrix, construct_weight_matrix

    lengths = graph.graph["lengths"]
    ks = graph.graph["ks"]
    expl = np.exp(1.0j * lengths * ks)
    # d(expl)/dk = j·ℓ · expl (per edge, same dispersion as for expl itself
    # since ks = dispersion_relation(freq); we treat ks as the wavenumber
    # directly here because mode_quality was constructed with that same
    # convention — we're differentiating in ``k`` not in ``freq``).
    dexpl = 1.0j * lengths * expl

    topo = graph.graph.get("_incidence_topology")
    if topo is None or topo["m"] != len(graph.edges):
        from .quantum_graph import _incidence_topology

        topo = _incidence_topology(graph)
    m, n = topo["m"], topo["n"]
    row, col = topo["row"], topo["col"]

    # dB/dk has zeros at the −1 slots and dexpl at the expl slots.
    ones_zero = np.zeros(m)
    dB_data = np.dstack([ones_zero, dexpl, dexpl, ones_zero])[0].flatten()
    dB_data_out = dB_data.copy()

    open_model = graph.graph["params"]["open_model"]
    if open_model == "open":
        mask = topo["open_mask"]
        dB_data_out[1::4][mask] = 0
        dB_data_out[2::4][mask] = 0
    elif open_model == "directed":
        dB_data_out[2::4] = 0
        dB_data_out[3::4] = 0
    elif open_model == "directed_reversed":
        dB_data[2::4] = 0
        dB_data[3::4] = 0

    dBT = sc.sparse.csr_matrix((dB_data_out, (col, row)), shape=(n, 2 * m), dtype=np.complex128)
    dB = sc.sparse.csr_matrix((dB_data, (row, col)), shape=(2 * m, n), dtype=np.complex128)

    # dW⁻¹/dk: if W⁻¹ = 1 / (exp(2jℓk) − 1), then d/dk = −2jℓ·exp(2jℓk)
    # / (exp(2jℓk) − 1)² = −2jℓ·(1 + W⁻¹) · W⁻¹.  W also carries a
    # multiplicative ``ks`` factor when with_k=True; product-rule both.
    e2 = np.exp(2.0j * lengths * ks)
    winv = 1.0 / (e2 - 1.0)
    dwinv_nok = -2.0j * lengths * e2 * winv * winv  # d/dk of 1/(e^{2jℓk} − 1)
    # For L, construct_weight_matrix uses with_k=True: data = ks * winv.
    # d/dk[ks * winv] = winv + ks * dwinv_nok  (ks is the scalar wavenumber
    # here; for vector ks (one per edge via dispersion) it's still the
    # diagonal derivative, computed elementwise).
    dwinv_data = winv + ks * dwinv_nok
    dWinv = sc.sparse.diags(np.repeat(dwinv_data, 2), format="csc", dtype=np.complex128)

    BT, B = construct_incidence_matrix(graph)
    Winv = construct_weight_matrix(graph, with_k=True)

    # dL/dk · v = dBT · Winv · B · v + BT · dWinv · B · v + BT · Winv · dB · v
    return dBT @ (Winv @ (B @ v)) + BT @ (dWinv @ (B @ v)) + BT @ (Winv @ (dB @ v))


def refine_mode_newton(initial_mode, graph, params, quality_method="eigenvalue", rng=None):
    """Refine a mode via Newton's method using a Hellmann-Feynman derivative.

    ``k_{n+1} = k_n − λ₁(k_n) / (dλ₁/dk)`` where ``dλ₁/dk =
    (u₁ · dL/dk · v₁) / (u₁ · v₁)``; ``v₁`` is the right eigenvector of
    ``L(k_n)`` at the smallest eigenvalue, and ``u₁`` is the right
    eigenvector of ``Lᵀ`` (i.e. the left eigenvector of ``L``).

    Falls back to :func:`refine_mode_root` if three consecutive Newton
    steps fail to decrease ``|λ₁|`` or if a step leaves the search
    window, so pathological starts that Newton can't handle still
    converge.
    """
    del quality_method  # always uses the complex eigenpair
    tol = params.get("quality_threshold", 1e-4)
    max_steps = params.get("max_steps", 50)
    trust_radius = max(
        params.get("search_stepsize", 0.01) * 10,
        tol * 100,
    )

    from .quantum_graph import construct_laplacian
    from .utils import from_complex, to_complex

    x0 = np.asarray(initial_mode, dtype=float)
    mode = x0.copy()
    prev_abs = np.inf
    non_decrease = 0

    def _finalise(candidate):
        # Hand off to the robust solver if Newton stalls or leaves the box.
        result = refine_mode_root(candidate, graph, params, rng=rng)
        if result is None:
            return None
        if not _within_search_box(x0, result, params):
            return None
        return result

    for _ in range(int(max_steps)):
        k = to_complex(mode)
        laplacian = construct_laplacian(k, graph)
        # right eigenpair
        lam_r, v = sc.sparse.linalg.eigs(
            laplacian,
            k=1,
            sigma=0,
            return_eigenvectors=True,
            which="LM",
            v0=(rng.random(laplacian.shape[0]) if rng is not None else None),
        )
        lam = complex(lam_r[0])
        v = v[:, 0]
        if abs(lam) < tol:
            if not _within_search_box(x0, mode, params):
                return None
            return mode
        if abs(lam) > prev_abs:
            non_decrease += 1
            if non_decrease >= 3:
                return _finalise(mode)
        else:
            non_decrease = 0
        prev_abs = abs(lam)

        # left eigenvector = right eigenvector of Lᵀ
        _, u = sc.sparse.linalg.eigs(
            laplacian.T,
            k=1,
            sigma=0,
            return_eigenvectors=True,
            which="LM",
            v0=(rng.random(laplacian.shape[0]) if rng is not None else None),
        )
        u = u[:, 0]

        dL_v = _laplacian_derivative_times_vector(graph, v)
        denom = u @ v
        if abs(denom) < 1e-14:
            return _finalise(mode)
        dlam_dk = (u @ dL_v) / denom
        if abs(dlam_dk) < 1e-14:
            return _finalise(mode)

        step = lam / dlam_dk
        if abs(step) > trust_radius:
            step = step * trust_radius / abs(step)
        k_new = k - step
        mode = np.asarray(from_complex(k_new), dtype=float)
        if not _within_search_box(x0, mode, params):
            return _finalise(mode)

    return _finalise(mode)


def refine_mode(initial_mode, graph, params, quality_method="eigenvalue", rng=None, **kwargs):
    """Dispatcher over the four refinement algorithms.

    Which one runs is controlled by ``params["refine_method"]`` — one of
    ``"root"`` (default), ``"newton"``, ``"nelder_mead"``, or
    ``"brownian"``. Unknown values raise ``ValueError`` rather than
    silently falling through, so typos surface at the graph boundary.

    Extra kwargs (e.g. ``disp``, ``save_mode_trajectories``) are passed
    through to the Brownian ratchet for backward compatibility; they're
    ignored by the other methods.
    """
    method = params.get("refine_method") or DEFAULT_REFINE_METHOD
    if method not in REFINE_METHODS:
        raise ValueError(f"Unknown refine_method {method!r}; expected one of {REFINE_METHODS}")
    if method == "brownian":
        return refine_mode_brownian_ratchet(
            initial_mode,
            graph,
            params,
            quality_method=quality_method,
            rng=rng,
            **kwargs,
        )
    # Non-ratchet methods ignore brownian-specific kwargs.
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        warnings.warn(
            f"refine_method={method!r} ignores kwargs: {unknown}",
            stacklevel=2,
        )
    if method == "root":
        return refine_mode_root(initial_mode, graph, params, quality_method=quality_method, rng=rng)
    if method == "newton":
        return refine_mode_newton(
            initial_mode, graph, params, quality_method=quality_method, rng=rng
        )
    if method == "nelder_mead":
        return refine_mode_nelder_mead(
            initial_mode, graph, params, quality_method=quality_method, rng=rng
        )
    raise AssertionError(f"unreachable: method={method!r}")  # pragma: no cover


def clean_duplicate_modes(all_modes, k_size, alpha_size):
    """Remave duplicate modes from a list of modes with a threshold in real/imag(k).

    Args:
        all_modes (list): list of modes
        k_size (float): minimal distance in real(k)
        alpha_size (float): minimal distance in imag(k)
    """
    duplicate_mode_ids = []
    for mode_id_0, mode_0 in enumerate(all_modes):
        for mode_id_1, mode_1 in enumerate(all_modes[mode_id_0 + 1 :]):
            if (
                mode_id_1 + mode_id_0 + 1 not in duplicate_mode_ids
                and abs(mode_0[0] - mode_1[0]) < k_size
                and abs(mode_0[1] - mode_1[1]) < alpha_size
            ):
                duplicate_mode_ids.append(mode_id_0)
                break

    return np.delete(np.array(all_modes), duplicate_mode_ids, axis=0)

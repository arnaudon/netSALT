"""Search algorithms for mode detection.

Each ``refine_mode_*`` function takes an initial guess for a complex
wavenumber and drives ``|λ₁(L(k))|`` to zero. Two implementations:

* :func:`refine_mode_root` — reframe as 2 real equations in 2 real
  unknowns and feed to MINPACK's ``hybr`` solver. Default.
* :func:`refine_mode_brownian_ratchet` — random-walk descent. Legacy
  fallback; typically burns hundreds of ARPACK ``eigs`` calls but
  never relies on derivative information.

:func:`refine_mode` dispatches based on ``params["refine_method"]``.

Newton's method (Hellmann-Feynman derivative) and Nelder-Mead
(simplex) used to live here too. They were removed in favour of
``root``: empirically Newton's 25-30% wall-time advantage on small
graphs narrowed to 10-15% on 300-node graphs (see
``benchmark/bench_refine_scaling.py``), and the analytic-derivative
machinery created a coupling to ``graph.graph["dispersion_relation"]``
that became a maintenance hazard when adding new physics. Nelder-Mead
was strictly worse than root on every benchmark.
"""

import logging
import warnings

import numpy as np
from scipy.optimize import root
from skimage.feature import peak_local_max

from .quantum_graph import mode_quality

L = logging.getLogger(__name__)

REFINE_METHODS = ("root", "brownian")
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
    extent here so ``root`` honours the same locality that the Brownian
    ratchet achieves by taking small steps. If no search window is set
    (caller invoking the refiner directly), fall back to the
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


def refine_mode(initial_mode, graph, params, quality_method="eigenvalue", rng=None, **kwargs):
    """Dispatcher over the two refinement algorithms.

    Which one runs is controlled by ``params["refine_method"]`` —
    ``"root"`` (default, MINPACK ``hybr``) or ``"brownian"`` (legacy
    random-walk ratchet). Unknown values raise ``ValueError`` so
    typos surface at the graph boundary.

    Extra kwargs (e.g. ``disp``, ``save_mode_trajectories``) are
    passed through to the Brownian ratchet for backward compatibility;
    they emit a ``UserWarning`` when paired with ``method="root"``.
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
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        warnings.warn(
            f"refine_method={method!r} ignores kwargs: {unknown}",
            stacklevel=2,
        )
    return refine_mode_root(initial_mode, graph, params, quality_method=quality_method, rng=rng)


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

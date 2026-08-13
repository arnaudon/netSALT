"""Does the full-SALT solver reduce to the linear model just above threshold?

That reduction is a *physics* requirement, not a convention: at
``D0 -> D0_thr`` the modal intensity goes to zero, the hole burning switches
off, and the saturated operator becomes the unsaturated one the near-threshold
(single-pole) model linearises about. Both solvers must agree there. Further
above threshold they must *disagree*, and in a direction: the operator-level
solver sees the real spatial hole burning, which the near-threshold model only
carries to first order.

So this script measures the ratio ``newton / linear`` for the dominant mode at
several distances above threshold, over a ladder of graphs of increasing size.
It also reports the worst SALT residual of each sweep -- the acceptance test
from :func:`netsalt.salt_residuals` -- because an agreeable-looking ratio from
a solve that never converged means nothing.

One caveat the numbers cannot show on their own: the newton amplitude is
reported in the linear model's unit, via an analytic first-order change of
variables (``modes_df.attrs["salt_unit_scale"]``, which comes out near 1). The
*onset slope* is therefore shared by construction. What is genuinely predicted
is how the ratio departs from 1 as the pump rises.

Run from this directory::

    OMP_NUM_THREADS=1 python compare_linear_vs_salt.py            # the whole ladder
    OMP_NUM_THREADS=1 python compare_linear_vs_salt.py two_ring   # one graph
"""

from __future__ import annotations

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES))

from _common import (  # noqa: E402
    compute_mode_competition_matrix,
    compute_modal_intensities_full_salt_newton,
    curves,
    linear_on_grid,
    threshold_modes,
)

#: ``(label, example dir, builder, passive-mode method)``, smallest first.
LADDER = [
    ("line (Fabry-Perot)", "line_fabry_perot", "build", "grid"),
    ("ring + leads", "ring_leads", "build", "grid"),
    ("binary tree", "tree", "build", "grid"),
    ("long line", "long_line", "build", "grid"),
    ("two rings", "two_ring", "build_two_ring", "contour"),
    ("chaotic ring", "chaotic_ring", "build_chaotic_ring", "contour"),
    ("dense ring", "dense_ring", "build_dense_ring", "contour"),
    ("ring chain", "ring_chain", "build_ring_chain", "contour"),
    ("mini buffon", "mini_buffon", "build", "contour"),
]

#: Pumps at which to compare, as ``D0 / D0_thr - 1``. The first is as close to
#: threshold as the pump grid resolves.
EPSILONS = (0.05, 0.2, 0.5, 1.0)
D0_MAX_FACTOR = 1.0 + max(EPSILONS)
D0_STEPS = 21


def load_builder(example_dir, builder_name):
    """Import an example's ``run.py`` and hand back its graph builder."""
    path = EXAMPLES / example_dir / "run.py"
    spec = importlib.util.spec_from_file_location(f"_ex_{example_dir}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # safe: every run.py guards on __main__
    return getattr(module, builder_name)


def compare(label, example_dir, builder_name, method):
    with warnings.catch_warnings():
        # The equal-length jitter warning fires per graph and is not what this
        # script is about; issue #45 covers it.
        warnings.simplefilter("ignore")
        graph = load_builder(example_dir, builder_name)()
        n_nodes, n_edges = len(graph), len(graph.edges)
        tdf = threshold_modes(graph, method=method)
    thresholds = np.asarray(tdf["lasing_thresholds"]).ravel()
    finite = thresholds[np.isfinite(thresholds)]
    if not len(finite):
        print(f"{label}: no mode reaches threshold -- skipped")
        return None
    first = float(finite.min())
    dominant = int(np.argmin(np.where(np.isfinite(thresholds), thresholds, np.inf)))

    d0_max = D0_MAX_FACTOR * first
    grid = np.linspace(first, d0_max, D0_STEPS)

    competition = compute_mode_competition_matrix(graph, tdf)
    t0 = time.perf_counter()
    linear = linear_on_grid(tdf, competition, grid, len(tdf))
    t_linear = time.perf_counter() - t0

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        newton_df = compute_modal_intensities_full_salt_newton(
            graph, tdf.copy(), d0_max, D0_steps=D0_STEPS
        )
        residual_warnings = [c for c in caught if "SALT residual" in str(c.message)]
    t_newton = time.perf_counter() - t0
    pumps, newton = curves(newton_df)

    diagnostics = newton_df.attrs.get("salt_diagnostics")
    worst_residual = float(diagnostics["max_residual"].max()) if diagnostics is not None else np.nan
    work_nodes = int(newton_df.attrs.get("salt_work_nodes", 0))

    peak = max(linear.max(), newton.max(), 1e-30)
    n_lin = int(np.sum(linear[:, -1] > 1e-2 * peak))
    n_nwt = int(np.sum(newton[:, -1] > 1e-2 * peak))

    ratios = []
    for eps in EPSILONS:
        d0 = first * (1.0 + eps)
        lin_val = float(np.interp(d0, grid, linear[dominant]))
        nwt_val = float(np.interp(d0, pumps, newton[dominant]))
        ratios.append(nwt_val / lin_val if lin_val > 1e-12 else np.nan)

    scales = newton_df.attrs.get("salt_unit_scale", {})
    scale = scales.get(dominant, float("nan"))

    print(
        f"{label:22s} {n_nodes:5d} {n_edges:5d} {work_nodes:6d} {len(finite):5d} "
        f"{n_lin:4d} {n_nwt:4d}  "
        + " ".join(f"{r:7.3f}" if np.isfinite(r) else "    nan" for r in ratios)
        + f"  {worst_residual:9.2e} {scale:6.3f} {t_linear:7.1f} {t_newton:7.1f}"
        + ("  RESIDUAL-WARN" if residual_warnings else "")
    )
    return {
        "label": label,
        "ratios": ratios,
        "worst_residual": worst_residual,
        "n_lin": n_lin,
        "n_nwt": n_nwt,
    }


def main(selected=None):
    ladder = [row for row in LADDER if selected is None or row[1] in selected]
    header_eps = " ".join(f"e={e:<5g}" for e in EPSILONS)
    print(
        "graph                  nodes edges  work  thr  lin  nwt  "
        + header_eps
        + "   worst_res  scale  t_lin  t_nwt"
    )
    print("-" * 118)
    results = []
    for label, example_dir, builder, method in ladder:
        try:
            row = compare(label, example_dir, builder, method)
        except Exception as exc:  # keep the ladder going; report the failure
            print(f"{label:22s} FAILED: {type(exc).__name__}: {exc}")
            continue
        if row:
            results.append(row)

    print()
    print("ratio = newton / linear for the dominant mode; -> 1 as e -> 0 is the")
    print("physics requirement (the hole burning switches off at threshold).")
    near = [r["ratios"][0] for r in results if np.isfinite(r["ratios"][0])]
    if near:
        print(
            f"near-threshold (e={EPSILONS[0]}): median {np.median(near):.3f}, "
            f"range [{min(near):.3f}, {max(near):.3f}] over {len(near)} graphs"
        )


if __name__ == "__main__":
    main(sys.argv[1:] or None)

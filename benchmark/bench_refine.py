"""Benchmark the four mode-refinement algorithms.

For each (graph, dispersion, perturbation) combination, runs all four
refiners from the same initial guess and reports:

* wall time (median of 3 runs)
* ``mode_quality`` call count
* final ``|λ₁|``
* distance to the canonical mode (Beyn ground truth)

Also asserts that all converged refiners land on the same point within
``agreement_tol``, so the suite doubles as a regression check.

Usage::

    .venv/bin/python benchmark/bench_refine.py
    .venv/bin/python benchmark/bench_refine.py --output benchmark/results_refine.md
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import netsalt.algorithm as alg  # noqa: E402
from _common import (  # noqa: E402
    buffon_planar_graph,
    count_calls,
    line_graph,
    line_graph_with_pump,
    time_block,
)
from netsalt.algorithm import (  # noqa: E402
    refine_mode_brownian_ratchet,
    refine_mode_root,
)
from netsalt.contour import find_modes_contour  # noqa: E402
from netsalt.quantum_graph import mode_quality  # noqa: E402

REFINERS = [
    ("brownian", refine_mode_brownian_ratchet, {"rng": np.random.default_rng(0)}),
    ("root", refine_mode_root, {}),
]

# Tolerance for mode-position agreement between methods (loose because
# different solvers stop at different points along the convergence
# basin's floor).
AGREEMENT_TOL = 5e-3


def true_modes(
    graph, k_min=0.5, k_max=20.0, alpha_min=0.0, alpha_max=5.0, n_quad=200, probe_dim=None
):
    """Use Beyn (high probe_dim, fine quadrature) as the ground-truth
    mode locator. Beyn is configured tighter than what any of the
    refiners produce, so it serves as the consensus reference.
    """
    if probe_dim is None:
        probe_dim = min(60, len(graph))
    return find_modes_contour(
        graph,
        bounds=(k_min, k_max, alpha_min, alpha_max),
        n_quad=n_quad,
        probe_dim=probe_dim,
        rng=np.random.default_rng(0),
    )


def run_refiner(name, fn, kwargs, init, graph, params, n_repeats=3):
    """Time and count one refiner; return (median_seconds, n_eval, result)."""
    times = []
    last_result = None
    last_count = 0
    for _ in range(n_repeats):
        with count_calls(alg, "mode_quality") as counter, time_block() as t:
            try:
                last_result = fn(np.array(init, dtype=float), graph, dict(params), **kwargs)
            except Exception as exc:
                return None, 0, f"raised {type(exc).__name__}: {exc}"
        times.append(t.seconds)
        last_count = counter[0]
    return statistics.median(times), last_count, last_result


def perturb(mode, dk, da):
    return [mode[0] + dk, mode[1] + da]


def bench_graph(graph_name, graph, params, perturbations, ground_truth=None):
    """Run all refiners on every (mode, perturbation) pair on ``graph``.

    ``ground_truth`` is a (n_modes, 2) array of canonical mode locations;
    if None, computed via Beyn.
    """
    if ground_truth is None:
        ground_truth = true_modes(graph)
    rows = []
    consistency_failures = []

    # Pick 3 well-separated true modes spread across the rectangle so
    # benchmarks don't all converge on the same easy basin.
    n = len(ground_truth)
    if n == 0:
        print(f"[{graph_name}] no true modes inside scan rectangle, skipping")
        return rows, consistency_failures
    pick_idx = np.linspace(0, n - 1, min(3, n)).astype(int)
    sample = ground_truth[pick_idx]

    for mode in sample:
        for label, (dk, da) in perturbations.items():
            init = perturb(mode, dk, da)
            results = {}
            for name, fn, kwargs in REFINERS:
                t, ne, res = run_refiner(name, fn, kwargs, init, graph, params)
                results[name] = (t, ne, res)
                final_q = mode_quality(res, graph) if isinstance(res, np.ndarray) else float("nan")
                pos = f"{res[0]:.5f},{res[1]:.5f}" if isinstance(res, np.ndarray) else str(res)
                err = (
                    np.linalg.norm(np.asarray(res) - mode)
                    if isinstance(res, np.ndarray)
                    else float("nan")
                )
                rows.append(
                    {
                        "graph": graph_name,
                        "true_mode": f"{mode[0]:.4f},{mode[1]:.4f}",
                        "perturb": label,
                        "method": name,
                        "time_ms": (t * 1e3) if t is not None else float("nan"),
                        "n_eval": ne,
                        "final_q": final_q,
                        "position": pos,
                        "err_to_truth": err,
                    }
                )

            # Cross-method agreement check: all converged refiners land
            # in the same neighbourhood.
            converged = {n: r for n, (_, _, r) in results.items() if isinstance(r, np.ndarray)}
            if len(converged) >= 2:
                refs = list(converged.values())
                center = np.mean(refs, axis=0)
                for name, r in converged.items():
                    if np.linalg.norm(r - center) > AGREEMENT_TOL:
                        consistency_failures.append(
                            f"[{graph_name}] init={init}: {name} → "
                            f"{r.tolist()} disagrees with consensus {center.tolist()}"
                        )
    return rows, consistency_failures


def render_md(rows: list[dict]) -> str:
    """Render rows as a grouped markdown table."""
    lines = []
    by_graph = {}
    for r in rows:
        by_graph.setdefault(r["graph"], []).append(r)

    for graph_name, group in by_graph.items():
        lines.append(f"### {graph_name}\n")
        lines.append(
            "| true_mode | perturb | method | time (ms) | evals | final `\\|λ\\|` | error |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for r in group:
            lines.append(
                f"| {r['true_mode']} | {r['perturb']} | {r['method']} | "
                f"{r['time_ms']:.1f} | {r['n_eval']} | "
                f"{r['final_q']:.2e} | {r['err_to_truth']:.2e} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="markdown file to write the result table to",
    )
    args = parser.parse_args()

    perturbations = {
        "near (0.005, 0.005)": (0.005, 0.005),
        "mid  (0.02,  0.02 )": (0.02, 0.02),
        "far  (0.05,  0.05 )": (0.05, 0.05),
    }

    rows = []
    failures = []

    workloads = []

    # Line graphs: classic regular topology, modes degenerate per family.
    # Use n_edges ≥ 15 — shorter segments destabilise the laplacian for
    # this dielectric / total-length combination.
    for n_edges in (15, 20):
        g = line_graph(n_edges=n_edges)
        workloads.append((f"line, n_edges={n_edges}, dielectric=4", g))

    # Pump dispersion (D0=0 reduces to dielectric, but the dispersion
    # function itself is the more complex one used in production — useful
    # for catching dispersion-related regressions in root).
    workloads.append(
        (
            "line + pump dispersion (D0=0), n_edges=20",
            line_graph_with_pump(n_edges=20),
        )
    )

    # Random planar buffon graph — irregular topology, modes
    # nondegenerate, each at its own ``alpha``. Stress-tests refiners on
    # something less symmetric than the line.
    workloads.append(
        (
            "buffon planar (n_lines=6 → ~60 nodes)",
            buffon_planar_graph(n_lines=6, total_length=1.0, seed=2),
        )
    )

    for gname, graph in workloads:
        params = dict(graph.graph["params"])
        params.update(
            {
                "quality_threshold": 1e-5,
                "search_stepsize": 0.01,
                "max_steps": 500,
                "k_min": 0.5,
                "k_max": 20.0,
                "alpha_min": 0.0,
                "alpha_max": 5.0,
            }
        )
        graph.graph["params"] = params
        truth = true_modes(graph)
        if len(truth) == 0:
            print(f"[{gname}] no true modes found, skipping")
            continue
        r, f = bench_graph(gname, graph, params, perturbations, ground_truth=truth)
        rows.extend(r)
        failures.extend(f)

    md = render_md(rows)
    print(md)
    if failures:
        print("\n!! consistency failures: !!")
        for line in failures:
            print(" ", line)
    else:
        print(f"\n[ok] all converged refiners agree within {AGREEMENT_TOL:.0e} of each other.")

    if args.output:
        Path(args.output).write_text(md + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

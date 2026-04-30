"""Refinement scaling on production-like batches.

``find_threshold_lasing_modes`` and ``pump_trajectories`` refine
*every* mode at *every* ``D0`` step, so the relevant cost is
**per-refine wall time × N_modes × N_steps**. This benchmark sweeps:

* graph size (buffon ``n_lines = 10, 15, 20`` → ~125, 189, 340 nodes)
* mode batch size (refine N=10, 30, 50 distinct modes per graph)

For each (graph, batch) combination it runs ``root`` (default) and
``brownian`` (legacy ratchet), each starting from a small (0.01)
perturbation of a true Beyn-found mode, and reports total wall time
+ total ``mode_quality`` calls. Per-refine time is independent of
batch size (linear scaling), so the per-refine ratio is the
production-relevant number.

Earlier versions also benchmarked Newton (Hellmann-Feynman + Armijo)
and Nelder-Mead. Both were removed in favour of root: Newton's
25-30% speedup at small graph sizes shrank to 10-15% at 340 nodes
and required carrying around a derivative + fallback path that
coupled to ``graph.graph["dispersion_relation"]``. Nelder-Mead was
strictly worse than root.

Usage::

    .venv/bin/python benchmark/bench_refine_scaling.py
    .venv/bin/python benchmark/bench_refine_scaling.py --output benchmark/results_refine_scaling.md
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import netsalt.algorithm as alg  # noqa: E402
from _common import buffon_planar_graph, count_calls, time_block  # noqa: E402
from netsalt.algorithm import (  # noqa: E402
    refine_mode_brownian_ratchet,
    refine_mode_root,
)
from netsalt.contour import find_modes_contour  # noqa: E402
from netsalt.quantum_graph import mode_quality  # noqa: E402

REFINERS = [
    ("root", refine_mode_root, {}),
    ("brownian", refine_mode_brownian_ratchet, {"rng": np.random.default_rng(0)}),
]


def discover_modes(graph, bounds, n_quad, probe_dim, n_k):
    """Use Beyn subdivided as the source of truth for refine inputs."""
    return find_modes_contour(
        graph,
        bounds=bounds,
        n_k=n_k,
        n_alpha=1,
        n_quad=n_quad,
        probe_dim=probe_dim,
        rng=np.random.default_rng(0),
    )


def refine_batch(name, fn, kwargs, modes, perturb, graph, params):
    """Refine every ``mode`` in the batch and return total wall time
    and total ``mode_quality`` calls."""
    converged = 0
    bad_residual = 0
    total_q = 0
    with count_calls(alg, "mode_quality") as counter, time_block() as t:
        for true_mode in modes:
            init = np.array([true_mode[0] + perturb, true_mode[1] + perturb], dtype=float)
            res = fn(init, graph, dict(params), **kwargs)
            if isinstance(res, np.ndarray):
                converged += 1
                q = mode_quality(res, graph)
                total_q += 1  # the mode_quality counter saw this too
                if q > params.get("quality_threshold", 1e-4):
                    bad_residual += 1
            # else: refiner returned None — counted as not converged
    return {
        "method": name,
        "wall_s": t.seconds,
        "evals": counter[0],
        "n_modes": len(modes),
        "converged": converged,
        "bad_residual": bad_residual,
    }


def bench_one_graph(label, graph, bounds, n_quad, probe_dim, n_k, batch_sizes, perturb):
    """Run all refiners on subsets of ``batch_sizes`` true modes."""
    print(f"\n=== {label} ({len(graph)} nodes, {len(graph.edges)} edges) ===")
    truth = discover_modes(graph, bounds, n_quad, probe_dim, n_k)
    print(f"discovered {len(truth)} modes via Beyn")
    if len(truth) == 0:
        print("  no modes — skipping")
        return []

    rows = []
    for batch_size in batch_sizes:
        if batch_size > len(truth):
            print(f"  batch_size={batch_size} > {len(truth)} discovered — skipping")
            continue
        # Spread the batch across the rectangle so we don't sample only easy modes.
        idx = np.linspace(0, len(truth) - 1, batch_size).astype(int)
        modes = truth[idx]
        params = dict(graph.graph["params"])
        params.update(
            {
                "quality_threshold": 1e-5,
                "search_stepsize": 0.01,
                "max_steps": 500,
            }
        )
        # WorkerModes-style search window centred per-mode would be more
        # realistic, but for a static workload setting a window around
        # the rectangle is enough — the perturbation is small.
        params["k_min"] = bounds[0]
        params["k_max"] = bounds[1]
        params["alpha_min"] = bounds[2]
        params["alpha_max"] = bounds[3]
        graph.graph["params"] = params

        for name, fn, kwargs in REFINERS:
            row = refine_batch(name, fn, kwargs, modes, perturb, graph, params)
            row["graph"] = label
            row["graph_nodes"] = len(graph)
            row["batch_size"] = batch_size
            rows.append(row)
            ms_per_refine = row["wall_s"] * 1e3 / max(row["n_modes"], 1)
            evals_per_refine = row["evals"] / max(row["n_modes"], 1)
            print(
                f"  batch={batch_size:>2}, {name:>11}: "
                f"{row['wall_s']:>6.2f}s total, "
                f"{ms_per_refine:>5.1f} ms/refine, "
                f"{evals_per_refine:>5.1f} evals/refine, "
                f"{row['converged']}/{row['n_modes']} converged "
                f"({row['bad_residual']} bad residual)"
            )
    return rows


def render_md(rows):
    """One table per graph, columns per method, rows per batch size."""
    by_graph = {}
    for r in rows:
        by_graph.setdefault(r["graph"], []).append(r)

    lines = []
    for label, group in by_graph.items():
        nodes = group[0]["graph_nodes"]
        lines.append(f"### {label} ({nodes} nodes)\n")
        lines.append(
            "| batch | method | total (s) | ms/refine | evals/refine | converged |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|")
        for r in group:
            ms_per_refine = r["wall_s"] * 1e3 / max(r["n_modes"], 1)
            evals_per_refine = r["evals"] / max(r["n_modes"], 1)
            lines.append(
                f"| {r['batch_size']} | {r['method']} | "
                f"{r['wall_s']:.2f} | {ms_per_refine:.1f} | "
                f"{evals_per_refine:.1f} | "
                f"{r['converged']}/{r['n_modes']} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_summary(rows):
    """Compute median ms/refine per (method, graph_size) for the
    largest batch on each graph — that's the production-relevant
    number for D0 tracking."""
    out = []
    by_graph = {}
    for r in rows:
        by_graph.setdefault(r["graph_nodes"], []).append(r)
    out.append("## Summary: ms/refine at the largest batch per graph size\n")
    out.append("| nodes | root | brownian | brownian/root |")
    out.append("|---:|---:|---:|---:|")
    for nodes in sorted(by_graph):
        group = by_graph[nodes]
        max_batch = max(r["batch_size"] for r in group)
        per_method = {}
        for r in group:
            if r["batch_size"] != max_batch:
                continue
            per_method[r["method"]] = r["wall_s"] * 1e3 / max(r["n_modes"], 1)
        if "root" not in per_method:
            continue
        ratio = per_method.get("brownian", float("nan")) / per_method["root"]
        out.append(
            f"| {nodes} | "
            f"{per_method.get('root', float('nan')):.1f} | "
            f"{per_method.get('brownian', float('nan')):.1f} | "
            f"{ratio:.1f}× |"
        )
    out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Three buffon sizes spanning the netsalt-typical range.
    # ``total_length`` is sized so each graph has ~50-150 modes in the
    # rectangle — enough to compare batch sizes 10 / 30 / 50.
    workloads = [
        # (label, n_lines, total_length, k_max, n_quad, probe_dim_target, n_k, batch_sizes, perturb)
        ("buffon n_lines=10", 10, 8.0, 30.0, 200, 30, 8, (10, 30, 50), 0.01),
        ("buffon n_lines=15", 15, 6.0, 30.0, 200, 40, 6, (10, 30, 50), 0.01),
        ("buffon n_lines=20", 20, 6.0, 30.0, 200, 60, 4, (10, 30, 50), 0.01),
    ]

    rows = []
    for (label, n_lines, L, kmax, n_quad, pd, n_k, batches, perturb) in workloads:
        g = buffon_planar_graph(n_lines=n_lines, total_length=L, seed=2)
        bounds = (0.5, kmax, 0.0, 5.0)
        rows.extend(
            bench_one_graph(label, g, bounds, n_quad, min(pd, len(g)), n_k, batches, perturb)
        )

    md = render_md(rows)
    summary = render_summary(rows)

    print()
    print(summary)

    if args.output:
        Path(args.output).write_text(
            md + "\n" + summary + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

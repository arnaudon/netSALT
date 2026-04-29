"""Benchmark mode-search algorithms.

Compares three approaches for locating *all* modes in a scan rectangle:

1. ``find_modes_contour`` — single Beyn contour
2. ``find_modes_contour_subdivided`` — partitioned Beyn (more accurate
   when the mode count exceeds ``probe_dim``)
3. Grid-scan + ``peak_local_max`` + per-mode ``refine_mode_root``

Reports wall time, the number of modes found, the worst ``|λ₁|``, and
the agreement between methods (modes within ``match_tol``).

Usage::

    .venv/bin/python benchmark/bench_search.py
    .venv/bin/python benchmark/bench_search.py --output benchmark/results_search.md
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _common import buffon_planar_graph, line_graph, time_block  # noqa: E402
from netsalt.algorithm import find_rough_modes_from_scan, refine_mode_root  # noqa: E402
from netsalt.contour import (  # noqa: E402
    find_modes_contour,
    find_modes_contour_subdivided,
)
from netsalt.modes import scan_frequencies  # noqa: E402
from netsalt.quantum_graph import mode_quality  # noqa: E402
from netsalt.utils import get_scan_grid  # noqa: E402


MATCH_TOL = 1e-2  # consider modes equal if Euclidean distance ≤ this


def median_time(fn, n_repeats=3):
    times = []
    last_result = None
    for _ in range(n_repeats):
        with time_block() as t:
            last_result = fn()
        times.append(t.seconds)
    return statistics.median(times), last_result


def grid_path(graph, params):
    """Reproduce the legacy mode-search pipeline."""
    qualities = scan_frequencies(graph)
    ks, alphas = get_scan_grid(graph)
    rough = find_rough_modes_from_scan(
        ks, alphas, qualities, min_distance=2, threshold_abs=1.0
    )
    refined = []
    for r in rough:
        out = refine_mode_root(np.array(r, dtype=float), graph, params)
        if out is not None:
            refined.append(out)
    return np.asarray(refined) if refined else np.empty((0, 2))


def match_modes(a, b, tol=MATCH_TOL):
    """Return (matched_count, only_a, only_b).

    A mode in ``a`` is matched to its nearest neighbour in ``b`` within
    ``tol`` — greedy, but fine for this benchmark's mode counts.
    """
    if len(a) == 0 or len(b) == 0:
        return 0, len(a), len(b)
    used_b = np.zeros(len(b), dtype=bool)
    matched = 0
    for ai in a:
        d = np.linalg.norm(b - ai, axis=1)
        d[used_b] = np.inf
        if d.min() <= tol:
            matched += 1
            used_b[d.argmin()] = True
    return matched, len(a) - matched, (~used_b).sum()


def worst_quality(modes, graph):
    if len(modes) == 0:
        return float("nan")
    return max(mode_quality(m, graph) for m in modes)


def bench_one(graph_name, graph, params, k_max, alpha_max, n_quad, probe_dim, n_k):
    """Run all three methods on this graph and report a row + agreement."""
    rows = []
    notes = []

    bounds = (0.5, k_max, 0.0, alpha_max)
    rng_seed = 0

    def f_single():
        return find_modes_contour(
            graph,
            bounds=bounds,
            n_quad=n_quad,
            probe_dim=probe_dim,
            rng=np.random.default_rng(rng_seed),
        )

    def f_subdiv():
        return find_modes_contour_subdivided(
            graph,
            bounds=bounds,
            n_k=n_k,
            n_alpha=1,
            n_quad=n_quad,
            probe_dim=probe_dim,
            rng=np.random.default_rng(rng_seed),
        )

    def f_grid():
        params_local = dict(params)
        params_local["k_min"] = bounds[0]
        params_local["k_max"] = bounds[1]
        params_local["alpha_min"] = bounds[2]
        params_local["alpha_max"] = bounds[3]
        params_local["k_n"] = 80
        params_local["alpha_n"] = 20
        params_local["n_workers"] = 1
        params_local["search_stepsize"] = (params_local["k_max"] - params_local["k_min"]) / 80
        graph.graph["params"] = params_local
        return grid_path(graph, params_local)

    methods = [
        ("contour", f_single),
        ("contour-subdiv", f_subdiv),
        ("grid+root", f_grid),
    ]
    results = {}
    for name, fn in methods:
        t, modes = median_time(fn, n_repeats=2 if name == "grid+root" else 3)
        results[name] = (t, modes)
        rows.append(
            {
                "graph": graph_name,
                "method": name,
                "time_ms": t * 1e3,
                "n_modes": len(modes),
                "worst_q": worst_quality(modes, graph),
            }
        )

    # Cross-method agreement against the subdivided contour (gold reference).
    ref_name = "contour-subdiv"
    ref_modes = results[ref_name][1]
    for name, (_, modes) in results.items():
        if name == ref_name:
            continue
        matched, missed, extras = match_modes(ref_modes, modes)
        notes.append(
            f"[{graph_name}] {name} vs {ref_name}: "
            f"matched={matched} missed={missed} extra={extras}"
        )

    return rows, notes


def render_md(rows):
    lines = []
    by_graph = {}
    for r in rows:
        by_graph.setdefault(r["graph"], []).append(r)
    for graph_name, group in by_graph.items():
        lines.append(f"### {graph_name}\n")
        lines.append("| method | time (ms) | n_modes | worst |λ| |")
        lines.append("|---|---:|---:|---:|")
        for r in group:
            lines.append(
                f"| {r['method']} | {r['time_ms']:.1f} | "
                f"{r['n_modes']} | {r['worst_q']:.2e} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rows = []
    notes = []

    # Workloads cover line graphs (regular, mode degenerate) of two
    # sizes and a buffon-style random planar graph (irregular, modes
    # well-separated). For each one, scan rectangles are sized to
    # ~10–20 modes — small enough that grid+root completes quickly.
    line_graph_15 = line_graph(n_edges=15, dielectric=4.0, total_length=1.0)
    line_graph_20 = line_graph(n_edges=20, dielectric=4.0, total_length=1.0)
    buffon_g = buffon_planar_graph(n_lines=6, total_length=1.0, seed=2)

    cases = [
        # (label, graph, kmax, alpha_max, n_quad, probe_dim, n_k)
        ("line n=15 (k ∈ [0.5,15], α ∈ [0,5])", line_graph_15, 15.0, 5.0, 120, 16, 2),
        ("line n=20 (k ∈ [0.5,15], α ∈ [0,5])", line_graph_20, 15.0, 5.0, 120, 21, 2),
        ("buffon n_lines=6, ~60 nodes (k ∈ [0.5,20], α ∈ [0,5])", buffon_g, 20.0, 5.0, 200, 40, 2),
    ]
    for label, graph, kmax, amax, n_quad, probe_dim, n_k in cases:
        params = dict(graph.graph["params"])
        params.update(
            {
                "quality_threshold": 1e-4,
                "search_stepsize": 0.05,
                "max_steps": 300,
                "n_workers": 1,
            }
        )
        graph.graph["params"] = params
        r, n = bench_one(label, graph, params, kmax, amax, n_quad, probe_dim, n_k)
        rows.extend(r)
        notes.extend(n)

    md = render_md(rows)
    print(md)
    print("\n# Cross-method agreement (vs subdivided contour, tol=%g)\n" % MATCH_TOL)
    for line in notes:
        print(" ", line)

    if args.output:
        out = md + "\n\n## Cross-method agreement\n\n"
        out += "\n".join(f"- {n}" for n in notes) + "\n"
        Path(args.output).write_text(out, encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

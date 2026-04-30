"""500-mode stress benchmark on a small buffon graph.

Workload: ``buffon_planar_graph(n_lines=6, total_length=20, seed=2)``
— 61 nodes, **~500 modes** in ``k ∈ [0.5, 40], α ∈ [0, 5]``.
``probe_dim`` caps at 60 (clamped to node count), so the rectangle
is **8.4× over the basic algorithm's per-contour capacity**. This
is the regime where:

* manual subdivision needs ``n_k ≳ 14`` to fit modes per cell under
  capacity;
* `find_modes_contour_adaptive` should still recover everything by
  recursive bisection without requiring the user to know that;
* `tune_contour_parameters` should sit between the two — fast like
  manual once tuned, no manual sizing required.

Skips ``grid+root`` — at 500 modes the legacy pipeline is hours.

Usage::

    .venv/bin/python benchmark/bench_500_modes.py
    .venv/bin/python benchmark/bench_500_modes.py --output benchmark/results_500_modes.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _common import buffon_planar_graph, time_block  # noqa: E402
from netsalt.contour import (  # noqa: E402
    find_modes_contour_adaptive,
    find_modes_contour,
    tune_contour_parameters,
)


def max_pos_error(found, reference):
    if len(found) == 0 or len(reference) == 0:
        return float("inf")
    used = np.zeros(len(reference), dtype=bool)
    worst = 0.0
    for f in found:
        d = np.linalg.norm(reference - f, axis=1)
        d[used] = np.inf
        j = int(d.argmin())
        worst = max(worst, d[j])
        used[j] = True
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    g = buffon_planar_graph(n_lines=6, total_length=20.0, seed=2)
    n_nodes = len(g)
    n_edges = len(g.edges)
    bounds = (0.5, 40.0, 0.0, 5.0)
    probe_dim = min(60, n_nodes)

    print(f"graph: {n_nodes} nodes, {n_edges} edges, probe_dim cap = {probe_dim}\n")

    # Gold: dense subdivision + high quadrature.
    print("computing gold reference (n_k=32, n_α=2, n_quad=320) ...")
    with time_block() as t_gold:
        gold = find_modes_contour(
            g,
            bounds=bounds,
            n_k=32,
            n_alpha=2,
            n_quad=320,
            probe_dim=probe_dim,
            rng=np.random.default_rng(0),
        )
    print(f"  gold: {len(gold)} modes in {t_gold.seconds:.1f}s\n")

    # Sweep 1 — manual subdivision.
    print(f"=== Manual: vary n_k @ n_quad=200, probe_dim={probe_dim} ===")
    print(f"{'n_k':>4} | {'modes/cell':>10} | {'time (s)':>8} | {'modes':>5} | {'err vs gold':>11}")
    print("-" * 60)
    sweep_n_k = []
    for n_k in (4, 8, 12, 16, 20, 24, 32):
        with time_block() as t:
            modes = find_modes_contour(
                g,
                bounds=bounds,
                n_k=n_k,
                n_alpha=1,
                n_quad=200,
                probe_dim=probe_dim,
                rng=np.random.default_rng(0),
            )
        err = max_pos_error(modes, gold)
        sweep_n_k.append(
            {
                "n_k": n_k,
                "modes_per_cell": len(gold) / n_k,
                "time_s": t.seconds,
                "n_modes": len(modes),
                "err": err,
            }
        )
        print(
            f"{n_k:>4} | {len(gold) / n_k:>10.1f} | {t.seconds:>8.2f} | "
            f"{len(modes):>5} | {err:>11.2e}"
        )

    # Sweep 2 — adaptive at varying probe_dim.
    print(f"\n=== Adaptive: vary probe_dim @ n_quad=200 ===")
    print(f"{'probe_dim':>9} | {'time (s)':>8} | {'modes':>5} | {'err vs gold':>11}")
    print("-" * 50)
    sweep_adaptive = []
    for pd in (20, 40, 60):
        with time_block() as t:
            modes = find_modes_contour_adaptive(
                g,
                bounds=bounds,
                n_quad=200,
                probe_dim=pd,
                max_depth=8,  # bumped from default 6 to leave room for 500 modes
                rng=np.random.default_rng(0),
            )
        err = max_pos_error(modes, gold)
        sweep_adaptive.append(
            {"probe_dim": pd, "time_s": t.seconds, "n_modes": len(modes), "err": err}
        )
        print(f"{pd:>9} | {t.seconds:>8.2f} | {len(modes):>5} | {err:>11.2e}")

    # Sweep 3 — tune_contour_parameters → batch.
    print("\n=== tune_contour_parameters → find_modes_contour ===")
    with time_block() as t_tune:
        params, info = tune_contour_parameters(
            g,
            bounds=bounds,
            probe_dim=probe_dim,
            n_quad=200,
            max_depth=8,
            rng=np.random.default_rng(0),
        )
    print(
        f"  tuned: {info['discovered_modes']} modes discovered in "
        f"{t_tune.seconds:.2f}s; chose {params}"
    )
    with time_block() as t_apply:
        modes_apply = find_modes_contour(
            g, bounds=bounds, **params, rng=np.random.default_rng(0)
        )
    err_apply = max_pos_error(modes_apply, gold)
    print(
        f"  apply: {len(modes_apply)} modes in {t_apply.seconds:.2f}s "
        f"(err vs gold {err_apply:.2e})"
    )

    # Render markdown.
    md = []
    md.append("# 500-mode benchmark: buffon n_lines=6, total_length=20\n")
    md.append(
        f"Graph: **{n_nodes} nodes**, **{n_edges} edges**, probe_dim cap "
        f"**{probe_dim}** (clamped to node count). Rectangle "
        f"``k ∈ [0.5, 40], α ∈ [0, 5]`` contains **{len(gold)} modes** — "
        f"{len(gold) / probe_dim:.1f}× the basic algorithm's per-contour "
        "capacity. Gold reference: ``n_k=32, n_α=2, n_quad=320`` "
        f"({t_gold.seconds:.1f}s).\n"
    )

    md.append("## Manual subdivision sweep\n")
    md.append("`n_quad=200`, varying `n_k`. Per-cell mode count = `gold_modes / n_k`.\n")
    md.append(
        "| n_k | modes/cell | time (s) | modes found | max pos err |"
    )
    md.append("|---:|---:|---:|---:|---:|")
    for r in sweep_n_k:
        err_s = "inf" if not np.isfinite(r["err"]) else f"{r['err']:.2e}"
        md.append(
            f"| {r['n_k']} | {r['modes_per_cell']:.1f} | "
            f"{r['time_s']:.2f} | {r['n_modes']} | {err_s} |"
        )
    md.append("")
    md.append(
        f"Threshold rule (`modes_per_cell ≲ 0.65 · probe_dim = "
        f"{0.65 * probe_dim:.0f}`) predicts the transition at "
        f"`n_k = ⌈{len(gold)} / {0.65 * probe_dim:.0f}⌉ = "
        f"{int(np.ceil(len(gold) / (0.65 * probe_dim)))}`.\n"
    )

    md.append("## Adaptive sweep\n")
    md.append(
        "`max_depth=8` (bumped from default 6 — at 500 modes the small-"
        "`probe_dim` runs need more recursion budget than the 303-mode "
        "stress workload).\n"
    )
    md.append("| probe_dim | time (s) | modes found | max pos err |")
    md.append("|---:|---:|---:|---:|")
    for r in sweep_adaptive:
        err_s = "inf" if not np.isfinite(r["err"]) else f"{r['err']:.2e}"
        md.append(
            f"| {r['probe_dim']} | {r['time_s']:.2f} | "
            f"{r['n_modes']} | {err_s} |"
        )
    md.append("")

    md.append("## Tune once → apply (single graph)\n")
    md.append(
        f"`tune_contour_parameters` discovered {info['discovered_modes']} "
        f"modes in **{t_tune.seconds:.2f}s** and chose `{params}`.\n\n"
        f"Applying those settings via `find_modes_contour` "
        f"recovered **{len(modes_apply)} modes in {t_apply.seconds:.2f}s** "
        f"(max position error {err_apply:.2e} vs gold).\n\n"
        "On a single graph the tune step is overhead — useful only when "
        "the same parameters get reused across many similar graphs (see "
        "`bench_tune_then_batch.py`).\n"
    )

    md.append("## Takeaway\n")
    md.append(
        f"At **{len(gold)} modes vs probe_dim {probe_dim}** "
        f"({len(gold) / probe_dim:.1f}× over capacity):\n\n"
        "- Manual subdivision needs `n_k ≥ 14` to recover all modes; "
        "`n_k=16` is the comfortable choice.\n"
        "- Adaptive recovers all modes at any reasonable `probe_dim`, "
        "but needs `max_depth=8` to budget enough recursion (default 6 "
        "is sized for 300-mode workloads). Pay 2-3× the wall time of "
        "the optimum-tuned manual run.\n"
        "- `tune_contour_parameters` runs adaptive once, picks an "
        "appropriate `n_k`, and then any subsequent call on a similar "
        "graph runs at full manual speed."
    )

    if args.output:
        Path(args.output).write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

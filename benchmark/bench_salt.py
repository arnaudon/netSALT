"""Modal-intensity solver benchmark: linear / self_consistent / full_salt.

Compares the three L--I (intensity-vs-pump) solvers added for issue #42 on
**speed** and **accuracy**:

* ``linear`` — the original near-threshold SALT model (one pump-independent
  competition matrix, a linear solve, piecewise-linear curves).
* ``self_consistent`` — competition matrix rebuilt from the mode profiles at
  the *operating* pump (relaxes the frozen-threshold-profile approximation).
* ``full_salt`` — experimental nonlinear SALT with the per-edge spatial
  hole-burning denominator (relaxes gain clamping too); bends the L--I over.

The expensive, shared pipeline steps (passive modes, pump, trajectories,
thresholds, competition matrix) are run once through the normal cached pipeline;
only the modal-intensity step is swapped between solvers, so the table isolates
the cost and the drift of the three intensity models.

Usage::

    python benchmark/bench_salt.py                       # default: line_PRA
    python benchmark/bench_salt.py examples/line_PRA/config.yaml

Writes ``benchmark/bench_salt_ll.pdf`` (overlaid L--I curves) and, for
``full_salt``, ``benchmark/bench_salt_oversample.pdf`` (within-edge
convergence). Requires the example to be runnable from its own directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from netsalt.config_loader import load_config
from netsalt.modes import (
    compute_modal_intensities,
    compute_modal_intensities_full_salt,
    compute_modal_intensities_full_salt_newton,
    compute_modal_intensities_self_consistent,
    compute_mode_competition_matrix,
)
from netsalt.pipeline import (
    _attach_pump_to_graph,
    step_compute_mode_competition_matrix,
    step_compute_mode_trajectories,
    step_create_pump_profile,
    step_create_quantum_graph,
    step_find_passive_modes,
    step_find_threshold_modes,
    step_scan_frequencies,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def time_block():
    class _T:
        def __enter__(self):
            self.t0 = perf_counter()
            return self

        def __exit__(self, *a):
            self.seconds = perf_counter() - self.t0

    return _T()


def _ll_curve(df):
    """Extract ``(pumps, per_mode, total)`` from an intensities dataframe."""
    cols = [c for c in df.columns if isinstance(c, tuple) and c[0] == "modal_intensities"]
    pumps = np.array([c[1] for c in cols], dtype=float)
    order = np.argsort(pumps)
    pumps = pumps[order]
    data = df[[cols[i] for i in order]].to_numpy(dtype=float)  # (n_modes, n_pumps)
    total = np.nansum(data, axis=0)
    return pumps, data, total


def _shared_steps(p, lasing_modes_id):
    """Run the cached pipeline up to (and including) the competition matrix."""
    qg = step_create_quantum_graph(p)
    qualities = step_scan_frequencies(p, qg)
    passive_modes_df = step_find_passive_modes(p, qg, qualities)
    pump = step_create_pump_profile(p, qg, passive_modes_df, lasing_modes_id)
    trajectories_df = step_compute_mode_trajectories(p, qg, passive_modes_df, pump, lasing_modes_id)
    threshold_modes_df = step_find_threshold_modes(p, qg, trajectories_df, pump, lasing_modes_id)
    competition = step_compute_mode_competition_matrix(
        p, qg, threshold_modes_df, pump, lasing_modes_id
    )
    qg = _attach_pump_to_graph(p, qg, pump)
    return qg, threshold_modes_df, competition


def _solvers(p):
    """Map method name -> zero-arg callable returning an intensities dataframe."""
    d0_max = p.get("intensities_D0_max") or p.get("D0_max", 0.1)
    steps = p.get("salt_D0_steps", 30)

    def run_linear(qg, tdf, comp):
        return compute_modal_intensities(tdf.copy(), d0_max, comp)

    def run_self(qg, tdf, comp):
        return compute_modal_intensities_self_consistent(qg, tdf.copy(), d0_max, D0_steps=steps)

    def run_full(qg, tdf, comp, oversample_size=None):
        return compute_modal_intensities_full_salt(
            qg, tdf.copy(), d0_max, D0_steps=steps, oversample_size=oversample_size
        )

    return {"linear": run_linear, "self_consistent": run_self, "full_salt": run_full}


def benchmark(config_path: Path):
    config_path = config_path.resolve()
    workdir = config_path.parent
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        p = load_config(config_path.name)
        lasing_modes_id = p.get("lasing_modes_id")
        print(f"=== {config_path.relative_to(REPO)} ===")

        qg, tdf, comp = _shared_steps(p, lasing_modes_id)
        thresholds = np.asarray(tdf["lasing_thresholds"]).ravel()
        n_lasing = int(np.sum(thresholds < np.inf))
        print(f"graph: {len(qg)} nodes, {len(qg.edges)} edges; {n_lasing} lasing modes")

        solvers = _solvers(p)
        results = {}
        print(
            f"\n{'method':>16} | {'time (s)':>8} | {'n_active':>8} | {'tot@max':>10} | {'Δ vs lin':>9}"
        )
        print("-" * 64)

        linear_total_max = None
        for name in ("linear", "self_consistent", "full_salt"):
            with time_block() as t:
                df = solvers[name](qg, tdf, comp)
            pumps, per_mode, total = _ll_curve(df)
            results[name] = (pumps, per_mode, total)
            tot_max = total[-1] if len(total) else float("nan")
            n_active = int(np.sum(np.nan_to_num(per_mode[:, -1]) > 0)) if per_mode.size else 0
            if name == "linear":
                linear_total_max = tot_max
                delta = "-"
            else:
                delta = f"{(tot_max - linear_total_max):+.3e}"
            print(f"{name:>16} | {t.seconds:>8.3f} | {n_active:>8} | {tot_max:>10.3e} | {delta:>9}")

        _plot_ll(results, HERE / "bench_salt_ll.pdf")
        print(f"\nwrote {(HERE / 'bench_salt_ll.pdf').relative_to(REPO)}")

        _oversample_study(qg, tdf, p, HERE / "bench_salt_oversample.pdf")
        _newton_single_mode_study(qg, tdf, p, HERE / "bench_salt_newton.pdf")
    finally:
        os.chdir(cwd)


def _newton_single_mode_study(qg, tdf, p, out):
    """Operator-level Newton (single mode) vs the linear dominant-mode L--I.

    ``full_salt_newton`` solves only the dominant mode, so it is compared against
    that *same* mode's linear curve rather than the multi-mode totals. The key
    check is that its onset slope matches the linear ``1/(T_μμ·D0_thr)``.
    """
    d0_max = p.get("intensities_D0_max") or p.get("D0_max", 0.1)
    steps = p.get("salt_D0_steps", 30)
    thresholds = np.asarray(tdf["lasing_thresholds"]).ravel()
    if not np.any(thresholds < np.inf):
        return
    target = int(np.argmin(thresholds))
    d0_thr = float(thresholds[target])

    t_self = compute_mode_competition_matrix(qg, tdf)[target, target]
    lin_slope = 1.0 / (t_self * d0_thr)

    with time_block() as t:
        df = compute_modal_intensities_full_salt_newton(qg, tdf.copy(), d0_max, D0_steps=steps)
    sub = df["modal_intensities"]
    pumps = np.array(sorted(sub.columns))
    a = sub.loc[target, pumps].to_numpy(dtype=float)
    above = pumps > d0_thr + 1e-9
    newton_slope = a[above][0] / (pumps[above][0] - d0_thr) if above.any() else float("nan")

    print("\noperator-level Newton (single dominant mode):")
    print(f"  solve time            : {t.seconds:.1f} s  ({len(pumps)} pumps)")
    print(f"  onset slope newton/lin: {newton_slope / lin_slope:.3f}  (1.0 = reduces to linear)")
    print(f"  intensity @ max pump  : {a[-1]:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(pumps, np.clip(lin_slope * (pumps / d0_thr - 1.0), 0, None), "--", label="linear")
    plt.plot(pumps, a, "o-", ms=3, label="full_salt_newton")
    plt.xlabel("pump $D_0$")
    plt.ylabel(f"dominant-mode intensity (mode {target})")
    plt.legend()
    plt.title("Single-mode: operator-level Newton vs linear")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"wrote {out.relative_to(REPO)}")


def _plot_ll(results, out):
    plt.figure(figsize=(6, 4))
    for name, (pumps, _per_mode, total) in results.items():
        plt.plot(pumps, total, marker="o", ms=3, label=name)
    plt.xlabel("pump $D_0$")
    plt.ylabel("total modal intensity")
    plt.legend()
    plt.title("L--I curves: linear vs self_consistent vs full_salt")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _oversample_study(qg, tdf, p, out):
    """full_salt L--I at several oversample sizes -> within-edge convergence."""
    d0_max = p.get("intensities_D0_max") or p.get("D0_max", 0.1)
    steps = p.get("salt_D0_steps", 30)
    # sizes below the native edge length so oversample_graph actually subdivides
    sizes = [None, 0.05, 0.02]
    plt.figure(figsize=(6, 4))
    print("\nfull_salt within-edge (oversample) convergence:")
    for size in sizes:
        try:
            df = compute_modal_intensities_full_salt(
                qg, tdf.copy(), d0_max, D0_steps=steps, oversample_size=size
            )
        except Exception as exc:  # best-effort: oversampling may fail on some graphs
            print(f"  oversample_size={size}: skipped ({exc})")
            continue
        pumps, _per_mode, total = _ll_curve(df)
        label = "edge_size (native)" if size is None else f"oversample={size}"
        plt.plot(pumps, total, marker=".", label=label)
        print(f"  oversample_size={size}: tot@max = {total[-1]:.4e}")
    plt.xlabel("pump $D_0$")
    plt.ylabel("total modal intensity")
    plt.legend()
    plt.title("full_salt: within-edge saturation convergence")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"wrote {out.relative_to(REPO)}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    config = Path(argv[0]) if argv else REPO / "examples" / "line_PRA" / "config.yaml"
    benchmark(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())

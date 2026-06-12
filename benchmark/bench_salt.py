"""Modal-intensity solver benchmark: linear vs full_salt_newton.

Compares the two L--I (intensity-vs-pump) solvers (issue #42) on **speed** and
**accuracy**:

* ``linear`` — the near-threshold SALT model (one pump-independent
  competition matrix, a linear solve, piecewise-linear curves).
* ``full_salt_newton`` — the operator-level nonlinear SALT (solves the
  saturated eigenproblem per pump; reduces to ``linear`` at threshold).

The expensive, shared pipeline steps (passive modes, pump, trajectories,
thresholds, competition matrix) are run once through the normal cached pipeline;
only the modal-intensity step is swapped between solvers, so the table isolates
the cost and the drift of the two intensity models.

Usage::

    python benchmark/bench_salt.py                       # default: line_PRA
    python benchmark/bench_salt.py examples/line_PRA/config.yaml

Writes ``benchmark/bench_salt_ll.pdf`` (overlaid L--I curves) and
``benchmark/bench_salt_newton.pdf`` (the newton study). Requires the example to
be runnable from its own directory.
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
    compute_modal_intensities_full_salt_newton,
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

    def run_newton(qg, tdf, comp):
        return compute_modal_intensities_full_salt_newton(qg, tdf.copy(), d0_max, D0_steps=steps)

    return {"linear": run_linear, "full_salt_newton": run_newton}


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
        for name in ("linear", "full_salt_newton"):
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

        _newton_study(qg, tdf, p, HERE / "bench_salt_newton.pdf")
    finally:
        os.chdir(cwd)


def _newton_study(qg, tdf, p, out, steps=8):
    """Operator-level full-SALT Newton vs the linear model.

    Highlights two things: (1) the dominant mode's onset slope reduces to the
    linear ``1/(T_μμ·D0_thr)`` near threshold, and (2) above threshold full SALT
    deviates -- bent curves and competition-shifted secondary modes. With the
    within-edge hole burning resolved (default auto-oversampling) the lasing set
    agrees with the linear/competition-matrix count (e.g. both modes on
    ``line_PRA``, matching Ge-Chong-Stone Eq. 28); a too-coarse mesh over-clamps
    and spuriously drops modes. The Newton solve is expensive (a nested
    frequency/profile + amplitude solve per pump, on an oversampled graph), so it
    runs on a coarse ``steps`` grid.
    """
    d0_max = p.get("intensities_D0_max") or p.get("D0_max", 0.1)
    thresholds = np.asarray(tdf["lasing_thresholds"]).ravel()
    if not np.any(thresholds < np.inf):
        return
    target = int(np.argmin(thresholds))
    d0_thr = float(thresholds[target])

    t_lin = compute_mode_competition_matrix(qg, tdf)
    lin_df = compute_modal_intensities(tdf.copy(), d0_max, t_lin)
    lin_last = np.nan_to_num(lin_df["modal_intensities"].to_numpy()[:, -1])
    lin_slope = 1.0 / (t_lin[target, target] * d0_thr)

    with time_block() as t:
        df = compute_modal_intensities_full_salt_newton(qg, tdf.copy(), d0_max, D0_steps=steps)
    sub = df["modal_intensities"]
    pumps = np.array(sorted(sub.columns))
    new_last = np.nan_to_num(sub.to_numpy()[:, -1])
    a = sub.loc[target, pumps].to_numpy(dtype=float)
    above = pumps > d0_thr + 1e-9
    newton_slope = a[above][0] / (pumps[above][0] - d0_thr) if above.any() else float("nan")

    print(f"\noperator-level Newton full SALT ({t.seconds:.0f} s, {len(pumps)} pumps):")
    print(f"  dominant-mode onset slope newton/linear: {newton_slope / lin_slope:.3f}  (1.0 = ok)")
    print(f"  linear lases modes : {sorted(int(i) for i in np.where(lin_last > 0)[0])}")
    print(f"  newton lases modes : {sorted(int(i) for i in np.where(new_last > 1e-6)[0])}")
    print(
        "  (counts should agree with resolved hole burning; full SALT bends the curves above threshold)"
    )

    plt.figure(figsize=(6, 4))
    plt.plot(
        pumps, np.clip(lin_slope * (pumps / d0_thr - 1.0), 0, None), "--", label="linear (mode)"
    )
    plt.plot(pumps, a, "o-", ms=3, label="full_salt_newton (mode)")
    plt.xlabel("pump $D_0$")
    plt.ylabel(f"dominant-mode intensity (mode {target})")
    plt.legend()
    plt.title("Operator-level Newton vs linear (dominant mode)")
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
    plt.title("L--I curves: linear vs full_salt_newton")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    config = Path(argv[0]) if argv else REPO / "examples" / "line_PRA" / "config.yaml"
    benchmark(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())

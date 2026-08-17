"""Which lasing solution does the varying-operator SALT solver land on?

Backs AUDIT.md section 10. The production solver gets the buffon's L--I curve
right at 1.02x and 1.05x threshold and wrong above it; this script measures the
*branch* rather than the solver, and shows the two disagree.

Two things are computed on the same graph, through the same residual
(``netsalt.salt_varying._lam_varying``):

``branch``
    Amplitude continuation. Fix ``a``, solve for ``(k, D0)`` -- the same two
    equations ``Re/Im lambda_1 = 0``, a different pair of unknowns held -- and
    walk ``a`` up from zero. This traces the physical branch, the one continuously
    connected to ``a = 0`` at threshold, and reports ``D0(a)``.

``solver``
    :func:`~netsalt.salt_varying.solve_salt_varying` at the same pumps, each
    started from the previous solution, exactly as the sweep drives it.

Measured on ``examples/buffon/buffon_narrow`` (208 nodes, 243 edges,
``n_steps = 512``, lowest mode at ``k = 10.67933130``):

| D0/D0_thr | branch (truth) | solver | solver residual | converged |
| --- | --- | --- | --- | --- |
| 1.02 | 1.431 | 1.419 | 8.3e-07 | yes |
| 1.05 | 3.092 | 3.063 | 6.9e-07 | yes |
| 1.10 | 5.623 | 22.73 | 6.8e-07 | yes |
| 1.26 | 15.558 | 85.67 | 6.6e-01 | no |
| 2.09 | 68.077 | 412.2 | 5.8e-01 | no |

The branch column is independently anchored: its near-threshold slope is 75.5
against the analytic ``1/T00 = 74.01``, and ``max |k - k0| = 7.3e-05`` over the
whole continuation, comfortably inside the ``2.09e-04`` k cap, so it never leaves
this mode.

The load-bearing observation is that ``D0(a)`` is **strictly increasing** at all
56 continuation points from ``a = 0.2`` to ``a = 70``. A strictly monotone
``D0(a)`` is a bijection, so each pump has exactly one amplitude on this branch --
and the continuation places the solver's ``a = 22.73`` at ``D0 = 1.38x``
threshold, not at the ``1.10x`` it was asked for. The solver's answer is
therefore off the physical branch, however small its residual: it is the
amplitude belonging to a different pump.

That is what makes this a solver problem rather than an equations problem, and
it is why a small residual is not evidence here. Run it as::

    OMP_NUM_THREADS=1 python probe_varying_amplitude_branch.py <out-dir>

where ``<out-dir>`` is the ``out/`` of a completed
``examples/buffon/buffon_narrow`` run (``bash run.sh`` there first).
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import yaml
from scipy.optimize import least_squares

from netsalt.io import load_graph, load_modes
from netsalt.physics import gamma
from netsalt.salt_varying import (
    _lam_varying,
    edge_field_profiles,
    node_solution_varying,
    saturated_eps_profiles,
    solve_salt_varying,
)

#: The pumps to report, as multiples of the mode's own lasing threshold.
PUMPS = (1.02, 1.05, 1.10, 1.26, 2.09)

#: k excursion allowed per solve. On this graph the Weyl mean spacing is 8.4e-4
#: and `_BRANCH_SAFETY` is 0.25, so the production cap is 2.09e-04.
CAP = 2.09e-04


def load(out_dir: str):
    graph = load_graph(f"{out_dir}/quantum_graph.json")
    modes_df = load_modes(f"{out_dir}/lasing_thresholds_modes.h5")
    with open(f"{out_dir}/pump_profile.yaml") as handle:
        pump = np.asarray(yaml.safe_load(handle), dtype=float)
    thresholds = np.asarray(modes_df["lasing_thresholds"]).ravel()
    threshold_modes = modes_df["threshold_lasing_modes"].to_numpy()
    index = int(np.nanargmin(np.where(np.isfinite(thresholds), thresholds, np.inf)))
    return (
        graph,
        pump,
        index,
        float(np.real(threshold_modes[index])),
        float(thresholds[index]),
    )


def branch(graph, pump, k0, threshold, n_steps, amplitudes):
    """Continue in the amplitude; return ``[(a, D0/threshold, k - k0)]``."""

    def field_at(k, profiles):
        _, psi = node_solution_varying(float(k), graph, profiles, n_steps=n_steps)
        return edge_field_profiles(float(k), graph, psi, profiles, n_steps=n_steps, pump=pump)

    field = field_at(k0, None)
    k, D0 = k0, threshold
    out = []
    for a in amplitudes:
        frozen = [list(field)]

        def residual(x, _frozen=frozen, _a=a):
            profiles = saturated_eps_profiles(
                graph, [float(x[0])], [_a], _frozen, float(x[1]), pump
            )
            value = _lam_varying(graph, float(x[0]), profiles, n_steps)
            return np.array([value.real, value.imag])

        for _ in range(12):
            result = least_squares(
                residual,
                [k, D0],
                method="trf",
                bounds=([k0 - 0.05, 1e-12], [k0 + 0.05, np.inf]),
                xtol=1e-14,
                ftol=1e-14,
                x_scale="jac",
            )
            k, D0 = float(result.x[0]), float(result.x[1])
            profiles = saturated_eps_profiles(graph, [k], [a], frozen, D0, pump)
            gain = gamma(complex(k), graph.graph["params"])
            for profile in profiles:
                if profile is not None:
                    profile.gain = gain
            new = field_at(k, profiles)
            field = [0.3 * old + 0.7 * fresh for old, fresh in zip(field, new, strict=True)]
            frozen = [list(field)]
            if np.linalg.norm(result.fun) < 1e-9:
                break
        out.append((float(a), D0 / threshold, k - k0))
    return out


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "../buffon/buffon_narrow/out"
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    graph, pump, index, k0, threshold = load(out_dir)
    print(f"mode {index}  k0 = {k0:.8f}  D0_thr = {threshold:.8g}  n_steps = {n_steps}\n")

    grid = np.concatenate(
        [
            np.linspace(0.2, 2.0, 10),
            np.linspace(2.5, 6.0, 8),
            np.linspace(7.0, 20.0, 14),
            np.linspace(24.0, 70.0, 24),
        ]
    )
    print(f"amplitude continuation over {len(grid)} points ...", flush=True)
    t0 = time.time()
    traced = branch(graph, pump, k0, threshold, n_steps, grid)
    print(f"  {time.time() - t0:.0f} s", flush=True)

    ratios = [r[1] for r in traced]
    rising = all(b > a for a, b in zip(ratios, ratios[1:], strict=False))
    print(f"  D0(a) strictly increasing: {rising}")
    print(f"  max |k - k0| = {max(abs(r[2]) for r in traced):.3e}  (k cap {CAP:.2e})")
    if rising:
        print("  -> D0(a) is a bijection: exactly one amplitude per pump on this branch.\n")

    print(f"{'D0/thr':>7} {'branch':>10} {'solver':>10} {'err':>9} {'res':>9} {'conv':>5}")
    k, a = k0, 1e-3
    for f in PUMPS:
        truth = float(np.interp(f, ratios, [r[0] for r in traced]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solution = solve_salt_varying(
                graph,
                [k],
                [max(a, 1e-3)],
                threshold * f,
                pump,
                n_steps=n_steps,
                k_window_cap=[CAP],
            )
        k, a = float(solution.ks[0]), float(solution.amplitudes[0])
        err = 100.0 * (a / truth - 1.0) if truth else float("nan")
        print(
            f"{f:7.2f} {truth:10.3f} {a:10.3f} {err:+8.1f}% "
            f"{solution.residuals[0]:9.1e} {str(solution.converged):>5}"
        )


if __name__ == "__main__":
    main()

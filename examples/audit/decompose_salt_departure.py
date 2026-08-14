"""Why does full SALT land above the near-threshold model on some graphs and below on others?

``compare_linear_vs_salt.py`` measures the ratio ``newton / linear`` for the
dominant mode and finds it mostly above 1, but below on ``two_ring`` (0.977)
and on ``line_PRA`` in its two-mode regime (0.93). Both signs are expected. The
departure is a sum of three terms with different sign rules, and this script
separates them by re-solving the dominant mode *in isolation* with the
mechanisms switched on one at a time.

**(i) Self-saturation — strictly positive.** The lasing condition is
``D0 ∫ p·w_μ / (1 + Σ_ν u_ν f_ν) = 1`` with ``u = Γa`` and ``f = |Ê|²``; the
single-pole model truncates it at first order. Since
``1/(1+S) = (1-S) + S²/(1+S)`` and ``S²/(1+S) ≥ 0``, the truncation *under-counts*
the gain still available, so the exact solution needs more saturation to clamp:
``a_exact > a_linear``, growing roughly linearly in the pump excess.

**(ii) Field relaxation — sign-indefinite.** Freezing the hole-burning profile
at its threshold shape is a second approximation. Letting it relax changes the
gain integral at fixed amplitude by a second-order amount that *would* have a
fixed sign if the eigenproblem were variational. SALT's operator is
non-Hermitian, so the relaxed profile is stationary but not extremal, and no
sign theorem applies.

**(iii) Competition — negative for the dominant mode.** Writing the
second-order correction as ``T δu = R`` with ``R_μ = ∫ p·w_μ·S² ≥ 0``: ``T`` has
positive entries, so ``T⁻¹`` has negative off-diagonals, and a competitor's
positive correction subtracts from the dominant mode.

(Frequency pulling is a fourth candidate and is numerically dead: the largest
shift on the ladder is 0.0016·γ⊥.)

The columns below separate (i) from (ii): "frozen field" holds ``f`` at its
threshold shape so only (i) acts; "relaxed" lets it move, so the difference
between the two rows *is* (ii). Term (iii) is the gap between this script's
isolated ratio and ``compare_linear_vs_salt.py``'s full-sweep ratio, since the
sweep lets competitors lase and this script does not.

Run from this directory::

    OMP_NUM_THREADS=1 python decompose_salt_departure.py two_ring
    OMP_NUM_THREADS=1 python decompose_salt_departure.py dense_ring 24
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES))

from _common import compute_mode_competition_matrix, threshold_modes  # noqa: E402

import netsalt.modes as modes  # noqa: E402
import netsalt.quantum_graph as quantum_graph  # noqa: E402
from netsalt.quantum_graph import graph_with_pump, oversample_graph  # noqa: E402
from netsalt.utils import from_complex  # noqa: E402

#: ``example dir -> (builder, passive-mode method)``.
BUILDERS = {
    "line_fabry_perot": ("build", "grid"),
    "ring_leads": ("build", "grid"),
    "tree": ("build", "grid"),
    "long_line": ("build", "grid"),
    "two_ring": ("build_two_ring", "contour"),
    "chaotic_ring": ("build_chaotic_ring", "contour"),
    "dense_ring": ("build_dense_ring", "contour"),
    "ring_chain": ("build_ring_chain", "contour"),
    "mini_buffon": ("build", "contour"),
}

EPSILONS = (0.2, 0.5, 1.0)


def load_builder(example_dir, builder_name):
    path = EXAMPLES / example_dir / "run.py"
    spec = importlib.util.spec_from_file_location(f"_ex_{example_dir}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # safe: every run.py guards on __main__
    return getattr(module, builder_name)


def setup(example_dir, resolution=12):
    builder_name, method = BUILDERS[example_dir]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = load_builder(example_dir, builder_name)()
        if isinstance(graph, tuple):
            graph = graph[0]  # some builders also return their node bookkeeping
        modes_df = threshold_modes(graph, method=method)
        thresholds = np.asarray(modes_df["lasing_thresholds"]).ravel()
        size = modes._auto_oversample_size(graph, modes_df, resolution=resolution)
        work = oversample_graph(graph, size) if size else graph
        competition = compute_mode_competition_matrix(work, modes_df)
    return work, modes_df, thresholds, competition


def main(example_dir, resolution=12):
    work, modes_df, thresholds, competition = setup(example_dir, resolution)
    params = work.graph["params"]
    pump = np.asarray(params["pump"], dtype=float)
    pump_mask = modes._get_mask_matrices(params)[1]
    k_a = float(params["k_a"])

    finite = np.where(np.isfinite(thresholds), thresholds, np.inf)
    dominant = int(np.argmin(finite))
    first = float(finite[dominant])
    t_self = abs(competition[dominant, dominant])

    mode0 = np.asarray(
        from_complex(modes_df["threshold_lasing_modes"].to_numpy()[dominant]), dtype=float
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field0 = modes._single_mode_field_intensity(graph_with_pump(work, first), mode0, pump_mask)
        scale = modes._newton_onset_unit_scale(work, mode0, field0, first, t_self)

    print(f"=== {example_dir}  lambda/{resolution}  work graph {len(work)} nodes ===")
    print(
        f"  dominant mode {dominant}: D0_thr={first:.6g}, T_dd={t_self:.5g}, "
        f"k_thr={mode0[0]:.6f} (k-k_a={mode0[0] - k_a:+.4f}), unit scale={scale:.5f}"
    )
    print(
        "\n  eps   D0        linear I    variant          newton I    ratio      dk        "
        "|df|/|f|  residual"
    )

    # The oversampled operator is banded; keep it on the ARPACK path the solver uses.
    saved = quantum_graph.DENSE_EIG_MAX
    quantum_graph.DENSE_EIG_MAX = min(saved, modes.NEWTON_DENSE_EIG_MAX)
    try:
        for eps in EPSILONS:
            d0 = first * (1.0 + eps)
            i_linear = (d0 / first - 1.0) / t_self
            for label, kwargs in (
                ("frozen field (i)", {"outer": 1, "max_steps": 400, "damping": 0.0}),
                ("relaxed (i+ii)  ", {"outer": 25, "max_steps": 30, "damping": 0.7}),
                ("relaxed, hard   ", {"outer": 200, "max_steps": 400, "damping": 0.5}),
            ):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    solution = modes.solve_salt_fixed_set(
                        work,
                        [mode0],
                        [max(i_linear, 1e-3)],
                        [field0.copy()],
                        d0,
                        pump,
                        pump_mask,
                        seed=42,
                        residual_tol=1e-9,
                        **kwargs,
                    )
                amplitude = float(solution.amplitudes[0]) * scale
                k = float(solution.ks[0][0])
                field = np.asarray(solution.fields[0])
                reshape = np.linalg.norm(field - field0) / np.linalg.norm(field0)
                print(
                    f"  {eps:<5} {d0:<9.5g} {i_linear:<11.6f} {label}  {amplitude:<11.6f} "
                    f"{amplitude / i_linear:<10.4f} {k - mode0[0]:<+9.2e} {reshape:<9.4f} "
                    f"{solution.residuals[0]:.2e}"
                )
            print()
    finally:
        quantum_graph.DENSE_EIG_MAX = saved

    print("Read: 'frozen field' isolates term (i) and is > 1 on every ladder graph.")
    print("The 'relaxed' - 'frozen' gap is term (ii), which has no fixed sign.")
    print("The gap to compare_linear_vs_salt.py's ratio is term (iii), competition,")
    print("which pushes the dominant mode down and is ~0 when T is near-diagonal.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in BUILDERS:
        print(f"usage: {Path(__file__).name} <{'|'.join(BUILDERS)}> [resolution]")
        raise SystemExit(2)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)

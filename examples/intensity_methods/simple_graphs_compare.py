"""Geometry + newton-vs-linear L--I for the three simple graphs.

The same picture style as ``dense_ring_compare.py`` / ``chaotic_ring_multimode.py``
(graph geometry on the left, ``linear`` dashed vs ``full_salt_newton`` solid on the
right), applied to the three small textbook cavities built in
``compare_intensity_methods.py``:

* **line (Fabry--Perot)** -- an open 1-D path; the two end edges are leads;
* **ring + leads** -- a closed loop made open by two pendant leads;
* **binary tree** -- a depth-3 splitter, one input lead and several leaf leads.

With the within-edge hole burning resolved (``full_salt_newton`` auto-oversamples),
the operator-level solver **agrees with ``linear`` on the lasing count** and tracks
it near threshold: the line and ring lase **two** modes under both, the tree one.
Above threshold the solid (newton) curves bend below the dashed (linear) ones --
the genuine full-SALT gain saturation that the near-threshold linear model omits.
(With the bare per-edge-mean hole burning the operator over-clamped and spuriously
reported a single mode here; resolving the standing wave fixes it, consistent with
Ge-Chong-Stone Eq. 28 -- see ``line_PRA`` and ``doc/source/lasing.rst``.)

Run::

    OMP_NUM_THREADS=1 python simple_graphs_compare.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from compare_intensity_methods import (
    D0_MAX,
    _threshold_modes,
    make_line,
    make_ring_with_leads,
    make_tree,
)

from netsalt.modes import (
    compute_modal_intensities,
    compute_modal_intensities_full_salt_newton,
    compute_mode_competition_matrix,
)

HERE = Path(__file__).resolve().parent
D0_STEPS = 26
GRAPHS = {
    "line (Fabry-Perot)": make_line,
    "ring + leads": make_ring_with_leads,
    "binary tree": make_tree,
}


def _curves(df):
    cols = np.array(
        sorted(c[1] for c in df.columns if isinstance(c, tuple) and c[0] == "modal_intensities")
    )
    return cols, np.nan_to_num(df[[("modal_intensities", c) for c in cols]].to_numpy(dtype=float))


def _draw_geometry(ax, graph, name):
    """Cavity (inner) edges grey, leads red-dashed, using the stored positions."""
    pos = {n: np.asarray(graph.nodes[n]["position"], dtype=float) for n in graph.nodes}
    for u, v in graph.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        if graph[u][v]["inner"]:
            ax.plot(x, y, color="0.55", lw=2.2, zorder=1)
        else:
            ax.plot(x, y, color="crimson", lw=2.0, ls="--", zorder=1)
    for n in graph.nodes:
        ax.scatter(*pos[n], s=80, color="white", edgecolor="black", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{name}  ({len(graph)} nodes)")


def main():
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(GRAPHS), 2, figsize=(11, 4.2 * len(GRAPHS)))
    print(f"{'graph':22s} {'linear':>8s} {'newton':>8s}")
    for row, (name, builder) in zip(axes, GRAPHS.items(), strict=True):
        graph = builder()
        tdf = _threshold_modes(graph)
        n_modes = len(tdf)
        competition = compute_mode_competition_matrix(graph, tdf)
        thr = np.asarray(tdf["lasing_thresholds"]).ravel()
        first = float(thr[thr < np.inf].min())
        grid = np.linspace(first, D0_MAX, D0_STEPS)
        linear = np.zeros((n_modes, grid.size))
        for j, d0 in enumerate(grid):
            df = compute_modal_intensities(tdf.copy(), d0, competition)
            cols = sorted(
                c[1] for c in df.columns if isinstance(c, tuple) and c[0] == "modal_intensities"
            )
            linear[:, j] = np.nan_to_num(df[("modal_intensities", cols[-1])].to_numpy(dtype=float))
        n_cols, newton = _curves(
            compute_modal_intensities_full_salt_newton(graph, tdf.copy(), D0_MAX, D0_steps=D0_STEPS)
        )

        peak = max(linear.max(), newton.max(), 1e-9)
        active = [m for m in range(n_modes) if max(linear[m].max(), newton[m].max()) > 1e-2 * peak]
        n_lin = int(np.sum(linear[:, -1] > 1e-2 * peak))
        n_nwt = int(np.sum(newton[:, -1] > 1e-2 * peak))
        print(f"{name:22s} {n_lin:>8d} {n_nwt:>8d}")

        _draw_geometry(row[0], graph, name)
        ax = row[1]
        for m in active:
            col = cmap(m % 10)
            ax.plot(grid, linear[m], "--", color=col, lw=1.3, alpha=0.8)
            ax.plot(n_cols, newton[m], ".-", color=col, lw=1.8, ms=4, label=f"mode {m}")
        ax.set_xlabel("pump $D_0$")
        ax.set_ylabel("modal intensity")
        ax.set_title(f"dashed = linear ({n_lin}), solid = newton ({n_nwt})")
        if active:
            ax.legend(fontsize=8, ncol=2)

    fig.suptitle("Simple cavities: full_salt_newton vs linear", y=1.0)
    fig.tight_layout()
    out = HERE / "simple_graphs_compare.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

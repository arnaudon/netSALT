"""Compare netsalt's modal-intensity solvers to Ge-Chong-Stone Fig. 6.

The ``line_PRA`` graph is exactly the 1D slab of Ge-Chong-Stone, PRA 82,
063824 (arXiv:1008.0628), Figs. 3/5/6: length 1, n = 1.5 on x < 0.25 and n = 3
on x > 0.25, open on both sides, gain k_a = 15, gamma_perp = 3, left half
pumped. This script runs the shared netsalt pipeline (cached in ``out/``), then
overlays

* ``linear``           -- netsalt's competition matrix; should coincide with the
                          paper's single-pole approximation (SPA) lines;
* ``full_salt_newton`` -- the operator-level SALT; should track the paper's
                          exact symbols (Eq. 28): dominant suppressed below the
                          SPA, second mode above it, once the second mode is on;

against ``data/ge_fig6_digitized.csv`` (regenerate with
``digitize_pra_fig6.py``). Writes ``figures/pra_fig6_compare.pdf`` and prints a
checkpoint table. Run from this directory (after ``python create_graph.py`` or
``bash run.sh``)::

    OMP_NUM_THREADS=1 python compare_to_pra_fig6.py
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
D0_MAX = 1.27  # the Fig. 6 pump range


def load_reference():
    series = defaultdict(list)
    with open(HERE / "data" / "ge_fig6_digitized.csv") as f:
        for row in csv.DictReader(r for r in f if not r.startswith("#")):
            series[row["series"]].append((float(row["D0"]), float(row["intensity"])))
    return {k: np.array(v) for k, v in series.items()}


def run_netsalt():
    from netsalt.config_loader import load_config
    from netsalt.modes import (
        compute_modal_intensities,
        compute_modal_intensities_full_salt_newton,
        compute_mode_competition_matrix,
    )
    from netsalt.pipeline import (
        _attach_pump_to_graph,
        step_compute_mode_trajectories,
        step_create_pump_profile,
        step_create_quantum_graph,
        step_find_passive_modes,
        step_find_threshold_modes,
        step_scan_frequencies,
    )

    p = load_config(HERE / "config.yaml")
    ids = p.get("lasing_modes_id")
    qg = step_create_quantum_graph(p)
    qualities = step_scan_frequencies(p, qg)
    passive = step_find_passive_modes(p, qg, qualities)
    pump = step_create_pump_profile(p, qg, passive, ids)
    traj = step_compute_mode_trajectories(p, qg, passive, pump, ids)
    tdf = step_find_threshold_modes(p, qg, traj, pump, ids)
    qg = _attach_pump_to_graph(p, qg, pump)

    T = compute_mode_competition_matrix(qg, tdf)
    lin = compute_modal_intensities(tdf.copy(), D0_MAX, T)
    new = compute_modal_intensities_full_salt_newton(qg, tdf.copy(), D0_MAX, D0_steps=16)
    return tdf, lin, new


def curves(df):
    cols = [c for c in df.columns if isinstance(c, tuple) and c[0] == "modal_intensities"]
    pumps = np.array(sorted(c[1] for c in cols), dtype=float)
    data = np.nan_to_num(df[[("modal_intensities", c) for c in pumps]].to_numpy(float))
    return pumps, data


def main():
    os.chdir(HERE)
    ref = load_reference()
    tdf, lin, new = run_netsalt()

    thresholds = np.asarray(tdf["lasing_thresholds"]).ravel()
    pl, dl = curves(lin)
    pn, dn = curves(new)
    lasing = [i for i in np.argsort(thresholds) if dn[i].max() > 0][:2]
    dom, sec = lasing[0], (lasing[1] if len(lasing) > 1 else None)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(*ref["spa_dominant"].T, "-", c="firebrick", lw=1, alpha=0.6, label="paper SPA")
    ax.plot(*ref["spa_second"].T, "-", c="steelblue", lw=1, alpha=0.6)
    ax.plot(
        *ref["exact_dominant"].T, "s", mfc="none", c="firebrick", ms=6, label="paper exact (Eq. 28)"
    )
    ax.plot(*ref["exact_second"].T, "o", mfc="none", c="steelblue", ms=6)
    ax.plot(pl, dl[dom], "--", c="firebrick", lw=2, label=f"netsalt linear (mode {dom})")
    ax.plot(pn, dn[dom], "x-", c="darkred", lw=1, ms=7, label=f"netsalt newton (mode {dom})")
    if sec is not None:
        ax.plot(pl, dl[sec], "--", c="steelblue", lw=2, label=f"netsalt linear (mode {sec})")
        ax.plot(pn, dn[sec], "x-", c="navy", lw=1, ms=7, label=f"netsalt newton (mode {sec})")
    ax.set_xlabel(r"pump strength $D_0$")
    ax.set_ylabel(r"modal intensity $I_\mu$")
    ax.set_xlim(0.58, 1.3)
    ax.set_ylim(0, 0.33)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("line_PRA vs Ge-Chong-Stone PRA 82, 063824 Fig. 6")
    fig.tight_layout()
    out = HERE / "figures" / "pra_fig6_compare.pdf"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")

    # checkpoint table at the figure edge
    d0c = 1.258

    def at(p, row):
        on = row > 0
        return float(np.interp(d0c, p[on], row[on])) if on.sum() > 1 else 0.0

    def ref_at(name):
        d = ref[name]
        return float(np.interp(d0c, d[:, 0], d[:, 1]))

    print(f"\ncheckpoint at D0 = {d0c}:")
    print(
        f"  dominant: paper exact {ref_at('exact_dominant'):.3f}  newton {at(pn, dn[dom]):.3f}"
        f"  | paper SPA {ref_at('spa_dominant'):.3f}  linear {at(pl, dl[dom]):.3f}"
    )
    if sec is not None:
        print(
            f"  second  : paper exact {ref_at('exact_second'):.3f}  newton {at(pn, dn[sec]):.3f}"
            f"  | paper SPA {ref_at('spa_second'):.3f}  linear {at(pl, dl[sec]):.3f}"
        )
    print(f"  first threshold: netsalt {thresholds[dom]:.4f} (paper ~0.61)")
    if sec is not None:
        ithr = np.asarray(new["interacting_lasing_thresholds"]).ravel()
        print(
            f"  2nd interacting threshold: linear "
            f"{np.asarray(lin['interacting_lasing_thresholds']).ravel()[sec]:.4f}, "
            f"newton <= {ithr[sec]:.4f} (paper exact 0.892, SPA 0.899)"
        )


if __name__ == "__main__":
    main()

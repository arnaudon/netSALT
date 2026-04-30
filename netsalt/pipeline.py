"""Plain-Python pipeline replacing ``netsalt/tasks/`` (Luigi).

Each function is a single step with file-cache semantics: if the output
file already exists and ``params["force"]`` is falsy, the cached file is
loaded and returned; otherwise the function runs and saves its output.

Three top-level entry points mirror the old Luigi wrappers:

- :func:`compute_passive_modes` — graph + scan + passive modes + plots
- :func:`compute_lasing_modes` — passive pipeline + pump + thresholds + intensities + plots
- :func:`compute_controllability` — single-mode lasing across the top-N modes

Configuration is a :class:`NetSaltParams` instance built from a YAML file
via :func:`netsalt.config_loader.load_config`. Output / figure
directories default to ``out/`` and ``figures/`` and can be overridden
with ``outdir`` / ``figdir`` in the config.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

from .io import (
    load_graph,
    load_modes,
    load_qualities,
    save_graph,
    save_modes,
    save_qualities,
)
from .modes import (
    compute_modal_intensities,
    compute_mode_competition_matrix,
    find_passive_modes,
    find_threshold_lasing_modes,
    pump_trajectories,
    scan_frequencies,
)
from .params import NetSaltParams
from .physics import (
    dispersion_relation_pump,
    set_dielectric_constant,
    set_dispersion_relation,
)
from .plotting import (
    plot_ll_curve,
    plot_modes,
    plot_pump_profile,
    plot_quantum_graph,
    plot_scan,
    plot_stem_spectra,
)
from .pump import (
    make_threshold_pump,
    optimize_pump_diff_evolution,
    optimize_pump_linear_programming,
)
from .quantum_graph import (
    create_quantum_graph as _create_quantum_graph,
)
from .quantum_graph import (
    oversample_graph,
    set_total_length,
    simplify_graph,
    update_parameters,
)

matplotlib.use("Agg")


# --------------------------------------------------------------------------- helpers


def _outdir(p: NetSaltParams) -> Path:
    return Path(p.get("outdir", "out"))


def _figdir(p: NetSaltParams) -> Path:
    return Path(p.get("figdir", "figures"))


def _reload_graph(p: NetSaltParams):
    """Reload the saved quantum graph from disk — used by steps that mutate
    ``qg.graph['params']`` so they don't leak that mutation back into the
    caller's in-memory copy. ``oversample_graph`` and the compute helpers
    in :mod:`netsalt.modes` rewrite ``params['pump']`` / ``params['inner']``
    in place; the original Luigi pipeline avoided cross-task contamination
    by loading the graph fresh in every task. Mirror that here."""
    return load_graph(str(_outdir(p) / "quantum_graph.json"))


def _apply_lasing_ids(name: str, ids: Iterable[Any] | None) -> str:
    """Append ``_id1_id2`` before the extension, mirroring the old
    ``NetSaltTask.add_lasing_modes_id`` filename mangler."""
    if not ids:
        return name
    path = Path(name)
    suffix = "_".join(str(_id) for _id in ids)
    return f"{path.with_suffix('')}_{suffix}{path.suffix}"


def _force(p: NetSaltParams) -> bool:
    return bool(p.get("force"))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- compute steps


def step_create_quantum_graph(p: NetSaltParams):
    """Build the quantum graph from a raw networkx graph and dielectric setup."""
    out = _outdir(p) / "quantum_graph.json"
    if out.exists() and not _force(p):
        return load_graph(str(out))
    _ensure_parent(out)

    graph_path = p["graph_path"]
    dielectric_mode = p.get("dielectric_mode", "refraction_params")
    dielectric_method = p.get("dielectric_method", "uniform")

    # Compose the params dict that the library helpers expect on the graph.
    qg_params: dict[str, Any] = {
        "open_model": p.get("graph_mode", "open"),
        dielectric_mode: {
            "method": dielectric_method,
            "inner_value": p.get("dielectric_inner_value", 1.5),
            "loss": p.get("dielectric_loss", 0.005),
            "outer_value": p.get("dielectric_outer_value", 1.0),
        },
        "node_loss": p.get("node_loss", 0),
        "plot_edgesize": p.get("edge_size", 0.1),
        "k_a": p.get("k_a", 15.0),
        "gamma_perp": p.get("gamma_perp", 3.0),
        "n_workers": p.get("n_workers", 1),
    }

    quantum_graph = load_graph(graph_path)
    if dielectric_method != "custom" and not p.get("keep_degree_two", True):
        quantum_graph = simplify_graph(quantum_graph)
    positions = np.array([quantum_graph.nodes[u]["position"] for u in quantum_graph.nodes])
    _create_quantum_graph(
        quantum_graph,
        qg_params,
        positions=positions,
        noise_level=p.get("noise_level", 0.001),
    )

    set_total_length(
        quantum_graph,
        p.get("inner_total_length"),
        max_extent=p.get("max_extent"),
        inner=True,
    )

    custom_index = None
    if dielectric_method == "custom":
        custom_index_path = p.get("custom_index_path", "index.yaml")
        with open(custom_index_path) as yml:
            custom_index = yaml.safe_load(yml)

    set_dielectric_constant(quantum_graph, qg_params, custom_values=custom_index)
    set_dispersion_relation(quantum_graph, dispersion_relation_pump)

    # Merge user's full param set onto the graph (this picks up k_min, alpha_*,
    # quality_threshold, mode_search_method, refine_method, etc. that the
    # downstream compute steps read from ``qg.graph["params"]``).
    update_parameters(quantum_graph, qg_params)
    update_parameters(quantum_graph, p.to_dict())

    quantum_graph = oversample_graph(quantum_graph, qg_params["plot_edgesize"])
    save_graph(quantum_graph, str(out))
    return quantum_graph


def step_scan_frequencies(p: NetSaltParams, qg):
    """Scan the (k, alpha) grid; saves the qualities array as HDF5."""
    out = _outdir(p) / "qualities.h5"
    if out.exists() and not _force(p):
        return load_qualities(str(out))
    _ensure_parent(out)

    qualities = scan_frequencies(qg, quality_method=p.get("quality_method", "eigenvalue"))
    save_qualities(qualities, filename=str(out))
    return qualities


def step_find_passive_modes(p: NetSaltParams, qg, qualities):
    """Locate passive modes via contour integration (default) or grid scan."""
    out = _outdir(p) / "passive_modes.h5"
    if out.exists() and not _force(p):
        return load_modes(str(out))
    _ensure_parent(out)

    method = p.get("mode_search_method") or "contour"
    if method == "grid":
        modes_df = find_passive_modes(
            qg,
            qualities,
            method="grid",
            quality_method=p.get("quality_method", "eigenvalue"),
            min_distance=p.get("min_distance", 2),
            threshold_abs=p.get("threshold_abs", 0.1),
        )
    else:
        modes_df = find_passive_modes(qg, method=method)
    save_modes(modes_df, filename=str(out))
    return modes_df


def step_optimize_pump(p: NetSaltParams, qg, modes_df, lasing_modes_id):
    """Optimize a pump profile to lase a specified set of modes."""
    out = _outdir(p) / _apply_lasing_ids("optimized_pump.npz", lasing_modes_id)
    if out.exists() and not _force(p):
        return np.load(str(out))
    _ensure_parent(out)

    method = p.get("optimize_pump_method", "linear_programming")
    if method == "differential_evolution":
        optimal_pump, pump_overlapps, costs, final_cost = optimize_pump_diff_evolution(
            modes_df,
            qg,
            lasing_modes_id,
            pump_min_frac=p.get("optimize_pump_min_frac", 1.0),
            maxiter=p.get("optimize_pump_maxiter", 1000),
            popsize=p.get("optimize_pump_popsize", 5),
            seed=p.get("optimize_pump_seed", 42),
            n_seeds=p.get("optimize_pump_n_seeds", 10),
            disp=p.get("optimize_pump_disp", False),
        )
    elif method == "linear_programming":
        optimal_pump, pump_overlapps, costs, final_cost = optimize_pump_linear_programming(
            modes_df,
            qg,
            lasing_modes_id,
            eps_min=p.get("optimize_pump_eps_min", 5.0),
            eps_max=p.get("optimize_pump_eps_max", 10.0),
            eps_n=p.get("optimize_pump_eps_n", 10),
            cost_diff_min=p.get("optimize_pump_cost_diff_min", 1e-4),
        )
    else:
        raise ValueError(f"unknown optimisation mode {method!r}")

    np.savez(
        str(out),
        optimal_pump=optimal_pump,
        pump_overlapps=pump_overlapps,
        costs=np.asarray(costs),
        final_cost=final_cost,
        lasing_modes_id=np.asarray(lasing_modes_id),
    )
    return np.load(str(out))


def step_create_pump_profile(p: NetSaltParams, qg, passive_modes_df, lasing_modes_id):
    """Build the per-edge pump profile based on ``params["pump_mode"]``."""
    out = _outdir(p) / _apply_lasing_ids("pump_profile.yaml", lasing_modes_id)
    if out.exists() and not _force(p):
        with open(out) as yml:
            return yaml.safe_load(yml)
    _ensure_parent(out)

    mode = p.get("pump_mode", "uniform")
    if mode == "uniform":
        pump = np.zeros(len(qg.edges()))
        for i, (u, v) in enumerate(qg.edges()):
            if qg[u][v]["inner"]:
                pump[i] = 1
        pump = pump.tolist()
    elif mode == "optimized":
        results = step_optimize_pump(p, qg, passive_modes_df, lasing_modes_id)
        pump = results["optimal_pump"].tolist()
    elif mode == "threshold":
        pump = make_threshold_pump(qg, lasing_modes_id, passive_modes_df)
    elif mode == "custom":
        custom_path = p.get("pump_custom_path", "pump_profile.yaml")
        with open(custom_path) as yml:
            pump = yaml.safe_load(yml)
    else:
        raise ValueError(f"Unknown pump_mode {mode!r}")

    with open(out, "w") as yml:
        yaml.safe_dump(pump, yml)
    return pump


def _attach_pump_to_graph(p: NetSaltParams, qg, pump):
    """Mutate ``qg.graph['params']`` so the compute helpers see the pump array."""
    qg.graph["params"].update(
        {
            "D0_max": p.get("D0_max", 0.05),
            "D0_steps": p.get("D0_steps", 10),
            "pump": np.array(pump),
        }
    )
    return qg


def step_compute_mode_trajectories(p: NetSaltParams, qg, passive_modes_df, pump, lasing_modes_id):
    """Track each passive mode as the pump strength D0 varies."""
    out = _outdir(p) / _apply_lasing_ids("mode_trajectories.h5", lasing_modes_id)
    if out.exists() and not _force(p):
        return load_modes(str(out))
    _ensure_parent(out)

    qg = _attach_pump_to_graph(p, _reload_graph(p), pump)
    if p.get("skip_trajectories"):
        modes_df = passive_modes_df
    else:
        modes_df = pump_trajectories(
            passive_modes_df,
            qg,
            return_approx=True,
            quality_method=p.get("quality_method", "eigenvalue"),
        )
    save_modes(modes_df, filename=str(out))
    return modes_df


def step_find_threshold_modes(p: NetSaltParams, qg, trajectories_df, pump, lasing_modes_id):
    """Find lasing thresholds (D0_threshold) and the modes at threshold."""
    out = _outdir(p) / _apply_lasing_ids("lasing_thresholds_modes.h5", lasing_modes_id)
    if out.exists() and not _force(p):
        return load_modes(str(out))
    _ensure_parent(out)

    qg = _attach_pump_to_graph(p, _reload_graph(p), pump)
    modes_df = find_threshold_lasing_modes(
        trajectories_df, qg, quality_method=p.get("quality_method", "eigenvalue")
    )
    save_modes(modes_df, filename=str(out))
    return modes_df


def step_compute_mode_competition_matrix(
    p: NetSaltParams, qg, threshold_modes_df, pump, lasing_modes_id
):
    """Compute the inter-mode gain competition matrix."""
    out = _outdir(p) / _apply_lasing_ids("mode_competition_matrix.h5", lasing_modes_id)
    if out.exists() and not _force(p):
        return pd.read_hdf(str(out), "mode_competition_matrix").to_numpy()
    _ensure_parent(out)

    qg = _attach_pump_to_graph(p, _reload_graph(p), pump)
    matrix = compute_mode_competition_matrix(qg, threshold_modes_df)
    pd.DataFrame(data=matrix, index=None, columns=None).to_hdf(
        str(out), key="mode_competition_matrix"
    )
    return matrix


def step_compute_modal_intensities(
    p: NetSaltParams, threshold_modes_df, competition_matrix, lasing_modes_id
):
    """Compute modal intensities over the pump-strength sweep."""
    out = _outdir(p) / _apply_lasing_ids("modal_intensities.h5", lasing_modes_id)
    if out.exists() and not _force(p):
        return load_modes(str(out))
    _ensure_parent(out)

    D0_max = p.get("intensities_D0_max")
    if D0_max is None:
        D0_max = p.get("D0_max", 0.1)
    modes_df = compute_modal_intensities(threshold_modes_df, D0_max, competition_matrix)
    save_modes(modes_df, filename=str(out))
    return modes_df


# --------------------------------------------------------------------------- plot steps


def _save_fig(out: Path):
    plt.tight_layout()
    plt.savefig(str(out))
    plt.close("all")


def plot_quantum_graph_fig(p: NetSaltParams, qg):
    out = _figdir(p) / "quantum_graph.pdf"
    if out.exists() and not _force(p):
        return out
    _ensure_parent(out)

    print("graph properties:")
    deg = nx.degree_histogram(qg)
    print("degree distribution", deg)
    cycles = nx.cycle_basis(qg)
    print("length cycle basis", len(cycles))
    print("number of nodes", len(qg.nodes()))
    print("number of edges", len(qg.edges()))
    print("number of inner edges", sum(qg.graph["params"]["inner"]))
    lengths = [qg[u][v]["length"] for u, v in qg.edges if qg[u][v]["inner"]]
    print("min edge length", np.min(lengths))
    print("max edge length", np.max(lengths))
    print("mean edge length", np.mean(lengths))
    print("total edge length", sum(lengths))

    cmap = plt.get_cmap("Pastel1_r")
    newcolors = cmap(np.take(np.linspace(0, 1, 9), [0, 4, 2, 3, 1, 8, 6, 7, 5]))
    newcmp = ListedColormap(newcolors)
    plot_quantum_graph(
        qg,
        edge_colors=qg.graph["params"]["dielectric_constant"],
        color_map=newcmp,
        cbar_min=1,
        cbar_max=np.max(np.abs(qg.graph["params"]["dielectric_constant"])),
    )
    _save_fig(out)
    return out


def plot_scan_fig(p: NetSaltParams, qg, qualities):
    out = _figdir(p) / "scan_frequencies.pdf"
    if out.exists() and not _force(p):
        return out
    _ensure_parent(out)
    plot_scan(qg, qualities, filename=str(out))
    _save_fig(out)
    return out


def plot_passive_modes_fig(p: NetSaltParams, qg, passive_modes_df):
    out = _figdir(p) / "passive_modes"
    if out.exists() and not _force(p):
        return out
    _ensure_parent(out)
    out.mkdir(exist_ok=True)

    qg_plot = oversample_graph(_reload_graph(p), p.get("plot_passive_edge_size", 1.0))
    mode_ids = p.get("plot_passive_mode_ids") or []
    if mode_ids:
        modes_df = passive_modes_df.loc[list(mode_ids)]
    else:
        modes_df = passive_modes_df.head(p.get("plot_passive_n_modes", 10))
    plot_modes(
        qg_plot, modes_df, df_entry="passive", folder=str(out), ext=p.get("plot_ext", ".pdf")
    )
    return out


def plot_scan_with_modes_fig(p: NetSaltParams, qg, qualities, passive_modes_df):
    out = _figdir(p) / "scan_frequencies_with_modes.pdf"
    if out.exists() and not _force(p):
        return out
    _ensure_parent(out)
    plot_scan(qg, qualities, passive_modes_df, filename=str(out))
    _save_fig(out)
    return out


def plot_scan_with_mode_trajectories_fig(
    p: NetSaltParams, qg, qualities, trajectories_df, lasing_modes_id
):
    out = _figdir(p) / _apply_lasing_ids("mode_trajectories.pdf", lasing_modes_id)
    if out.exists() and not _force(p):
        return out
    _ensure_parent(out)
    plot_scan(qg, qualities, trajectories_df, relax_upper=True)
    plt.savefig(str(out), bbox_inches="tight")
    plt.close("all")
    return out


def plot_scan_with_threshold_modes_fig(
    p: NetSaltParams, qg, qualities, threshold_modes_df, lasing_modes_id
):
    out = _figdir(p) / _apply_lasing_ids("threshold_modes.pdf", lasing_modes_id)
    if out.exists() and not _force(p):
        return out
    _ensure_parent(out)
    plot_scan(qg, qualities, threshold_modes_df, relax_upper=True, with_approx=False)
    plt.savefig(str(out), bbox_inches="tight")
    plt.close("all")
    return out


def plot_threshold_modes_fig(p: NetSaltParams, qg, threshold_modes_df, pump, lasing_modes_id):
    out = _figdir(p) / _apply_lasing_ids("threshold_modes", lasing_modes_id)
    if out.exists() and not _force(p):
        return out
    _ensure_parent(out)
    out.mkdir(exist_ok=True)

    qg = _attach_pump_to_graph(p, _reload_graph(p), pump)
    qg.graph["params"]["plot_edgesize"] = p.get("plot_threshold_edge_size", 1.0)
    qg = oversample_graph(qg, qg.graph["params"]["plot_edgesize"])

    mode_ids = p.get("plot_threshold_mode_ids") or []
    if mode_ids:
        modes_df = threshold_modes_df.loc[list(mode_ids)]
    else:
        modes_df = threshold_modes_df.head(p.get("plot_threshold_n_modes", 10))
    # ``use_inf_as_na`` was removed in pandas 3.0 — drop +/-inf thresholds explicitly.
    thresholds = modes_df["lasing_thresholds"].replace([np.inf, -np.inf], np.nan)
    modes_df = modes_df[~thresholds.isna()]
    plot_modes(
        qg,
        modes_df,
        df_entry="threshold_lasing_modes",
        folder=str(out),
        ext=p.get("plot_ext", ".pdf"),
    )
    return out


def plot_ll_curve_fig(p: NetSaltParams, qg, intensities_df, lasing_modes_id):
    out = _figdir(p) / _apply_lasing_ids("ll_curve.pdf", lasing_modes_id)
    if out.exists() and not _force(p):
        return out
    _ensure_parent(out)
    plot_ll_curve(qg, intensities_df, with_legend=True)
    _save_fig(out)
    return out


def plot_stem_spectra_fig(p: NetSaltParams, qg, intensities_df, lasing_modes_id):
    out = _figdir(p) / _apply_lasing_ids("stem_spectra.pdf", lasing_modes_id)
    if out.exists() and not _force(p):
        return out
    _ensure_parent(out)
    plot_stem_spectra(qg, intensities_df)
    _save_fig(out)
    return out


def plot_pump_profile_fig(p: NetSaltParams, qg, pump, lasing_modes_id):
    out = _figdir(p) / _apply_lasing_ids("pump_profile.pdf", lasing_modes_id)
    if out.exists() and not _force(p):
        return out
    _ensure_parent(out)
    plot_pump_profile(qg, pump, node_size=5)
    _save_fig(out)
    return out


def plot_optimized_pump_fig(p: NetSaltParams, qg, lasing_modes_id):
    """Multi-page PDF summarising the optimized pump (only meaningful when
    ``pump_mode == 'optimized'``)."""
    out = _figdir(p) / _apply_lasing_ids("optimized_pump.pdf", lasing_modes_id)
    if out.exists() and not _force(p):
        return out
    _ensure_parent(out)

    optimized_pump_path = _outdir(p) / _apply_lasing_ids("optimized_pump.npz", lasing_modes_id)
    results = np.load(str(optimized_pump_path))
    with PdfPages(str(out)) as pdf:
        plot_pump_profile(qg, results["optimal_pump"])
        plt.tight_layout()
        pdf.savefig()
        plt.close("all")

        plt.figure()
        plt.hist(results["costs"], bins=20)
        plt.tight_layout()
        pdf.savefig()
        plt.close("all")

        plt.figure(figsize=(20, 5))
        for lasing_mode in results["lasing_modes_id"]:
            plt.plot(results["pump_overlapps"][lasing_mode])
        plt.twinx()
        plt.plot(results["optimal_pump"], "r+")
        plt.gca().set_ylim(0.5, 1.5)
        plt.tight_layout()
        pdf.savefig()
        plt.close("all")
    return out


def plot_mode_competition_matrix_fig(p: NetSaltParams, competition_matrix, lasing_modes_id):
    out = _figdir(p) / _apply_lasing_ids("mode_competition_matrix.pdf", lasing_modes_id)
    if out.exists() and not _force(p):
        return out
    _ensure_parent(out)
    plt.figure(figsize=0.5 * np.array(np.shape(competition_matrix)))
    sns.heatmap(competition_matrix, ax=plt.gca(), square=True)
    plt.savefig(str(out))
    plt.close("all")
    return out


def plot_controllability_fig(p: NetSaltParams, single_mode_matrix):
    out = _figdir(p) / "single_mode_control.pdf"
    if out.exists() and not _force(p):
        return out
    _ensure_parent(out)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        single_mode_matrix,
        ax=plt.gca(),
        cmap="Blues",
        cbar_kws={"label": "modal amplitude", "shrink": 0.7},
        vmin=0,
    )
    plt.ylabel("Mode ids to single lase")
    plt.xlabel("Modal ids")
    plt.savefig(str(out), bbox_inches="tight")
    plt.close("all")
    return out


# --------------------------------------------------------------------------- entry points


def compute_passive_modes(p: NetSaltParams):
    """Compute passive modes of a quantum graph and produce the standard plots."""
    qg = step_create_quantum_graph(p)
    qualities = step_scan_frequencies(p, qg)
    passive_modes_df = step_find_passive_modes(p, qg, qualities)

    plot_quantum_graph_fig(p, qg)
    plot_scan_fig(p, qg, qualities)
    plot_passive_modes_fig(p, qg, passive_modes_df)
    plot_scan_with_modes_fig(p, qg, qualities, passive_modes_df)
    return passive_modes_df


def compute_lasing_modes(p: NetSaltParams, lasing_modes_id=None):
    """Full lasing-mode pipeline: passive + pump + thresholds + intensities + plots.

    Mirrors the old :class:`ComputeLasingModes` Luigi wrapper, which does
    *not* emit the passive-pipeline plots — those live in
    :func:`compute_passive_modes` and are only produced when that workflow
    is invoked explicitly.
    """
    if lasing_modes_id is None:
        lasing_modes_id = p.get("lasing_modes_id")

    qg = step_create_quantum_graph(p)
    qualities = step_scan_frequencies(p, qg)
    passive_modes_df = step_find_passive_modes(p, qg, qualities)

    pump = step_create_pump_profile(p, qg, passive_modes_df, lasing_modes_id)
    plot_pump_profile_fig(p, qg, pump, lasing_modes_id)

    trajectories_df = step_compute_mode_trajectories(p, qg, passive_modes_df, pump, lasing_modes_id)
    plot_scan_with_mode_trajectories_fig(p, qg, qualities, trajectories_df, lasing_modes_id)

    threshold_modes_df = step_find_threshold_modes(p, qg, trajectories_df, pump, lasing_modes_id)
    plot_scan_with_threshold_modes_fig(p, qg, qualities, threshold_modes_df, lasing_modes_id)
    plot_threshold_modes_fig(p, qg, threshold_modes_df, pump, lasing_modes_id)

    competition = step_compute_mode_competition_matrix(
        p, qg, threshold_modes_df, pump, lasing_modes_id
    )
    plot_mode_competition_matrix_fig(p, competition, lasing_modes_id)

    intensities_df = step_compute_modal_intensities(
        p, threshold_modes_df, competition, lasing_modes_id
    )
    plot_ll_curve_fig(p, qg, intensities_df, lasing_modes_id)
    plot_stem_spectra_fig(p, qg, intensities_df, lasing_modes_id)

    if p.get("pump_mode") == "optimized":
        plot_optimized_pump_fig(p, qg, lasing_modes_id)

    return intensities_df


def compute_controllability(p: NetSaltParams):
    """Run single-mode pump optimisation across the top-N passive modes."""
    qg = step_create_quantum_graph(p)
    qualities = step_scan_frequencies(p, qg)
    passive_modes_df = step_find_passive_modes(p, qg, qualities)

    plot_quantum_graph_fig(p, qg)
    plot_scan_fig(p, qg, qualities)
    plot_passive_modes_fig(p, qg, passive_modes_df)
    plot_scan_with_modes_fig(p, qg, qualities, passive_modes_df)

    n_top = p.get("n_top_modes", 4)
    top_ids = passive_modes_df.head(n_top).index

    spectra_matrix = []
    for mode_id in top_ids:
        ids = [mode_id]
        pump = step_create_pump_profile(p, qg, passive_modes_df, ids)
        plot_optimized_pump_fig(p, qg, ids)
        with _temp_param(p, "skip_trajectories", True):
            trajectories_df = step_compute_mode_trajectories(p, qg, passive_modes_df, pump, ids)
        threshold_modes_df = step_find_threshold_modes(p, qg, trajectories_df, pump, ids)
        competition = step_compute_mode_competition_matrix(p, qg, threshold_modes_df, pump, ids)
        intensities_df = step_compute_modal_intensities(p, threshold_modes_df, competition, ids)
        plot_ll_curve_fig(p, qg, intensities_df, ids)

        spectra = intensities_df[
            "modal_intensities", intensities_df["modal_intensities"].columns[-1]
        ].to_numpy()
        spectra_matrix.append(spectra)

    spectra_matrix = np.array(spectra_matrix)
    out = _outdir(p) / "single_mode_matrix.npy"
    _ensure_parent(out)
    np.save(str(out), spectra_matrix)
    plot_controllability_fig(p, spectra_matrix)
    return spectra_matrix


# --------------------------------------------------------------------------- helpers


class _temp_param:
    """Temporarily set a key on ``params`` for the scope of a ``with`` block."""

    def __init__(self, params: NetSaltParams, key: str, value: Any):
        self.params = params
        self.key = key
        self.value = value
        self.had_value = False
        self.previous: Any = None

    def __enter__(self):
        self.had_value = self.key in self.params
        if self.had_value:
            self.previous = self.params[self.key]
        self.params[self.key] = self.value
        return self.params

    def __exit__(self, exc_type, exc, tb):
        if self.had_value:
            self.params[self.key] = self.previous
        else:
            del self.params[self.key]
        return False

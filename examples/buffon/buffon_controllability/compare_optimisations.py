import pandas as pd
import yaml
from tqdm import tqdm
from pathlib import Path

# import seaborn as sns
import numpy as np
from netsalt.io import load_graph
from netsalt.plotting import (
    plot_single_mode,
    plot_pump_traj,
    plot_spectra,
    _plot_single_mode,
    plot_quantum_graph,
)
from netsalt.modes import mean_mode_on_edges, mode_on_nodes
from netsalt.quantum_graph import get_total_inner_length
from netsalt.modes import (
    compute_overlapping_factor,
    from_complex,
    compute_mode_competition_matrix,
    mode_on_nodes,
    mean_mode_on_edges,
    compute_mode_IPR,
)
from netsalt.pump import pump_cost, compute_pump_overlapping_matrix
import matplotlib.pyplot as plt


def get_ratios(base_path):
    clip = 20
    clip_ipr = 15
    graph = load_graph("../buffon_uniform/out/quantum_graph.gpickle")
    graph.graph["params"]["k_min"] = 10.35
    graph.graph["params"]["k_max"] = 11.0
    graph.graph["params"]["k_n"] = 1500
    graph.graph["params"]["alpha_min"] = 10.35
    graph.graph["params"]["alpha_max"] = 11.0
    graph.graph["params"]["alpha_n"] = 1500
    graph.graph["params"]["quality_threshold"] = 2e-3
    graph.graph["params"]["pump"] = np.ones(len(graph.edges))

    tot_L = get_total_inner_length(graph)
    base_path = Path(base_path)
    ratios = []
    qs = []
    iprs = []
    ids = []
    areas = []
    n_modes = 200
    for mode_id in tqdm(range(n_modes)):
        p = base_path / f"modal_intensities_{mode_id}.h5"
        with open(base_path / f"pump_profile_{mode_id}.yaml") as p_f:
            pump = yaml.safe_load(p_f)
        area = (
            sum(graph[u][v]["length"] for i, (u, v) in enumerate(graph.edges) if pump[u] == 1)
            / tot_L
        )
        df = pd.read_hdf(p)
        iprs.append(np.clip(compute_mode_IPR(graph, df, mode_id), 0, clip_ipr))
        _df = df["modal_intensities"]
        _df[_df.isna()] = 0
        m = max(_df.loc[_df.index != mode_id, _df.columns[-1]])
        if m > 0:
            ratio = _df.loc[mode_id, _df.columns[-1]] / m
        else:
            ratio = clip
        ratios.append(np.clip(ratio, 0, clip) if ratio is not np.isnan else 0)
        qs.append(np.real(df.loc[mode_id, "q_factor"].to_list()[0]))
        ids.append(mode_id)
        areas.append(area)

    df = pd.DataFrame(index=ids)
    df["q-value"] = qs
    df["ratio"] = ratios
    df["IPR"] = iprs
    df["area"] = areas
    return df


if __name__ == "__main__":

    area_thresh = 2e-2
    df = get_ratios("buffon_control_optimized/out/")
    df_thresh = get_ratios("buffon_control_threshold/out/")
    df_uni = pd.read_hdf("buffon_uniform/out/modal_intensities.h5")
    print(df, df_thresh)
    df["ratio_threshold"] = df_thresh["ratio"]
    df["area_threshold"] = df_thresh["area"]

    plt.figure(figsize=(3, 3))
    plt.hist(df_thresh["area"], bins=20, histtype="step", label="threshold", color="k")
    plt.hist(df["area"], bins=20, histtype="step", label="optimized", color="b")
    plt.legend()
    plt.xlabel("pump fractional area")
    plt.tight_layout()
    plt.savefig("optimized_pump_area.pdf")

    plt.figure()
    ax = plt.gca()
    df.plot.scatter(x="IPR", y="ratio", ax=ax, label="optimisation", c="C0", s=2)
    df.plot.scatter(x="IPR", y="ratio_threshold", ax=ax, label="threshold", c="C1", s=2)
    plt.xscale("log")
    plt.legend()
    plt.savefig("ratios_iprs.pdf")

    _df = df.copy()
    _df["diff"] = df["ratio"] - df["ratio_threshold"]
    # _df.loc[abs(_df["diff"]) > 10] = 10
    # _df.loc[_df["diff"].isna()] = 0
    plt.figure()
    _df.plot.scatter(x="IPR", y="diff", ax=ax, label="optimisation", c="C0", s=2)

    ax = plt.gca()
    plt.axhline(0)
    plt.xscale("log")
    plt.legend()
    plt.savefig("diff_ratios_iprs.pdf")

    plt.figure(figsize=(3, 3))
    plt.hist(df_thresh["ratio"], bins=20, histtype="step", ls="--", color="k")
    plt.hist(
        df_thresh.loc[df_thresh.area > area_thresh, "ratio"], bins=20, histtype="step", color="k"
    )
    plt.hist(df["ratio"], bins=20, histtype="step", ls="--", color="b")
    plt.hist(df.loc[df.area > area_thresh, "ratio"], bins=20, histtype="step", color="b")
    plt.axvline(1, ls="-", lw=0.8, c="k")
    plt.axvline(2, ls="-.", lw=0.8, c="k")
    plt.xlabel("ratio")
    plt.tight_layout()
    plt.savefig("ratio_hist.pdf")

    plt.figure()
    ax = plt.gca()
    df.plot.scatter(x="q-value", y="ratio", ax=ax, label="optimisation", c="C0", s=2)
    df.plot.scatter(x="q-value", y="ratio_threshold", ax=ax, label="threshold", c="C1", s=2)
    plt.savefig("ratios_qs.pdf")

    plt.figure()
    ax = plt.gca()
    _df.plot.scatter(x="q-value", y="diff", ax=ax, label="optimisation", c="area", s=2)
    plt.axhline(0)
    plt.axhline(_df["diff"].mean(), ls="--")
    plt.savefig("diff_ratios_qs.pdf")

    plt.figure(figsize=(5, 3))
    plt.hist(_df["diff"], bins=50)
    plt.xlabel("difference in ratios (optimised - threshold)")
    plt.axvline(_df["diff"].mean(), ls="--", c="k")
    print("mean diff:", _df["diff"].mean())
    plt.axvline(0, ls="--", c="k")
    plt.tight_layout()
    plt.savefig("diff_ratios_hist.pdf")
    plt.figure()
    _df.plot.scatter(x="area_threshold", y="ratio_threshold", ax=plt.gca(), c="r", s=2)
    _df.plot.scatter(x="area", y="ratio", ax=plt.gca(), c="k", s=2)
    plt.xscale("log")
    plt.savefig("area_ratio.pdf")

    plt.figure(figsize=(6, 3))
    ax = plt.gca()
    df_thresh[df_thresh.area < area_thresh].plot.scatter(
        x="IPR", y="ratio", ax=ax, c="k", marker="+", s=20
    )
    df_thresh.plot.scatter(x="IPR", y="ratio", ax=ax, c="k", marker=".", label="mode matching")

    df[df.area < area_thresh].plot.scatter(x="IPR", y="ratio", ax=ax, c="b", marker="+", s=20)
    df.plot.scatter(x="IPR", y="ratio", ax=ax, c="b", marker=".", label="optimisation")
    plt.axhline(2, c="k", ls="-.", lw=0.8)
    plt.axhline(1, c="k", ls="--", lw=0.8)
    plt.axhline(0, c="k", ls="-", lw=0.8)
    plt.tight_layout()

    plt.savefig("ratios_iprs_single.pdf")

    plt.figure(figsize=(6, 3))
    ax = plt.gca()
    df_thresh[df_thresh.area < area_thresh].plot.scatter(
        x="q-value", y="ratio", ax=ax, c="k", marker="+", s=20
    )
    df_thresh.plot.scatter(x="q-value", y="ratio", ax=ax, c="k", marker=".", label="mode matching")

    df[df.area < area_thresh].plot.scatter(x="q-value", y="ratio", ax=ax, c="b", marker="+", s=20)
    df.plot.scatter(x="q-value", y="ratio", ax=ax, c="b", marker=".", label="optimisation")
    plt.axhline(2, c="k", ls="-.", lw=0.8)
    plt.axhline(1, c="k", ls="--", lw=0.8)
    plt.axhline(0, c="k", ls="-", lw=0.8)
    plt.tight_layout()
    plt.savefig("ratios_qs_single.pdf")

    print("opt >2", len(df[df.ratio > 2]), len(df), len(df[df.ratio > 2]) / len(df))
    df.loc[df.area < area_thresh, "ratio"] = np.nan
    df_thresh.loc[df_thresh.area < area_thresh, "ratio"] = np.nan
    print("opt >1", len(df[df.ratio > 1]), len(df), len(df[df.ratio > 1]) / len(df))
    print(
        "thresh>2",
        len(df_thresh[df_thresh.ratio > 2]),
        len(df_thresh),
        len(df_thresh[df_thresh.ratio > 2]) / len(df_thresh),
    )
    print(
        "thresh>1",
        len(df_thresh[df_thresh.ratio > 1]),
        len(df_thresh),
        len(df_thresh[df_thresh.ratio > 1]) / len(df_thresh),
    )

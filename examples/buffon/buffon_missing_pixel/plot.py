import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
from netsalt.io import load_graph
from netsalt.modes import (
    compute_overlapping_factor,
    from_complex,
    compute_mode_competition_matrix,
    mode_on_nodes,
    mean_mode_on_edges,
    compute_mode_IPR,
)
from tqdm import tqdm

if __name__ == "__main__":
    df_missing = pd.read_hdf("out/modal_intensities.h5")
    df_uni = pd.read_hdf("../buffon_uniform/out/modal_intensities.h5")

    plt.figure(figsize=(4, 2))
    _df = df_uni["modal_intensities"]
    _df[~_df.isna()] = 1
    _df[_df.isna()] = 0
    _df.sum(0).plot(ax=plt.gca(), label="uniform pump")
    print(_df.sum(0))
    _df = df_missing["modal_intensities"]
    _df[~_df.isna()] = 1
    _df[_df.isna()] = 0
    _df.sum(0).plot(ax=plt.gca(), label="missing edges")
    plt.xlabel("D0")
    plt.ylabel("Number of lasing modes")
    plt.legend()
    plt.tight_layout()
    plt.savefig("number_modes.pdf")

    mask = (
        ~df_uni[("modal_intensities", 0.01)].isna()
        | ~df_missing[("modal_intensities", 0.01)].isna()
    )

    graph = load_graph("out/quantum_graph.gpickle")
    graph.graph["params"]["k_min"] = 10.35
    graph.graph["params"]["k_max"] = 11.0
    graph.graph["params"]["k_n"] = 1500
    graph.graph["params"]["alpha_min"] = 10.35
    graph.graph["params"]["alpha_max"] = 11.0
    graph.graph["params"]["alpha_n"] = 1500
    graph.graph["params"]["quality_threshold"] = 2e-3
    pump = yaml.safe_load(open("pump.yaml"))
    graph.graph["params"]["pump"] = 1 - np.array(pump)

    _id = 1
    _id2 = 6

    for gid in tqdm(df_uni.index):
        df_uni.loc[gid, "over"] = np.real(
            compute_overlapping_factor(df_uni.loc[gid, "passive"].to_list()[0], graph)
        )

    col = ("modal_intensities", 0.01)
    df_missing[col][df_missing[col].isna()] = 0
    df_uni[col][df_uni[col].isna()] = 0
    print(df_missing[col])
    m_miss = df_missing[col]
    m_uni = df_uni[col]
    plt.figure(figsize=(4, 3))
    c = df_uni["over"]

    plt.plot(df_uni[col], df_uni[col], c="k", lw=0.5)
    plt.axvline(0, c="k", lw=0.5)
    plt.axhline(0, c="k", lw=0.5)

    plt.scatter(
        df_uni.loc[_id, col],
        df_missing.loc[_id, col],
        c="r",
        marker="+",
        s=100,
    )

    plt.scatter(
        df_uni.loc[_id2, col],
        df_missing.loc[_id2, col],
        c="g",
        marker="+",
        s=100,
    )

    print(df_uni[col], df_uni["over"])
    plt.scatter(df_uni.loc[mask, col], df_missing.loc[mask, col], c=df_uni.loc[mask, "over"], s=15)
    plt.colorbar(label="overlap with missing edges")
    plt.xlabel("modal amplitude (uniform)")
    plt.ylabel("modal amplitude (missing edges)")
    plt.tight_layout()
    plt.savefig("modal_amps.pdf")

    plt.figure(figsize=(4, 3))
    df_uni["threshold_diff"] = np.real(
        df_missing["lasing_thresholds"] - df_uni["lasing_thresholds"]
    )

    plt.scatter(df_uni["over"], df_uni["threshold_diff"], c="k", s=8, label="all")
    plt.scatter(
        df_uni.loc[mask, "over"], df_uni.loc[mask, "threshold_diff"], c="b", s=8, label="lasing"
    )

    plt.scatter(
        np.real(df_uni.loc[_id, "over"].to_list()[0]),
        np.real(df_uni.loc[_id, "threshold_diff"].to_list()[0]),
        c="r",
        s=20,
        label=f"mode {_id}",
    )
    plt.scatter(
        np.real(df_uni.loc[_id2, "over"].to_list()[0]),
        np.real(df_uni.loc[_id2, "threshold_diff"].to_list()[0]),
        c="g",
        s=20,
        label=f"mode {_id2}",
    )
    plt.legend(loc="best")
    plt.xlabel("overlap with missing edges")
    plt.ylabel("diff lasing threshold")
    plt.tight_layout()
    plt.savefig("thresh.pdf")

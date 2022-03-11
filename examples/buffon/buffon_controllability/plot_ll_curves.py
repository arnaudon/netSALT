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
from netsalt.plotting import plot_ll_curve
from tqdm import tqdm

if __name__ == "__main__":
    first_lase = []
    ids = list(range(250))
    count = 0
    for i in ids:
        df_opt = pd.read_hdf(f"buffon_control/out/modal_intensities_{i}.h5")["modal_intensities"]
        df_opt[~df_opt.isna()] = 1
        df_opt[df_opt.isna()] = 0
        first_lase.append(df_opt.sum(1).idxmax())
        if df_opt.sum(1).idxmax() == i:
            count += 1
    print(count, len(ids))
    plt.figure()
    plt.scatter(ids, first_lase)
    plt.savefig("first_lasing.pdf")

    df_opt = pd.read_hdf("buffon_control/out/modal_intensities_41.h5")
    plt.figure(figsize=(4, 3))
    ax = plt.gca()
    plot_ll_curve(None, df_opt, ax=ax, with_colors=False, with_legend=False)
    ax.set_xlim(0.003, 0.01)
    plt.tight_layout()
    plt.savefig("ll_curve_41.pdf")

    df_opt = pd.read_hdf("buffon_control/out/modal_intensities_65.h5")
    plt.figure(figsize=(4, 3))
    ax = plt.gca()
    plot_ll_curve(None, df_opt, ax=ax, with_colors=False, with_legend=False)
    ax.set_xlim(0.003, 0.01)
    plt.tight_layout()
    plt.savefig("ll_curve_65.pdf")

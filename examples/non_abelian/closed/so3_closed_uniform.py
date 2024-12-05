import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import networkx as nx

from netsalt.quantum_graph import oversample_graph, create_quantum_graph, mode_quality
from netsalt.plotting import plot_single_mode

from make_graph import make_graph
from netsalt.non_abelian import construct_so3_laplacian


if __name__ == "__main__":
    graph, pos = make_graph()
    params = {
        "open_model": "open",
        "n_workers": 7,
        "k_n": 2000,
        "k_min": 0.00001,
        "k_max": 5.2,
        "alpha_n": 20,
        "alpha_min": 0.00,
        "alpha_max": 0.2,
        "quality_threshold": 1e-3,
        "c": len(graph.edges) * [1.0],
        "laplacian_constructor": construct_so3_laplacian,
    }

    nx.draw(graph, pos=pos)
    nx.draw_networkx_labels(graph, pos=pos)

    create_quantum_graph(graph, params=params, positions=pos)
    ks = np.linspace(5, 7, 2000)
    if not Path("so3_qualities.csv").exists():
        res = []
        for k in tqdm(ks):
            res.append(mode_quality([k, 0], graph))

        modes = ks[peak_local_max(1 / (1e-10 + np.array(res))).flatten()]
        np.savetxt("so3_modes_uniform.csv", modes)
        np.savetxt("so3_qualities.csv", res)
    else:
        res = np.loadtxt("so3_qualities.csv")
        modes = np.loadtxt("so3_modes_uniform.csv")

    plt.figure(figsize=(4, 2))
    for mode in modes:
        plt.axvline(mode, c="k")
    plt.semilogy(ks, res, "-")
    plt.axis([ks[0], ks[-1], 1e-3, 1])
    plt.xlabel("wavenumber")
    plt.ylabel("mode quality")
    plt.tight_layout()
    plt.savefig("so3_close_scan_uniform.pdf")

    modes_df = pd.DataFrame()
    modes_df.loc[:, "passive"] = modes
    over_graph = oversample_graph(graph, 0.01)

    plt.figure(figsize=(4, 3))
    plot_single_mode(over_graph, modes_df, 1, ax=plt.gca(), norm_type="real")
    plt.tight_layout()
    plt.savefig("so3_close_mode_1_uniform.pdf")

    plt.figure(figsize=(4, 3))
    plot_single_mode(over_graph, modes_df, 1, ax=plt.gca(), norm_type="real_x")
    plt.tight_layout()
    plt.savefig("so3_close_mode_1_x_uniform.pdf")

    plt.figure(figsize=(4, 3))
    plot_single_mode(over_graph, modes_df, 1, ax=plt.gca(), norm_type="real_y")
    plt.tight_layout()
    plt.savefig("so3_close_mode_1_y_uniform.pdf")

    plt.figure(figsize=(4, 3))
    plot_single_mode(over_graph, modes_df, 1, ax=plt.gca(), norm_type="real_z")
    plt.tight_layout()
    plt.savefig("so3_close_mode_1_z_uniform.pdf")

    plt.figure(figsize=(4, 3))
    plot_single_mode(over_graph, modes_df, 2, ax=plt.gca(), norm_type="real")
    plt.tight_layout()
    plt.savefig("so3_close_mode_2_uniform.pdf")


    plt.figure(figsize=(4, 3))
    plot_single_mode(over_graph, modes_df, 2, ax=plt.gca(), norm_type="real_x")
    plt.tight_layout()
    plt.savefig("so3_close_mode_2_x_uniform.pdf")


    plt.figure(figsize=(4, 3))
    plot_single_mode(over_graph, modes_df, 2, ax=plt.gca(), norm_type="real_y")
    plt.tight_layout()
    plt.savefig("so3_close_mode_2_y_uniform.pdf")


    plt.figure(figsize=(4, 3))
    plot_single_mode(over_graph, modes_df, 2, ax=plt.gca(), norm_type="real_z")
    plt.tight_layout()
    plt.savefig("so3_close_mode_2_z_uniform.pdf")
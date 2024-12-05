import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import networkx as nx

from netsalt.quantum_graph import oversample_graph, create_quantum_graph, mode_quality
from netsalt.plotting import plot_single_mode
from netsalt.physics import dispersion_relation_linear, set_dispersion_relation

from make_graph import make_graph


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
    }

    nx.draw(graph, pos=pos)
    nx.draw_networkx_labels(graph, pos=pos)

    create_quantum_graph(graph, params=params, positions=pos)
    set_dispersion_relation(graph, dispersion_relation_linear)

    res = []
    ks = np.linspace(5, 7, 2000)
    for k in tqdm(ks):
        res.append(mode_quality([k, 0], graph))

    modes = ks[peak_local_max(1 / (1e-10 + np.array(res))).flatten()]
    print(modes)
    plt.figure(figsize=(4, 2))
    for mode in modes:
        plt.axvline(mode, c="k")
    plt.semilogy(ks, res, "-")
    plt.axis([ks[0], ks[-1], 1e-3, 1])
    plt.xlabel("wavenumber")
    plt.ylabel("mode quality")
    plt.tight_layout()
    plt.savefig("close_scan_abelian.pdf")

    modes_df = pd.DataFrame()
    modes_df.loc[:, "passive"] = modes
    over_graph = oversample_graph(graph, 0.01)
    over_graph.graph["params"]["c"] = len(over_graph.edges) * [1.0]
    set_dispersion_relation(over_graph, dispersion_relation_linear)
    plt.figure(figsize=(4, 3))
    plot_single_mode(over_graph, modes_df, 1, ax=plt.gca(), norm_type="real")
    plt.tight_layout()
    plt.savefig("close_mode_1_abelian.pdf")

    plt.figure(figsize=(4, 3))
    plot_single_mode(over_graph, modes_df, 2, ax=plt.gca(), norm_type="real")
    plt.tight_layout()
    plt.savefig("close_mode_2_abelian.pdf")
    plt.show()

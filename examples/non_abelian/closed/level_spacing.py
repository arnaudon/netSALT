import numpy as np
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

from netsalt.quantum_graph import create_quantum_graph, mode_quality
from netsalt.physics import dispersion_relation_linear, set_dispersion_relation

from make_graph import make_graph


def _qual(k, graph=None):
    return mode_quality([k, 0], graph)


if __name__ == "__main__":
    graph, pos = make_graph()
    k_max = 200
    k_res = 500
    params = {"open_model": "open", "quality_threshold": 1e-3, "c": len(graph.edges) * [1.0]}

    create_quantum_graph(graph, params=params, positions=pos)
    set_dispersion_relation(graph, dispersion_relation_linear)

    res = []
    ks = np.linspace(10, k_max, k_res * k_max)

    with Pool() as pool:
        res = list(tqdm(pool.imap(partial(_qual, graph=graph), ks), total=len(ks)))

    modes = ks[sorted(peak_local_max(1 / (1e-10 + np.array(res))).flatten())]
    print(len(modes))

    plt.figure(figsize=(20, 2))
    for mode in modes:
        plt.axvline(mode, c="k")

    plt.semilogy(ks, res, "-")
    plt.axis([ks[0], ks[-1], 1e-3, 1])
    plt.xlabel("wavenumber")
    plt.ylabel("mode quality")
    plt.tight_layout()

    modes_inter = np.diff(modes)
    mean_modes_inter = np.mean(modes_inter)

    plt.figure(figsize=(5, 3))
    plt.hist(modes_inter / mean_modes_inter, bins=40, histtype="step", density=True, label="data")
    s = np.linspace(0, 4, 100)
    plt.plot(s, np.pi * s / 2 * np.exp(-np.pi / 4 * s**2), label="GOE")
    plt.plot(s, np.exp(-s), label="Poisson")
    plt.xlabel("s")
    plt.ylabel("P(s)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("closed_level_spacing_abelian.pdf")
    plt.show()
